/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <string>

#include "Driver/CompilerOptions.h"
#include "Kernel.h"
#include "Utils/Utils.h"

using namespace clang;

KernelAWBTransform::KernelAWBTransform(Rewriter& rewriter, Analyzer& analyzer, bool isCalledFromKernel)
    : rewriter_(rewriter), analyzer_(analyzer), isCalledFromKernel_(isCalledFromKernel) {}

std::string KernelAWBTransform::granularityAsString() {
    if(CompilerOptions::transformType() == CompilerOptions::AW) {
        return "warp";
    } else if(CompilerOptions::transformType() == CompilerOptions::AB) {
        return "block";
    } else {
        assert(0 && "Unreachable");
    }
}

bool KernelAWBTransform::VisitFunctionDecl(FunctionDecl *kernel) {

    if(kernel->getAttr<CUDAGlobalAttr>()) {
        if(isCalledFromKernel_) {

            /*
             * Transform the signature of the original kernel:
             *
             *      __global__ void kernel_name(T1 p1, T2 p2)
             *
             * into the signature of the transformed kernel called from a kernel:
             *
             *      __global__ void kernel_name_warp[block]_k(T1* p1_array, T2* p2_array,
             *                      unsigned int* __gDim_array, unsigned int* __bDim_array,
             *                      __GlobalMemPool __memPool, int __parent_size)
             *
             */

            // Declare and initialize commonly used variables
            std::stringstream ssParams;
            bool isScalarBlockDim = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isBlockDimInvariant(kernel);
            std::vector<bool> isScalarParam(kernel->getNumParams());
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                isScalarParam[p] = CompilerOptions::scalarizeInvariantParameters() && analyzer_.isParamInvariant(kernel, p);
            }

            // Append suffix to kernel name
            SourceLocation nameLoc = kernel->getNameInfo().getLoc();
            rewriter_.InsertTextAfterToken(nameLoc, "_" + granularityAsString() + "_k");

            // Change parameters to parameter arrays
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                ParmVarDecl* param = kernel->getParamDecl(p);
                ssParams << param->getType().getAsString();
                if(!isScalarParam[p]) {
                    ssParams << "*";
                }
                ssParams << " " << param->getNameAsString();
                if(!isScalarParam[p]) {
                    ssParams << "_array";
                }
                ssParams << ", ";
            }

            // Add grid and block dimension arrays to parameter list
            ssParams << "unsigned int* __gDim_array, ";
            if(!isScalarBlockDim) {
                ssParams << "unsigned int* __bDim_array, ";
            }

            // Add memory pool
            ssParams << "__GlobalMemPool __memPool";
            if(CompilerOptions::transformType() == CompilerOptions::AB) {
                ssParams << ", int __parent_size";
            }

            // Update parameters
            SourceLocation beginParamLoc = kernel->getParamDecl(0)->getSourceRange().getBegin();
            SourceLocation endParamLoc =
                kernel->getParamDecl(kernel->getNumParams() - 1)->getSourceRange().getEnd();
            rewriter_.ReplaceText(SourceRange(beginParamLoc, endParamLoc), ssParams.str());

            if(kernel->doesThisDeclarationHaveABody()) {

                /*
                 * Insert code around original kernel body to extract parameters:
                 *
                 *      unsigned int __parentIdx = __find_parent_idx_warp[block](__gDim_array, blockIdx.x, __parent_size);
                 *
                 *      T1 p1 = p1_array[__parentIdx];
                 *      T2 p2 = p2_array[__parentIdx];
                 *
                 *      unsigned int __gridDim_x =
                 *                      __gDim_array[__parentIdx] - ((__parentIdx == 0)?0:__gDim_array[__parentIdx - 1]);
                 *      unsigned int __blockDim_x = __bDim_array[__parentIdx];
                 *      unsigned int __blockIdx_x = blockIdx.x -
                 *                      ((__parentIdx == 0)?0:__gDim_array[__parentIdx - 1]);
                 *
                 *      if(threadIdx.x < __blockDim_x) {
                 *          <original kernel body>
                 *      }
                 *
                 */

                // Find parent index
                std::stringstream ssPrologue;
                ssPrologue << "unsigned int __parentIdx = __find_parent_idx_" << granularityAsString() << "(__gDim_array, blockIdx.x";
                if(CompilerOptions::transformType() == CompilerOptions::AB) {
                    ssPrologue << ", __parent_size";
                }
                ssPrologue << ");\n";

                // Load non-invariant parameters
                for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                    if(!isScalarParam[p]) {
                        ParmVarDecl* param = kernel->getParamDecl(p);
                        ssPrologue << param->getType().getAsString() << " "
                            << param->getNameAsString() << " = "
                            << param->getNameAsString() << "_array[__parentIdx];\n";
                    }
                }

                // Compute __gridDim_x, __blockDim_x, and __blockIdx_x
                if(analyzer_.usesGridDim(kernel)) {
                    ssPrologue << "unsigned int __gridDim_x = __gDim_array[__parentIdx] - ((__parentIdx == 0)?0:__gDim_array[__parentIdx - 1]);\n";
                }
                ssPrologue << "unsigned int __blockDim_x = ";
                if(isScalarBlockDim) {
                    ssPrologue << "blockDim.x;\n";
                } else {
                    ssPrologue << "__bDim_array[__parentIdx];\n";
                }
                ssPrologue << "unsigned int __blockIdx_x = blockIdx.x - ((__parentIdx == 0)?0:__gDim_array[__parentIdx - 1]);\n";

                // Guard kernel body with bounds check
                ssPrologue << "if(threadIdx.x < __blockDim_x) {\n";

                SourceLocation beginBody = cast<CompoundStmt>(kernel->getBody())->body_front()->getLocStart();
                rewriter_.InsertTextBefore(beginBody, ssPrologue.str());

                // Close guard
                std::stringstream ssEpilogue;
                ssEpilogue << ";\n}\n";
                SourceLocation endBody = cast<CompoundStmt>(kernel->getBody())->body_back()->getLocEnd();
                rewriter_.InsertTextAfterToken(endBody, ssEpilogue.str());

            }

        } else {

            /*
             * Transform the signature of the original kernel:
             *
             *      __global__ void kernel_name(T1 p1, T2 p2)
             *
             * into the signature of the transformed kernel called from the host:
             *
             *      __global__ void kernel_name_warp[block]_h(T1 p1, T2 p2, __GlobalMemPool __memPool)
             *
             */

            // Append _warp[block]_h to kernel name
            SourceLocation nameLoc = kernel->getNameInfo().getLoc();
            rewriter_.InsertTextAfterToken(nameLoc, "_" + granularityAsString() + "_h");

            // Add memory pool
            std::stringstream ssParams;
            ssParams << ", __GlobalMemPool __memPool";

            // Add new parameters to kernel declaration
            SourceLocation lastParamLoc =
                kernel->getParamDecl(kernel->getNumParams() - 1)->getSourceRange().getEnd();
            rewriter_.InsertTextAfterToken(lastParamLoc, ssParams.str());
        }
    } else if(!kernel->isImplicit()) {
        SourceManager& sm = rewriter_.getSourceMgr();
        if(sm.isInMainFile(sm.getExpansionLoc(kernel->getLocStart()))) {
            rewriter_.RemoveText(SourceRange(kernel->getLocStart(), kernel->getLocEnd()));
        }
    }

    return true;

}

bool KernelAWBTransform::VisitStmt(Stmt *s) {

    if(CUDAKernelCallExpr* kernelCall = dyn_cast<CUDAKernelCallExpr>(s)) {

        if(analyzer_.isCallFromKernelCandidate(kernelCall)) {

            /*
             * Transform the call to the original kernel:
             *
             *     kernel_name <<< gDim, bDim, smem >>> (p1, p2);
             *
             * into an aggregated call to the transformed kernel:
             *
             *      unsigned int _i_ = threadIdx.x%WARP_SIZE[threadIdx.x];
             *      unsigned int __granularity_size = WARP_SIZE[blockDim.x];
             *      __LocalMemPool __localPool = __memPool.warp[block]_allocate(__granularity_size*(sizeof(T1) + sizeof(T2) + 2*sizeof(unsigned int)));
             *      T1*           p1_array     = __localPool.allocate<T1>(__granularity_size);
             *      T2*           p2_array     = __localPool.allocate<T2>(__granularity_size);
             *      unsigned int* __gDim_array = __localPool.allocate<unsigned int>(__granularity_size);
             *      unsigned int* __bDim_array = __localPool.allocate<unsigned int>(__granularity_size);
             *      unsigned int* __smem_array = __localPool.allocate<unsigned int>(__granularity_size);
             *      p1_array[_i_] = p1;
             *      p2_array[_i_] = p2;
             *      __gDim_array[_i_] = gDim;
             *      __bDim_array[_i_] = bDim;
             *      __smem_array[_i_] = smem;
             *      unsigned int __sumGridDim = warp[block]_inclusive_scan(__gDim_array);
             *      unsigned int __maxBlockDim = warp[block]_max(__bDim_array);
             *      unsigned int __maxSmem = warp[block]_max(__smem_array);
             *      if(_i_ == __granularity_size - 1) {
             *          if(__sumGridDim > 0) {
             *              kernel_name_warp[block]_k <<< __sumGridDim, __maxBlockDim, __maxSmem >>>
             *                  (p1_array, p2_array, __gDim_array, __bDim_array, __memPool, __granularity_size);
             *          }
             *      }
             *
             */

            // Declare and initialize commonly used variables
            std::stringstream ss;
            FunctionDecl* kernel = kernelCall->getDirectCallee();
            CallExpr* config = kernelCall->getConfig();
            std::string gridDimConfig = toString(config->getArg(0));
            std::string blockDimConfig = toString(config->getArg(1));
            bool isScalarBlockDim = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isBlockDimInvariant(kernel);
            std::string smemConfig = toString(config->getArg(2));
            bool isSmemConfigExplicit = !dyn_cast<CXXDefaultArgExpr>(config->getArg(2));
            bool isScalarSmemConfig = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isInvariant(config->getArg(2));
            std::vector<bool> isScalarParam(kernel->getNumParams());
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                isScalarParam[p] = CompilerOptions::scalarizeInvariantParameters() && analyzer_.isParamInvariant(kernel, p);
            }

            // Create a new scope to avoid name collisions
            ss << "{\n";

            // Identity
            if(CompilerOptions::transformType() == CompilerOptions::AW) {
                ss << "unsigned int _i_ = threadIdx.x%WARP_SIZE;\n";
                ss << "unsigned int __granularity_size = WARP_SIZE;\n";
            } else {
                ss << "unsigned int _i_ = threadIdx.x;\n";
                ss << "unsigned int __granularity_size = ";
                if(isCalledFromKernel_) {
                    ss << "__blockDim_x";
                } else {
                    ss << "blockDim.x";
                }
                ss << ";\n";
            }

            // Allocate local memory pool
            ss << "__LocalMemPool __localPool = __memPool." << granularityAsString() << "_allocate(__granularity_size*(";
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                if(!isScalarParam[p]) {
                    ParmVarDecl* param = kernel->getParamDecl(p);
                    std::string paramType = param->getType().getAsString();
                    ss << "sizeof(" << paramType << ") + ";
                }
            }
            ss << "3*sizeof(unsigned int)));\n";

            // Allocate memory from local pool for kernel parameters and configurations
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                if(!isScalarParam[p]) {
                    ParmVarDecl* param = kernel->getParamDecl(p);
                    std::string paramType = param->getType().getAsString();
                    std::string paramName = param->getNameAsString();
                    ss << paramType << "* " << paramName << "_array = __localPool.allocate<" << paramType << ">(__granularity_size);\n";
                }
            }
            ss << "unsigned int* __gDim_array = __localPool.allocate<unsigned int>(__granularity_size);\n";
            if(!isScalarBlockDim) {
                ss << "unsigned int* __bDim_array = __localPool.allocate<unsigned int>(__granularity_size);\n";
            }
            if(isSmemConfigExplicit && !isScalarSmemConfig) {
                ss << "unsigned int* __smem_array = __localPool.allocate<unsigned int>(__granularity_size);\n";
            }

            // Store kernel parameters in arrays
            for(unsigned int a = 0; a < kernelCall->getNumArgs(); ++a) {
                if(!isScalarParam[a]) {
                    std::string arg = toString(kernelCall->getArg(a));
                    std::string paramName = kernel->getParamDecl(a)->getNameAsString();
                    ss << paramName << "_array[_i_] = " << arg << ";\n";
                }
            }

            // Store kernel configurations in arrays
            ss << "__gDim_array[_i_] = " << gridDimConfig << ";\n";
            if(!isScalarBlockDim) {
                ss << "__bDim_array[_i_] = " << blockDimConfig << ";\n";
            }
            if(isSmemConfigExplicit && !isScalarSmemConfig) {
                ss << "__smem_array[_i_] = " << smemConfig << ";\n";
            }

            // Scan numBlocks
            ss << "unsigned int __sumGridDim = " << granularityAsString() << "_inclusive_scan(__gDim_array);\n";

            // Max numThreads
            if(isScalarBlockDim) {
                ss << "unsigned int __maxBlockDim = " << blockDimConfig << ";\n";
            } else {
                ss << "unsigned int __maxBlockDim = " << granularityAsString() << "_max(__bDim_array);\n";
            }

            // Max smem size
            if(isSmemConfigExplicit) {
                if(isScalarSmemConfig) {
                    ss << "unsigned int __maxSmem = " << smemConfig << ";\n";
                } else {
                    ss << "unsigned int __maxSmem = " << granularityAsString() << "_max(__smem_array);\n";
                }
            }

            // Kernel call
            ss << "if(_i_ == __granularity_size - 1) {\n";
            ss << "if(__sumGridDim > 0) {\n";
            ss << kernel->getNameAsString() << "_" << granularityAsString() << "_k <<< __sumGridDim, __maxBlockDim";
            if(isSmemConfigExplicit) {
                ss << ", __maxSmem";
            }
            ss << " >>> (";
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                if(isScalarParam[p]) {
                    std::string arg = toString(kernelCall->getArg(p));
                    ss << arg << ", ";
                } else {
                    std::string paramName = kernel->getParamDecl(p)->getNameAsString();
                    ss << paramName << "_array, ";
                }
            }
            ss << "__gDim_array, ";
            if(!isScalarBlockDim) {
                ss << "__bDim_array, ";
            }
            ss << "__memPool";
            if(CompilerOptions::transformType() == CompilerOptions::AB) {
                ss << ", __granularity_size";
            }
            ss << ");\n";
            ss << "}\n";
            ss << "}\n";

            // Close scope
            ss << "}\n";

            // Replace original call
            rewriter_.ReplaceText(SourceRange(kernelCall->getLocStart(), kernelCall->getLocEnd()), ss.str());

        }

    } else if(CallExpr* CE = dyn_cast<CallExpr>(s)) {

        if(analyzer_.isAPIFromKernelCandidate(CE)) {
            if(FunctionDecl* f = CE->getDirectCallee()) {
                std::string funcName = f->getDeclName().getAsString();
                std::stringstream ss;
                ss << "__" << funcName << "_" << granularityAsString() << "(";
                for(unsigned int a = 0; a < CE->getNumArgs(); ++a) {
                    std::string arg = toString(CE->getArg(a));
                    if(a > 0) ss << ", ";
                    ss << arg;
                }
                ss << ")";
                rewriter_.ReplaceText(SourceRange(CE->getLocStart(), CE->getLocEnd()), ss.str());
            }
        }

    } else if(MemberExpr* memberExpr = dyn_cast<MemberExpr>(s)) {
        if(isCalledFromKernel_) {

            // Replace uses of gridDim.c, blockDim.x, and blockIdx.x with __gridDim_x, __blockDim_x, and __blockIdx_x
            SourceLocation beginExpr = memberExpr->getLocStart();
            SourceLocation endExpr = memberExpr->getLocEnd();
            if(toString(memberExpr) == "gridDim.__fetch_builtin_x") { // gridDim.x
                rewriter_.ReplaceText(SourceRange(beginExpr, endExpr), "__gridDim_x");
            } else if(toString(memberExpr) == "blockDim.__fetch_builtin_x") { // blockDim.x
                rewriter_.ReplaceText(SourceRange(beginExpr, endExpr), "__blockDim_x");
            } else if(toString(memberExpr) == "blockIdx.__fetch_builtin_x") { // blockIdx.x
                rewriter_.ReplaceText(SourceRange(beginExpr, endExpr), "__blockIdx_x");
            }
        }
    }

    return true;
}

