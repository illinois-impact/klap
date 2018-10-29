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

#include "Host.h"
#include "Analysis/KernelCallFinder.h"
#include "Driver/CompilerOptions.h"
#include "Utils/Utils.h"

using namespace clang;

HostAGTransform::HostAGTransform(Rewriter& rewriter, Analyzer& analyzer)
    : rewriter_(rewriter), analyzer_(analyzer) {}

bool HostAGTransform::VisitStmt(Stmt *s) {

    if(CUDAKernelCallExpr* kernelCall = dyn_cast<CUDAKernelCallExpr>(s)) {
        if(analyzer_.isCallFromHostCandidate(kernelCall)) {

            /*
             * Transform the call to the original kernel:
             *
             *     kernel_name <<< gDim, bDim, smem >>> (p1, p2);
             *
             * into a call to the transformed kernel:
             *
             *      static __MemPool __memPool(memPoolSize_);
             *      __GridMemPool __memPoolHost(__memPool);
             *      __GridMemPool __memPoolDevice = __memPoolHost;
             *
             *      unsigned int __gDim = gDim;
             *      unsigned int __bDim = bDim;
             *
             *      // The following code is generated for each child kernel of kernel_name, where the child kernel signature is:
             *      //         __global__ void child_name(S1 q1, S2 q2)
             *      S1* q1_array_child = __memPoolHost.grid_allocate<S1>(__gDim*__bDim);
             *      S2* q2_array_child = __memPoolHost.grid_allocate<S2>(__gDim*__bDim);
             *      unsigned int* gDim_array_child = __memPoolHost.grid_allocate<unsigned int>(__gDim*__bDim);
             *      unsigned int* bDim_array_child = __memPoolHost.grid_allocate<unsigned int>(__gDim*__bDim);
             *      unsigned int* smem_array_child = __memPoolHost.grid_allocate<unsigned int>(__gDim*__bDim);
             *      cudaMemset(gDim_array_child, 0, 3*__gDim*__bDim*sizeof(unsigned int));
             *
             *      // The kernel is called
             *      kernel_name_kernel_h <<< __gDim, __bDim, smem >>> (p1, p2, __memPoolDevice);
             *
             *      // The following code is generated for each child kernel of kernel_name, where the child kernel signature is:
             *      //         __global__ void child_name(S1 q1, S2 q2)
             *      unsigned int __gDim_child = kernel_inclusive_scan(gDim_array_child, __gDim*__bDim);
             *      unsigned int __bDim_child = kernel_max(bDim_array_child, __gDim*__bDim);
             *      unsigned int __smem_child = kernel_max(smem_array_child, __gDim*__bDim);
             *      child_name_kernel_k_h(__gDim_child, __bDim_child, __smem_child, q1_array_child, q2_array_child, gDim_array_child, bDim_array_child, __gDim*__bDim, __memPoolHost);
             *
             */

            // Find all the child kernel calls in the kernel
            FunctionDecl* kernel = kernelCall->getDirectCallee();
            KernelCallFinder kcFinder(kernel);
            std::set<CUDAKernelCallExpr*> childKernelCalls = kcFinder.getKernelCalls();

            // Declare and initialize commonly used variables
            std::stringstream ss;
            CallExpr* config = kernelCall->getConfig();
            std::string gridDimConfig = toString(config->getArg(0));
            std::string blockDimConfig = toString(config->getArg(1));
            std::string smemConfig = toString(config->getArg(2));
            bool isSmemConfigExplicit = !dyn_cast<CXXDefaultArgExpr>(config->getArg(2));
            std::map<FunctionDecl*, std::vector<bool>> isScalarChildParam;
            std::map<FunctionDecl*, bool> isScalarChildBlockDim;
            std::map<FunctionDecl*, bool> isChildSmemConfigExplicit;
            std::map<FunctionDecl*, bool> isScalarChildSmemConfig;
            for(CUDAKernelCallExpr* childKernelCall : childKernelCalls) {
                FunctionDecl* childKernel = childKernelCall->getDirectCallee();
                isScalarChildParam[childKernel].resize(childKernel->getNumParams());
                for(unsigned int p = 0; p < childKernel->getNumParams(); ++p) {
                    isScalarChildParam[childKernel][p] = CompilerOptions::scalarizeInvariantParameters() && analyzer_.isParamInvariant(childKernel, p);
                }
                isScalarChildBlockDim[childKernel] = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isBlockDimInvariant(childKernel);
                isChildSmemConfigExplicit[childKernel] = !dyn_cast<CXXDefaultArgExpr>(childKernelCall->getConfig()->getArg(2));
                isScalarChildSmemConfig[childKernel] = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isInvariant(childKernelCall->getConfig()->getArg(2));
            }

            // Create a new scope to avoid name collisions
            ss << "{\n";

            // Create a memory pool
            ss << "static __MemPool __memPool(" << CompilerOptions::memoryPoolSize() << ");\n";
            ss << "__GridMemPool __memPoolHost(__memPool);\n";
            ss << "__GridMemPool __memPoolDevice = __memPoolHost;\n";

            // Store configurations in variables
            ss << "unsigned int __gDim = " << gridDimConfig << ";\n";
            ss << "unsigned int __bDim = " << blockDimConfig << ";\n";

            // Extract and initialize parameter/configuration arrays of all children
            for(CUDAKernelCallExpr* childKernelCall : childKernelCalls) {

                FunctionDecl* childKernel = childKernelCall->getDirectCallee();

                // Extract pointers to the arrays of each parameter
                for(unsigned int p = 0; p < childKernel->getNumParams(); ++p) {
                    ParmVarDecl* param = childKernel->getParamDecl(p);
                    std::string paramType = param->getType().getAsString();
                    std::string paramName = param->getNameAsString();
                    ss << paramType << "* " << paramName << "_array_child = __memPoolHost.grid_allocate<" << paramType << ">(\n";
                    if(isScalarChildParam[childKernel][p]) {
                        ss << "2";
                    } else {
                        ss << " __gDim*__bDim";
                    }
                    ss << ");\n";
                }

                // Extract pointers to the arrays of the configurations
                ss << "unsigned int* gDim_array_child = __memPoolHost.grid_allocate<unsigned int>(__gDim*__bDim);\n";
                std::string memsetLength = "(__gDim*__bDim";
                ss << "unsigned int* bDim_array_child = __memPoolHost.grid_allocate<unsigned int>(";
                if(isScalarChildBlockDim[childKernel]) {
                    ss << "2";
                    memsetLength += " + 2";
                } else {
                    ss << "__gDim*__bDim";
                    memsetLength += " + __gDim*__bDim";
                }
                ss << ");\n";
                if(isChildSmemConfigExplicit[childKernel]) {
                    ss << "unsigned int* smem_array_child = __memPoolHost.grid_allocate<unsigned int>(";
                    if(isScalarChildSmemConfig[childKernel]) {
                        ss << "2";
                        memsetLength += " + 2";
                    } else {
                        ss << "__gDim*__bDim";
                        memsetLength += " + __gDim*__bDim";
                    }
                    ss << ");\n";
                }
                memsetLength += ")*sizeof(unsigned int)";
                if(CompilerOptions::useAtomicsBasedScan()) {
                    ss << "union scan_counter* _sc_ = __memPoolHost.grid_allocate<union scan_counter>(1);\n";
                    memsetLength += " + sizeof(union scan_counter)";
                }
                ss << "cudaMemset(gDim_array_child, 0, " << memsetLength << ");\n\n";

            }

            // Kernel call
            ss << kernel->getNameAsString() << "_kernel_h <<< __gDim, __bDim";
            if(isSmemConfigExplicit) {
                ss << ", " << smemConfig;
            }
            ss << " >>> (";
            for(unsigned int a = 0; a < kernelCall->getNumArgs(); ++a) {
                std::string arg = toString(kernelCall->getArg(a));
                ss << arg << ", "; // Original arguments
            }
            ss << "__memPoolDevice);\n"; // Configuration arrays

            // Compute configurations and generate calls to all the children
            for(CUDAKernelCallExpr* childKernelCall : childKernelCalls) {

                FunctionDecl* childKernel = childKernelCall->getDirectCallee();

                // Compute configurations of aggregated kernel launch
                if(CompilerOptions::useAtomicsBasedScan()) {
                    ss << "union scan_counter sc_child; cudaMemcpy(&sc_child, _sc_, sizeof(union scan_counter), cudaMemcpyDeviceToHost);\n";
                    ss << "unsigned int __gDim_child = sc_child.nb;\n";
                } else {
                    ss << "unsigned int __gDim_child = kernel_inclusive_scan(gDim_array_child, __gDim*__bDim);\n";
                }
                if(CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isBlockDimInvariant(childKernel)) {
                    ss << "unsigned int __bDim_child; cudaMemcpy(&__bDim_child, &bDim_array_child[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);\n";
                } else {
                    ss << "unsigned int __bDim_child = kernel_max(bDim_array_child, __gDim*__bDim);\n";
                }
                if(isChildSmemConfigExplicit[childKernel]) {
                    if(isScalarChildSmemConfig[childKernel]) {
                        ss << "unsigned int __smem_child; cudaMemcpy(&__smem_child, &smem_array_child[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);\n";
                    } else {
                        ss << "unsigned int __smem_child = kernel_max(smem_array_child, __gDim*__bDim);\n";
                    }
                }

                // Call function that launches aggregated kernel
                ss << childKernel->getNameAsString() << "_kernel_k_h(__gDim_child, __bDim_child, ";
                if(isChildSmemConfigExplicit[childKernel]) {
                    ss << "__smem_child, ";
                }
                for(unsigned int p = 0; p < childKernel->getNumParams(); ++p) {
                    std::string paramName = childKernel->getParamDecl(p)->getNameAsString();
                    ss << paramName << "_array_child, ";
                }
                ss << "gDim_array_child, ";
                if(!CompilerOptions::scalarizeInvariantConfigurations() || !analyzer_.isBlockDimInvariant(childKernel)) {
                    ss << "bDim_array_child, ";
                }
                if(CompilerOptions::useAtomicsBasedScan()) {
                    ss << "sc_child.idx";
                } else {
                    ss << "__gDim*__bDim";
                }
                ss << ", __memPoolHost);\n"; // FIXME: Hold mem_pool for each child to pass the right mem_pool

            }

            // Close scope
            ss << "}\n";

            // Replace original call
            rewriter_.ReplaceText(SourceRange(kernelCall->getLocStart(), kernelCall->getLocEnd()), ss.str());

        }
    }

    return true;
}
