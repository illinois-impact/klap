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

DivergenceEliminationTransform::DivergenceEliminationTransform(Rewriter& rewriter, Analyzer& analyzer)
    : rewriter_(rewriter), analyzer_(analyzer) {}

bool DivergenceEliminationTransform::VisitStmt(Stmt *s) {

    if(CUDAKernelCallExpr* kernelCall = dyn_cast<CUDAKernelCallExpr>(s)) {

        if(analyzer_.isCallFromKernelCandidate(kernelCall)) {

            // Declare and initialize commonly used variables
            std::stringstream ssBefore, ssReplace, ssAfter;
            FunctionDecl* kernel = kernelCall->getDirectCallee();
            CallExpr* config = kernelCall->getConfig();
            Expr* gridDimConfig = config->getArg(0);
            Expr* blockDimConfig = config->getArg(1);
            Expr* smemConfig = config->getArg(2);

            // Find outermost divergent control statement
            std::stack<const Stmt*> divergenceStack = analyzer_.getDivergenceStack(kernelCall);
            const Stmt* outermostDivergentStmt = NULL;
            if(divergenceStack.empty()) {
                return true; // No transformation is needed because context is not divergent
            } else {
                do {
                    outermostDivergentStmt = divergenceStack.top();
                    if(!dyn_cast<IfStmt>(outermostDivergentStmt)) {
                        assert(0 && "Divergent loops are not supported!");
                    }
                    divergenceStack.pop();
                } while(!divergenceStack.empty());
                // FIXME: Check if there is a call to cudaDeviceSynchronize() between the kernel call and the end of the outermost divergent control statement to ensure that postponing is legal
            }

            // Declare variables to store invariant configs
            ssBefore << "unsigned int __nb = 0;\n";
            if(!analyzer_.isInvariant(blockDimConfig)) {
                ssBefore << "unsigned int __nt;\n";
            }
            if(!dyn_cast<CXXDefaultArgExpr>(smemConfig)) {
                if(!analyzer_.isInvariant(smemConfig)) {
                    ssBefore << "unsigned int __sm;\n";
                }
            }

            // Declare variables to store variant parameters
            for(unsigned int a = 0; a < kernelCall->getNumArgs(); ++a) {
                if(!analyzer_.isInvariant(kernelCall->getArg(a))) {
                    std::string paramType = kernel->getParamDecl(a)->getType().getAsString();
                    ssBefore << paramType << " __param" << a << ";\n";
                }
            }

            // Store variant configs
            ssReplace << "__nb = " << toString(gridDimConfig) << ";\n";
            if(!analyzer_.isInvariant(blockDimConfig)) {
                ssReplace << "__nt = " << toString(blockDimConfig) << ";\n";
            }
            if(!dyn_cast<CXXDefaultArgExpr>(smemConfig)) {
                if(!analyzer_.isInvariant(smemConfig)) {
                    ssReplace << "__sm = " << toString(smemConfig) << ";\n";
                }
            }

            // Store variant parameters
            for(unsigned int a = 0; a < kernelCall->getNumArgs(); ++a) {
                if(!analyzer_.isInvariant(kernelCall->getArg(a))) {
                    std::string arg = toString(kernelCall->getArg(a));
                    ssReplace << "__param" << a << " = " << arg << ";\n";
                }
            }

            // Move original kernel replacing variant parameters
            ssAfter << "\n" << kernel->getNameAsString() << " <<< __nb, ";
            if(analyzer_.isInvariant(blockDimConfig)) {
                ssAfter << toString(blockDimConfig);
            } else {
                ssAfter << "__nt";
            }
            if(!dyn_cast<CXXDefaultArgExpr>(smemConfig)) {
                if(analyzer_.isInvariant(smemConfig)) {
                    ssAfter << ", " << toString(smemConfig);
                } else {
                    ssAfter << ", __sm";
                }
            }
            ssAfter << " >>> (";
            for(unsigned int a = 0; a < kernelCall->getNumArgs(); ++a) {
                if(a > 0) ssAfter << ", ";
                if(analyzer_.isInvariant(kernelCall->getArg(a))) {
                    ssAfter << toString(kernelCall->getArg(a));
                } else {
                    ssAfter << "__param" << a;
                }
            }
            ssAfter << ");\n";

            // Rewrite
            rewriter_.InsertTextBefore(outermostDivergentStmt->getLocStart(), ssBefore.str());
            rewriter_.ReplaceText(SourceRange(kernelCall->getLocStart(), kernelCall->getLocEnd()), ssReplace.str());
            rewriter_.InsertTextAfterToken(outermostDivergentStmt->getLocEnd(), ssAfter.str());

        }

    } else if(CallExpr* CE = dyn_cast<CallExpr>(s)) {

        if(analyzer_.isAPIFromKernelCandidate(CE)) {
            if(FunctionDecl* f = CE->getDirectCallee()) {
                if(f->getDeclName().getAsString() == "cudaMalloc") {

                    std::stringstream ssBefore, ssReplace, ssAfter;

                    // Find outermost divergent control statement
                    std::stack<const Stmt*> divergenceStack = analyzer_.getDivergenceStack(CE);
                    const Stmt* outermostDivergentStmt = NULL;
                    if(divergenceStack.empty()) {
                        return true; // No transformation is needed because context is not divergent
                    } else {
                        do {
                            outermostDivergentStmt = divergenceStack.top();
                            if(!dyn_cast<IfStmt>(outermostDivergentStmt)) {
                                assert(0 && "Divergent loops are not supported!");
                            }
                            divergenceStack.pop();
                        } while(!divergenceStack.empty());
                        // FIXME: Check if there is a call to cudaDeviceSynchronize() between the API call and the end of the outermost divergent control statement to ensure that postponing is legal
                    }

                    // Declare variables to store parameters
                    ssBefore << "void** __malloc_ptr = NULL;\n";
                    ssBefore << "size_t __malloc_size = 0;\n";

                    // Store parameters
                    ssReplace << "__malloc_ptr = " << toString(CE->getArg(0)) << ";\n";
                    ssReplace << "__malloc_size = " << toString(CE->getArg(1)) << ";\n";

                    // Move original call replacing parameters
                    ssAfter << "\ncudaMalloc(__malloc_ptr, __malloc_size);\n";

                    // Rewrite
                    rewriter_.InsertTextBefore(outermostDivergentStmt->getLocStart(), ssBefore.str());
                    rewriter_.ReplaceText(SourceRange(CE->getLocStart(), CE->getLocEnd()), ssReplace.str());
                    rewriter_.InsertTextAfterToken(outermostDivergentStmt->getLocEnd(), ssAfter.str());

                } else if(f->getDeclName().getAsString() == "cudaFree") {
                    // Do nothing, aggregated cudaFree works in the presence of divergence
                } else {
                    assert(0 && f && "Unreachable");
                }
            }
        }

    }

    return true;
}

