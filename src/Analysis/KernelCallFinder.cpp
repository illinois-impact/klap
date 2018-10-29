/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "KernelCallFinder.h"

class KernelCallFinderInternal : public RecursiveASTVisitor<KernelCallFinderInternal> {

    public:

        KernelCallFinderInternal(std::set<CUDAKernelCallExpr*>& kernelCalls, std::set<CallExpr*>& apiCalls)
            : kernelCalls_(kernelCalls), apiCalls_(apiCalls) { }

        bool VisitStmt(Stmt *s) {
            if(CUDAKernelCallExpr* kernelCall = dyn_cast<CUDAKernelCallExpr>(s)) {
                kernelCalls_.insert(kernelCall);
            } else if(CallExpr* CE = dyn_cast<CallExpr>(s)) {
                if(FunctionDecl* f = CE->getDirectCallee()) {
                    std::string funcName = f->getDeclName().getAsString();
                    if(funcName == "cudaMalloc" || funcName == "cudaFree") {
                        apiCalls_.insert(CE);
                    }
                }
            }
            return true;
        }

    private:

        std::set<CUDAKernelCallExpr*>& kernelCalls_;
        std::set<CallExpr*>& apiCalls_;

};


KernelCallFinder::KernelCallFinder(FunctionDecl* f) {
    KernelCallFinderInternal finder(kernelCalls_, apiCalls_);
    finder.TraverseDecl(f);
}

std::set<CUDAKernelCallExpr*> KernelCallFinder::getKernelCalls() {
    return kernelCalls_;
}

std::set<CallExpr*> KernelCallFinder::getAPICalls() {
    return apiCalls_;
}

