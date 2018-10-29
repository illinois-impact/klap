/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "Analysis/KernelCallFinder.h"
#include "Driver/CompilerOptions.h"
#include "TransformCandidateFinder.h"
#include "Utils/Utils.h"

class TransformCandidateFinderInternal : public RecursiveASTVisitor<TransformCandidateFinderInternal> {

    public:

        TransformCandidateFinderInternal(std::set<CUDAKernelCallExpr*>& callFromHostCandidates, std::set<CUDAKernelCallExpr*>& callFromKernelCandidates, std::set<CallExpr*>& apiFromKernelCandidates, std::set<FunctionDecl*>& kernelsCalledFromKernelWithSmem)
            : callFromHostCandidates_(callFromHostCandidates), callFromKernelCandidates_(callFromKernelCandidates), apiFromKernelCandidates_(apiFromKernelCandidates), kernelsCalledFromKernelWithSmem_(kernelsCalledFromKernelWithSmem) { }

        bool VisitFunctionDecl(FunctionDecl* funcDecl) {

            if(funcDecl != NULL) {
                if(!funcDecl->getAttr<CUDAGlobalAttr>() && !funcDecl->getAttr<CUDADeviceAttr>()) { // If host function

                    // Find all the kernel calls in the host function
                    KernelCallFinder callsFromHostFinder(funcDecl);
                    std::set<CUDAKernelCallExpr*> callsFromHost = callsFromHostFinder.getKernelCalls();

                    // For each kernel call in the host function, find if it calls other kernels or API functions
                    for(CUDAKernelCallExpr* callFromHost : callsFromHost) {
                        KernelCallFinder callsFromKernelFinder(callFromHost->getDirectCallee());
                        if(callsFromKernelFinder.getKernelCalls().size() > 0
                           || (CompilerOptions::aggregateMallocFree() && callsFromKernelFinder.getAPICalls().size() > 0)) {
                            callFromHostCandidates_.insert(callFromHost);
                        }
                    }

                } else if(funcDecl->getAttr<CUDAGlobalAttr>()) {

                    // Find all the kernel calls in the kernel
                    KernelCallFinder callsFromKernelFinder(funcDecl);
                    std::set<CUDAKernelCallExpr*> callsFromKernel = callsFromKernelFinder.getKernelCalls();
                    for(CUDAKernelCallExpr* callFromKernel : callsFromKernel) {
                        callFromKernelCandidates_.insert(callFromKernel);
                        bool hasExplicitSmemConfig = !dyn_cast<CXXDefaultArgExpr>(callFromKernel->getConfig()->getArg(2));
                        if(hasExplicitSmemConfig) {
                            kernelsCalledFromKernelWithSmem_.insert(callFromKernel->getDirectCallee());
                        }
                    }
                    if(CompilerOptions::aggregateMallocFree()) {
                        std::set<CallExpr*> apisFromKernel = callsFromKernelFinder.getAPICalls();
                        for(CallExpr* CE : apisFromKernel) {
                            apiFromKernelCandidates_.insert(CE);
                        }
                    }

                }

            }
            return true;

        }

    private:

        std::set<CUDAKernelCallExpr*>& callFromHostCandidates_;
        std::set<CUDAKernelCallExpr*>& callFromKernelCandidates_;
        std::set<CallExpr*>& apiFromKernelCandidates_;
        std::set<FunctionDecl*>& kernelsCalledFromKernelWithSmem_;

};

TransformCandidateFinder::TransformCandidateFinder(TranslationUnitDecl* TU) {
    TransformCandidateFinderInternal transformCandidateFinder(callFromHostCandidates_, callFromKernelCandidates_, apiFromKernelCandidates_, kernelsCalledFromKernelWithSmem_);
    transformCandidateFinder.TraverseDecl(TU);
}

bool TransformCandidateFinder::isCallFromHostCandidate(CUDAKernelCallExpr* kernelCall) {
    return callFromHostCandidates_.count(kernelCall);
}

bool TransformCandidateFinder::isCallFromKernelCandidate(CUDAKernelCallExpr* kernelCall) {
    return callFromKernelCandidates_.count(kernelCall);
}

bool TransformCandidateFinder::isAPIFromKernelCandidate(CallExpr* CE) {
    return apiFromKernelCandidates_.count(CE);
}

bool TransformCandidateFinder::isCalledFromKernelWithSmem(FunctionDecl* kernel) {
    return kernelsCalledFromKernelWithSmem_.count(kernel);
}

