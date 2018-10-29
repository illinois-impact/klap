/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_TRANSFORM_CANDIDATE_FINDER_H_
#define _KLAP_TRANSFORM_CANDIDATE_FINDER_H_

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include <map>
#include <set>
#include <string>

using namespace clang;

class TransformCandidateFinder {

    public:

        TransformCandidateFinder(TranslationUnitDecl* TU);

        /** Returns true if a kernel call is from the host and is a candidate for transformation */
        bool isCallFromHostCandidate(CUDAKernelCallExpr* kernelCall);

        /** Returns true if a kernel call is from a kernel and is a candidate for transformation */
        bool isCallFromKernelCandidate(CUDAKernelCallExpr* kernelCall);

        /** Returns true if an API call is from a kernel and is a candidate for transformation */
        bool isAPIFromKernelCandidate(CallExpr* CE);

        /** Returns true if a kernel is called from another kernel in the translation unit with dynamic shared memory configured */
        bool isCalledFromKernelWithSmem(FunctionDecl* kernel);

    private:

        std::set<CUDAKernelCallExpr*> callFromHostCandidates_;
        std::set<CUDAKernelCallExpr*> callFromKernelCandidates_;
        std::set<CallExpr*> apiFromKernelCandidates_;
        std::set<FunctionDecl*> kernelsCalledFromKernelWithSmem_;

};

#endif

