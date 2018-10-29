/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_ANALYZER_H_
#define _KLAP_ANALYZER_H_

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "Analysis/InvarianceAnalyzer.h"
#include "Analysis/TransformCandidateFinder.h"
#include "Analysis/SpecialStmtFinder.h"

class Analyzer {

    public:

        Analyzer(TranslationUnitDecl* TU);

        /** Returns true if a kernel call is from the host and is a candidate for transformation */
        bool isCallFromHostCandidate(CUDAKernelCallExpr* kernelCall);

        /** Returns true if a kernel call is from a kernel and is a candidate for transformation */
        bool isCallFromKernelCandidate(CUDAKernelCallExpr* kernelCall);

        /** Returns true if a kernel is called from another kernel in the translation unit with dynamic shared memory configured */
        bool isCalledFromKernelWithSmem(FunctionDecl* kernel);

        /** Returns true if an API call is from a kernel and is a candidate for transformation */
        bool isAPIFromKernelCandidate(CallExpr* CE);

        /** Returns true if a kernel's parameter is invariant in all callers in the translation unit */
        bool isParamInvariant(FunctionDecl* kernel, unsigned int paramIdx);

        /** Returns true if a kernel's block dimension is invariant in all callers in the translation unit */
        bool isBlockDimInvariant(FunctionDecl* kernel);

        /** Returns true if a statement or decl is invariant */
        bool isInvariant(Stmt* stmt);
        bool isInvariant(VarDecl* vdecl);

        /** Returns the stack of divergent control statements surrounding a kernel call */
        std::stack<const Stmt*> getDivergenceStack(const CallExpr* call);

        /** Returns true if a kernel uses the gridDim special variable */
        bool usesGridDim(FunctionDecl* kernel);

    private:

        TransformCandidateFinder candidateFinder_;
        InvarianceAnalyzer invarianceAnalyzer_;
        SpecialStmtFinder specialStmtFinder_;

};

#endif

