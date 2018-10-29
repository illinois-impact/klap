/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_KERNEL_CALL_FINDER_H_
#define _KLAP_KERNEL_CALL_FINDER_H_

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include <set>

using namespace clang;

/** Finds kernel and API calls contained in a function/kernel */
class KernelCallFinder {

    public:

        KernelCallFinder(FunctionDecl* f);

        /** Returns the set of kernel calls found in the function. */
        std::set<CUDAKernelCallExpr*> getKernelCalls();

        /** Returns the set of calls to the CUDA runtime API found in the function. */
        std::set<CallExpr*> getAPICalls();

    private:

        std::set<CUDAKernelCallExpr*> kernelCalls_;
        std::set<CallExpr*> apiCalls_;

};

#endif

