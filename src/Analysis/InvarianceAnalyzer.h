/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_INVARIANCE_ANALYZER_H_
#define _KLAP_INVARIANCE_ANALYZER_H_

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include <map>
#include <set>
#include <stack>
#include <string>

using namespace clang;

class InvarianceAnalyzer {

    public:

        InvarianceAnalyzer(TranslationUnitDecl* TU);

        /** Returns true if a kernel's parameter is invariant in all callers in the translation unit */
        bool isParamInvariant(FunctionDecl* kernel, unsigned int paramIdx);

        /** Returns true if a kernel's block dimension is invariant in all callers in the translation unit */
        bool isBlockDimInvariant(FunctionDecl* kernel);

        /** Returns true if a statement or decl is invariant */
        bool isInvariant(Stmt* stmt);
        bool isInvariant(VarDecl* vdecl);

        /** Returns the stack of divergent control statements surrounding a kernel call or API call from a kernel */
        std::stack<const Stmt*> getDivergenceStack(const CallExpr* call);

    private:

        std::map<const FunctionDecl*, std::set<unsigned int> > variantKernelParams_;
        std::set<const FunctionDecl*> kernelsWithVariantBlockDim_;
        std::set<const Stmt*> variantStmts_;
        std::set<const VarDecl*> variantDecls_;
        std::map<const CallExpr*, std::stack<const Stmt*> > callExprDivergenceStack_;

};

#endif

