/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_SPECIAL_STMT_FINDER_H_
#define _KLAP_SPECIAL_STMT_FINDER_H_

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include <map>
#include <set>
#include <stack>
#include <string>

using namespace clang;

class SpecialStmtFinder {

    public:

        SpecialStmtFinder(TranslationUnitDecl* TU);

        /** Returns true if a kernel uses the gridDim special variable */
        bool usesGridDim(FunctionDecl* kernel);

    private:

        std::set<const FunctionDecl*> kernelsUsingGridDim_;

};

#endif

