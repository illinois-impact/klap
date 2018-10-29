/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_HOST_KT_TRANSFORM_H_
#define _KLAP_HOST_KT_TRANSFORM_H_

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include <set>

#include "Analysis/Analyzer.h"

using namespace clang;

class HostAGTransform : public RecursiveASTVisitor<HostAGTransform> {

    public:

        HostAGTransform(Rewriter& rewriter, Analyzer& analyzer);

        bool VisitStmt(Stmt *s);

    private:

        Rewriter &rewriter_;
        Analyzer& analyzer_;

};

#endif
