/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "SpecialStmtFinder.h"
#include "Utils/Utils.h"

class StmtSpecialStmtFinder : public RecursiveASTVisitor<StmtSpecialStmtFinder> {

    public:

        StmtSpecialStmtFinder() { }

        bool VisitMemberExpr(const MemberExpr* expr) {
            if(toString(expr) == "gridDim.__fetch_builtin_x") {
                kernelUsesGridDim_ = true;
            }
            return true;
        }

        bool kernelUsesGridDim() { return kernelUsesGridDim_; }

    private:

        bool kernelUsesGridDim_ = false;

};

class SpecialStmtFinderInternal : public RecursiveASTVisitor<SpecialStmtFinderInternal> {

    public:

        SpecialStmtFinderInternal(std::set<const FunctionDecl*>& kernelsUsingGridDim)
            : kernelsUsingGridDim_(kernelsUsingGridDim) { }

        bool VisitFunctionDecl(FunctionDecl* funcDecl) {

            if(funcDecl->getAttr<CUDAGlobalAttr>() && funcDecl->doesThisDeclarationHaveABody()) {
                StmtSpecialStmtFinder finder;
                finder.TraverseDecl(funcDecl);
                if(finder.kernelUsesGridDim()) {
                    kernelsUsingGridDim_.insert(funcDecl);
                }
            }
            return true;

        }

    private:

        std::set<const FunctionDecl*>& kernelsUsingGridDim_;

};

SpecialStmtFinder::SpecialStmtFinder(TranslationUnitDecl* TU) {
    SpecialStmtFinderInternal specialStmtFinder(kernelsUsingGridDim_);
    specialStmtFinder.TraverseDecl(TU);
}

bool SpecialStmtFinder::usesGridDim(FunctionDecl* kernel) {
    return kernelsUsingGridDim_.count(kernel);
}

