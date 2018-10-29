/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "InvarianceAnalyzer.h"
#include "Utils/Utils.h"

class StmtInvarianceAnalyzer : public ConstStmtVisitor<StmtInvarianceAnalyzer> {

    public:

        StmtInvarianceAnalyzer(std::map<const FunctionDecl*, std::set<unsigned int> >& variantKernelParams, std::set<const FunctionDecl*>& kernelsWithVariantBlockDim, std::set<const Stmt*>& variantStmts, std::set<const VarDecl*>& variantDecls, std::map<const CallExpr*, std::stack<const Stmt*> >& callExprDivergenceStack)
            : variantKernelParams_(variantKernelParams), kernelsWithVariantBlockDim_(kernelsWithVariantBlockDim), variantStmts_(variantStmts), variantDecls_(variantDecls), callExprDivergenceStack_(callExprDivergenceStack)  { }

        void VisitDeclStmt(const DeclStmt* stmt) {
            for(auto decl = stmt->decl_begin(); decl != stmt->decl_end(); ++decl) {
                if(const VarDecl* vdecl = dyn_cast<VarDecl>(*decl)) {
                    if (const Expr* init = vdecl->getInit()) {
                        Visit(init);
                        if(isVariant(init) || isContextDivergent()) {
                            setVariant(vdecl);
                        }
                    }
                }
            }
        }

        void VisitCompoundStmt(const CompoundStmt* stmt) {
            for(auto child = stmt->child_begin(); child != stmt->child_end(); ++child) {
                if (*child) {
                    Visit(*child);
                }
            }
        }

        void VisitCallExpr(const CallExpr* call) {

            const CUDAKernelCallExpr* kernelCall = dyn_cast<CUDAKernelCallExpr>(call);
            const FunctionDecl* callee = call->getDirectCallee();

            // Process arguments
            for(unsigned int a = 0; a < call->getNumArgs(); ++a) {
                const Expr* arg = call->getArg(a);
                Visit(arg);
                if(isVariant(arg)) {
                    setVariant(call);
                    if(kernelCall) {
                        variantKernelParams_[callee].insert(a);
                    }
                }
            }

            // Conservatively treat all calls to non-pure functions as variant
            if(!isPure(callee)) {
                setVariant(call);
            }

            // Handle L-value if C++ assignment
            if(const CXXOperatorCallExpr* cxxOpCall = dyn_cast<CXXOperatorCallExpr>(call)) {
                if(cxxOpCall->isAssignmentOp()) {
                    VisitLValue(cxxOpCall, cxxOpCall->getArg(0), cxxOpCall->getArg(1));
                }
            }

            // Process configs if kernel call
            if(kernelCall) {
                const CallExpr* config = kernelCall->getConfig();
                const Expr* gridDimConfig = config->getArg(0);
                Visit(gridDimConfig);
                const Expr* blockDimConfig = config->getArg(1);
                Visit(blockDimConfig);
                if(isVariant(blockDimConfig)) {
                    kernelsWithVariantBlockDim_.insert(callee);
                }
                const Expr* smemConfig = config->getArg(2);
                Visit(smemConfig);
            }

            // Track divergence stack if kernel or API call
            if(kernelCall) {
                callExprDivergenceStack_[kernelCall] = divergentControlStmts_;
            } else if(callee->getDeclName().getAsString() == "cudaMalloc") {
                callExprDivergenceStack_[call] = divergentControlStmts_;
            }

        }

        void VisitBinaryOperator(const BinaryOperator *BO) {
            const Expr* rhs = BO->getRHS();
            const Expr* lhs = BO->getLHS();
            Visit(rhs);
            Visit(lhs);
            if(isVariant(rhs) || isVariant(lhs)) {
                setVariant(BO);
            }
            if(BO->isAssignmentOp()) {
                VisitLValue(BO, lhs, rhs);
            }
        }

        void VisitDeclRefExpr(const DeclRefExpr* E) {
            std::string name = toString(E);
            if(name == "blockIdx" || name == "threadIdx") {
                setVariant(E); // TODO: distinguish between block-invariance and grid-invariance
            } else if(name == "blockDim" || name == "gridDim") {
                // Do nothing, reserved dims are invariant
            } else if(dyn_cast<EnumConstantDecl>(E->getDecl())) {
                // Do nothing, enum values are invariant
            } else if(const VarDecl* vdecl = dyn_cast<VarDecl>(E->getDecl())) {
                if(isVariant(vdecl)) {
                    setVariant(E);
                }
            } else {
                assert(0 && "Unsupported DeclRef");
            }
        }

        void VisitArraySubscriptExpr(const ArraySubscriptExpr* E) {
            const Expr* base = E->getBase();
            const Expr* idx = E->getIdx();
            Visit(base);
            Visit(idx);
            if(isVariant(base) || isVariant(idx) || isContextDivergent()) {
                setVariant(E);
            }
        }

        void VisitCastExpr(const CastExpr* E) {
            const Expr* sub = E->getSubExpr();
            Visit(sub);
            if(isVariant(sub)) {
                setVariant(E);
            }
        }

        void VisitParenExpr(const ParenExpr* E) {
            const Expr* sub = E->getSubExpr();
            Visit(sub);
            if(isVariant(sub)) {
                setVariant(E);
            }
        }

        void VisitIntegerLiteral(const IntegerLiteral *E) {
            // Do nothing, literal is invariant and has no subexpressions to visit
        }

        void VisitFloatingLiteral(const FloatingLiteral *E) {
            // Do nothing, literal is invariant and has no subexpressions to visit
        }

        void VisitStringLiteral(const StringLiteral *E) {
            // Do nothing, literal is invariant and has no subexpressions to visit
        }

        void VisitCharacterLiteral(const CharacterLiteral *E) {
            // Do nothing, literal is invariant and has no subexpressions to visit
        }

        void VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr* E) {
            // Do nothing, literal is invariant and has no subexpressions to visit
        }

        void VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
            // Do nothing, literal is invariant and has no subexpressions to visit
        }

        void VisitGNUNullExpr(const GNUNullExpr* expr) {
            // Do nothing, literal is invariant and has no subexpressions to visit
        }

        void VisitUnaryOperator(const UnaryOperator* E) {
            Expr* sub = E->getSubExpr();
            Visit(sub);
            if(isVariant(sub)) {
                setVariant(E);
            }
            if(E->isIncrementDecrementOp()) {
                VisitLValue(E, sub, sub);
            }
            // FIXME: If opcode is AddrOf '&', result should be variant and variable itself should be variant since can no longer track writes to it
        }

        void VisitReturnStmt(const ReturnStmt* Stmt) {
            const Expr* ret = Stmt->getRetValue();
            if(ret) {
                Visit(ret);
            }
        }

        void VisitMemberExpr(const MemberExpr* expr) {
            const Expr* base = expr->getBase();
            Visit(base);
            if(isVariant(base)) {
                setVariant(expr);
            }
        }

        void VisitNullStmt(const NullStmt* stmt) {
            // Do nothing, null statement has no effect
        }

        void VisitConditionalOperator(const ConditionalOperator* CO) {
            Expr* cond = CO->getCond();
            Expr* trueExpr = CO->getTrueExpr();
            Expr* falseExpr = CO->getFalseExpr();
            Visit(cond);
            if(isVariant(cond)) {
                pushDivergenceStack(CO);
            }
            Visit(trueExpr);
            Visit(falseExpr);
            if(isVariant(cond)) {
                popDivergenceStack();
            }
            if(isVariant(cond) || isVariant(trueExpr) || isVariant(falseExpr)) {
                setVariant(CO);
            }
        }

        void VisitIfStmt(const IfStmt* if_stmt) {
            const Expr* cond = if_stmt->getCond();
            const Stmt* thenExpr = if_stmt->getThen();
            const Stmt* elseExpr = if_stmt->getElse();
            Visit(cond);
            if(isVariant(cond)) {
                pushDivergenceStack(if_stmt);
            }
            Visit(thenExpr);
            if(elseExpr) {
                Visit(elseExpr);
            }
            if(isVariant(cond)) {
                popDivergenceStack();
            }
        }

        void VisitForStmt(const ForStmt* for_stmt) {
            const Stmt* init = for_stmt->getInit();
            const Expr* cond = for_stmt->getCond();
            const Stmt* body = for_stmt->getBody();
            const Expr* inc = for_stmt->getInc();
            Visit(init);
            Visit(cond);
            if(isVariant(cond)) {
                pushDivergenceStack(for_stmt);
            }
            Visit(body);
            Visit(inc);
            if(isVariant(cond)) {
                popDivergenceStack();
            }
        }

        void VisitWhileStmt(const WhileStmt* while_stmt) {
            const Expr* cond = while_stmt->getCond();
            const Stmt* body = while_stmt->getBody();
            Visit(cond);
            if(isVariant(cond)) {
                pushDivergenceStack(while_stmt);
            }
            Visit(body);
            if(isVariant(cond)) {
                popDivergenceStack();
            }
        }

        void VisitDoStmt(const DoStmt* while_stmt) {
            const Expr* cond = while_stmt->getCond();
            const Stmt* body = while_stmt->getBody();
            Visit(cond);
            if(isVariant(cond)) {
                pushDivergenceStack(while_stmt);
            }
            Visit(body);
            if(isVariant(cond)) {
                popDivergenceStack();
            }
        }

        void VisitContinueStmt(const ContinueStmt* C) {
            assert(0 && "TODO: unsupported continue statement");
        }

        void VisitBreakStmt(const BreakStmt* B) {
            assert(0 && "TODO: unsupported break statement");
        }

        void VisitCaseStmt(const CaseStmt* stmt) {
            assert(0 && "TODO: unsupported case statement");
        }

        void VisitSwitchStmt(const SwitchStmt* stmt) {
            assert(0 && "TODO: unsupported switch statement");
        }

        void VisitDefaultStmt(const DefaultStmt* stmt) {
            assert(0 && "TODO: unsupported default statement");
        }

        void VisitPseudoObjectExpr(const PseudoObjectExpr* expr) {
            const Expr* synExpr = expr->getSyntacticForm();
            Visit(synExpr);
            if(isVariant(synExpr)) {
                setVariant(expr);
            }
        }

        void VisitMSPropertyRefExpr(const MSPropertyRefExpr* expr) {
            const Expr* base = expr->getBaseExpr();
            Visit(base);
            if(isVariant(base)) {
                setVariant(expr);
            }
        }

        void VisitOpaqueValueExpr(const OpaqueValueExpr* expr) {
            const Expr* source = expr->getSourceExpr();
            Visit(source);
            if(isVariant(source)) {
                setVariant(expr);
            }
        }

        void VisitInitListExpr(const InitListExpr* E) {
            for(unsigned I = 0; I != E->getNumInits(); ++I) {
                const Expr* expr = E->getInit(I);
                Visit(expr);
                if(isVariant(expr)) {
                    setVariant(E);
                }
            }
        }

        void VisitExprWithCleanups(const ExprWithCleanups* E) {
            const Expr* sub = E->getSubExpr();
            Visit(sub);
            if(isVariant(sub)) {
                setVariant(E);
            }
        }

        void VisitCXXConstructExpr(const CXXConstructExpr* E) {
            for(unsigned int a = 0; a < E->getNumArgs(); ++a) {
                const Expr* arg = E->getArg(a);
                Visit(arg);
                if(isVariant(arg)) {
                    setVariant(E);
                }
            }
        }

        void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr* expr) {
            const Expr* sub = expr->GetTemporaryExpr();
            Visit(sub);
            if(isVariant(sub)) {
                setVariant(expr);
            }
        }

        void VisitAttributedStmt(const AttributedStmt* stmt) {
            const Stmt* sub = stmt->getSubStmt();
            Visit(sub);
            if(isVariant(sub)) {
                setVariant(stmt);
            }
        }

        void VisitCXXDefaultArgExpr(const CXXDefaultArgExpr* expr) {
            // Do nothing, treat default arguments are invariant
        }

        void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr* expr) {
            // Do nothing, treat operations on types as invariant
        }

        void VisitStmt(const Stmt* stmt) {
            stmt->dumpColor();
            llvm::errs() << toString(stmt) << "\n";
            assert(0 && "Unsupported statement!");
        }

        bool hasChanged() { return changed_; }

        void resetChanged() { changed_ = false; }

    private:

        std::map<const FunctionDecl*, std::set<unsigned int> >& variantKernelParams_;
        std::set<const FunctionDecl*>& kernelsWithVariantBlockDim_;
        std::set<const Stmt*>& variantStmts_;
        std::set<const VarDecl*>& variantDecls_;
        std::map<const CallExpr*, std::stack<const Stmt*> >& callExprDivergenceStack_;
        bool changed_;
        std::stack<const Stmt*> divergentControlStmts_;

        bool isVariant(const Stmt* stmt) {
            return variantStmts_.count(stmt);
        }

        void setVariant(const Stmt* stmt) {
            if(!variantStmts_.count(stmt)) {
                variantStmts_.insert(stmt);
                changed_ = true;
            }
        }

        bool isVariant(const VarDecl* decl) {
            return variantDecls_.count(decl);
        }

        void setVariant(const VarDecl* decl) {
            if(!variantDecls_.count(decl)) {
                variantDecls_.insert(decl);
                changed_ = true;
            }
        }

        void pushDivergenceStack(const Stmt* stmt) {
            divergentControlStmts_.push(stmt);
        }

        void popDivergenceStack() {
            divergentControlStmts_.pop();
        }

        bool isContextDivergent() {
            return !divergentControlStmts_.empty();
        }

        bool isPure(const FunctionDecl* f) {
            std::string funcName = f->getDeclName().getAsString();
            if(funcName == "ceil") {
                return true;
            } else {
                return false;
            }
        }

        void VisitLValue(const Expr* E, const Expr* lhs, const Expr* rhs) {
            if(isContextDivergent()) {
                setVariant(E);
            }
            if(isVariant(rhs) || isContextDivergent()) {
                if(const DeclRefExpr* lhsDeclRef = dyn_cast<DeclRefExpr>(lhs)) {
                    if(const VarDecl* vdecl = dyn_cast<VarDecl>(lhsDeclRef->getDecl())) {
                        setVariant(vdecl);
                    } else {
                        assert(0 && "Unsupported assignment to DeclRef");
                    }
                } else if(const MemberExpr* memberExpr = dyn_cast<MemberExpr>(lhs)) {
                    const Expr* base = memberExpr->getBase();
                    VisitLValue(E, base, rhs); // Modifying individual field is like modifying entire object
                } else if(dyn_cast<ArraySubscriptExpr>(lhs)) {
                    // Do nothing, modified value is memory location not local variable
                } else if(const UnaryOperator* UO = dyn_cast<UnaryOperator>(lhs)) {
                    if(UO->getOpcode() == UO_Deref) {
                        // Do nothing, modified value is memory location not local variable
                    } else {
                        assert(0 && "Unsuppoerted lvalue");
                    }
                } else {
                    assert(0 && "Unsuppoerted lvalue");
                }
            }
        }

};

class InvarianceAnalyzerInternal : public RecursiveASTVisitor<InvarianceAnalyzerInternal> {

    public:

        InvarianceAnalyzerInternal(std::map<const FunctionDecl*, std::set<unsigned int> >& variantKernelParams, std::set<const FunctionDecl*>& kernelsWithVariantBlockDim, std::set<const Stmt*>& variantStmts, std::set<const VarDecl*>& variantDecls, std::map<const CallExpr*, std::stack<const Stmt*> >& callExprDivergenceStack)
            : variantKernelParams_(variantKernelParams), kernelsWithVariantBlockDim_(kernelsWithVariantBlockDim), variantStmts_(variantStmts), variantDecls_(variantDecls), callExprDivergenceStack_(callExprDivergenceStack) { }

        bool VisitFunctionDecl(FunctionDecl* funcDecl) {

            if(funcDecl->getAttr<CUDAGlobalAttr>() && funcDecl->doesThisDeclarationHaveABody()) {
                StmtInvarianceAnalyzer analyzer(variantKernelParams_, kernelsWithVariantBlockDim_, variantStmts_, variantDecls_, callExprDivergenceStack_);
                Stmt* body = funcDecl->getBody();
                assert(body != NULL);
                do {
                    analyzer.resetChanged();
                    analyzer.Visit(body);
                } while(analyzer.hasChanged());
            }
            return true;

        }

    private:

        std::map<const FunctionDecl*, std::set<unsigned int> >& variantKernelParams_;
        std::set<const FunctionDecl*>& kernelsWithVariantBlockDim_;
        std::set<const Stmt*>& variantStmts_;
        std::set<const VarDecl*>& variantDecls_;
        std::map<const CallExpr*, std::stack<const Stmt*> >& callExprDivergenceStack_;

};

InvarianceAnalyzer::InvarianceAnalyzer(TranslationUnitDecl* TU) {
    InvarianceAnalyzerInternal invarianceAnalyzer(variantKernelParams_, kernelsWithVariantBlockDim_, variantStmts_, variantDecls_, callExprDivergenceStack_);
    invarianceAnalyzer.TraverseDecl(TU);
}

bool InvarianceAnalyzer::isParamInvariant(FunctionDecl* kernel, unsigned int paramIdx) {
    return !variantKernelParams_[kernel].count(paramIdx);
}

bool InvarianceAnalyzer::isBlockDimInvariant(FunctionDecl* kernel) {
    return !kernelsWithVariantBlockDim_.count(kernel);
}

bool InvarianceAnalyzer::isInvariant(Stmt* stmt) {
    return !variantStmts_.count(stmt);
}

bool InvarianceAnalyzer::isInvariant(VarDecl* vdecl) {
    return !variantDecls_.count(vdecl);
}

std::stack<const Stmt*> InvarianceAnalyzer::getDivergenceStack(const CallExpr* call) {
    return callExprDivergenceStack_[call];
}

