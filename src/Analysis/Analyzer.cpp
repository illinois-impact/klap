/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "Analysis/Analyzer.h"

Analyzer::Analyzer(TranslationUnitDecl* TU)
    : candidateFinder_(TU), invarianceAnalyzer_(TU), specialStmtFinder_(TU) { }

bool Analyzer::isCallFromHostCandidate(CUDAKernelCallExpr* kernelCall){
    return candidateFinder_.isCallFromHostCandidate(kernelCall);
}

bool Analyzer::isCallFromKernelCandidate(CUDAKernelCallExpr* kernelCall){
    return candidateFinder_.isCallFromKernelCandidate(kernelCall);
}

bool Analyzer::isCalledFromKernelWithSmem(FunctionDecl* kernel) {
    return candidateFinder_.isCalledFromKernelWithSmem(kernel);
}

bool Analyzer::isAPIFromKernelCandidate(CallExpr* CE){
    return candidateFinder_.isAPIFromKernelCandidate(CE);
}

bool Analyzer::isParamInvariant(FunctionDecl* kernel, unsigned int paramIdx){
    return invarianceAnalyzer_.isParamInvariant(kernel, paramIdx);
}

bool Analyzer::isBlockDimInvariant(FunctionDecl* kernel){
    return invarianceAnalyzer_.isBlockDimInvariant(kernel);
}

bool Analyzer::isInvariant(Stmt* stmt) {
    return invarianceAnalyzer_.isInvariant(stmt);
}

bool Analyzer::isInvariant(VarDecl* vdecl) {
    return invarianceAnalyzer_.isInvariant(vdecl);
}

std::stack<const Stmt*> Analyzer::getDivergenceStack(const CallExpr* call) {
    return invarianceAnalyzer_.getDivergenceStack(call);
}

bool Analyzer::usesGridDim(FunctionDecl* kernel) {
    return specialStmtFinder_.usesGridDim(kernel);
}

