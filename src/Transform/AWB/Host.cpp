/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <string>

#include "Driver/CompilerOptions.h"
#include "Host.h"
#include "Utils/Utils.h"

using namespace clang;

HostAWBTransform::HostAWBTransform(Rewriter& rewriter, Analyzer& analyzer)
    : rewriter_(rewriter), analyzer_(analyzer) {}

std::string HostAWBTransform::granularityAsString() {
    if(CompilerOptions::transformType() == CompilerOptions::AW) {
        return "warp";
    } else if(CompilerOptions::transformType() == CompilerOptions::AB) {
        return "block";
    } else {
        assert(0 && "Unreachable");
    }
}

bool HostAWBTransform::VisitStmt(Stmt *s) {

    if(CUDAKernelCallExpr* kernelCall = dyn_cast<CUDAKernelCallExpr>(s)) {
        if(analyzer_.isCallFromHostCandidate(kernelCall)) {
            /*
             * Transform the call to the original kernel:
             *
             *     kernel_name <<< gDim, bDim, smem >>> (p1, p2);
             *
             * into a call to the transformed kernel:
             *
             *     static __MemPool __memPool(memPoolSize_);
             *     __memPool.reset();
             *
             *     kernel_name_warp[block]_h <<< gDim, bDim, smem >>> (p1, p2, __memPool);
             *
             */

            std::stringstream ss;

            // Create a new scope to avoid name collisions
            ss << "{\n";

            // Create a memory pool
            ss << "static __MemPool __memPool(" << CompilerOptions::memoryPoolSize() << ");";
            ss << "__memPool.reset();\n";

            // Change the name of the called kernel
            std::string kernelName = kernelCall->getDirectCallee()->getNameAsString();
            ss << kernelName << "_" << granularityAsString() << "_h";

            // Keep configurations and arguments as is
            CallExpr* config = kernelCall->getConfig();
            ss << " <<< ";
            for(unsigned int a = 0; a < config->getNumArgs(); ++a) {
                Expr* arg = config->getArg(a);
                if(!dyn_cast<CXXDefaultArgExpr>(arg)) {
                    if(a > 0) ss << ", ";
                    ss << toString(arg);
                }
            }
            ss << " >>> (";
            for(unsigned int a = 0; a < kernelCall->getNumArgs(); ++a) {
                if(a > 0) ss << ", ";
                ss << toString(kernelCall->getArg(a));
            }

            // Append memory pool argument
            ss << ", __memPool);\n";

            // Close scope
            ss << "}\n";

            // Replace original call
            rewriter_.ReplaceText(SourceRange(kernelCall->getLocStart(), kernelCall->getLocEnd()), ss.str());

        }
    }

    return true;
}
