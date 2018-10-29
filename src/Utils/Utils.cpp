/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "llvm/Support/raw_ostream.h"

#include "Utils.h"

std::string toString(const Stmt* s) {
    std::string ss;
    llvm::raw_string_ostream os(ss);
    LangOptions lo;
    lo.CPlusPlus = true;
    clang::PrintingPolicy pp(lo);
    s->printPretty(os, NULL, pp);
    return os.str();
}
