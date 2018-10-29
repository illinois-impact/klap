/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_UTILS_H_
#define _KLAP_UTILS_H_

#include "clang/AST/AST.h"

#include <string>

using namespace clang;

/** Prints a statement to a string */
std::string toString(const Stmt* s);

#endif
