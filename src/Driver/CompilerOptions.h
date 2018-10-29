/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_COMPILER_OPTIONS_H_
#define _KLAP_COMPILER_OPTIONS_H_

#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;

class CompilerOptions {

    public:

        static llvm::cl::OptionCategory KLAPCategory;

        enum TransformType {
            DE,             // Divergence elimination pre-processing step
            AW, AB, AG      // Aggregation levels of granularity
        };

        enum WriteMode { OVERWRITE, APPEND };

        /* Returns the type of the transformation being performed */
        static TransformType transformType();

        /* Writes to the output file */
        static void writeToOutputFile(Rewriter& rewriter, CompilerOptions::WriteMode mode);

        /* Returns the size of the memory pool to be allocated from the host */
        static unsigned long long memoryPoolSize();

        /* Returns whether to use the atoms-based scan */
        static bool useAtomicsBasedScan();

        /* Returns whether to scalarize invariant parameters during aggregation */
        static bool scalarizeInvariantParameters();

        /* Returns whether to scalarize invariant configurations during aggregation */
        static bool scalarizeInvariantConfigurations();

        /* Returns whether to aggregate calls to cudaMalloc */
        static bool aggregateMallocFree();

    private:


};

#endif

