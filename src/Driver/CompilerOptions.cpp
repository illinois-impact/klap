/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "llvm/Support/raw_ostream.h"

#include "CompilerOptions.h"

llvm::cl::OptionCategory CompilerOptions::KLAPCategory("Kernel Launch Aggregation & Promotion (KLAP) options");

static llvm::cl::opt<std::string>
transformTypeOp("t",
        llvm::cl::desc("Transformation type (de, aw, ab, ag)"),
        llvm::cl::cat(CompilerOptions::KLAPCategory));

static llvm::cl::opt<std::string>
outFileNameOp("o",
        llvm::cl::desc("Output file name"),
        llvm::cl::cat(CompilerOptions::KLAPCategory));

static llvm::cl::opt<unsigned long long>
memPoolSizeOp("m",
        llvm::cl::desc("Memory pool size"),
        llvm::cl::init(1 << 30),
        llvm::cl::cat(CompilerOptions::KLAPCategory));

static llvm::cl::opt<bool>
useAtomicsBasedScanOp("a",
        llvm::cl::desc("Use atomics-based scan"),
        llvm::cl::init(true),
        llvm::cl::cat(CompilerOptions::KLAPCategory));

static llvm::cl::opt<bool>
scalarizeInvariantParametersOp("s",
        llvm::cl::desc("Scalairze invariant parameters"),
        llvm::cl::init(true),
        llvm::cl::cat(CompilerOptions::KLAPCategory));

static llvm::cl::opt<bool>
scalarizeInvariantConfigurationsOp("b",
        llvm::cl::desc("Scalairze invariant configurations"),
        llvm::cl::init(true),
        llvm::cl::cat(CompilerOptions::KLAPCategory));

static llvm::cl::opt<bool>
aggregateMallocFreeOp("g",
        llvm::cl::desc("Aggregate cudaMalloc/cudaFree"),
        llvm::cl::init(true),
        llvm::cl::cat(CompilerOptions::KLAPCategory));

CompilerOptions::TransformType CompilerOptions::transformType() {
    if(transformTypeOp == "de") {
        return DE;
    } else if(transformTypeOp == "aw") {
        return AW;
    } else if (transformTypeOp == "ab") {
        return AB;
    } else if (transformTypeOp == "ag") {
        return AG;
    } else if (transformTypeOp.empty()) {
        llvm::errs() << "No transform type provided.\n";
        llvm::errs() << "Use the -t option to provide a transform type.\n";
        llvm::errs() << "Possible values: de, aw, ab, ag.\n";
        exit(0);
    } else {
        llvm::errs() << "Unrecognized transformation type: " << transformTypeOp << ".\n";
        llvm::errs() << "Possible values: de, aw, ab, ag.\n";
        exit(0);
    }
}

void CompilerOptions::writeToOutputFile(Rewriter& rewriter, CompilerOptions::WriteMode mode) {
    SourceManager &SM = rewriter.getSourceMgr();
    if(outFileNameOp != "") {
        std::error_code EC;
        llvm::sys::fs::OpenFlags flags = llvm::sys::fs::F_Text;
        if(mode == APPEND) {
            flags |= llvm::sys::fs::F_Append;
        }
        llvm::raw_fd_ostream FileStream(outFileNameOp, EC, flags);
        if(mode == OVERWRITE && (transformType() == AW || transformType() == AB || transformType() == AG)) {
            FileStream << "#include \"klap.h\"\n";
        }
        if (EC) {
            llvm::errs() << "Error: Could not write to " << EC.message() << "\n";
        } else {
            rewriter.getEditBuffer(SM.getMainFileID()).write(FileStream);
        }
    } else {
        rewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());
    }
}

unsigned long long CompilerOptions::memoryPoolSize() {
    return memPoolSizeOp;
}

bool CompilerOptions::useAtomicsBasedScan() {
    return useAtomicsBasedScanOp;
}

bool CompilerOptions::scalarizeInvariantParameters() {
    return scalarizeInvariantParametersOp;
}

bool CompilerOptions::scalarizeInvariantConfigurations() {
    return scalarizeInvariantConfigurationsOp;
}

bool CompilerOptions::aggregateMallocFree() {
    return aggregateMallocFreeOp;
}

