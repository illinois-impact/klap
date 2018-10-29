
set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(klap
  Analysis/Analyzer.cpp
  Analysis/InvarianceAnalyzer.cpp
  Analysis/KernelCallFinder.cpp
  Analysis/SpecialStmtFinder.cpp
  Analysis/TransformCandidateFinder.cpp
  Driver/Driver.cpp
  Driver/CompilerOptions.cpp
  Transform/DE/Kernel.cpp
  Transform/AWB/Kernel.cpp
  Transform/AWB/Host.cpp
  Transform/AG/Kernel.cpp
  Transform/AG/Host.cpp
  Utils/Utils.cpp
  )
target_link_libraries(klap
  PRIVATE
  clangTooling
  clangBasic
  clangASTMatchers
  )

install(TARGETS klap
  RUNTIME DESTINATION bin)

