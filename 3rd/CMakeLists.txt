add_library(cutlass INTERFACE)
target_include_directories(cutlass INTERFACE ./cutlass)
target_compile_features(cutlass INTERFACE cxx_std_17)