// Workaround addressing __noinline__ conflicts between gcc 12/13 libstdc++ and CUDA/HIP code
// https://github.com/llvm/llvm-project/issues/57544
#if defined(__clang__) && defined(__CUDA__)
#undef __noinline__
#endif