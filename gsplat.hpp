#if defined(USE_HIP) || defined(USE_CUDA)
#include "vendor/gsplat/bindings.h"
#else
#include "vendor/gsplat-cpu/bindings.h"
#endif