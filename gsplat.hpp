#ifndef GSPLAT_H
#define GSPLAT_H

#include <gsplat/config.h>

#if defined(USE_HIP) || defined(USE_CUDA)
#include <gsplat/bindings.h>
#endif

#if defined(USE_MPS)
#include <gsplat-metal/bindings.h>
#endif

#include <gsplat-cpu/bindings.h>

#endif
