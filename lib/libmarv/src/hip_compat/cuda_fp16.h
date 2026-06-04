#ifndef MARV_HIP_COMPAT_CUDA_FP16_H
#define MARV_HIP_COMPAT_CUDA_FP16_H
// Redirect CUDA's <cuda_fp16.h> to the HIP half-precision header on ROCm.
#include <hip/hip_fp16.h>
#endif
