#ifndef MARV_CUDA_TO_HIP_H
#define MARV_CUDA_TO_HIP_H

// ROCm/HIP compatibility shim for libmarv (CUDASW++/cudasw4). It is force
// included on the HIP build (see CMakeLists) so the sources keep their CUDA
// spelling; here the CUDA runtime symbols, the fp16 / cooperative-groups
// headers, and the packed-integer SIMD intrinsics are mapped to their HIP /
// ROCm equivalents. The CUDA build never sees this header.

// On Windows, <windows.h> defines min/max as function-like macros which break
// C++ standard library uses of std::numeric_limits<T>::max() etc. Suppress them.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// On Windows (ROCm 7.14+ TheRock), <amd_hip_bf16.h> defines __shfl_*_sync
// bfloat16 overloads that conflict with the generic warp-sync template
// already declared by <amd_warp_sync_functions.h>, causing a clang
// "redefinition of default argument" error. Suppress the templates there.
// On Linux ROCm 7.2.x the generic template is NOT present and __syncwarp is
// defined inside the same guard, so suppressing it breaks __syncwarp; do not
// set the flag on Linux.
#ifdef _WIN32
#ifndef HIP_DISABLE_WARP_SYNC_BUILTINS
#define HIP_DISABLE_WARP_SYNC_BUILTINS
#endif
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// __grid_constant__ is a CUDA kernel-parameter attribute with no HIP analogue;
// it is a pure optimization hint, so define it away on ROCm.
#ifndef __grid_constant__
#define __grid_constant__
#endif

// HIP's cooperative-groups tile shfl_up/down/xor forward to the maskless
// __shfl_* builtins, which on ROCm only overload the scalar arithmetic types.
// libmarv shuffles its packed score vectors (short2, half2) and an int3
// max-with-position triple through those tiles. Provide generic overloads that
// shuffle any trivially copyable value as a sequence of 32-bit words, so the
// cg member functions resolve for these types. Defined before the cooperative
// groups header is included (this header is force included first).
#include <cstring>
#include <type_traits>
namespace marv_amd {
    template<class T, class ShuffleWord>
    __device__ __forceinline__ T shfl_generic(T var, ShuffleWord shuffle){
        constexpr int words = (sizeof(T) + sizeof(int) - 1) / sizeof(int);
        union { T value; int w[words]; } src, dst;
        src.value = var;
        for(int i = 0; i < words; i++){ dst.w[i] = shuffle(src.w[i]); }
        return dst.value;
    }
}
#define MARV_DEFINE_VECTOR_SHFL(VECTYPE)                                                            \
    __device__ __forceinline__ VECTYPE __shfl(VECTYPE var, int src, int width){                      \
        return marv_amd::shfl_generic(var, [&](int w){ return __shfl(w, src, width); }); }           \
    __device__ __forceinline__ VECTYPE __shfl_up(VECTYPE var, unsigned d, int width){                \
        return marv_amd::shfl_generic(var, [&](int w){ return __shfl_up(w, d, width); }); }           \
    __device__ __forceinline__ VECTYPE __shfl_down(VECTYPE var, unsigned d, int width){              \
        return marv_amd::shfl_generic(var, [&](int w){ return __shfl_down(w, d, width); }); }         \
    __device__ __forceinline__ VECTYPE __shfl_xor(VECTYPE var, int mask, int width){                 \
        return marv_amd::shfl_generic(var, [&](int w){ return __shfl_xor(w, mask, width); }); }

// ROCm already overloads __shfl* for __half2 and the scalar arithmetic types;
// add the remaining packed types libmarv shuffles: short2 (score vector),
// int2 / float2 (Smith-Waterman M/E score pairs), int3 (max-with-position).
MARV_DEFINE_VECTOR_SHFL(short2)
MARV_DEFINE_VECTOR_SHFL(int2)
MARV_DEFINE_VECTOR_SHFL(float2)
MARV_DEFINE_VECTOR_SHFL(int3)
#undef MARV_DEFINE_VECTOR_SHFL

// HIP's __shfl_*_sync require a 64-bit lane mask, while libmarv's legacy
// (non-PSSM) kernels call them with the CUDA 32-bit full-warp mask. These are
// width-bounded logical-warp shuffles (the width is always passed explicitly
// or is the 32-lane CUDA warp), so route them to the maskless __shfl_* builtins,
// which operate within the requested width on the active lanes. The mask
// argument is dropped.
#define __shfl_sync(mask, ...)      __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...)   __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...)  __shfl_xor(__VA_ARGS__)

// ---- error / runtime types and enums ----
#define cudaError_t                                hipError_t
#define cudaSuccess                                hipSuccess
#define cudaGetLastError                           hipGetLastError
#define cudaPeekAtLastError                        hipPeekAtLastError
#define cudaGetErrorString                         hipGetErrorString
#define cudaDeviceSynchronize                      hipDeviceSynchronize

#define cudaStream_t                               hipStream_t
#define cudaStreamLegacy                           ((hipStream_t)1)
#define cudaStreamCreate                           hipStreamCreate
#define cudaStreamCreateWithFlags                  hipStreamCreateWithFlags
#define cudaStreamDestroy                          hipStreamDestroy
#define cudaStreamSynchronize                      hipStreamSynchronize
#define cudaStreamQuery                            hipStreamQuery
#define cudaStreamWaitEvent                        hipStreamWaitEvent
#define cudaStreamNonBlocking                      hipStreamNonBlocking

#define cudaEvent_t                                hipEvent_t
#define cudaEventCreate                            hipEventCreate
#define cudaEventCreateWithFlags                   hipEventCreateWithFlags
#define cudaEventDestroy                           hipEventDestroy
#define cudaEventRecord                            hipEventRecord
#define cudaEventSynchronize                       hipEventSynchronize
#define cudaEventQuery                             hipEventQuery
#define cudaEventElapsedTime                       hipEventElapsedTime
#define cudaEventDisableTiming                     hipEventDisableTiming

// ---- device management ----
#define cudaSetDevice                              hipSetDevice
#define cudaGetDevice                              hipGetDevice
#define cudaGetDeviceCount                         hipGetDeviceCount
#define cudaGetDeviceProperties                    hipGetDeviceProperties
#define cudaDeviceProp                             hipDeviceProp_t
#define cudaDeviceGetAttribute                     hipDeviceGetAttribute
#define cudaDevAttrComputeCapabilityMajor          hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor          hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrMultiProcessorCount             hipDeviceAttributeMultiprocessorCount
#define cudaDeviceAttrReadOnlyHostRegisterSupported hipDeviceAttributeCanUseHostPointerForRegisteredMem
#define cudaDevAttrPageableMemoryAccessUsesHostPageTables hipDeviceAttributePageableMemoryAccessUsesHostPageTables
#define cudaDeviceSetCacheConfig                   hipDeviceSetCacheConfig
#define cudaFuncCachePreferShared                  hipFuncCachePreferShared
#define cudaDeviceCanAccessPeer                    hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess                 hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess                hipDeviceDisablePeerAccess
#define cudaErrorPeerAccessAlreadyEnabled          hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled              hipErrorPeerAccessNotEnabled

// ---- memory ----
#define cudaMalloc                                 hipMalloc
#define cudaFree                                   hipFree
#define cudaMallocHost                             hipHostMalloc
#define cudaFreeHost                               hipHostFree
#define cudaMallocManaged                          hipMallocManaged
#define cudaMallocAsync                            hipMallocAsync
#define cudaFreeAsync                              hipFreeAsync
#define cudaMemcpy                                 hipMemcpy
#define cudaMemcpyAsync                            hipMemcpyAsync
#define cudaMemcpyToSymbol                         hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync                    hipMemcpyToSymbolAsync
#define cudaMemset                                 hipMemset
#define cudaMemsetAsync                            hipMemsetAsync
#define cudaMemGetInfo                             hipMemGetInfo
#define cudaMemcpyHostToHost                       hipMemcpyHostToHost
#define cudaMemcpyHostToDevice                     hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost                     hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice                   hipMemcpyDeviceToDevice
#define cudaMemcpyDefault                          hipMemcpyDefault
#define cudaMemcpyKind                             hipMemcpyKind
#define cudaHostRegister                           hipHostRegister
#define cudaHostUnregister                         hipHostUnregister
#define cudaHostRegisterDefault                    hipHostRegisterDefault
#define cudaMemAdvise                              hipMemAdvise
#define cudaMemAdviseSetAccessedBy                 hipMemAdviseSetAccessedBy

// ---- memory pools ----
#define cudaMemPool_t                              hipMemPool_t
#define cudaMemPoolSetAttribute                    hipMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold            hipMemPoolAttrReleaseThreshold
#define cudaDeviceGetDefaultMemPool                hipDeviceGetDefaultMemPool

// ---- kernels / func attributes ----
// HIP's hipFuncSetAttribute takes a const void* function pointer and has no
// typed template overload, so cast the kernel entry explicitly.
#define cudaFuncSetAttribute(func, attr, value) hipFuncSetAttribute((const void*)(func), attr, value)
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaLaunchHostFunc                         hipLaunchHostFunc
#define cudaHostFn_t                               hipHostFn_t

// ---- IPC ----
#define cudaIpcMemHandle_t                         hipIpcMemHandle_t
#define cudaIpcGetMemHandle                        hipIpcGetMemHandle
#define cudaIpcOpenMemHandle                       hipIpcOpenMemHandle
#define cudaIpcCloseMemHandle                      hipIpcCloseMemHandle
#define cudaIpcMemLazyEnablePeerAccess             hipIpcMemLazyEnablePeerAccess

// libmarv only includes <cuda_fp16.h> and the cooperative groups headers; map
// them and supply the SIMD intrinsic emulations.
#include <hip/hip_fp16.h>
#include "marv_simd_amd.cuh"

// libmarv uses a handful of libcudacxx types (cuda::std::uint8_t / int8_t) that
// have no ROCm equivalent; alias the cuda::std namespace to the standard
// library, which provides the same fixed-width integer types.
#include <cstdint>
namespace cuda { namespace std = ::std; }

// cudasw4.cuh uses cub::SwitchDevice (a RAII current-device guard that CUDA's
// thrust backend pulls in transitively); ROCm's hipCUB does not provide it, so
// supply the equivalent.
namespace cub {
    struct SwitchDevice {
        int old_device = 0;
        explicit SwitchDevice(int new_device){
            (void)hipGetDevice(&old_device);
            (void)hipSetDevice(new_device);
        }
        ~SwitchDevice(){ (void)hipSetDevice(old_device); }
        SwitchDevice(const SwitchDevice&) = delete;
        SwitchDevice& operator=(const SwitchDevice&) = delete;
    };
}

#endif
