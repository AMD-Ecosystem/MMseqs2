#ifndef MARV_SIMD_AMD_CUH
#define MARV_SIMD_AMD_CUH

// AMD/ROCm emulation of the NVIDIA SIMD-within-a-word integer intrinsics that
// libmarv (CUDASW++/cudasw4) routes its gapless and Smith-Waterman score
// recurrences through. ROCm 7.2.1 provides none of the packed-integer video
// or DPX intrinsics, so they are emulated here in scalar form: unpack the
// 2x16 / 4x8 SIMD lanes, apply the per-lane signed/unsigned arithmetic with
// the exact saturation, tie-break, and relu semantics of the CUDA intrinsics,
// then repack. The lane geometry tracks the SIMD WORD width (two 16-bit lanes,
// four 8-bit lanes); it is independent of the hardware wavefront width.
//
// References: CUDA Math API "SIMD Intrinsics" and "Integer Intrinsics".
// __vibmax_*: per-lane max and a boolean per lane that is true when the first
// operand wins (a >= b). __vimax3_*: per-lane max of three. __viaddmax_*:
// per-lane max(a + b, c) with a wrapping (non-saturating) add. The _relu forms
// additionally clamp the result at 0. __v*us4: per-byte unsigned saturating
// add/sub. __vadd2/__vadd4: per-lane wrapping add.

#include <hip/hip_runtime.h>
#include <cstdint>

namespace marv_amd {

    __device__ __forceinline__
    short2 unpack_s2(unsigned int u){
        short2 v;
        v.x = static_cast<short>(u & 0xFFFFu);
        v.y = static_cast<short>((u >> 16) & 0xFFFFu);
        return v;
    }

    __device__ __forceinline__
    unsigned int pack_s2(int x, int y){
        return (static_cast<unsigned int>(x) & 0xFFFFu)
             | ((static_cast<unsigned int>(y) & 0xFFFFu) << 16);
    }

    __device__ __forceinline__
    unsigned short hi16(unsigned int u){ return static_cast<unsigned short>((u >> 16) & 0xFFFFu); }
    __device__ __forceinline__
    unsigned short lo16(unsigned int u){ return static_cast<unsigned short>(u & 0xFFFFu); }

    __device__ __forceinline__ int  imax(int a, int b){ return a > b ? a : b; }
    __device__ __forceinline__ int  imax3(int a, int b, int c){ return imax(a, imax(b, c)); }
    __device__ __forceinline__ unsigned umax(unsigned a, unsigned b){ return a > b ? a : b; }

} // namespace marv_amd


// ---- 16x2 signed/unsigned ----

__device__ __forceinline__
unsigned int __vadd2(unsigned int a, unsigned int b){
    // per-halfword wrapping add
    short2 va = marv_amd::unpack_s2(a);
    short2 vb = marv_amd::unpack_s2(b);
    return marv_amd::pack_s2(int(va.x) + int(vb.x), int(va.y) + int(vb.y));
}

__device__ __forceinline__
unsigned int __vmaxs2(unsigned int a, unsigned int b){
    // per-halfword signed max
    short2 va = marv_amd::unpack_s2(a);
    short2 vb = marv_amd::unpack_s2(b);
    return marv_amd::pack_s2(marv_amd::imax(va.x, vb.x), marv_amd::imax(va.y, vb.y));
}

__device__ __forceinline__
unsigned int __vibmax_u16x2(unsigned int a, unsigned int b, bool* pred_lo, bool* pred_hi){
    // per-halfword UNSIGNED max; pred true when first operand wins (a >= b)
    unsigned short alo = marv_amd::lo16(a), ahi = marv_amd::hi16(a);
    unsigned short blo = marv_amd::lo16(b), bhi = marv_amd::hi16(b);
    *pred_lo = (alo >= blo);
    *pred_hi = (ahi >= bhi);
    unsigned short rlo = *pred_lo ? alo : blo;
    unsigned short rhi = *pred_hi ? ahi : bhi;
    return (static_cast<unsigned int>(rlo)) | (static_cast<unsigned int>(rhi) << 16);
}

__device__ __forceinline__
unsigned int __vimax3_s16x2(unsigned int a, unsigned int b, unsigned int c){
    short2 va = marv_amd::unpack_s2(a);
    short2 vb = marv_amd::unpack_s2(b);
    short2 vc = marv_amd::unpack_s2(c);
    return marv_amd::pack_s2(marv_amd::imax3(va.x, vb.x, vc.x),
                             marv_amd::imax3(va.y, vb.y, vc.y));
}

__device__ __forceinline__
unsigned int __vimax3_s16x2_relu(unsigned int a, unsigned int b, unsigned int c){
    short2 va = marv_amd::unpack_s2(a);
    short2 vb = marv_amd::unpack_s2(b);
    short2 vc = marv_amd::unpack_s2(c);
    return marv_amd::pack_s2(marv_amd::imax(0, marv_amd::imax3(va.x, vb.x, vc.x)),
                             marv_amd::imax(0, marv_amd::imax3(va.y, vb.y, vc.y)));
}

__device__ __forceinline__
unsigned int __viaddmax_s16x2(unsigned int a, unsigned int b, unsigned int c){
    // per-halfword signed max(a + b, c) with wrapping add
    short2 va = marv_amd::unpack_s2(a);
    short2 vb = marv_amd::unpack_s2(b);
    short2 vc = marv_amd::unpack_s2(c);
    short sx = static_cast<short>(static_cast<unsigned short>(int(va.x) + int(vb.x)));
    short sy = static_cast<short>(static_cast<unsigned short>(int(va.y) + int(vb.y)));
    return marv_amd::pack_s2(marv_amd::imax(sx, vc.x), marv_amd::imax(sy, vc.y));
}

__device__ __forceinline__
unsigned int __viaddmax_s16x2_relu(unsigned int a, unsigned int b, unsigned int c){
    short2 va = marv_amd::unpack_s2(a);
    short2 vb = marv_amd::unpack_s2(b);
    short2 vc = marv_amd::unpack_s2(c);
    short sx = static_cast<short>(static_cast<unsigned short>(int(va.x) + int(vb.x)));
    short sy = static_cast<short>(static_cast<unsigned short>(int(va.y) + int(vb.y)));
    return marv_amd::pack_s2(marv_amd::imax(0, marv_amd::imax(sx, vc.x)),
                             marv_amd::imax(0, marv_amd::imax(sy, vc.y)));
}


// ---- 32-bit scalar (DPX) ----

__device__ __forceinline__
int __vibmax_s32(int a, int b, bool* pred){
    *pred = (a >= b);
    return *pred ? a : b;
}

__device__ __forceinline__
int __vimax3_s32(int a, int b, int c){
    return marv_amd::imax3(a, b, c);
}

__device__ __forceinline__
int __viaddmax_s32(int a, int b, int c){
    return marv_amd::imax(a + b, c);
}

__device__ __forceinline__
int __viaddmax_s32_relu(int a, int b, int c){
    return marv_amd::imax(0, marv_amd::imax(a + b, c));
}


// ---- 8x4 unsigned (video intrinsics) ----

namespace marv_amd {
    __device__ __forceinline__ unsigned byte_of(unsigned int u, int i){ return (u >> (8 * i)) & 0xFFu; }
    __device__ __forceinline__ unsigned int pack4(unsigned b0, unsigned b1, unsigned b2, unsigned b3){
        return (b0 & 0xFFu) | ((b1 & 0xFFu) << 8) | ((b2 & 0xFFu) << 16) | ((b3 & 0xFFu) << 24);
    }
    __device__ __forceinline__ unsigned addus8(unsigned a, unsigned b){ unsigned s = a + b; return s > 255u ? 255u : s; }
    __device__ __forceinline__ unsigned subus8(unsigned a, unsigned b){ return a > b ? a - b : 0u; }
}

__device__ __forceinline__
unsigned int __vmaxu4(unsigned int a, unsigned int b){
    using namespace marv_amd;
    return pack4(umax(byte_of(a,0), byte_of(b,0)), umax(byte_of(a,1), byte_of(b,1)),
                 umax(byte_of(a,2), byte_of(b,2)), umax(byte_of(a,3), byte_of(b,3)));
}

__device__ __forceinline__
unsigned int __vminu4(unsigned int a, unsigned int b){
    using namespace marv_amd;
    auto umin = [](unsigned x, unsigned y){ return x < y ? x : y; };
    return pack4(umin(byte_of(a,0), byte_of(b,0)), umin(byte_of(a,1), byte_of(b,1)),
                 umin(byte_of(a,2), byte_of(b,2)), umin(byte_of(a,3), byte_of(b,3)));
}

__device__ __forceinline__
unsigned int __vadd4(unsigned int a, unsigned int b){
    using namespace marv_amd;
    return pack4(byte_of(a,0) + byte_of(b,0), byte_of(a,1) + byte_of(b,1),
                 byte_of(a,2) + byte_of(b,2), byte_of(a,3) + byte_of(b,3));
}

__device__ __forceinline__
unsigned int __vaddus4(unsigned int a, unsigned int b){
    using namespace marv_amd;
    return pack4(addus8(byte_of(a,0), byte_of(b,0)), addus8(byte_of(a,1), byte_of(b,1)),
                 addus8(byte_of(a,2), byte_of(b,2)), addus8(byte_of(a,3), byte_of(b,3)));
}

__device__ __forceinline__
unsigned int __vsubus4(unsigned int a, unsigned int b){
    using namespace marv_amd;
    return pack4(subus8(byte_of(a,0), byte_of(b,0)), subus8(byte_of(a,1), byte_of(b,1)),
                 subus8(byte_of(a,2), byte_of(b,2)), subus8(byte_of(a,3), byte_of(b,3)));
}


// ---- half2 ----
// ROCm provides scalar __hmax but not __hmax2. Emulate per lane; __hmax
// propagates the non-NaN operand, matching CUDA's __hmax2 NaN handling.

__device__ __forceinline__
__half2 __hmax2(__half2 a, __half2 b){
    return __halves2half2(__hmax(__low2half(a), __low2half(b)),
                          __hmax(__high2half(a), __high2half(b)));
}

#endif
