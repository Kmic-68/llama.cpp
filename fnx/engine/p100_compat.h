// p100_compat.h: what the split engine's kernels ask of a card and sm_60 does not have (force-included in every CUDA source).
#pragma once
#include <cuda_runtime.h>

// the four-way signed byte dot product, instruction from sm_61 on, in two forms that give the same sums:
//   fnx_dp4a_vmad: vmad with byte selectors, 4 instructions (the C form took about 12 and 4 more per sum).
//                  "+&r": the accumulator never shares a register with a, b (Strata PR #4274: with "+r" a forced alias gave wrong sums);
//   fnx_dp4a_xmad: (2026-11-12, idea 63, from KaneQuah/Kmic-68's llama.cpp fork 99eb48176d) PRMT sign-extends byte pairs into 16-bit
//                  halves, XMAD multiplies the halves (mad.wide.s16), 8 instructions. GP100 issues them at about twice the rate of the
//                  4 VMAD (mikro/kane/dp4a_hitrost.cu: 456-465 -> 874-952 Gdp4a/s a card), and the PRMTs of an operand that several
//                  calls share are computed once (the asm is not volatile). Bit-exact (1.07e9 cases, 0 mismatches). It holds more
//                  registers: a kernel that falls a step in blocks an SM by it (64 / 85 / 128 registers at 256 threads) gets slower
//                  (split68: all kernels on it, fresh writing -1.9 %), one that stays on its step faster (dense, 8 and 12 rows of
//                  window, 4 rows a warp: -18 and -14 %).
// __dp4a is the vmad form; the kernels that gain from the other name it themselves (FNX_DP4A_FORMS: both exist; dense.cuh, hc.cuh).
// -DFNX_DP4A_XMAD: __dp4a is the xmad form (split68's build).
#define FNX_DP4A_FORMS 1
__device__ __forceinline__ int fnx_dp4a_c(int a, int b, int c) {
    return c + (int) (signed char) a * (int) (signed char) b +
           (int) (signed char) (a >> 8) * (int) (signed char) (b >> 8) +
           (int) (signed char) (a >> 16) * (int) (signed char) (b >> 16) + (a >> 24) * (b >> 24);
}
__device__ __forceinline__ int fnx_dp4a_vmad(int a, int b, int c) {
#if defined(__CUDA_ARCH__)
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;" : "+&r"(c) : "r"(a), "r"(b));
    asm("vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;" : "+&r"(c) : "r"(a), "r"(b));
    asm("vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;" : "+&r"(c) : "r"(a), "r"(b));
    asm("vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;" : "+&r"(c) : "r"(a), "r"(b));
    return c;
#else
    return fnx_dp4a_c(a, b, c);
#endif
}
__device__ __forceinline__ int fnx_dp4a_xmad(int a, int b, int c) {
#if defined(__CUDA_ARCH__)
    int a01, a23, b01, b23;
    asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(a01) : "r"(a));   // byte 0, its sign, byte 1, its sign
    asm("prmt.b32 %0, %1, 0, 0xB3A2;" : "=r"(a23) : "r"(a));   // bytes 2 and 3 the same way
    asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(b01) : "r"(b));
    asm("prmt.b32 %0, %1, 0, 0xB3A2;" : "=r"(b23) : "r"(b));
    asm("{ .reg .s16 al,ah,bl,bh;\n\t"
        "mov.b32 {al,ah}, %1;\n\t"
        "mov.b32 {bl,bh}, %2;\n\t"
        "mad.wide.s16 %0, al, bl, %0;\n\t"
        "mad.wide.s16 %0, ah, bh, %0;\n\t}"
        : "+&r"(c) : "r"(a01), "r"(b01));
    asm("{ .reg .s16 al,ah,bl,bh;\n\t"
        "mov.b32 {al,ah}, %1;\n\t"
        "mov.b32 {bl,bh}, %2;\n\t"
        "mad.wide.s16 %0, al, bl, %0;\n\t"
        "mad.wide.s16 %0, ah, bh, %0;\n\t}"
        : "+&r"(c) : "r"(a23), "r"(b23));
    return c;
#else
    return fnx_dp4a_c(a, b, c);
#endif
}
#ifdef FNX_DP4A_XMAD
#define __dp4a fnx_dp4a_xmad
#else
#define __dp4a fnx_dp4a_vmad
#endif

// the wait loops' short sleep, sm_70 on; here they simply spin
__device__ __forceinline__ void fnx_nanosleep(unsigned) {}
#define __nanosleep fnx_nanosleep
