#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void init_fastdiv(uint64_t d_64, uint32_t *mp_out, uint32_t *L_out) {
    if (d_64 == 0 || d_64 > 0xFFFFFFFFull) { fprintf(stderr,"ASSERT d=%llu\n",(unsigned long long)d_64); exit(1); }
    uint32_t d = (uint32_t)d_64;
    uint32_t L = 0;
    while (L < 32 && ((uint32_t)1u << (L==32?0:L)) < d) L++;
    uint32_t mp = (uint32_t)(((uint64_t)1 << 32) * (((uint64_t)1 << L) - d) / d + 1);
    *mp_out = mp; *L_out = L;
}
static inline uint32_t fastdiv(uint32_t n, uint32_t mp, uint32_t L) {
    uint32_t hi = (uint32_t)(((uint64_t)n * (uint64_t)mp) >> 32);
    return (hi + n) >> L;
}
/* also the modulo form used by fast_div_modulo / concat / cpy remainder */
static inline uint32_t fastmod(uint32_t n, uint32_t mp, uint32_t L, uint32_t d) {
    return n - fastdiv(n,mp,L)*d;
}

int main(int argc, char **argv) {
    static const uint64_t ds[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,31,32,33,63,64,65,100,127,128,129,
        255,256,257,1000,1023,1024,1025,2048,3072,4096,5120,8192,11008,18944,32768,65535,65536,65537,
        1048576,1048577,1431655765ULL,2147483647ULL,2147483648ULL,2147483649ULL,3000000000ULL,4294967295ULL};
    const int nd = (int)(sizeof(ds)/sizeof(ds[0]));
    long long tested = 0;
    for (int k=0;k<nd;k++) {
        uint32_t d=(uint32_t)ds[k], mp, L; init_fastdiv(ds[k],&mp,&L);
        uint64_t minfail = ~0ULL; uint64_t nfail = 0; uint64_t minfailmod = ~0ULL;
        #pragma omp parallel for schedule(static) reduction(min:minfail,minfailmod) reduction(+:nfail)
        for (long long nn=0; nn<4294967296LL; nn++) {
            uint32_t n=(uint32_t)nn;
            uint32_t got = fastdiv(n,mp,L);
            if (got != n/d) { nfail++; if ((uint64_t)n < minfail) minfail=(uint64_t)n; }
            if (fastmod(n,mp,L,d) != n%d) { if ((uint64_t)n < minfailmod) minfailmod=(uint64_t)n; }
        }
        tested += 4294967296LL;
        printf("d=%-11u L=%2u mp=%-11u  div_fails=%llu first_bad_n=%s%llu  first_bad_mod_n=%s%llu\n",
            d,L,mp,(unsigned long long)nfail,
            minfail==~0ULL?"none(":"", (unsigned long long)(minfail==~0ULL?0:minfail),
            minfailmod==~0ULL?"none(":"", (unsigned long long)(minfailmod==~0ULL?0:minfailmod));
        fflush(stdout);
    }
    printf("total (n,d) cases tested: %lld\n", tested);
    return 0;
}
