#include <stdio.h>
#include <stdint.h>
/* For every d in [1,2^32): compute mp,L and the smallest n where mulhi(n,mp)+n >= 2^32
   (verified experimentally to be exactly the first n where fastdiv != n/d). Report the min. */
int main(void){
    uint64_t global_min = ~0ULL; uint32_t argmin=0;
    uint64_t min_pow2ok = ~0ULL;
    uint64_t cnt_L32 = 0;
    #pragma omp parallel for schedule(static) reduction(min:global_min)
    for (long long dd=1; dd<4294967296LL; dd++){
        uint32_t d=(uint32_t)dd;
        uint32_t L=0; while (L<32 && ((uint32_t)1u<<(L&31))<d) L++;
        if (L==32) continue;                 /* separate catastrophic regime, handled elsewhere */
        uint32_t mp=(uint32_t)((((uint64_t)1<<32)*((((uint64_t)1)<<L)-d))/d + 1);
        if (mp==1) continue;                 /* exact power of two: never overflows */
        /* smallest n with floor(n*mp/2^32)+n >= 2^32 */
        uint64_t nn = ((uint64_t)1<<32);
        uint64_t first = ( ( ((__uint128_t)1<<64) + ((uint64_t)mp + nn) - 1 ) / ((__uint128_t)mp + nn) );
        if (first < global_min) global_min = first;
    }
    /* recompute argmin serially over a reduced candidate set: d = 2^k+1 */
    for (int k=1;k<31;k++){
        uint32_t d=(1u<<k)+1; uint32_t L=0; while (L<32 && (1u<<(L&31))<d) L++;
        uint32_t mp=(uint32_t)((((uint64_t)1<<32)*((((uint64_t)1)<<L)-d))/d + 1);
        uint64_t first = (((__uint128_t)1<<64) + mp + (1ULL<<32) - 1)/((__uint128_t)mp + (1ULL<<32));
        printf("d=2^%d+1=%-11u  first_bad_n=%llu\n",k,d,(unsigned long long)first);
    }
    printf("MIN over all d with L<32 and mp!=1 of first_bad_n = %llu  (2^31=%llu)\n",
        (unsigned long long)global_min,(unsigned long long)(1ULL<<31));
    (void)argmin;(void)min_pow2ok;(void)cnt_L32;
    return 0;
}
