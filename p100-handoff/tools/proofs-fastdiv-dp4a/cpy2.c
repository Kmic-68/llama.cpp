#include <stdio.h>
#include <stdint.h>
typedef struct { uint32_t x,y,z; } uint3;
static uint3 init_fastdiv(uint64_t d64){uint32_t d=(uint32_t)d64,L=0;while(L<32&&((uint32_t)1u<<(L&31))<d)L++;uint32_t mp=(uint32_t)((((uint64_t)1<<32)*((((uint64_t)1)<<L)-d))/d+1);uint3 r={mp,L,d};return r;}
static inline uint32_t fastdiv(uint32_t n,uint3 f){uint32_t hi=(uint32_t)(((uint64_t)n*(uint64_t)f.x)>>32);return f.y>=32?0u:((hi+n)>>f.y);}
int main(void){
    /* ne00*ne01*ne02 = 2^30+1 = 1073741825, ne03 = 3  => ne = 3221225475 <= UINT32_MAX (guard passes) */
    int64_t n00=25,n01=533,n02=80581,n03=3;
    int64_t P=n00*n01*n02, ne=P*n03;
    printf("ne00=%lld ne01=%lld ne02=%lld ne03=%lld  P=%lld  ne=%lld  UINT32_MAX=%u  guard passes: %s\n",
      (long long)n00,(long long)n01,(long long)n02,(long long)n03,(long long)P,(long long)ne,4294967295u,
      (ne<=4294967295LL && P<=4294967295LL)?"YES":"no");
    uint3 A=init_fastdiv(P);
    long long bad=0,firstbad=-1,checked=0;
    for(int64_t i=2147483640LL;i<ne;i+= (i<2147483700LL?1:104729)){
        uint32_t i03f=fastdiv((uint32_t)i,A);
        int64_t  i03t=i/P;
        checked++;
        if((int64_t)i03f!=i03t){ bad++; if(firstbad<0){firstbad=i; printf("FIRST MISMATCH i=%lld : fastdiv->i03=%u  true i03=%lld\n",(long long)i,i03f,(long long)i03t);} }
    }
    printf("checked=%lld mismatches=%lld\n",checked,bad);
    return 0;
}
