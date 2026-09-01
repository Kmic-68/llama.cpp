#include <stdio.h>
#include <stdint.h>
int main(void){
  const uint64_t ds[]={1073741825ULL,536870913ULL,2049ULL,4097ULL,2147483649ULL};
  for(int k=0;k<5;k++){
    uint32_t d=(uint32_t)ds[k]; uint32_t L=0; while(L<32 && ((uint32_t)1u<<(L&31))<d) L++;
    uint32_t mp=(uint32_t)((((uint64_t)1<<32)*((((uint64_t)1)<<L)-d))/d+1);
    uint64_t minfail=~0ULL, nf=0;
    #pragma omp parallel for schedule(static) reduction(min:minfail) reduction(+:nf)
    for(long long nn=0;nn<4294967296LL;nn++){
      uint32_t n=(uint32_t)nn;
      uint32_t hi=(uint32_t)(((uint64_t)n*(uint64_t)mp)>>32);
      /* model PTX shr.u32 clamp-at-32 semantics for L==32 */
      uint32_t got = (L>=32)?0u:((hi+n)>>L);
      if(got!=n/d){nf++; if((uint64_t)n<minfail)minfail=(uint64_t)n;}
    }
    printf("d=%-11u L=%2u mp=%-11u fails=%llu first_bad=%llu\n",d,L,mp,(unsigned long long)nf,(unsigned long long)minfail);
  }
  return 0;
}
