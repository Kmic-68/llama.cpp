#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
typedef struct { uint32_t x,y,z; } uint3;
static uint3 init_fastdiv(uint64_t d64){ if(!d64||d64>0xFFFFFFFFull){fprintf(stderr,"ASSERT d=%llu\n",(unsigned long long)d64);exit(2);} uint32_t d=(uint32_t)d64,L=0; while(L<32&&((uint32_t)1u<<(L&31))<d)L++; uint32_t mp=(uint32_t)((((uint64_t)1<<32)*((((uint64_t)1)<<L)-d))/d+1); uint3 r={mp,L,d}; return r;}
static inline uint32_t fastdiv(uint32_t n,uint3 f){uint32_t hi=(uint32_t)(((uint64_t)n*(uint64_t)f.x)>>32); return f.y>=32?0u:((hi+n)>>f.y);}

/* original cpy_scalar offsets (int64 division) */
static void orig(int64_t i,int64_t ne00,int64_t ne01,int64_t ne02,int64_t nb00,int64_t nb01,int64_t nb02,int64_t nb03,
                 int64_t ne10,int64_t ne11,int64_t ne12,int64_t nb10,int64_t nb11,int64_t nb12,int64_t nb13,
                 int64_t*xo,int64_t*do_){
    const int64_t i03=i/(ne00*ne01*ne02); const int64_t i02=(i-i03*ne00*ne01*ne02)/(ne00*ne01);
    const int64_t i01=(i-i03*ne00*ne01*ne02-i02*ne01*ne00)/ne00;
    const int64_t i00=i-i03*ne00*ne01*ne02-i02*ne01*ne00-i01*ne00;
    *xo=i00*nb00+i01*nb01+i02*nb02+i03*nb03;
    const int64_t i13=i/(ne10*ne11*ne12); const int64_t i12=(i-i13*ne10*ne11*ne12)/(ne10*ne11);
    const int64_t i11=(i-i13*ne10*ne11*ne12-i12*ne10*ne11)/ne10;
    const int64_t i10=i-i13*ne10*ne11*ne12-i12*ne10*ne11-i11*ne10;
    *do_=i10*nb10+i11*nb11+i12*nb12+i13*nb13;
}
static void fastv(uint32_t i,uint3 A,uint3 B,uint3 C,int64_t nb00,int64_t nb01,int64_t nb02,int64_t nb03,
                  uint3 D,uint3 E,uint3 F,int64_t nb10,int64_t nb11,int64_t nb12,int64_t nb13,
                  int64_t*xo,int64_t*do_){
    uint32_t i03=fastdiv(i,A); uint32_t r3=i-i03*A.z; uint32_t i02=fastdiv(r3,B); uint32_t r2=r3-i02*B.z;
    uint32_t i01=fastdiv(r2,C); uint32_t i00=r2-i01*C.z;
    *xo=(int64_t)i00*nb00+(int64_t)i01*nb01+(int64_t)i02*nb02+(int64_t)i03*nb03;
    uint32_t i13=fastdiv(i,D); uint32_t s3=i-i13*D.z; uint32_t i12=fastdiv(s3,E); uint32_t s2=s3-i12*E.z;
    uint32_t i11=fastdiv(s2,F); uint32_t i10=s2-i11*F.z;
    *do_=(int64_t)i10*nb10+(int64_t)i11*nb11+(int64_t)i12*nb12+(int64_t)i13*nb13;
}
static void run(const char*name,int64_t n00,int64_t n01,int64_t n02,int64_t n03,int64_t ts){
    int64_t ne=n00*n01*n02*n03;
    int64_t nb00=ts,nb01=ts*n00,nb02=ts*n00*n01,nb03=ts*n00*n01*n02;
    uint3 A=init_fastdiv((uint32_t)(n00*n01*n02)),B=init_fastdiv((uint32_t)(n00*n01)),C=init_fastdiv((uint32_t)n00);
    long long bad=0; int64_t firstbad=-1;
    #pragma omp parallel for schedule(static) reduction(+:bad)
    for(int64_t i=0;i<ne;i++){
        int64_t xo,dof,xo2,dof2;
        orig(i,n00,n01,n02,nb00,nb01,nb02,nb03,n00,n01,n02,nb00,nb01,nb02,nb03,&xo,&dof);
        fastv((uint32_t)i,A,B,C,nb00,nb01,nb02,nb03,A,B,C,nb00,nb01,nb02,nb03,&xo2,&dof2);
        if(xo!=xo2||dof!=dof2){bad++; if(firstbad<0)
            #pragma omp critical
            {if(firstbad<0)firstbad=i;}
        }
    }
    printf("%-28s ne=%-12lld (=%.2f G) mismatches=%lld first=%lld\n",name,(long long)ne,ne/1e9,bad,(long long)firstbad);
    fflush(stdout);
}
int main(void){
    run("qwen ffn  [5120,1,1,1]",5120,1,1,1,4);
    run("kv copy   [128,32,4096,1]",128,32,4096,1,2);
    run("big f16   [4096,4096,32,1]",4096,4096,32,1,2);   /* 2^29 */
    run("2^31-ish  [4097,131072,4,1]",4097,131072,4,1,2); /* 2.147e9 > 2^31 */
    return 0;
}
