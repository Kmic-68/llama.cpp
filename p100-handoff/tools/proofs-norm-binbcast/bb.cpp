// Exhaustive check: when the new flat fast-path predicate holds, does the stock
// general path (collapse + k_bin_bcast index math) map dst element -> same src elements?
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cstring>
typedef long long i64;
struct T { i64 ne[4]; size_t nb[4]; };
static const size_t TS = 4; // f32

// ggml_is_contiguous_0, blck_size == 1
static bool is_contig(const T&t){
    size_t next=TS;
    if(t.ne[0]!=1 && t.nb[0]!=next) return false;
    next*=t.ne[0];
    for(int i=1;i<4;i++){ if(t.ne[i]!=1 && t.nb[i]!=next) return false; next*=t.ne[i]; }
    return true;
}
static bool is_perm(const T&t){ return t.nb[0]>t.nb[1]||t.nb[1]>t.nb[2]||t.nb[2]>t.nb[3]; }
static bool same_shape(const T&a,const T&b){ for(int i=0;i<4;i++) if(a.ne[i]!=b.ne[i]) return false; return true; }

// Model the general path: returns for each dst flat index the src0 and src1 element offsets.
static void general_map(const T&s0,const T&s1,const T&d,std::vector<i64>&o0,std::vector<i64>&o1,std::vector<i64>&od){
    i64 cne[4],cne0[4],cne1[4]; size_t cnb[4],cnb0[4],cnb1[4];
    for(int i=0;i<4;i++){cne[i]=d.ne[i];cne0[i]=s0.ne[i];cne1[i]=s1.ne[i];cnb[i]=d.nb[i];cnb0[i]=s0.nb[i];cnb1[i]=s1.nb[i];}
    int nr[4]; for(int i=0;i<4;i++) nr[i]=(int)(s1.ne[i]/d.ne[i]);
    auto collapse=[](i64 c[4]){c[0]*=c[1];c[1]=c[2];c[2]=c[3];c[3]=1;};
    auto collapse_nb=[](size_t b[4],const i64 c[4]){b[1]*=c[1];b[2]*=c[2];b[3]*=c[3];};
    if(is_contig(s0)&&is_contig(s1)&&!is_perm(s0)&&!is_perm(s1)){
        for(int i=0;i<4;i++){ if(nr[i]!=1) break;
            if(i>0){ collapse_nb(cnb,cne);collapse_nb(cnb0,cne0);collapse_nb(cnb1,cne1);
                     collapse(cne);collapse(cne0);collapse(cne1); } }
    }
    i64 ne0=cne[0],ne1=cne[1],ne2=cne[2],ne3=cne[3];
    size_t s1_=cnb[1]/TS,s2_=cnb[2]/TS,s3_=cnb[3]/TS;
    size_t s00=cnb0[0]/TS,s01=cnb0[1]/TS,s02=cnb0[2]/TS,s03=cnb0[3]/TS;
    size_t s10=cnb1[0]/TS,s11=cnb1[1]/TS,s12=cnb1[2]/TS,s13=cnb1[3]/TS;
    i64 n10=cne1[0],n11=cne1[1],n12=cne1[2],n13=cne1[3];
    for(i64 i3=0;i3<ne3;i3++)for(i64 i2=0;i2<ne2;i2++)for(i64 i1=0;i1<ne1;i1++)for(i64 i0=0;i0<ne0;i0++){
        i64 i11=i1%n11,i12=i2%n12,i13=i3%n13,i10=i0%n10;
        i64 isrc0=i3*s03+i2*s02+i1*s01;
        i64 isrc1=i13*s13+i12*s12+i11*s11;
        i64 idst =i3*s3_+i2*s2_+i1*s1_;
        o0.push_back(isrc0+i0*(i64)s00);
        o1.push_back(isrc1+i10*(i64)s10);
        od.push_back(idst+i0);
    }
}
int main(){
    const i64 vals[]={1,2,3,5};
    int bad=0,checked=0;
    for(int a=0;a<4;a++)for(int b=0;b<4;b++)for(int c=0;c<4;c++)for(int e=0;e<4;e++){
        T t; t.ne[0]=vals[a];t.ne[1]=vals[b];t.ne[2]=vals[c];t.ne[3]=vals[e];
        // canonical contiguous strides
        t.nb[0]=TS; for(int i=1;i<4;i++) t.nb[i]=t.nb[i-1]*t.ne[i-1];
        // also probe the "free" nb[0] when ne[0]==1
        std::vector<size_t> nb0cand={TS};
        if(t.ne[0]==1){ nb0cand.push_back(0); nb0cand.push_back(TS*7); }
        for(size_t nb0 : nb0cand){
            T s0=t,s1=t,d=t; s0.nb[0]=nb0; s1.nb[0]=nb0;
            if(!(is_contig(s0)&&is_contig(s1)&&is_contig(d)&&same_shape(s0,d)&&same_shape(s1,d))) continue;
            checked++;
            std::vector<i64> o0,o1,od; general_map(s0,s1,d,o0,o1,od);
            i64 n=1; for(int i=0;i<4;i++) n*=t.ne[i];
            if((i64)od.size()!=n){printf("COUNT MISMATCH\n");bad++;continue;}
            bool ok=true;
            for(i64 k=0;k<n;k++){ if(od[k]!=k||o0[k]!=k||o1[k]!=k){ok=false;break;} }
            if(!ok){ bad++;
                printf("DIVERGE ne=[%lld,%lld,%lld,%lld] nb0=%zu : general maps dst0..3 -> src0 %lld,%lld,%lld  dst %lld,%lld,%lld\n",
                    t.ne[0],t.ne[1],t.ne[2],t.ne[3],nb0,
                    o0[0],(n>1?o0[1]:-1),(n>2?o0[2]:-1),od[0],(n>1?od[1]:-1),(n>2?od[2]:-1));
            }
        }
    }
    printf("checked=%d diverging=%d\n",checked,bad);
}
