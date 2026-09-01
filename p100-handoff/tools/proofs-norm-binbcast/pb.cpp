#include <cstdio>
#include <algorithm>
static int pb_select(int ne11,int nbatch_fa,int ntiles_dst,int nsm,int mbps){
    int ntiles_KV=(ne11+nbatch_fa-1)/nbatch_fa;
    int parallel_blocks=std::min(mbps,ntiles_KV);
    int bpw=nsm*mbps,nwb=0,ebp=0;
    for(int t=parallel_blocks;t<=ntiles_KV;++t){
        int nb=ntiles_dst*t; int nw=(nb+bpw-1)/bpw; int e=100*nb/(nw*bpw);
        if(ebp>=95&&nw>nwb) break;
        if(e>ebp){nwb=nw;ebp=e;parallel_blocks=t;}
    }
    return parallel_blocks;
}
int main(){
    const int nsm=56;
    int Ds[2]={64,256};
    for(int di=0;di<2;++di){int D=Ds[di];
      int diff=0,tot=0;
      for(int mbps=1;mbps<=16;++mbps)
      for(int ntd=1;ntd<=64;++ntd)
      for(int ne11=256;ne11<=16384;ne11+=256){
        int a=pb_select(ne11,D,ntd,nsm,mbps);      // stock
        int b=pb_select(ne11,128,ntd,nsm,mbps);    // patched
        tot++;
        if(a!=b){ if(diff<6) printf("D=%d mbps=%d ntiles_dst=%d ne11=%d : stock pb=%d patched pb=%d\n",D,mbps,ntd,ne11,a,b); diff++; }
      }
      printf("D=%d: %d/%d configs differ (%.1f%%)\n\n",D,diff,tot,100.0*diff/tot);
    }
}
