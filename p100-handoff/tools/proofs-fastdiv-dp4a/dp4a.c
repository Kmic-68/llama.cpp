#include <stdio.h>
#include <stdint.h>
/* emulate PRMT.b32 d, a, b, mode  (generic mode, msb sign-replicate) */
static uint32_t prmt(uint32_t a,uint32_t b,uint32_t mode){
    uint8_t src[8]; for(int i=0;i<4;i++)src[i]=(a>>(8*i))&0xFF; for(int i=0;i<4;i++)src[4+i]=(b>>(8*i))&0xFF;
    uint32_t d=0;
    for(int i=0;i<4;i++){ uint32_t nib=(mode>>(4*i))&0xF; uint8_t v=src[nib&7];
        uint8_t out = (nib&8) ? ((v&0x80)?0xFF:0x00) : v;
        d |= (uint32_t)out<<(8*i); }
    return d;
}
static int32_t patched(int32_t a,int32_t b,int32_t c){
    uint32_t a01=prmt((uint32_t)a,0,0x9180), a23=prmt((uint32_t)a,0,0xB3A2);
    uint32_t b01=prmt((uint32_t)b,0,0x9180), b23=prmt((uint32_t)b,0,0xB3A2);
    int16_t al=(int16_t)(a01&0xFFFF), ah=(int16_t)(a01>>16), bl=(int16_t)(b01&0xFFFF), bh=(int16_t)(b01>>16);
    uint32_t cc=(uint32_t)c;
    cc += (uint32_t)((int32_t)al*(int32_t)bl); cc += (uint32_t)((int32_t)ah*(int32_t)bh);
    int16_t al2=(int16_t)(a23&0xFFFF), ah2=(int16_t)(a23>>16), bl2=(int16_t)(b23&0xFFFF), bh2=(int16_t)(b23>>16);
    cc += (uint32_t)((int32_t)al2*(int32_t)bl2); cc += (uint32_t)((int32_t)ah2*(int32_t)bh2);
    return (int32_t)cc;
}
static int32_t reference(int32_t a,int32_t b,int32_t c){
    const int8_t*a8=(const int8_t*)&a; const int8_t*b8=(const int8_t*)&b;
    uint32_t cc=(uint32_t)c;
    cc+=(uint32_t)((int32_t)a8[0]*(int32_t)b8[0]); cc+=(uint32_t)((int32_t)a8[1]*(int32_t)b8[1]);
    cc+=(uint32_t)((int32_t)a8[2]*(int32_t)b8[2]); cc+=(uint32_t)((int32_t)a8[3]*(int32_t)b8[3]);
    return (int32_t)cc;
}
int main(void){
    long long n=0,bad=0;
    /* exhaustive over all 4 byte-lane value pairs independently (2^16 per lane x 4 lanes placed in situ) */
    for(int lane=0;lane<4;lane++) for(int x=-128;x<128;x++) for(int y=-128;y<128;y++){
        int32_t a=((uint32_t)(uint8_t)x)<<(8*lane), b=((uint32_t)(uint8_t)y)<<(8*lane);
        for(int ci=0;ci<3;ci++){ int32_t c=(int32_t[]){0,-1,0x7FFFFFFF}[ci];
            n++; if(patched(a,b,c)!=reference(a,b,c)) bad++; }
    }
    /* full 32-bit random */
    uint64_t s=88172645463325252ULL;
    for(long long k=0;k<20000000LL;k++){
        s^=s<<13;s^=s>>7;s^=s<<17; int32_t a=(int32_t)s;
        s^=s<<13;s^=s>>7;s^=s<<17; int32_t b=(int32_t)s;
        s^=s<<13;s^=s>>7;s^=s<<17; int32_t c=(int32_t)s;
        n++; if(patched(a,b,c)!=reference(a,b,c)) bad++;
    }
    /* exhaustive over all (a,b) with a,b in {0x00,0x01,0x7f,0x80,0x81,0xff}^4 */
    const uint8_t v[6]={0x00,0x01,0x7f,0x80,0x81,0xff};
    for(int i0=0;i0<6;i0++)for(int i1=0;i1<6;i1++)for(int i2=0;i2<6;i2++)for(int i3=0;i3<6;i3++)
    for(int j0=0;j0<6;j0++)for(int j1=0;j1<6;j1++)for(int j2=0;j2<6;j2++)for(int j3=0;j3<6;j3++){
        int32_t a=v[i0]|(v[i1]<<8)|(v[i2]<<16)|((uint32_t)v[i3]<<24);
        int32_t b=v[j0]|(v[j1]<<8)|(v[j2]<<16)|((uint32_t)v[j3]<<24);
        n++; if(patched(a,b,-12345)!=reference(a,b,-12345)) bad++;
    }
    printf("dp4a cases=%lld mismatches=%lld\n",n,bad);
    return 0;
}
