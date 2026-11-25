// Driver-API per-K validator/benchmark for the dense CH2 gemv_sf module.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"cudart %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} } while(0)
#define CU(x) do { CUresult r_=(x); if(r_!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r_,&s);fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,s?s:"?");exit(1);} } while(0)
static uint32_t rs_=777001u;
static uint32_t rnd(){rs_^=rs_<<13;rs_^=rs_>>17;rs_^=rs_<<5;return rs_;}
static std::vector<char> readfile(const char* p){std::ifstream f(p,std::ios::binary);if(!f){fprintf(stderr,"open %s failed\n",p);exit(1);}return std::vector<char>(std::istreambuf_iterator<char>(f),{});}
struct Mod{CUmodule m{};CUfunction ref{},cand{};};
static Mod loadmod(const char* path){
    auto b=readfile(path); Mod z;
    CU(cuModuleLoadData(&z.m,b.data()));
    CU(cuModuleGetFunction(&z.ref,z.m,"_Z20inkwell_exl3_gemv_sfPKjPKfPKtPfS5_Piiiiii"));
    CU(cuModuleGetFunction(&z.cand,z.m,"_Z24inkwell_exl3_gemv_sf_ch2PKjPKfPKtPfS5_Piiiiii"));
    return z;
}
static void launch(CUfunction f,CUdeviceptr trel,CUdeviceptr xh,CUdeviceptr svh,CUdeviceptr y,CUdeviceptr t,CUdeviceptr ctr,int tk,int tn,int ka,int hk,int nky){
    int pw=0; unsigned gx=(tn+7)/8,gy=nky;
    void* args[]={&trel,&xh,&svh,&y,&t,&ctr,&tk,&tn,&ka,&hk,&pw};
    CU(cuLaunchKernel(f,gx,gy,1,256,1,1,0,nullptr,args,nullptr));
}
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s dense.cubin [correctonly]\n",argv[0]);return 1;}
    bool correctonly=argc>2;
    CK(cudaFree(0)); CU(cuInit(0)); Mod z=loadmod(argv[1]);
    struct K{int ka,hk,tk,tn;const char*name;} ks[]={{3,0,144,192,"3.0/gate"},{3,1,192,144,"3.5/wo"},{4,0,144,12568,"4.0/unembed"},{5,0,144,24,"5.0/wr"}};
    bool exact=true;
    cudaEvent_t ev0,ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    for(auto K:ks){
        const int tk=K.tk,tn=K.tn,k=tk*16,n=tn*16,nw=8*K.ka+4*K.hk,nb=(tn+7)/8;
        int nky=1; while (nb*nky < 32 && nky < 8 && nky*2 <= tk) nky *= 2;
        const size_t words=(size_t)tk*tn*nw;
        std::vector<uint32_t> hw(words); for(auto& w:hw)w=rnd();
        std::vector<float> hx(k),ha(n),hb(n); for(auto& v:hx)v=((int)(rnd()%2001)-1000)/512.f;
        std::vector<uint16_t> hs(n); for(auto& v:hs)v=(uint16_t)(0x3c00+(rnd()&63));
        CUdeviceptr dtrel=0,dxh=0,dsvh=0,dy=0,dt=0,dctr=0; void*p=nullptr;
        CK(cudaMalloc(&p,words*4));dtrel=(CUdeviceptr)(uintptr_t)p;CK(cudaMemcpy(p,hw.data(),words*4,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&p,k*4));dxh=(CUdeviceptr)(uintptr_t)p;CK(cudaMemcpy(p,hx.data(),k*4,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&p,n*2));dsvh=(CUdeviceptr)(uintptr_t)p;CK(cudaMemcpy(p,hs.data(),n*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&p,n*4));dy=(CUdeviceptr)(uintptr_t)p;
        CK(cudaMalloc(&p,(size_t)nky*n*4));dt=(CUdeviceptr)(uintptr_t)p;
        CK(cudaMalloc(&p,(size_t)nb*4));dctr=(CUdeviceptr)(uintptr_t)p;CK(cudaMemset(p,0,(size_t)nb*4));
        launch(z.ref,dtrel,dxh,dsvh,dy,dt,dctr,tk,tn,K.ka,K.hk,nky);
        CK(cudaDeviceSynchronize());CK(cudaMemcpy(ha.data(),(void*)(uintptr_t)dy,n*4,cudaMemcpyDeviceToHost));
        launch(z.cand,dtrel,dxh,dsvh,dy,dt,dctr,tk,tn,K.ka,K.hk,nky);
        CK(cudaDeviceSynchronize());CK(cudaMemcpy(hb.data(),(void*)(uintptr_t)dy,n*4,cudaMemcpyDeviceToHost));
        size_t bad=0;float maxd=0;for(int i=0;i<n;++i){maxd=std::max(maxd,fabsf(ha[i]-hb[i]));if(memcmp(&ha[i],&hb[i],4))++bad;}
        printf("bitwise K=%s badbits=%zu/%d maxabs=%.3g\n",K.name,bad,n,maxd);
        if(bad)exact=false;
        if(!correctonly){
            std::vector<float> tv[2];
            auto job=[&](int v){launch(v?z.cand:z.ref,dtrel,dxh,dsvh,dy,dt,dctr,tk,tn,K.ka,K.hk,nky);};
            for(int v=0;v<2;++v)for(int i=0;i<5;++i)job(v);
            CK(cudaDeviceSynchronize());
            for(int rep=0;rep<31;++rep)for(int ii=0;ii<2;++ii){int v=(rep+ii)&1;CK(cudaEventRecord(ev0));job(v);CK(cudaEventRecord(ev1));CK(cudaEventSynchronize(ev1));float ms;CK(cudaEventElapsedTime(&ms,ev0,ev1));tv[v].push_back(ms);}
            float a=*std::min_element(tv[0].begin(),tv[0].end()),b=*std::min_element(tv[1].begin(),tv[1].end());
            printf("timing K=%s ref_ms=%.4f cand_ms=%.4f ref/cand=%.4f\n",K.name,a,b,a/b);
        }
        CK(cudaFree((void*)(uintptr_t)dtrel));CK(cudaFree((void*)(uintptr_t)dxh));CK(cudaFree((void*)(uintptr_t)dsvh));
        CK(cudaFree((void*)(uintptr_t)dy));CK(cudaFree((void*)(uintptr_t)dt));CK(cudaFree((void*)(uintptr_t)dctr));
    }
    CK(cudaEventDestroy(ev0));CK(cudaEventDestroy(ev1));CU(cuModuleUnload(z.m));
    return exact?0:2;
}
