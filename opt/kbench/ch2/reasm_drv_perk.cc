// Driver-API per-K validator/benchmark for externally reassembled cubins.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"cudart %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} } while(0)
#define CU(x) do { CUresult r_=(x); if(r_!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r_,&s);fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,s?s:"?");exit(1);} } while(0)

static uint32_t rs_=777001u;
static uint32_t rnd(){rs_^=rs_<<13;rs_^=rs_>>17;rs_^=rs_<<5;return rs_;}
static std::vector<char> readfile(const char* p){
    std::ifstream f(p,std::ios::binary);
    if(!f){fprintf(stderr,"open %s failed\n",p);exit(1);}
    return std::vector<char>(std::istreambuf_iterator<char>(f),{});
}

struct Mod { CUmodule m{}; CUfunction f{}; };
static Mod loadmod(const char* path){
    static const char* name="_Z23inkwell_exl3_gemv_moe_sPKmS0_PKiPKfS4_iPfiiiiiiiS2_ii";
    auto bytes=readfile(path);
    Mod z;
    CU(cuModuleLoadData(&z.m,bytes.data()));
    CU(cuModuleGetFunction(&z.f,z.m,name));
    return z;
}

static void launch(const Mod& z,CUdeviceptr dd,CUdeviceptr dids,CUdeviceptr dx,CUdeviceptr dt,
                   int n,int tk,int tn,int k,int nky,int nslots,int pw){
    CUdeviceptr zero=0;
    int xh2s=0,tslot=n,tzoff=0,skipmode=0;
    void* args[]={&dd,&zero,&dids,&dx,&zero,&xh2s,&dt,&tslot,&tzoff,&tk,&tn,&k,&n,&nky,&zero,&skipmode,&pw};
    unsigned gx=(tn+7)/8,gy=nslots,gz=nky;
    CU(cuLaunchKernel(z.f,gx,gy,gz,256,1,1,0,nullptr,args,nullptr));
}

struct ShapeData {
    int tk,tn,nslots,k,n;
    size_t words,outn;
    CUdeviceptr dx=0,dt=0,dd=0,dids=0;
    std::vector<CUdeviceptr> mem;
};

static ShapeData setup_shape(int ka,int hk,int tk,int tn){
    ShapeData d;
    d.tk=tk; d.tn=tn; d.nslots=8;
    int nw=8*ka+4*hk;
    d.words=(size_t)tk*tn*nw;
    d.k=tk*16; d.n=tn*16; d.outn=(size_t)d.nslots*d.n;
    std::vector<uint32_t> hw(d.words);
    for(auto& w:hw) w=rnd();
    std::vector<float> hx((size_t)d.nslots*d.k);
    for(auto& v:hx) v=((int)(rnd()%2001)-1000)/512.f;

    void* p=nullptr;
    CK(cudaMalloc(&p,hx.size()*4));
    CK(cudaMemcpy(p,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
    d.dx=(CUdeviceptr)(uintptr_t)p;
    p=nullptr;
    CK(cudaMalloc(&p,d.outn*4));
    d.dt=(CUdeviceptr)(uintptr_t)p;

    std::vector<uint64_t> hd(d.nslots*4,0);
    d.mem.resize(d.nslots);
    for(int s=0;s<d.nslots;++s){
        void* q=nullptr;
        CK(cudaMalloc(&q,d.words*4));
        CK(cudaMemcpy(q,hw.data(),d.words*4,cudaMemcpyHostToDevice));
        d.mem[s]=(CUdeviceptr)(uintptr_t)q;
        hd[s*4]=(uint64_t)d.mem[s];
        hd[s*4+3]=(uint64_t)(ka|(hk<<8));
    }
    p=nullptr;
    CK(cudaMalloc(&p,hd.size()*8));
    CK(cudaMemcpy(p,hd.data(),hd.size()*8,cudaMemcpyHostToDevice));
    d.dd=(CUdeviceptr)(uintptr_t)p;
    std::vector<int> hi(d.nslots);
    for(int i=0;i<d.nslots;++i) hi[i]=i;
    p=nullptr;
    CK(cudaMalloc(&p,d.nslots*4));
    CK(cudaMemcpy(p,hi.data(),d.nslots*4,cudaMemcpyHostToDevice));
    d.dids=(CUdeviceptr)(uintptr_t)p;
    return d;
}

static void teardown_shape(ShapeData& d){
    for(auto p:d.mem) CK(cudaFree((void*)(uintptr_t)p));
    if(d.dx) CK(cudaFree((void*)(uintptr_t)d.dx));
    if(d.dt) CK(cudaFree((void*)(uintptr_t)d.dt));
    if(d.dd) CK(cudaFree((void*)(uintptr_t)d.dd));
    if(d.dids) CK(cudaFree((void*)(uintptr_t)d.dids));
}

static void run_once(const Mod& z,ShapeData& d,int nky,int pw){
    CK(cudaMemset((void*)(uintptr_t)d.dt,0,d.outn*4));
    launch(z,d.dd,d.dids,d.dx,d.dt,d.n,d.tk,d.tn,d.k,nky,d.nslots,pw);
    CK(cudaDeviceSynchronize());
}

struct Cmp { size_t bad; float maxabs; };
static Cmp compare_output(const std::vector<float>& a,const std::vector<float>& b){
    Cmp c{0,0.f};
    for(size_t i=0;i<a.size();++i){
        c.maxabs=std::max(c.maxabs,fabsf(a[i]-b[i]));
        if(memcmp(&a[i],&b[i],4)!=0) ++c.bad;
    }
    return c;
}

int main(int argc,char**argv){
    if(argc<4){
        fprintf(stderr,"usage: %s unpatched.cubin patched_nops.cubin patched_nopless.cubin [correctonly]\n",argv[0]);
        return 1;
    }
    bool correctonly=argc>4;
    CK(cudaFree(0));
    CU(cuInit(0));
    Mod mods[3]={loadmod(argv[1]),loadmod(argv[2]),loadmod(argv[3])};
    const char* vname[3]={"unpatched","nops","nopless"};
    int pw[3]={0,1<<30,1<<30};

    struct K{int ka,hk,kk;const char*name;} ks[]={
        {1,1,3,"1.5"},{2,0,4,"2.0"},{2,1,5,"2.5"},
        {3,0,6,"3.0"},{3,1,7,"3.5"},{4,0,8,"4.0"}
    };
    struct Shape{int tk,tn;const char*name;} shapes[]={{144,72,"gu"},{72,144,"dn"}};
    bool exact=true;

    cudaEvent_t ev0,ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));

    for(auto& K:ks){
        for(auto& S:shapes){
            ShapeData d=setup_shape(K.ka,K.hk,S.tk,S.tn);

            std::vector<float> out[3];
            for(int v=0;v<3;++v){
                out[v].resize(d.outn);
                run_once(mods[v],d,1,pw[v]);
                CK(cudaMemcpy(out[v].data(),(void*)(uintptr_t)d.dt,d.outn*4,cudaMemcpyDeviceToHost));
            }
            for(int v=1;v<3;++v){
                Cmp c=compare_output(out[0],out[v]);
                printf("bitwise K=%s shape=%s variant=%s badbits=%zu/%zu maxabs=%.3g\n",
                       K.name,S.name,vname[v],c.bad,d.outn,c.maxabs);
                if(c.bad) exact=false;
            }

            if(!correctonly){
                auto job=[&](int v){
                    CK(cudaMemset((void*)(uintptr_t)d.dt,0,d.outn*4));
                    launch(mods[v],d.dd,d.dids,d.dx,d.dt,d.n,d.tk,d.tn,d.k,2,d.nslots,pw[v]);
                };
                for(int v=0;v<3;++v) for(int i=0;i<5;++i) job(v);
                CK(cudaDeviceSynchronize());
                std::vector<float> tv[3];
                for(int rep=0;rep<31;++rep){
                    for(int ii=0;ii<3;++ii){
                        int v=(rep+ii)%3;
                        CK(cudaEventRecord(ev0));
                        job(v);
                        CK(cudaEventRecord(ev1));
                        CK(cudaEventSynchronize(ev1));
                        float ms=0.f;
                        CK(cudaEventElapsedTime(&ms,ev0,ev1));
                        tv[v].push_back(ms);
                    }
                }
                CK(cudaDeviceSynchronize());
                float mn[3];
                for(int v=0;v<3;++v) mn[v]=*std::min_element(tv[v].begin(),tv[v].end());
                printf("timing K=%s shape=%s unpatched_ms=%.4f nops_ms=%.4f nopless_ms=%.4f unpatched/nops=%.4f unpatched/nopless=%.4f nops/nopless=%.4f\n",
                       K.name,S.name,mn[0],mn[1],mn[2],mn[0]/mn[1],mn[0]/mn[2],mn[1]/mn[2]);
            }
            teardown_shape(d);
        }
    }
    CK(cudaEventDestroy(ev0));
    CK(cudaEventDestroy(ev1));
    for(auto& z:mods) if(z.m) CU(cuModuleUnload(z.m));
    return exact?0:2;
}
