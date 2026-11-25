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
#include <functional>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"cudart %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} } while(0)
#define CU(x) do { CUresult r_=(x); if(r_!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r_,&s);fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,s?s:"?");exit(1);} } while(0)
static uint32_t rs_=777001u;
static uint32_t rnd(){rs_^=rs_<<13;rs_^=rs_>>17;rs_^=rs_<<5;return rs_;}
static std::vector<char> readfile(const char* p){std::ifstream f(p,std::ios::binary);if(!f){fprintf(stderr,"open %s failed\n",p);exit(1);}return std::vector<char>(std::istreambuf_iterator<char>(f),{});}
static float timeit(std::function<void()> f,int reps=31){
 cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
 for(int i=0;i<5;++i)f();CK(cudaDeviceSynchronize());std::vector<float> v;
 for(int i=0;i<reps;++i){CK(cudaEventRecord(a));f();CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));v.push_back(ms);}
 std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];
}
struct Mod {CUmodule m{}; CUfunction f{};};
static Mod loadmod(const char* path){
 static const char* name="_Z23inkwell_exl3_gemv_moe_sPKmS0_PKiPKfS4_iPfiiiiiiiS2_ii";
 auto bytes=readfile(path);Mod z;CU(cuModuleLoadData(&z.m,bytes.data()));CU(cuModuleGetFunction(&z.f,z.m,name));return z;
}
static void launch(const Mod& z,CUdeviceptr dd,CUdeviceptr dids,CUdeviceptr dx,CUdeviceptr dt,
                   int n,int tk,int tn,int k,int nky,int nslots,int pw){
 CUdeviceptr zero=0;int xh2s=0,tslot=n,tzoff=0,skipmode=0;
 void* args[]={&dd,&zero,&dids,&dx,&zero,&xh2s,&dt,&tslot,&tzoff,&tk,&tn,&k,&n,&nky,&zero,&skipmode,&pw};
 unsigned gx=(tn+7)/8,gy=nslots,gz=nky;
 CU(cuLaunchKernel(z.f,gx,gy,gz,256,1,1,0,nullptr,args,nullptr));
}
int main(int argc,char**argv){
 if(argc<3){fprintf(stderr,"usage: %s control.cubin candidate.cubin [correctonly]\n",argv[0]);return 1;}
 bool correctonly=argc>3;
 CK(cudaFree(0));CU(cuInit(0));
 Mod oldm=loadmod(argv[1]), newm=loadmod(argv[2]);
 struct K{int ka,hk,kk;const char*name;} ks[]={{1,1,3,"1.5"},{2,0,4,"2.0"},{2,1,5,"2.5"},{3,0,6,"3.0"},{3,1,7,"3.5"},{4,0,8,"4.0"}};
 printf("K exact gu_old gu_new gu_speed dn_old dn_new dn_speed\n");
 for(auto&K:ks){
  int nw=8*K.ka+4*K.hk; double ratios[2]={0,0},times[4]={0,0,0,0};
  for(int sh=0;sh<2;++sh){
   const int tk=sh?72:144,tn=sh?144:72,nslots=8;
   const size_t words=(size_t)tk*tn*nw;
   std::vector<uint32_t> hw(words);for(auto&w:hw)w=rnd();
   const int k=tk*16,n=tn*16;std::vector<float> hx((size_t)nslots*k);for(auto&v:hx)v=((int)(rnd()%2001)-1000)/512.f;
   float *dx,*dt;CK(cudaMalloc(&dx,hx.size()*4));CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
   CK(cudaMalloc(&dt,(size_t)nslots*n*4));
   std::vector<uint64_t> hd(nslots*4);std::vector<CUdeviceptr> mem(nslots);
   for(int s=0;s<nslots;++s){CK(cudaMalloc((void**)&mem[s],words*4));CK(cudaMemcpy((void*)mem[s],hw.data(),words*4,cudaMemcpyHostToDevice));hd[s*4]=(uint64_t)mem[s];hd[s*4+3]=K.ka|(K.hk<<8);}
   CUdeviceptr dd,dids;CK(cudaMalloc((void**)&dd,hd.size()*8));CK(cudaMemcpy((void*)dd,hd.data(),hd.size()*8,cudaMemcpyHostToDevice));
   std::vector<int> hi(nslots);for(int i=0;i<nslots;++i)hi[i]=i;CK(cudaMalloc((void**)&dids,nslots*4));CK(cudaMemcpy((void*)dids,hi.data(),nslots*4,cudaMemcpyHostToDevice));
   if(sh==0){
    size_t outn=(size_t)nslots*n;std::vector<float> a(outn),b(outn);
    CK(cudaMemset(dt,0,outn*4));launch(oldm,dd,dids,(CUdeviceptr)(uintptr_t)dx,(CUdeviceptr)(uintptr_t)dt,n,tk,tn,k,1,nslots,0);CK(cudaDeviceSynchronize());CK(cudaMemcpy(a.data(),dt,outn*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dt,0,outn*4));launch(newm,dd,dids,(CUdeviceptr)(uintptr_t)dx,(CUdeviceptr)(uintptr_t)dt,n,tk,tn,k,1,nslots,1<<30);CK(cudaDeviceSynchronize());CK(cudaMemcpy(b.data(),dt,outn*4,cudaMemcpyDeviceToHost));
    size_t bad=0;float maxd=0;for(size_t i=0;i<outn;++i){maxd=std::max(maxd,fabsf(a[i]-b[i]));if(memcmp(&a[i],&b[i],4))++bad;}
    printf("%s bad=%zu/%zu max=%.3g ",K.name,bad,outn,maxd);fflush(stdout);if(bad){printf("\n");return 2;}
   }
   if(!correctonly){
    double to=1e30,tnw=1e30;
    for(int r=0;r<2;++r){
     to=std::min(to,(double)timeit([&]{CK(cudaMemset(dt,0,(size_t)nslots*n*4));launch(oldm,dd,dids,(CUdeviceptr)(uintptr_t)dx,(CUdeviceptr)(uintptr_t)dt,n,tk,tn,k,2,nslots,0);}));
     tnw=std::min(tnw,(double)timeit([&]{CK(cudaMemset(dt,0,(size_t)nslots*n*4));launch(newm,dd,dids,(CUdeviceptr)(uintptr_t)dx,(CUdeviceptr)(uintptr_t)dt,n,tk,tn,k,2,nslots,1<<30);}));
    }
    ratios[sh]=to/tnw;times[sh*2]=to;times[sh*2+1]=tnw;
   }
   for(auto p:mem)CK(cudaFree((void*)p));CK(cudaFree(dx));CK(cudaFree(dt));CK(cudaFree((void*)dd));CK(cudaFree((void*)dids));
  }
  if(correctonly) printf("\n"); else printf("gu %.4f %.4f %.3fx dn %.4f %.4f %.3fx\n",times[0],times[1],ratios[0],times[2],times[3],ratios[1]);
 }
 return 0;
}
