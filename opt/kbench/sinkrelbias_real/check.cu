#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

#define CK(x) do { cudaError_t e=(x); if(e) { fprintf(stderr,"cuda %s at %d\n",cudaGetErrorString(e),__LINE__); exit(2);} } while(0)
#define CB(x) do { cublasStatus_t e=(x); if(e) { fprintf(stderr,"cublas %d at %d\n",(int)e,__LINE__); exit(2);} } while(0)

static std::vector<unsigned char> readfile(const std::string& p) {
    FILE* f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"open %s\n",p.c_str());exit(2);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<unsigned char> b(n); if(n && fread(b.data(),1,n,f)!=(size_t)n){fprintf(stderr,"read %s\n",p.c_str());exit(2);} fclose(f); return b;
}
template<class T> static T* upload(const std::vector<unsigned char>& b) {
    T* p=nullptr; CK(cudaMalloc(&p,b.size())); CK(cudaMemcpy(p,b.data(),b.size(),cudaMemcpyHostToDevice)); return p;
}
__device__ __forceinline__ float blo(unsigned u){return __uint_as_float(u<<16);}
__device__ __forceinline__ float bhi(unsigned u){return __uint_as_float(u&0xffff0000u);}

__global__ void mmvf_sink(const __nv_bfloat16* __restrict__ x,const float* __restrict__ y,float* __restrict__ dst,int ncols2){
    const int row=blockIdx.x,tid=threadIdx.x; extern __shared__ float buf[];
    if(tid<32)buf[tid]=0.f; __syncthreads();
    const __nv_bfloat162* x2=(const __nv_bfloat162*)(x+(size_t)row*(ncols2*2));
    const float2* y2=(const float2*)y; float sumf=0.f;
    for(int col2=tid;col2<ncols2;col2+=256){
        const __nv_bfloat162 a=x2[col2]; const float2 b=__ldg(&y2[col2]);
        sumf += __bfloat162float(a.x)*b.x; sumf += __bfloat162float(a.y)*b.y;
    }
#pragma unroll
    for(int m=16;m>0;m>>=1)sumf+=__shfl_xor_sync(0xffffffffu,sumf,m);
    buf[tid>>5]=sumf; __syncthreads();
    if(tid<32){sumf=buf[tid];
#pragma unroll
        for(int m=16;m>0;m>>=1)sumf+=__shfl_xor_sync(0xffffffffu,sumf,m);
        if(tid==0)dst[row]=sumf;
    }
}
__global__ void gather_ref(const float* __restrict__ mm,const float* __restrict__ sinks,const int* __restrict__ rp,float* __restrict__ y,int nkv,int nh,int ext){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nkv*nh)return; int h=i/nkv,j=i-h*nkv;
    y[i]=__fsub_rn(mm[(size_t)rp[j]+(size_t)h*ext],sinks[h]);
}

template<int V>
__device__ __forceinline__ float dot16(const uint4 q0,const uint4 q1,const float4 r0,const float4 r1,const float4 r2,const float4 r3){
    float w[16]={blo(q0.x),bhi(q0.x),blo(q0.y),bhi(q0.y),blo(q0.z),bhi(q0.z),blo(q0.w),bhi(q0.w),blo(q1.x),bhi(q1.x),blo(q1.y),bhi(q1.y),blo(q1.z),bhi(q1.z),blo(q1.w),bhi(q1.w)};
    float x[16]={r0.x,r0.y,r0.z,r0.w,r1.x,r1.y,r1.z,r1.w,r2.x,r2.y,r2.z,r2.w,r3.x,r3.y,r3.z,r3.w};
    if(V==0){float a=0;
#pragma unroll
        for(int k=0;k<16;k++)a=fmaf(w[k],x[k],a);return a;}
    if(V==1){float a=0;
#pragma unroll
        for(int k=0;k<16;k++)a=__fadd_rn(a,__fmul_rn(w[k],x[k]));return a;}
    if(V==2){float a=0;
#pragma unroll
        for(int k=15;k>=0;k--)a=fmaf(w[k],x[k],a);return a;}
    if(V==3){float p[8];
#pragma unroll
        for(int k=0;k<8;k++)p[k]=__fmul_rn(w[2*k],x[2*k])+__fmul_rn(w[2*k+1],x[2*k+1]);
#pragma unroll
        for(int s=4;s;s>>=1)for(int k=0;k<s;k++)p[k]+=p[k+s];return p[0];}
    if(V==4){float a=0,b=0;
#pragma unroll
        for(int k=0;k<8;k++){a=fmaf(w[2*k],x[2*k],a);b=fmaf(w[2*k+1],x[2*k+1],b);}return a+b;}
    float a[4]={0,0,0,0};
#pragma unroll
    for(int k=0;k<16;k++)a[k&3]=fmaf(w[k],x[k],a[k&3]);
    return (a[0]+a[1])+(a[2]+a[3]);
}

template<int V>
__global__ void fused(const __nv_bfloat16* __restrict__ ws,const float* __restrict__ inp,const __nv_bfloat16* __restrict__ rb,const float* __restrict__ r,const int* __restrict__ rp,float* __restrict__ out,int nkv,int ncols2){
    const int h=blockIdx.x,tid=threadIdx.x; __shared__ float buf[32],ss,sr[16];
    if(tid<32)buf[tid]=0.f; if(tid<16)sr[tid]=r[h*16+tid];__syncthreads();
    const __nv_bfloat162* x2=(const __nv_bfloat162*)(ws+(size_t)h*(ncols2*2)); const float2* y2=(const float2*)inp;float sumf=0.f;
    for(int c=tid;c<ncols2;c+=256){const __nv_bfloat162 a=x2[c];const float2 b=__ldg(&y2[c]);sumf+=__bfloat162float(a.x)*b.x;sumf+=__bfloat162float(a.y)*b.y;}
#pragma unroll
    for(int m=16;m>0;m>>=1)sumf+=__shfl_xor_sync(0xffffffffu,sumf,m);
    buf[tid>>5]=sumf;__syncthreads();
    if(tid<32){sumf=buf[tid];
#pragma unroll
      for(int m=16;m>0;m>>=1)sumf+=__shfl_xor_sync(0xffffffffu,sumf,m);if(tid==0)ss=sumf;}__syncthreads();
    float4 r0=((const float4*)sr)[0],r1=((const float4*)sr)[1],r2=((const float4*)sr)[2],r3=((const float4*)sr)[3];
    for(int j=tid;j<nkv;j+=256){int p=rp[j];const uint4* q=(const uint4*)(rb+(size_t)p*16);uint4 q0=__ldg(q),q1=__ldg(q+1);out[(size_t)h*nkv+j]=__fsub_rn(dot16<V>(q0,q1,r0,r1,r2,r3),ss);}
}

static void stats(const char* name,std::vector<float>& a,std::vector<float>& b){
    size_t bad=0;double ma=0,mr=0;for(size_t i=0;i<a.size();i++){float d=fabsf(a[i]-b[i]);if(d>ma)ma=d;float den=fabsf(a[i]);if(den>1e-30&&d/den>mr)mr=d/den;if(a[i]!=b[i])bad++;}
    printf("%s bit_diffs=%zu/%zu max_abs=%.3g max_rel=%.3g\n",name,bad,a.size(),ma,mr);
}
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: check dir nkv ystride\n");return 2;}std::string dir=argv[1];int nkv=atoi(argv[2]);
    auto hinp=readfile(dir+"/sr_pos64_l5p0_inp0.f32"),hr=readfile(dir+"/sr_pos64_l5p0_r.f32"),hprod=readfile(dir+"/sr_pos64_l5p0_kqb.f32"),hs=readfile(dir+"/sink.bf16"),hrb=readfile(dir+"/rel.bf16"),hrf=readfile(dir+"/rel.f32");
    int ystride=(int)(hprod.size()/4/24);
    float *dinp=upload<float>(hinp),*dr=upload<float>(hr),*ds=upload<float>(std::vector<unsigned char>(24*4)),*dt=upload<float>(hrf),*dy=upload<float>(std::vector<unsigned char>((size_t)nkv*24*4)),*dc=upload<float>(std::vector<unsigned char>((size_t)nkv*24*4)),*dprod=nullptr;
    (void)ds;(void)dt;(void)dy;(void)dc;(void)dprod;
    __nv_bfloat16 *dws=upload<__nv_bfloat16>(hs),*drb=upload<__nv_bfloat16>(hrb);
    int *drp;CK(cudaMalloc(&drp,nkv*4));std::vector<int> hp(nkv);for(int j=0;j<nkv;j++)hp[j]=std::min(63-j,1024);CK(cudaMemcpy(drp,hp.data(),nkv*4,cudaMemcpyHostToDevice));
    float *dsink,*dtable,*dref,*dcand;CK(cudaMalloc(&dsink,24*4));CK(cudaMalloc(&dtable,1025*24*4));CK(cudaMalloc(&dref,nkv*24*4));CK(cudaMalloc(&dcand,nkv*24*4));
    cublasHandle_t cublas;CB(cublasCreate(&cublas));float alpha=1,beta=0;
    CB(cublasSgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,1025,24,16,&alpha,(const float*)dt,16,dr,16,&beta,dtable,1025));
    mmvf_sink<<<24,256,128>>>(dws,dinp,dsink,1152);
    gather_ref<<<(nkv*24+255)/256,256>>>(dtable,dsink,drp,dref,nkv,24,1025);
    CK(cudaDeviceSynchronize());
    std::vector<float> href(nkv*24),hc(nkv*24),hpd((size_t)ystride*24);
    CK(cudaMemcpy(href.data(),dref,nkv*24*4,cudaMemcpyDeviceToHost));
    // production output has stride ystride; compare valid prefix per head
    std::vector<float> hprodvalid(nkv*24),hfile=std::vector<float>((size_t)ystride*24);
    memcpy(hfile.data(),hprod.data(),std::min(hprod.size(),hfile.size()*4));
    for(int h=0;h<24;h++)for(int j=0;j<nkv;j++)hprodvalid[h*nkv+j]=hfile[(size_t)h*ystride+j];
    stats("reference-vs-production",href,hprodvalid);
    for(int v=0;v<6;v++){
      switch(v){
        case 0:fused<0><<<24,256>>>(dws,dinp,drb,dr,drp,dcand,nkv,1152);break;
        case 1:fused<1><<<24,256>>>(dws,dinp,drb,dr,drp,dcand,nkv,1152);break;
        case 2:fused<2><<<24,256>>>(dws,dinp,drb,dr,drp,dcand,nkv,1152);break;
        case 3:fused<3><<<24,256>>>(dws,dinp,drb,dr,drp,dcand,nkv,1152);break;
        case 4:fused<4><<<24,256>>>(dws,dinp,drb,dr,drp,dcand,nkv,1152);break;
        default:fused<5><<<24,256>>>(dws,dinp,drb,dr,drp,dcand,nkv,1152);break;
      }
      CK(cudaDeviceSynchronize());CK(cudaMemcpy(hc.data(),dcand,nkv*24*4,cudaMemcpyDeviceToHost));
      char nm[64];snprintf(nm,sizeof nm,"reference-vs-fused-v%d",v);stats(nm,href,hc);
      snprintf(nm,sizeof nm,"production-vs-fused-v%d",v);stats(nm,hprodvalid,hc);
    }
    return 0;
}
