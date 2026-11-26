// Re-runnable Phase-1 prefill kernel tests.  nvcc harness; prints PASS/FAIL per suite.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %d %s:%d\n",(int)e,__FILE__,__LINE__); exit(2);}}while(0)
#include "../../../crates/inkwell-cuda/src/prefill.cu"

static uint32_t rngs=1234567;
static uint32_t rnd(){rngs^=rngs<<13;rngs^=rngs>>17;rngs^=rngs<<5;return rngs;}
static float frnd(){return ((int)(rnd()%2001)-1000)*0.001f;}
template<class F> static float timeit(F f,int n=10){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));f();f();CK(cudaDeviceSynchronize());CK(cudaEventRecord(a));for(int i=0;i<n;i++)f();CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/n;}

// Scalar reference core: one output column per thread, exact tk-then-r order.
__global__ void ref_exl3_m1(const uint32_t* trel,const float* x,float* y,int M,int tk,int tn,int ka,int hk){
 int m=blockIdx.y,tn_i=blockIdx.x,c=threadIdx.x,w=((hk ? 256*ka+128 : 256*ka)/32);
 float KINV=__half2float(__ushort_as_half(0x1eee)),KOFF=1024.f*KINV+__half2float(__ushort_as_half(0xc931));
 float s=0,sx=0;
 for(int a=0;a<tk;a++){const uint32_t* ring=trel+((size_t)a*tn+tn_i)*w;
  for(int r=0;r<16;r++){int i=((((c&7)<<2)|((r>>1)&3))<<3)|(r&1)|((r&8)>>2)|((c&8)>>1);
   float xv=x[(size_t)m*tk*16+a*16+r];s+=xv*pf_bs_f32(pf_exl3_win(ring,i,ka,hk));sx+=xv;}}
 y[(size_t)m*tn*16+tn_i*16+c]=fmaf(KINV,s,KOFF*sx);
}

static int test_gemm(){
 struct S{const char*name;int K,N,ka,hk;} ss[]={{"MoE_gu",2304,1152,2,1},{"MoE_dn",1152,2304,2,1},{"Dense_wq",2304,3072,3,0},{"Dense_wo",3072,2304,3,1},{"Dense_rel",2304,384,5,0}};
 bool pass=true; printf("gemm_smem\n");
 for(auto sh:ss){
  int tk=sh.K/16,tn=sh.N/16,maxM=128;size_t nw=(size_t)tk*tn*((sh.hk ? 256*sh.ka+128 : 256*sh.ka)/32);
  std::vector<uint32_t> ht(nw);for(auto&x:ht)x=rnd();
  std::vector<float> hx(maxM*sh.K),hsu(sh.K),hsv(sh.N);for(auto&x:hx)x=frnd();
  std::vector<uint16_t> hu(sh.K),hv(sh.N);for(auto&x:hu)x=(uint16_t)(0x3800+(rnd()&0x3ff));for(auto&x:hv)x=(uint16_t)(0x3800+(rnd()&0x3ff));
  uint32_t*dw;uint16_t*du,*dv;float *dx,*dy,*dr,*ds;CK(cudaMalloc(&dw,nw*4));CK(cudaMalloc(&du,sh.K*2));CK(cudaMalloc(&dv,sh.N*2));
  CK(cudaMalloc(&dx,hx.size()*4));CK(cudaMalloc(&dy,(size_t)maxM*sh.N*4));CK(cudaMalloc(&dr,(size_t)maxM*sh.N*4));CK(cudaMalloc(&ds,(size_t)maxM*(sh.K+sh.N)*4));
  CK(cudaMemcpy(dw,ht.data(),nw*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(du,hu.data(),sh.K*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,hv.data(),sh.N*2,cudaMemcpyHostToDevice));
  for(int M: {1,16,64,127,128}){
   // Reference: same Hadamard kernels + scalar exact core.
   float*xh=ds,*tm=ds+(size_t)M*sh.K;
   inkwell_exl3_had_in_prefill<<<dim3(sh.K/128,M),128>>>(dx,du,xh,M,sh.K);
   ref_exl3_m1<<<dim3(tn,M),16>>>(dw,xh,tm,M,tk,tn,sh.ka,sh.hk);
   inkwell_exl3_had_out_prefill<<<dim3(sh.N/128,M),128>>>(tm,dv,dr,M,sh.N);
   ink_launch_exl3_gemm_smem(dw,du,dv,dx,dy,M,sh.K,sh.N,sh.ka,sh.hk,ds,nullptr);
   CK(cudaDeviceSynchronize());
   std::vector<float> a(M*sh.N),b(M*sh.N);CK(cudaMemcpy(a.data(),dr,a.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),dy,b.size()*4,cudaMemcpyDeviceToHost));
   int bd=0;float ma=0;for(size_t i=0;i<a.size();i++){bd+=memcmp(&a[i],&b[i],4)!=0;ma=fmaxf(ma,fabsf(a[i]-b[i]));}
   double ms=timeit([&]{ink_launch_exl3_gemm_smem(dw,du,dv,dx,dy,M,sh.K,sh.N,sh.ka,sh.hk,ds,nullptr);});
   double tf=2.0*M*sh.K*sh.N/(ms*1e9);
   printf("  %-8s M=%2d bitdiff=%d max=%.2e TF=%.2f\n",sh.name,M,bd,ma,tf);
   if(bd){pass=false;}
  }
  cudaFree(dw);cudaFree(du);cudaFree(dv);cudaFree(dx);cudaFree(dy);cudaFree(dr);cudaFree(ds);
 }
 printf("GEMM %s\n",pass?"PASS":"FAIL");return pass?0:1;
}

// sequential conv oracle, copied semantics from inkwell_conv_step.
__global__ void ref_conv_step(const float*x,float*state,int ro,const float*w,const float*dyn,int ch,int nh,float scale,float*out){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=ch)return;float*st=state+3*(size_t)ro;
 float x0=st[3*c],x1=st[3*c+1],x2=st[3*c+2],x3=x[c];st[3*c]=x1;st[3*c+1]=x2;st[3*c+2]=x3;
 float acc=w[4*c]*x0+w[4*c+1]*x1+w[4*c+2]*x2+w[4*c+3]*x3;
 if(dyn){int h=c/(ch/nh);acc+=dyn[4*h]*x3+dyn[4*h+1]*x2+dyn[4*h+2]*x1+dyn[4*h+3]*x0;}
 out[c]=(x3+acc)*scale;
}
static int test_conv(){
 int C=17,ch=1024,nh=8,ro=3;size_t sn=(size_t)3*(ro+ch);
 std::vector<float> hx(C*ch),hw(4*ch),hd(C*4*nh),hs(sn),hs2(sn),ho(C*ch),hr(C*ch);
 for(auto&x:hx)x=frnd();for(auto&x:hw)x=frnd();for(auto&x:hd)x=frnd();for(auto&x:hs)x=frnd();hs2=hs;
 float *dx,*dst,*dstr,*dw,*dd,*do1,*do2;CK(cudaMalloc(&dx,hx.size()*4));CK(cudaMalloc(&dst,sn*4));CK(cudaMalloc(&dstr,sn*4));CK(cudaMalloc(&dw,hw.size()*4));CK(cudaMalloc(&dd,hd.size()*4));CK(cudaMalloc(&do1,ho.size()*4));CK(cudaMalloc(&do2,hr.size()*4));
 CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dw,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dd,hd.data(),hd.size()*4,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(dst,hs.data(),sn*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dstr,hs.data(),sn*4,cudaMemcpyHostToDevice));
 ink_launch_conv_prefill(dx,dst,ro,dw,dd,C,ch,nh,1.f/3.f,do1,nullptr);
 for(int t=0;t<C;t++)ref_conv_step<<<(ch+255)/256,256>>>(dx+(size_t)t*ch,dstr,ro,dw,dd+(size_t)t*4*nh,ch,nh,1.f/3.f,do2+(size_t)t*ch);
 CK(cudaDeviceSynchronize());
 CK(cudaMemcpy(ho.data(),do1,ho.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hr.data(),do2,hr.size()*4,cudaMemcpyDeviceToHost));
 CK(cudaMemcpy(hs.data(),dst,sn*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hs2.data(),dstr,sn*4,cudaMemcpyDeviceToHost));
 int bd=0;for(size_t i=0;i<ho.size();i++)bd+=memcmp(&ho[i],&hr[i],4)!=0;for(size_t i=0;i<sn;i++)bd+=memcmp(&hs[i],&hs2[i],4)!=0;
 printf("conv bitdiff=%d\nCONV %s\n",bd,bd?"FAIL":"PASS");return bd!=0;
}

// Straightforward math oracle for flash attention (same equations, un tiled).
__global__ void ref_flash(const __half*K,const __half*V,const float*Q,const float*r,const float*rw,const float*snk,float*out,int C,int pos0,int cap,int swa,float scale){
 int q=blockIdx.x,h=blockIdx.y,d=threadIdx.x;if(q>=C||d>=128)return;int hk=h/3,qp=pos0+q;
 float mx=0.f,den=1.f,acc=0;
 int lo=swa?max(0,qp-swa+1):0;
 for(int k=lo;k<=qp;k++){
  int slot=swa?k%swa:k;const __half*kp=K+((size_t)hk*cap+slot)*128;float dot=0;
  for(int z=0;z<128;z++)dot=fmaf(Q[((size_t)q*24+h)*128+z],__half2float(kp[z]),dot);
  float b=-snk[(size_t)q*24+h];int dist=qp-k;if(dist<1024){float rd=0;for(int z=0;z<16;z++)rd=fmaf(r[((size_t)q*24+h)*16+z],rw[((size_t)h*1025+dist)*16+z],rd);b+=rd;}
  float sc=dot*scale+b;float nm=fmaxf(mx,sc),al=expf(mx-nm),p=expf(sc-nm);
  acc=acc*al+p*__half2float(V[((size_t)hk*cap+slot)*128+d]);den=den*al+p;mx=nm;
 }
 out[((size_t)q*24+h)*128+d]=acc/den;
}
extern "C" int ink_launch_attn_decode_split(const void*,const void*,const float*,const float*,const float*,void*,long long,long long,long long,long long,long long,long long,long long,long long,long long,long long,float,int,int,void*);
static int test_flash(){
 int C=512,cap=2048;size_t ks=(size_t)8*cap*128,qs=(size_t)C*24*128,rs=(size_t)C*24*16,ws=(size_t)24*1025*16,ss=(size_t)C*24;
 std::vector<__half> hk(ks),hv(ks);for(auto&x:hk)x=__float2half(frnd()*0.2f);for(auto&x:hv)x=__float2half(frnd()*0.2f);
 std::vector<float> hq(qs),hr(rs),hw(ws),hs(ss),ha(qs);for(auto&x:hq)x=frnd();for(auto&x:hr)x=frnd();for(auto&x:hw)x=frnd()*0.1f;for(auto&x:hs)x=fabsf(frnd())*0.1f;
 __half*dK,*dV;float *dq,*drr,*dw,*ds,*da;CK(cudaMalloc(&dK,ks*2));CK(cudaMalloc(&dV,ks*2));CK(cudaMalloc(&dq,qs*4));CK(cudaMalloc(&drr,rs*4));CK(cudaMalloc(&dw,ws*4));CK(cudaMalloc(&ds,ss*4));CK(cudaMalloc(&da,qs*4));
 CK(cudaMemcpy(dK,hk.data(),ks*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dV,hv.data(),ks*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dq,hq.data(),qs*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(drr,hr.data(),rs*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dw,hw.data(),ws*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,hs.data(),ss*4,cudaMemcpyHostToDevice));
 float *dbias,*dmask,*dout;CK(cudaMalloc(&dbias,(size_t)24*1600*4));CK(cudaMalloc(&dmask,1601*4));CK(cudaMalloc(&dout,24*128*4));
 const float scale=1.f/sqrtf(128.f);
 bool pass=true;std::pair<int,int> cases[]={{0,1024},{0,0},{64,1024},{1100,1024},{1500,1024},{700,0}};for(auto cs:cases){
  ink_launch_flash_prefill_gqa3(dK,dV,dq,drr,dw,ds,da,C,cs.first,cap,cs.second,scale,nullptr);
  CK(cudaDeviceSynchronize());CK(cudaMemcpy(ha.data(),da,qs*4,cudaMemcpyDeviceToHost));
  float ma=0;int rows[]={0,1,17,31,63};
  for(int qq:rows){
   const int qp=cs.first+qq;
   const int nkv=cs.second?std::min(qp+1,cs.second):qp+1;
   std::vector<float> bias((size_t)24*nkv),mask(nkv+1);
   for(int h=0;h<24;h++)for(int j=0;j<nkv;j++){
    bool filled=true;
    if(cs.second && qp<cs.second)filled=j<=qp;
    int dist=0;
    if(cs.second)dist=filled?((qp-j)%cs.second):0;
    else dist=std::min(qp-j,1024);
    float rd=0.f;
    if(dist<1024){for(int z=0;z<16;z++)rd+=hr[((size_t)qq*24+h)*16+z]*hw[((size_t)h*1025+dist)*16+z];}
    bias[(size_t)h*nkv+j]=rd-hs[(size_t)qq*24+h];
    if(h==0)mask[j]=(cs.second&&!filled)?-1e30f:0.f;
   }
   mask[nkv]=0.f;  // the sink is the fixed zero reference
   CK(cudaMemcpy(dbias,bias.data(),bias.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dmask,mask.data(),mask.size()*4,cudaMemcpyHostToDevice));
   int ok=ink_launch_attn_decode_split(dK,dV,dq+(size_t)qq*24*128,dbias,dmask,dout,
     24,8,nkv,128,256,(long long)cap*256,256,(long long)cap*256,128,nkv,scale,1,32,nullptr);
   if(!ok){printf("decode oracle launch failed\n");pass=false;continue;}
   CK(cudaDeviceSynchronize());
   std::vector<float> oref(24*128);CK(cudaMemcpy(oref.data(),dout,oref.size()*4,cudaMemcpyDeviceToHost));
   for(int h=0;h<24;h++)for(int d=0;d<128;d++){
    float x=ha[((size_t)qq*24+h)*128+d],y=oref[(size_t)h*128+d];ma=fmaxf(ma,fabsf(x-y));
   }
  }
  printf("flash pos=%d swa=%d decode_max_abs=%.3e\n",cs.first,cs.second,ma);if(ma>=5e-5)pass=false;
 }
 double ms=timeit([&]{ink_launch_flash_prefill_gqa3(dK,dV,dq,drr,dw,ds,da,C,1100,cap,1024,scale,nullptr);},10);
 double tf=(double)C*1024*24*4*128/(ms*1e9);printf("flash TF=%.2f\nFLASH %s\n",tf,pass?"PASS":"FAIL");
 cudaFree(dbias);cudaFree(dmask);cudaFree(dout);cudaFree(dK);cudaFree(dV);cudaFree(dq);cudaFree(drr);cudaFree(dw);cudaFree(ds);cudaFree(da);
 return !pass;
}

// Row scalar GEMM oracle for grouped MoE core.
__global__ void ref_group_gemm(const uint64_t*D,const int*rex,const float*x,float*y,int P,int tk,int tn){
 int row=blockIdx.y,tn_i=blockIdx.x,c=threadIdx.x;const uint64_t*d=D+rex[row]*4;int ka=d[3]&255,hk=(d[3]>>8)&1,w=((hk ? 256*ka+128 : 256*ka)/32);const uint32_t*tr=(const uint32_t*)(uintptr_t)d[0];
 float KINV=__half2float(__ushort_as_half(0x1eee)),KOFF=1024.f*KINV+__half2float(__ushort_as_half(0xc931));float s=0,sx=0;
 for(int a=0;a<tk;a++){const uint32_t*ring=tr+((size_t)a*tn+tn_i)*w;for(int r=0;r<16;r++){int i=((((c&7)<<2)|((r>>1)&3))<<3)|(r&1)|((r&8)>>2)|((c&8)>>1);float xv=x[(size_t)row*tk*16+a*16+r];s+=xv*pf_bs_f32(pf_exl3_win(ring,i,ka,hk));sx+=xv;}}
 y[(size_t)row*tn*16+tn_i*16+c]=fmaf(KINV,s,KOFF*sx);
}
static int test_moe(){
 // Grouped-GEMM core exactness on the real down shape, using 4 physical weight
 // blobs recycled through all 128 descriptor entries.
 int C=512,P=C*8,K=1152,N=2304,tk=K/16,tn=N/16,ka=2,hk=1,w=((hk ? 256*ka+128 : 256*ka)/32);
 std::vector<uint32_t> hw((size_t)tk*tn*w);for(auto&x:hw)x=rnd();
 uint32_t*dw;uint64_t*dd;float *dx,*dy,*dr;int *dte,*dts,*dtc,*dtn,*drex;
 CK(cudaMalloc(&dw,hw.size()*4));CK(cudaMemcpy(dw,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));
 std::vector<uint64_t> hd(128*4);for(int e=0;e<128;e++){hd[4*e]=(uint64_t)(uintptr_t)dw;hd[4*e+3]=ka|(hk<<8);}CK(cudaMalloc(&dd,hd.size()*8));CK(cudaMemcpy(dd,hd.data(),hd.size()*8,cudaMemcpyHostToDevice));
 std::vector<float> hx(P*K);for(auto&x:hx)x=frnd();CK(cudaMalloc(&dx,hx.size()*4));CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
 CK(cudaMalloc(&dy,(size_t)P*N*4));CK(cudaMalloc(&dr,(size_t)P*N*4));CK(cudaMalloc(&dte,4096));CK(cudaMalloc(&dts,4096));CK(cudaMalloc(&dtc,4096));CK(cudaMalloc(&dtn,4));CK(cudaMalloc(&drex,P*4));
 // Mixed routed rows: alternating 24/32+8-sized expert tiles exercise both
 // the 32-row kernel and the 64-row weight-reuse kernel at realistic partial
 // tile sizes while keeping the same total expanded row count.
 std::vector<int> ht(128),hs(128),hc(128),hn{128},rex(P);
 int pos=0;
 for(int e=0;e<128;e++){
  ht[e]=e;hs[e]=pos;hc[e]=(e&1)?40:24;
  for(int i=0;i<hc[e];i++)rex[pos+i]=e;
  pos+=hc[e];
 }
 if(pos!=P){printf("bad mixed row count %d\n",pos);return 1;}
 CK(cudaMemcpy(dte,ht.data(),ht.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dts,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dtc,hc.data(),hc.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dtn,hn.data(),4,cudaMemcpyHostToDevice));CK(cudaMemcpy(drex,rex.data(),P*4,cudaMemcpyHostToDevice));
 ref_group_gemm<<<dim3(tn,P),16>>>(dd,drex,dx,dr,P,tk,tn);CK(cudaDeviceSynchronize());
 std::vector<float>a(P*N),b(P*N);CK(cudaMemcpy(b.data(),dr,b.size()*4,cudaMemcpyDeviceToHost));
 dim3 g((N/16+3)/4,P/64+128);
 bool pass=true;
 auto check=[&](const char*name,int kind,int mode32,int mode64){
  CK(cudaMemset(dy,0,(size_t)P*N*4));
  if(kind==0) inkwell_exl3_moe_grouped_gemm_kernel<<<g,256>>>(dd,dte,dts,dtc,dtn,dx,dy,tk,tn,0,mode32);
  else if(kind==1) inkwell_exl3_moe_grouped_gemm64_kernel<<<g,256>>>(dd,dte,dts,dtc,dtn,dx,dy,tk,tn,0,mode64);
  else {
   inkwell_exl3_moe_grouped_gemm_kernel<<<g,256>>>(dd,dte,dts,dtc,dtn,dx,dy,tk,tn,0,mode32);
   inkwell_exl3_moe_grouped_gemm64_kernel<<<g,256>>>(dd,dte,dts,dtc,dtn,dx,dy,tk,tn,0,mode64);
  }
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(a.data(),dy,a.size()*4,cudaMemcpyDeviceToHost));
  int bd=0;float ma=0;for(size_t i=0;i<a.size();i++){bd+=memcmp(&a[i],&b[i],4)!=0;ma=fmaxf(ma,fabsf(a[i]-b[i]));}
  double ms=timeit([&]{
   if(kind==0) inkwell_exl3_moe_grouped_gemm_kernel<<<g,256>>>(dd,dte,dts,dtc,dtn,dx,dy,tk,tn,0,mode32);
   else if(kind==1) inkwell_exl3_moe_grouped_gemm64_kernel<<<g,256>>>(dd,dte,dts,dtc,dtn,dx,dy,tk,tn,0,mode64);
   else {
    inkwell_exl3_moe_grouped_gemm_kernel<<<g,256>>>(dd,dte,dts,dtc,dtn,dx,dy,tk,tn,0,mode32);
    inkwell_exl3_moe_grouped_gemm64_kernel<<<g,256>>>(dd,dte,dts,dtc,dtn,dx,dy,tk,tn,0,mode64);
   }
  });
  double tf=2.0*P*K*N/(ms*1e9);printf("moe %-8s bitdiff=%d max=%.2e TF=%.2f\n",name,bd,ma,tf);
  pass=pass&&bd==0;
 };
 check("group32",0,0,0);
 check("group64",1,0,0); // mode zero makes the BM64 core process every tile here.
 check("adaptive",2,1,2);
 printf("MOE %s\n",pass?"PASS":"FAIL");return pass?0:1;
}

// Full grouped-MoE pipeline oracle: identical epilogues, but scalar row GEMVs.
static int test_moe_pipeline(){
 int C=2,P=C*8,KG=2304,NG=1152,ND=2304,ka=2,hk=1,w=20,NV=4;
 int tkg=KG/16,tng=NG/16,tkd=NG/16,tnd=ND/16;
 size_t wg=(size_t)tkg*tng*w,wd=(size_t)tkd*tnd*w;
 std::vector<uint32_t> htrg(wg*NV),htru(wg*NV),htrd(wd*NV);for(auto&x:htrg)x=rnd();for(auto&x:htru)x=rnd();for(auto&x:htrd)x=rnd();
 std::vector<uint16_t> hsug(KG*NV),hsuu(KG*NV),hsvg(NG*NV),hsvu(NG*NV),hsud(NG*NV),hsvd(ND*NV);
 for(auto&x:hsug)x=(uint16_t)(0x3800+(rnd()&0x3ff));for(auto&x:hsuu)x=(uint16_t)(0x3800+(rnd()&0x3ff));
 for(auto&x:hsvg)x=(uint16_t)(0x3800+(rnd()&0x3ff));for(auto&x:hsvu)x=(uint16_t)(0x3800+(rnd()&0x3ff));
 for(auto&x:hsud)x=(uint16_t)(0x3800+(rnd()&0x3ff));for(auto&x:hsvd)x=(uint16_t)(0x3800+(rnd()&0x3ff));
 uint32_t *dtg,*dtu,*dtd;uint16_t *dsug,*dsuu,*dsvg,*dsvu,*dsud,*dsvd;uint64_t *dg,*du,*dd;
 CK(cudaMalloc(&dtg,htrg.size()*4));CK(cudaMalloc(&dtu,htru.size()*4));CK(cudaMalloc(&dtd,htrd.size()*4));
 CK(cudaMemcpy(dtg,htrg.data(),htrg.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dtu,htru.data(),htru.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dtd,htrd.data(),htrd.size()*4,cudaMemcpyHostToDevice));
 CK(cudaMalloc(&dsug,hsug.size()*2));CK(cudaMalloc(&dsuu,hsuu.size()*2));CK(cudaMalloc(&dsvg,hsvg.size()*2));CK(cudaMalloc(&dsvu,hsvu.size()*2));CK(cudaMalloc(&dsud,hsud.size()*2));CK(cudaMalloc(&dsvd,hsvd.size()*2));
 CK(cudaMemcpy(dsug,hsug.data(),hsug.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dsuu,hsuu.data(),hsuu.size()*2,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(dsvg,hsvg.data(),hsvg.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dsvu,hsvu.data(),hsvu.size()*2,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(dsud,hsud.data(),hsud.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dsvd,hsvd.data(),hsvd.size()*2,cudaMemcpyHostToDevice));
 std::vector<uint64_t> hg(128*4),hu(128*4),hd(128*4);
 for(int e=0;e<128;e++){int v=e%NV;hg[4*e]=(uint64_t)(uintptr_t)(dtg+v*wg);hg[4*e+1]=(uint64_t)(uintptr_t)(dsug+v*KG);hg[4*e+2]=(uint64_t)(uintptr_t)(dsvg+v*NG);hg[4*e+3]=ka|(hk<<8);
  hu[4*e]=(uint64_t)(uintptr_t)(dtu+v*wg);hu[4*e+1]=(uint64_t)(uintptr_t)(dsuu+v*KG);hu[4*e+2]=(uint64_t)(uintptr_t)(dsvu+v*NG);hu[4*e+3]=ka|(hk<<8);
  hd[4*e]=(uint64_t)(uintptr_t)(dtd+v*wd);hd[4*e+1]=(uint64_t)(uintptr_t)(dsud+v*NG);hd[4*e+2]=(uint64_t)(uintptr_t)(dsvd+v*ND);hd[4*e+3]=ka|(hk<<8);}
 CK(cudaMalloc(&dg,hg.size()*8));CK(cudaMalloc(&du,hu.size()*8));CK(cudaMalloc(&dd,hd.size()*8));
 CK(cudaMemcpy(dg,hg.data(),hg.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(du,hu.data(),hu.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dd,hd.data(),hd.size()*8,cudaMemcpyHostToDevice));
 std::vector<float> hx(C*KG),hw8(P);std::vector<int> hid(P);for(auto&x:hx)x=frnd();for(int i=0;i<P;i++){hid[i]=(int)(rnd()%128);hw8[i]=0.05f+0.001f*(rnd()%1000);}
 float *dx,*dw8,*wsa,*wsb,*oa,*ob;int *dids,*eo,*stok,*sslot,*inv,*te,*ts,*tc,*tn;
 CK(cudaMalloc(&dx,hx.size()*4));CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
 CK(cudaMalloc(&dw8,P*4));CK(cudaMemcpy(dw8,hw8.data(),P*4,cudaMemcpyHostToDevice));CK(cudaMalloc(&dids,P*4));CK(cudaMemcpy(dids,hid.data(),P*4,cudaMemcpyHostToDevice));
 CK(cudaMalloc(&eo,129*4));CK(cudaMalloc(&stok,P*4));CK(cudaMalloc(&sslot,P*4));CK(cudaMalloc(&inv,P*4));CK(cudaMalloc(&te,256*4));CK(cudaMalloc(&ts,256*4));CK(cudaMalloc(&tc,256*4));CK(cudaMalloc(&tn,4));
 float*dsw;CK(cudaMalloc(&dsw,P*4));
 ink_launch_exl3_moe_group_sort(dids,dw8,C,eo,stok,sslot,dsw,inv,te,ts,tc,tn,nullptr);
 size_t wfloats=(size_t)P*14976+64;CK(cudaMalloc(&wsa,wfloats*4));CK(cudaMalloc(&wsb,wfloats*4));CK(cudaMalloc(&oa,(size_t)C*ND*4));CK(cudaMalloc(&ob,(size_t)C*ND*4));
 ink_launch_exl3_moe_grouped_gemm(dg,du,dd,dx,C,stok,te,ts,tc,tn,dsw,inv,wsa,oa,nullptr);
 // Reference uses the same deterministic epilogues but scalar per-row GEMVs.
 float*xhg=wsb,*xhu=xhg+(size_t)P*KG,*tg=xhu+(size_t)P*KG,*tu=tg+(size_t)P*NG,*gu=tu+(size_t)P*NG,*xhd=gu+(size_t)P*2*NG,*td=xhd+(size_t)P*NG,*tdn=td+(size_t)P*ND;int*rex=(int*)(tdn+(size_t)P*ND);
 inkwell_moe_fill_row_expert<<<1,128>>>(te,ts,tc,tn,rex);
 inkwell_moe_had_in_rows<<<dim3(KG/128,P),128>>>(dg,rex,stok,dx,xhg,P,KG);
 inkwell_moe_had_in_rows<<<dim3(KG/128,P),128>>>(du,rex,stok,dx,xhu,P,KG);
 ref_group_gemm<<<dim3(tng,P),16>>>(dg,rex,xhg,tg,P,tkg,tng);
 ref_group_gemm<<<dim3(tng,P),16>>>(du,rex,xhu,tu,P,tkg,tng);
 inkwell_moe_had_out_pair_rows<<<dim3(NG/128,P,2),128>>>(dg,du,rex,tg,tu,gu,P,NG);
 inkwell_moe_situ_had_rows<<<dim3(NG/128,P),128>>>(dd,rex,gu,xhd,4.f,25.f,P,NG);
 ref_group_gemm<<<dim3(tnd,P),16>>>(dd,rex,xhd,td,P,tkd,tnd);
 inkwell_moe_had_out_rows<<<dim3(ND/128,P),128>>>(dd,rex,td,tdn,P,ND);
 inkwell_moe_weighted_scatter<<<dim3((ND+255)/256,C),256>>>(tdn,dsw,inv,ob,C,ND);
 CK(cudaDeviceSynchronize());
 std::vector<float> va(C*ND),vb(C*ND);CK(cudaMemcpy(va.data(),oa,va.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(vb.data(),ob,vb.size()*4,cudaMemcpyDeviceToHost));
 int bd=0;float ma=0;for(size_t i=0;i<va.size();i++){bd+=memcmp(&va[i],&vb[i],4)!=0;ma=fmaxf(ma,fabsf(va[i]-vb[i]));}
 printf("moe pipeline bitdiff=%d max=%.2e\nMOE_PIPELINE %s\n",bd,ma,bd?"FAIL":"PASS");
 cudaFree(dtg);cudaFree(dtu);cudaFree(dtd);cudaFree(dsug);cudaFree(dsuu);cudaFree(dsvg);cudaFree(dsvu);cudaFree(dsud);cudaFree(dsvd);cudaFree(dg);cudaFree(du);cudaFree(dd);
 cudaFree(dx);cudaFree(dw8);cudaFree(wsa);cudaFree(wsb);cudaFree(oa);cudaFree(ob);cudaFree(dids);cudaFree(eo);cudaFree(stok);cudaFree(sslot);cudaFree(inv);cudaFree(te);cudaFree(ts);cudaFree(tc);cudaFree(tn);cudaFree(dsw);
 return bd!=0;
}

int main(int argc,char**argv){std::string m=argc>1?argv[1]:"all";int rc=0;
 if(m=="gemm"||m=="all")rc|=test_gemm();if(m=="conv"||m=="all")rc|=test_conv();if(m=="flash"||m=="all")rc|=test_flash();if(m=="moe"||m=="all"){rc|=test_moe();rc|=test_moe_pipeline();}
 printf("OVERALL %s\n",rc?"FAIL":"PASS");return rc;}
