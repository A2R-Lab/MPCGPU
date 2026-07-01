// test_bdmv.cu — verify the cooperative bdmv (block-tridiagonal matvec) on the REAL dumped S.
// Replicates pcg.cuh's populate (zero block-0 L / last-block R strips) + loadbdVec + bdmv for p=ones,
// writes (S*p) per block, and the host compares to a direct block-tridiagonal multiply. If they differ,
// the kernel's matvec is wrong (=> PCG iterates a wrong operator => false convergence).
//   nvcc -O3 -I../include -I../GLASS -arch=sm_120 -DSTATE_SIZE=14 -DKNOT_POINTS=32 test_bdmv.cu -o test_bdmv.exe
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cooperative_groups.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"
namespace cg = cooperative_groups;

template<typename T, uint32_t state_size, uint32_t knot_points>
__global__ void k_bdmv(T* d_S, T* d_p, T* d_out){
    const uint32_t block_id=blockIdx.x, tid=threadIdx.x, bdim=blockDim.x;
    const uint32_t ss=state_size, states_sq=ss*ss;
    extern __shared__ T s[];
    T* s_S = s; T* s_vec = s_S + 3*states_sq;     // s_vec holds 3*state_size (halo)
    // populate S strip, zeroing absent L (block 0) / R (last) strips — exactly like pcg.cuh
    for(uint32_t ind=tid; ind<3*states_sq; ind+=bdim){
        if(block_id==0 && ind<states_sq){ s_S[ind]=0; continue; }
        if(block_id==knot_points-1 && ind>=2*states_sq){ s_S[ind]=0; continue; }
        s_S[ind]=d_S[block_id*states_sq*3+ind];
    }
    __syncthreads();
    loadbdVec<T,state_size,knot_points-1>(s_vec, block_id, &d_p[block_id*state_size]);
    __syncthreads();
    T* s_out = s_vec + 3*state_size;
    bdmv<T>(s_out, s_S, s_vec, state_size, knot_points-1, block_id);
    __syncthreads();
    for(uint32_t i=tid;i<state_size;i+=bdim) d_out[block_id*state_size+i]=s_out[i];
}

static inline int sidx(int i,int slot,int r,int c,int d){ return i*3*d*d+slot*d*d+c*d+r; }

int main(int argc, char** argv){
    const int d=STATE_SIZE,N=KNOT_POINTS; const uint32_t ss=d*d;
    const char* fn = (argc>1)? argv[1] : "/tmp/mpc_S.bin";
    std::vector<float> h_S(3*ss*N); FILE* f=fopen(fn,"rb"); fread(h_S.data(),4,3*ss*N,f); fclose(f);
    printf("matrix=%s\n", fn);
    std::vector<float> h_p(d*N,1.0f);   // p = ones
    float *d_S,*d_p,*d_out; gpuErrchk(cudaMalloc(&d_S,3*ss*N*4)); gpuErrchk(cudaMalloc(&d_p,d*N*4)); gpuErrchk(cudaMalloc(&d_out,d*N*4));
    gpuErrchk(cudaMemcpy(d_S,h_S.data(),3*ss*N*4,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_p,h_p.data(),d*N*4,cudaMemcpyHostToDevice));
    size_t smem = (3*ss + 3*d + d)*sizeof(float);
    void* args[]={&d_S,&d_p,&d_out};
    gpuErrchk(cudaLaunchCooperativeKernel((void*)k_bdmv<float,STATE_SIZE,KNOT_POINTS>, N, 64, args, smem));
    gpuErrchk(cudaDeviceSynchronize());
    std::vector<float> gpu(d*N); gpuErrchk(cudaMemcpy(gpu.data(),d_out,d*N*4,cudaMemcpyDeviceToHost));
    // host reference: (S*p)_i = L_i p_{i-1} + D_i p_i + R_i p_{i+1}
    double maxerr=0,maxval=0; int wi=-1;
    for(int i=0;i<N;i++) for(int r=0;r<d;r++){
        double sx=0;
        for(int c=0;c<d;c++){
            sx += h_S[sidx(i,1,r,c,d)]*h_p[i*d+c];
            if(i>0)   sx += h_S[sidx(i,0,r,c,d)]*h_p[(i-1)*d+c];
            if(i<N-1) sx += h_S[sidx(i,2,r,c,d)]*h_p[(i+1)*d+c];
        }
        double e=std::fabs(gpu[i*d+r]-sx);
        if(e>maxerr){maxerr=e;wi=i;} if(std::fabs(sx)>maxval)maxval=std::fabs(sx);
    }
    printf("bdmv GPU vs host: max|diff|=%.3e (rel %.3e) at block %d   |S*p|max=%.3e\n",
           maxerr, maxerr/(maxval+1e-30), wi, maxval);
    // per-block error to see if it's boundary-localized
    for(int i=0;i<N;i++){ double be=0; for(int r=0;r<d;r++){
        double sx=0; for(int c=0;c<d;c++){ sx+=h_S[sidx(i,1,r,c,d)]*h_p[i*d+c]; if(i>0)sx+=h_S[sidx(i,0,r,c,d)]*h_p[(i-1)*d+c]; if(i<N-1)sx+=h_S[sidx(i,2,r,c,d)]*h_p[(i+1)*d+c]; }
        be=std::max(be,(double)std::fabs(gpu[i*d+r]-sx)); }
        if(i<3||i>N-3||be>1e-2) printf("  block %2d: max|diff|=%.3e\n",i,be);
    }
    return 0;
}
