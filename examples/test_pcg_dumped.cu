// test_pcg_dumped.cu — run GBD-PCG standalone on the REAL dumped Schur (S,Pinv,gamma) from an MPCGPU
// solve (/tmp/mpc_{S,Pinv,gamma}.bin, float32, [L|D|R] strip layout). Isolates the cooperative kernel
// from the SQP setup: numpy PCG on the same (symmetric, post-fix) matrices converges in ~35; if this
// harness takes ~100, the residual is in the GBD-PCG kernel itself.
//   nvcc -O3 -I../include -I../GLASS -arch=sm_120 -DSTATE_SIZE=14 -DKNOT_POINTS=32 test_pcg_dumped.cu -o test_dumped.exe
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"

static std::vector<float> loadbin(const char* fn, size_t n){
    std::vector<float> v(n); FILE* f=fopen(fn,"rb");
    if(!f){ printf("cannot open %s\n",fn); exit(1);}
    size_t got=fread(v.data(),sizeof(float),n,f); fclose(f);
    if(got!=n){ printf("%s: read %zu != %zu\n",fn,got,n); exit(1);} return v;
}
static inline int sidx(int i,int slot,int r,int c,int d){ return i*3*d*d+slot*d*d+c*d+r; }

int main(int argc, char** argv){
    const int d=STATE_SIZE, N=KNOT_POINTS; const uint32_t ss=d*d;
    float rel = (argc>1)? atof(argv[1]) : 1e-4f;
    int maxit = (argc>2)? atoi(argv[2]) : 500;
    auto h_S    = loadbin("/tmp/mpc_S.bin",    3*ss*N);
    auto h_Pinv = loadbin("/tmp/mpc_Pinv.bin", 3*ss*N);
    auto h_gamma= loadbin("/tmp/mpc_gamma.bin", d*N);
    std::vector<float> h_lambda(d*N,0.0f);

    float *d_S,*d_Pinv,*d_gamma,*d_lambda,*d_r,*d_p,*d_v,*d_eta;
    gpuErrchk(cudaMalloc(&d_S,3*ss*N*sizeof(float)));    gpuErrchk(cudaMalloc(&d_Pinv,3*ss*N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_gamma,d*N*sizeof(float)));   gpuErrchk(cudaMalloc(&d_lambda,d*N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_r,d*N*sizeof(float)));       gpuErrchk(cudaMalloc(&d_p,d*N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_v,N*sizeof(float)));         gpuErrchk(cudaMalloc(&d_eta,N*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_S,h_S.data(),3*ss*N*sizeof(float),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Pinv,h_Pinv.data(),3*ss*N*sizeof(float),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_gamma,h_gamma.data(),d*N*sizeof(float),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lambda,h_lambda.data(),d*N*sizeof(float),cudaMemcpyHostToDevice));

    pcg_config<float> config; config.pcg_exit_tol=1e-6f; config.pcg_rel_tol=rel; config.pcg_max_iter=maxit;
    uint32_t iters = solvePCG<float>(d,N,d_S,d_Pinv,d_gamma,d_lambda,d_r,d_p,d_v,d_eta,&config);
    gpuErrchk(cudaMemcpy(h_lambda.data(),d_lambda,d*N*sizeof(float),cudaMemcpyDeviceToHost));

    // host residual ||gamma - S*lambda|| via block-tridiagonal multiply (col-major strips)
    double res=0,gn=0;
    for(int i=0;i<N;i++) for(int r=0;r<d;r++){
        double sx=0;
        for(int c=0;c<d;c++){
            sx += h_S[sidx(i,1,r,c,d)]*h_lambda[i*d+c];
            if(i>0)   sx += h_S[sidx(i,0,r,c,d)]*h_lambda[(i-1)*d+c];
            if(i<N-1) sx += h_S[sidx(i,2,r,c,d)]*h_lambda[(i+1)*d+c];
        }
        double e=h_gamma[i*d+r]-sx; res+=e*e; gn+=h_gamma[i*d+r]*h_gamma[i*d+r];
    }
    printf("GBD-PCG on dumped Schur: iters=%u  ||gamma-S*lambda||/||gamma|| = %.3e  (rel_tol=%.0e maxit=%d)\n",
           iters, std::sqrt(res/(gn+1e-30)), rel, maxit);
    return 0;
}
