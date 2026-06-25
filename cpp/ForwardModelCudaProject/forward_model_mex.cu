#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <thrust/complex.h>

using cfloat = thrust::complex<float>;

// ================================================================
// CUDA kernel: one (tx, rx, freq) per launch
// ================================================================
__global__
void forward_kernel(
    const float* __restrict__ TARGx,
    const float* __restrict__ TARGy,
    const float* __restrict__ TARGz,
    const float* __restrict__ s_loc,
    const float* __restrict__ TXSx,
    const float* __restrict__ TXSy,
    const float* __restrict__ TXSz,
    const float* __restrict__ RX0x,
    const float* __restrict__ RX0y,
    const float* __restrict__ RX0z,
    const float* __restrict__ k,
    int ntarg,
    int ntx,
    int nrx,
    int nf,
    int itx,
    int irx,
    int ifreq,
    cfloat* __restrict__ sigf
)
{
    extern __shared__ cfloat shmem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    cfloat local_sum(0.0f, 0.0f);

    float tx_x = TXSx[itx];
    float tx_y = TXSy[itx];
    float tx_z = TXSz[itx];

    float rx_x = RX0x[irx];
    float rx_y = RX0y[irx];
    float rx_z = RX0z[irx];

    float kf = k[ifreq];

    for (int i = idx; i < ntarg; i += blockDim.x * gridDim.x)
    {
        float x = TARGx[i];
        float y = TARGy[i];
        float z = TARGz[i];
        float s = s_loc[i];

        float dx1 = tx_x - x;
        float dy1 = tx_y - y;
        float dz1 = tx_z - z;

        float dx2 = rx_x - x;
        float dy2 = rx_y - y;
        float dz2 = rx_z - z;

        float r1 = sqrtf(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
        float r2 = sqrtf(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

        float phase = -kf * (r1 + r2);
        float amp = s / (r1 * r2);

        float cosp = __cosf(phase);
        float sinp = __sinf(phase);

        local_sum += cfloat(amp * cosp, amp * sinp);
    }

    shmem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
    {
        // nf x nrx, column-major: row = ifreq, col = irx
        int out_idx = ifreq + irx * nf;

        float* base = reinterpret_cast<float*>(&sigf[out_idx]);
        atomicAdd(base + 0, shmem[0].real());
        atomicAdd(base + 1, shmem[0].imag());
    }
}

// ================================================================
// MEX entry point
// ================================================================
extern "C"
void mexFunction(int nlhs, mxArray* plhs[],
    int nrhs, const mxArray* prhs[])
{
    if (nrhs != 5)
        mexErrMsgTxt("Usage: sigf = forward_model_mex(TARG, TXS, RX0, s, k)");

    const mxArray* TARG = prhs[0];
    const mxArray* TXS = prhs[1];
    const mxArray* RX0 = prhs[2];
    const mxArray* S = prhs[3];
    const mxArray* K = prhs[4];

    int ntarg = mxGetN(TARG);
    int ntx = mxGetM(TXS);
    int nrx = mxGetM(RX0);
    int nf = mxGetNumberOfElements(K);

    // TARG: ntarg x 3
    float* h_TARGx = (float*)mxGetData(TARG);
    float* h_TARGy = h_TARGx + ntarg;
    float* h_TARGz = h_TARGx + 2 * ntarg;

    // TXS: ntx x 3
    float* h_TXSx = (float*)mxGetData(TXS);
    float* h_TXSy = h_TXSx + ntx;
    float* h_TXSz = h_TXSx + 2 * ntx;

    // RX0: nrx x 3
    float* h_RX0x = (float*)mxGetData(RX0);
    float* h_RX0y = h_RX0x + nrx;
    float* h_RX0z = h_RX0x + 2 * nrx;

    float* h_s = (float*)mxGetData(S);
    float* h_k = (float*)mxGetData(K);

    // Output: nf x nrx
    plhs[0] = mxCreateNumericMatrix(nf, nrx, mxSINGLE_CLASS, mxCOMPLEX);
    cfloat* h_sigf = (cfloat*)mxGetData(plhs[0]);

    // Device pointers
    float* d_TARGx, * d_TARGy, * d_TARGz;
    float* d_TXSx, * d_TXSy, * d_TXSz;
    float* d_RX0x, * d_RX0y, * d_RX0z;
    float* d_s, * d_k;
    cfloat* d_sigf;

    cudaMalloc(&d_TARGx, ntarg * sizeof(float));
    cudaMalloc(&d_TARGy, ntarg * sizeof(float));
    cudaMalloc(&d_TARGz, ntarg * sizeof(float));
    cudaMalloc(&d_s, ntarg * sizeof(float));

    cudaMalloc(&d_TXSx, ntx * sizeof(float));
    cudaMalloc(&d_TXSy, ntx * sizeof(float));
    cudaMalloc(&d_TXSz, ntx * sizeof(float));

    cudaMalloc(&d_RX0x, nrx * sizeof(float));
    cudaMalloc(&d_RX0y, nrx * sizeof(float));
    cudaMalloc(&d_RX0z, nrx * sizeof(float));

    cudaMalloc(&d_k, nf * sizeof(float));
    cudaMalloc(&d_sigf, nf * nrx * sizeof(cfloat));

    cudaMemcpy(d_TARGx, h_TARGx, ntarg * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TARGy, h_TARGy, ntarg * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TARGz, h_TARGz, ntarg * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, ntarg * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_TXSx, h_TXSx, ntx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TXSy, h_TXSy, ntx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TXSz, h_TXSz, ntx * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_RX0x, h_RX0x, nrx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RX0y, h_RX0y, nrx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RX0z, h_RX0z, nrx * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_k, h_k, nf * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_sigf, 0, nf * nrx * sizeof(cfloat));

    int block = 256;
    int grid = (ntarg + block - 1) / block;
    size_t shmem = size_t(block) * sizeof(cfloat);

    for (int itx = 0; itx < ntx; ++itx)
        for (int irx = 0; irx < nrx; ++irx)
            for (int ifreq = 0; ifreq < nf; ++ifreq)
                forward_kernel << <grid, block, (int)shmem >> > (
                    d_TARGx, d_TARGy, d_TARGz,
                    d_s,
                    d_TXSx, d_TXSy, d_TXSz,
                    d_RX0x, d_RX0y, d_RX0z,
                    d_k,
                    ntarg, ntx, nrx, nf,
                    itx, irx, ifreq,
                    d_sigf
                    );

    cudaMemcpy(h_sigf, d_sigf, nf * nrx * sizeof(cfloat), cudaMemcpyDeviceToHost);

    cudaFree(d_TARGx); cudaFree(d_TARGy); cudaFree(d_TARGz);
    cudaFree(d_s);
    cudaFree(d_TXSx); cudaFree(d_TXSy); cudaFree(d_TXSz);
    cudaFree(d_RX0x); cudaFree(d_RX0y); cudaFree(d_RX0z);
    cudaFree(d_k);
    cudaFree(d_sigf);
}
