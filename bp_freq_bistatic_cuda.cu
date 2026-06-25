#include "mex.h"
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// CUDA error checking
// -----------------------------------------------------------------------------
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:CUDA",
                          "CUDA error %s at %s:%d",
                          cudaGetErrorString(code), file, line);
    }
}

// -----------------------------------------------------------------------------
// Kernel: frequency-domain bistatic backprojection
// -----------------------------------------------------------------------------
__global__
void bp_kernel(
    float tx_x, float tx_y, float tx_z,
    const float* __restrict__ rx_x,
    const float* __restrict__ rx_y,
    const float* __restrict__ rx_z,
    const float2* __restrict__ sig,     // [nf x nXcvr], column-major
    const float* __restrict__ xx,
    const float* __restrict__ yy,
    const float* __restrict__ zz,
    const float* __restrict__ f,
    float c,
    int nVox,
    int nXcvr,
    int nf,
    float2* __restrict__ outImg
)
{
    int vox = blockIdx.x * blockDim.x + threadIdx.x;
    if (vox >= nVox) return;

    float x = xx[vox];
    float y = yy[vox];
    float z = zz[vox];

    float2 acc = make_float2(0.f, 0.f);

    for (int i = 0; i < nXcvr; ++i)
    {
        float rx0 = rx_x[i];
        float rx1 = rx_y[i];
        float rx2 = rx_z[i];

        float dTV = sqrtf((tx_x - x)*(tx_x - x) +
                          (tx_y - y)*(tx_y - y) +
                          (tx_z - z)*(tx_z - z));

        float dVR = sqrtf((rx0 - x)*(rx0 - x) +
                          (rx1 - y)*(rx1 - y) +
                          (rx2 - z)*(rx2 - z));

        float T = (dTV + dVR) / c;

        for (int k = 0; k < nf; ++k)
        {
            float phase = 2.f * (float)M_PI * f[k] * T;
            float cs = cosf(phase);
            float sn = sinf(phase);

            float2 s = sig[k + nf * i];

            acc.x += s.x * cs - s.y * sn;
            acc.y += s.x * sn + s.y * cs;
        }
    }

    outImg[vox] = acc;
}

// -----------------------------------------------------------------------------
// MEX gateway
// -----------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // 12 inputs expected
    if (nrhs != 12)
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:args",
            "Usage: img = bp_freq_bistatic_cuda(tx_x,tx_y,tx_z, rx_x,rx_y,rx_z, sig, xx,yy,zz, f, c)");

    // TX scalars
    float tx_x = *((float*)mxGetData(prhs[0]));
    float tx_y = *((float*)mxGetData(prhs[1]));
    float tx_z = *((float*)mxGetData(prhs[2]));

    // RX vectors
    float* rx_x = (float*)mxGetData(prhs[3]);
    float* rx_y = (float*)mxGetData(prhs[4]);
    float* rx_z = (float*)mxGetData(prhs[5]);

    // sig (complex single)
    const mxArray* sig_in = prhs[6];

    // voxel grids
    float* xx = (float*)mxGetData(prhs[7]);
    float* yy = (float*)mxGetData(prhs[8]);
    float* zz = (float*)mxGetData(prhs[9]);

    // frequency vector
    float* f  = (float*)mxGetData(prhs[10]);

    // speed of light
    float  c  = *((float*)mxGetData(prhs[11]));

    // Sizes
    int nVox  = (int)mxGetNumberOfElements(prhs[7]);
    int nf    = (int)mxGetM(sig_in);
    int nXcvr = (int)mxGetN(sig_in);

    // Pack sig into float2
    float* sigR = (float*)mxGetData(sig_in);
    float* sigI = (float*)mxGetImagData(sig_in);

    size_t nSig = (size_t)nf * (size_t)nXcvr;
    float2* h_sig = (float2*)mxMalloc(nSig * sizeof(float2));

    for (size_t idx = 0; idx < nSig; ++idx) {
        h_sig[idx].x = sigR[idx];
        h_sig[idx].y = sigI ? sigI[idx] : 0.0f;
    }

    // Output allocation
    plhs[0] = mxCreateNumericMatrix(nVox, 1, mxSINGLE_CLASS, mxCOMPLEX);
    float* outR = (float*)mxGetData(plhs[0]);
    float* outI = (float*)mxGetImagData(plhs[0]);

    float2* h_out = (float2*)mxMalloc(nVox * sizeof(float2));

    // Device memory
    float *d_rx_x, *d_rx_y, *d_rx_z;
    float *d_xx, *d_yy, *d_zz, *d_f;
    float2 *d_sig, *d_out;

    CUDA_CHECK(cudaMalloc(&d_rx_x, nXcvr*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rx_y, nXcvr*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rx_z, nXcvr*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_sig, nSig*sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_xx,  nVox*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yy,  nVox*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zz,  nVox*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f,   nf*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, nVox*sizeof(float2)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_rx_x, rx_x, nXcvr*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx_y, rx_y, nXcvr*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx_z, rx_z, nXcvr*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_sig, h_sig, nSig*sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xx,  xx,    nVox*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_yy,  yy,    nVox*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_zz,  zz,    nVox*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f,   f,     nf*sizeof(float),    cudaMemcpyHostToDevice));

    mxFree(h_sig);

    // Launch kernel
    int block = 256;
    int grid  = (nVox + block - 1) / block;

    bp_kernel<<<grid, block>>>(
        tx_x, tx_y, tx_z,
        d_rx_x, d_rx_y, d_rx_z,
        d_sig,
        d_xx, d_yy, d_zz,
        d_f,
        c,
        nVox,
        nXcvr,
        nf,
        d_out
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nVox*sizeof(float2), cudaMemcpyDeviceToHost));

    for (int i = 0; i < nVox; ++i) {
        outR[i] = h_out[i].x;
        outI[i] = h_out[i].y;
    }

    mxFree(h_out);

    // Cleanup
    cudaFree(d_rx_x);
    cudaFree(d_rx_y);
    cudaFree(d_rx_z);
    cudaFree(d_sig);
    cudaFree(d_xx);
    cudaFree(d_yy);
    cudaFree(d_zz);
    cudaFree(d_f);
    cudaFree(d_out);
}
