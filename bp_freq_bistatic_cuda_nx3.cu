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
    const float*  __restrict__ tx,      // [3]
    const float*  __restrict__ rx,      // [nXcvr x 3]
    const float2* __restrict__ sig,     // [nf x nXcvr], column-major
    const float*  __restrict__ xx,      // [nVox]
    const float*  __restrict__ yy,      // [nVox]
    const float*  __restrict__ zz,      // [nVox]
    const float*  __restrict__ f,       // [nf]
    float c,
    int nVox,
    int nXcvr,
    int nf,
    float2* __restrict__ outImg         // [nVox]
)
{
    int vox = blockIdx.x * blockDim.x + threadIdx.x;
    if (vox >= nVox) return;

    float x = xx[vox];
    float y = yy[vox];
    float z = zz[vox];

    float2 acc = make_float2(0.f, 0.f);

    float tx0 = tx[0];
    float tx1 = tx[1];
    float tx2 = tx[2];

    for (int i = 0; i < nXcvr; ++i)
    {
        float rx0 = rx[3*i + 0];
        float rx1 = rx[3*i + 1];
        float rx2 = rx[3*i + 2];

        float dTV = sqrtf((tx0 - x)*(tx0 - x) +
                          (tx1 - y)*(tx1 - y) +
                          (tx2 - z)*(tx2 - z));

        float dVR = sqrtf((rx0 - x)*(rx0 - x) +
                          (rx1 - y)*(rx1 - y) +
                          (rx2 - z)*(rx2 - z));

        float T = (dTV + dVR) / c;

        for (int k = 0; k < nf; ++k)
        {
            float phase = 2.f * (float)M_PI * f[k] * T;
            float cs = cosf(phase);
            float sn = sinf(phase);

            // MATLAB: sig is nf x nXcvr, column-major → index = k + nf*i
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
    if (nrhs != 8)
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:args",
                          "Usage: img = bp_freq_bistatic_cuda(tx, rx, sig, xx, yy, zz, f, c)");

    const mxArray* tx_in  = prhs[0];
    const mxArray* rx_in  = prhs[1];
    const mxArray* sig_in = prhs[2];
    const mxArray* xx_in  = prhs[3];
    const mxArray* yy_in  = prhs[4];
    const mxArray* zz_in  = prhs[5];
    const mxArray* f_in   = prhs[6];
    const mxArray* c_in   = prhs[7];

    // -------------------------------------------------------------------------
    // Input checks
    // -------------------------------------------------------------------------
    if (!mxIsSingle(tx_in) || mxIsComplex(tx_in) || mxGetNumberOfElements(tx_in) != 3)
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:tx",
                          "tx must be 3-element real single.");

    if (!mxIsSingle(rx_in) || mxIsComplex(rx_in) || mxGetN(rx_in) != 3)
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:rx",
                          "rx must be [nXcvr x 3] real single.");

    if (!mxIsSingle(sig_in) || !mxIsComplex(sig_in))
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:sig",
                          "sig must be complex single (nf x nXcvr).");

    if (!mxIsSingle(xx_in) || mxIsComplex(xx_in) ||
        !mxIsSingle(yy_in) || mxIsComplex(yy_in) ||
        !mxIsSingle(zz_in) || mxIsComplex(zz_in))
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:xyz",
                          "xx, yy, zz must be real single vectors.");

    if (!mxIsSingle(f_in) || mxIsComplex(f_in))
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:f",
                          "f must be real single vector.");

    if (!mxIsSingle(c_in) || mxIsComplex(c_in) ||
        mxGetNumberOfElements(c_in) != 1)
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:c",
                          "c must be scalar real single.");

    // -------------------------------------------------------------------------
    // Extract pointers and sizes
    // -------------------------------------------------------------------------
    float* tx = (float*)mxGetData(tx_in);
    float* rx = (float*)mxGetData(rx_in);
    float* xx = (float*)mxGetData(xx_in);
    float* yy = (float*)mxGetData(yy_in);
    float* zz = (float*)mxGetData(zz_in);
    float* f  = (float*)mxGetData(f_in);
    float  c  = ((float*)mxGetData(c_in))[0];

    int nVox  = (int)mxGetNumberOfElements(xx_in);
    int nXcvr = (int)mxGetM(rx_in);      // rows of rx
    int nf    = (int)mxGetNumberOfElements(f_in);

    if ((int)mxGetNumberOfElements(yy_in) != nVox ||
        (int)mxGetNumberOfElements(zz_in) != nVox)
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:xyz_size",
                          "xx, yy, zz must have same number of elements.");

    // sig: nf x nXcvr complex single
    mwSize sigM = mxGetM(sig_in);
    mwSize sigN = mxGetN(sig_in);
    if ((int)sigM != nf || (int)sigN != nXcvr)
        mexErrMsgIdAndTxt("bp_freq_bistatic_cuda:sig_dim",
                          "sig must be nf x nXcvr (matching f and rx).");

    float* sigR = (float*)mxGetData(sig_in);
    float* sigI = (float*)mxGetImagData(sig_in);

    size_t nSig = (size_t)nf * (size_t)nXcvr;

    // Pack MATLAB complex into interleaved float2
    float2* h_sig = (float2*)mxMalloc(nSig * sizeof(float2));
    for (size_t idx = 0; idx < nSig; ++idx) {
        h_sig[idx].x = sigR[idx];
        h_sig[idx].y = sigI ? sigI[idx] : 0.0f;
    }

    // -------------------------------------------------------------------------
    // Allocate output (MATLAB complex: real + imag separate)
    // -------------------------------------------------------------------------
    plhs[0] = mxCreateNumericMatrix(nVox, 1, mxSINGLE_CLASS, mxCOMPLEX);
    float* outR = (float*)mxGetData(plhs[0]);
    float* outI = (float*)mxGetImagData(plhs[0]);

    // Host buffer for interleaved complex from device
    float2* h_out = (float2*)mxMalloc(nVox * sizeof(float2));

    // -------------------------------------------------------------------------
    // Device allocations
    // -------------------------------------------------------------------------
    float  *d_tx, *d_rx, *d_xx, *d_yy, *d_zz, *d_f;
    float2 *d_sig, *d_out;

    CUDA_CHECK(cudaMalloc(&d_tx, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rx, 3*nXcvr*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sig, nSig*sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_xx, nVox*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yy, nVox*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zz, nVox*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f,  nf*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, nVox*sizeof(float2)));

    // -------------------------------------------------------------------------
    // Host → Device copies
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_tx,  tx,    3*sizeof(float),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx,  rx,    3*nXcvr*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig, h_sig, nSig*sizeof(float2),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xx,  xx,    nVox*sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_yy,  yy,    nVox*sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_zz,  zz,    nVox*sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f,   f,     nf*sizeof(float),      cudaMemcpyHostToDevice));

    mxFree(h_sig);

    // -------------------------------------------------------------------------
    // Launch kernel
    // -------------------------------------------------------------------------
    int block = 256;
    int grid  = (nVox + block - 1) / block;

    bp_kernel<<<grid, block>>>(d_tx, d_rx, d_sig,
                               d_xx, d_yy, d_zz,
                               d_f, c,
                               nVox, nXcvr, nf,
                               d_out);

    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------------------
    // Device → Host copy, then split into real/imag
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nVox*sizeof(float2), cudaMemcpyDeviceToHost));

    for (int i = 0; i < nVox; ++i) {
        outR[i] = h_out[i].x;
        outI[i] = h_out[i].y;
    }

    mxFree(h_out);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_tx);
    cudaFree(d_rx);
    cudaFree(d_sig);
    cudaFree(d_xx);
    cudaFree(d_yy);
    cudaFree(d_zz);
    cudaFree(d_f);
    cudaFree(d_out);
}
