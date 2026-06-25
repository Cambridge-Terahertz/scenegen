#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            mexErrMsgIdAndTxt("forward_model_mex:cuda", cudaGetErrorString(err__)); \
        } \
    } while (0)

__global__
void forward_kernel(
    const float* __restrict__ TARG,   // [3 x ntarg], column-major
    const float* __restrict__ s_loc,  // [ntarg]
    const float* __restrict__ TXS,    // [ntx x 3], column-major (we use row 0)
    const float* __restrict__ RX0,    // [nrx x 3], column-major
    const float* __restrict__ f,      // [nf]
    float c,
    int ntarg,
    int nf,
    int nrx,
    cuFloatComplex* __restrict__ sigf // [nf x nrx], column-major
)
{
    int ifreq = blockIdx.x;  // frequency index
    int irx   = blockIdx.y;  // receiver index

    if (ifreq >= nf || irx >= nrx)
        return;

    // Use first transmitter TXS(1,:)
    float TXx = TXS[0 + 0 * nrx];  // row 0, col 0
    float TXy = TXS[0 + 1 * nrx];  // row 0, col 1
    float TXz = TXS[0 + 2 * nrx];  // row 0, col 2

    // RX0: nrx x 3, column-major
    float RXx = RX0[irx + 0 * nrx];
    float RXy = RX0[irx + 1 * nrx];
    float RXz = RX0[irx + 2 * nrx];

    float freq = f[ifreq];

    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    // TARG: 3 x ntarg, column-major
    for (int t = 0; t < ntarg; ++t) {
        float xt = TARG[0 + 3 * t];
        float yt = TARG[1 + 3 * t];
        float zt = TARG[2 + 3 * t];
        float s  = s_loc[t];

        float dx1 = TXx - xt;
        float dy1 = TXy - yt;
        float dz1 = TXz - zt;

        float dx2 = RXx - xt;
        float dy2 = RXy - yt;
        float dz2 = RXz - zt;

        float r1 = sqrtf(dx1*dx1 + dy1*dy1 + dz1*dz1);
        float r2 = sqrtf(dx2*dx2 + dy2*dy2 + dz2*dz2);

        float phase = -2.0f * CUDART_PI_F * freq * (r1 + r2) / c;
        float amp   = s / (r1 * r2);

        float cs = cosf(phase);
        float sn = sinf(phase);

        sum_real += amp * cs;
        sum_imag += amp * sn;
    }

    // sigf: nf x nrx, column-major
    int idx = ifreq + nf * irx;
    sigf[idx] = make_cuFloatComplex(sum_real, sum_imag);
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    mxInitGPU();

    if (nrhs != 6)
        mexErrMsgIdAndTxt("forward_model_mex:usage",
                          "Usage: sigf = forward_model_mex(TARGg, sg, TXSg, RX0g, fg, c)");

    const mxArray *TARG_in = prhs[0]; // 3 x ntarg, gpuArray(single)
    const mxArray *s_in    = prhs[1]; // ntarg x 1, gpuArray(single)
    const mxArray *TXS_in  = prhs[2]; // ntx x 3, gpuArray(single)
    const mxArray *RX0_in  = prhs[3]; // nrx x 3, gpuArray(single)
    const mxArray *f_in    = prhs[4]; // nf x 1, gpuArray(single)
    double c_d             = mxGetScalar(prhs[5]);
    float  c               = (float)c_d;

    mxGPUArray const *TARGg = mxGPUCreateFromMxArray(TARG_in);
    mxGPUArray const *sg    = mxGPUCreateFromMxArray(s_in);
    mxGPUArray const *TXSg  = mxGPUCreateFromMxArray(TXS_in);
    mxGPUArray const *RX0g  = mxGPUCreateFromMxArray(RX0_in);
    mxGPUArray const *fg    = mxGPUCreateFromMxArray(f_in);

    if (mxGPUGetClassID(TARGg) != mxSINGLE_CLASS ||
        mxGPUGetClassID(sg)    != mxSINGLE_CLASS ||
        mxGPUGetClassID(TXSg)  != mxSINGLE_CLASS ||
        mxGPUGetClassID(RX0g)  != mxSINGLE_CLASS ||
        mxGPUGetClassID(fg)    != mxSINGLE_CLASS)
    {
        mexErrMsgIdAndTxt("forward_model_mex:type",
                          "All GPU inputs must be single-precision (gpuArray(single)).");
    }

    const mwSize *dimsT = mxGPUGetDimensions(TARGg);
    if (dimsT[0] != 3)
        mexErrMsgIdAndTxt("forward_model_mex:TARG",
                          "TARG must be 3 x ntarg on GPU.");
    int ntarg = (int)dimsT[1];

    const mwSize *dimsRX = mxGPUGetDimensions(RX0g);
    if (dimsRX[1] != 3)
        mexErrMsgIdAndTxt("forward_model_mex:RX0",
                          "RX0 must be nrx x 3 on GPU.");
    int nrx = (int)dimsRX[0];

    int nf = (int)mxGPUGetNumberOfElements(fg);

    const float *d_TARG = (const float*)mxGPUGetDataReadOnly(TARGg);
    const float *d_s    = (const float*)mxGPUGetDataReadOnly(sg);
    const float *d_TXS  = (const float*)mxGPUGetDataReadOnly(TXSg);
    const float *d_RX0  = (const float*)mxGPUGetDataReadOnly(RX0g);
    const float *d_f    = (const float*)mxGPUGetDataReadOnly(fg);

    mwSize outDims[2] = { (mwSize)nf, (mwSize)nrx };
    mxGPUArray *sigf_g = mxGPUCreateGPUArray(2, outDims,
                                             mxSINGLE_CLASS, mxCOMPLEX,
                                             MX_GPU_INITIALIZE_VALUES);

    cuFloatComplex *d_sigf = (cuFloatComplex*)mxGPUGetData(sigf_g);

    dim3 grid(nf, nrx);
    dim3 block(1);

    forward_kernel<<<grid, block>>>(
        d_TARG,
        d_s,
        d_TXS,
        d_RX0,
        d_f,
        c,
        ntarg,
        nf,
        nrx,
        d_sigf
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    plhs[0] = mxGPUCreateMxArrayOnCPU(sigf_g);

    mxGPUDestroyGPUArray(TARGg);
    mxGPUDestroyGPUArray(sg);
    mxGPUDestroyGPUArray(TXSg);
    mxGPUDestroyGPUArray(RX0g);
    mxGPUDestroyGPUArray(fg);
    mxGPUDestroyGPUArray(sigf_g);
}
