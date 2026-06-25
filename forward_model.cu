// forward_model.cu
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>

__global__
void forward_model_kernel(
    const float* __restrict__ TARGx,   // [ntarg]
    const float* __restrict__ TARGy,   // [ntarg]
    const float* __restrict__ TARGz,   // [ntarg]
    const float* __restrict__ s_loc,   // [ntarg]
    const float* __restrict__ TXSx,    // [ntx]
    const float* __restrict__ TXSy,    // [ntx]
    const float* __restrict__ TXSz,    // [ntx]
    const float* __restrict__ RX0x,    // [1] or [nr]
    const float* __restrict__ RX0y,
    const float* __restrict__ RX0z,
    const float* __restrict__ f,       // [nf]
    float c,
    int ntarg,
    int nf,
    int ntx,
    cuFloatComplex* __restrict__ sigf  // [nf * ntx]
)
{
    int itx   = blockIdx.y;   // transmitter index
    int ifreq = blockIdx.x;   // frequency index
    int tid   = threadIdx.x + blockDim.x * blockIdx.z; // target index

    if (itx >= ntx || ifreq >= nf || tid >= ntarg) return;

    // Load TX position for this transmitter
    float TXx = TXSx[itx];
    float TXy = TXSy[itx];
    float TXz = TXSz[itx];

    // Assuming single RX (monostatic) – adapt if you have multiple RXs
    float RXx = RX0x[0];
    float RXy = RX0y[0];
    float RXz = RX0z[0];

    // Load target and scattering
    float xt = TARGx[tid];
    float yt = TARGy[tid];
    float zt = TARGz[tid];
    float s  = s_loc[tid];

    // Distances TX→target and target→RX
    float dx1 = TXx - xt;
    float dy1 = TXy - yt;
    float dz1 = TXz - zt;

    float dx2 = RXx - xt;
    float dy2 = RXy - yt;
    float dz2 = RXz - zt;

    float r1 = sqrtf(dx1*dx1 + dy1*dy1 + dz1*dz1);
    float r2 = sqrtf(dx2*dx2 + dy2*dy2 + dz2*dz2);

    // Phase term: exp(-2i*pi*f(ifreq)*(r1+r2)/c)
    float freq = f[ifreq];
    float phase = -2.0f * CUDART_PI_F * freq * (r1 + r2) / c;

    float amp = s / (r1 * r2);

    float cs = cosf(phase);
    float sn = sinf(phase);

    cuFloatComplex contrib = make_cuFloatComplex(amp * cs, amp * sn);

    // Flatten (ifreq, itx) → index
    int idx = ifreq * ntx + itx;

    // Atomic add to sigf[ifreq, itx]
    atomicAdd(&sigf[idx].x, contrib.x);
    atomicAdd(&sigf[idx].y, contrib.y);
}
