void run_forward_model_cuda(
    const float* h_TARGx,
    const float* h_TARGy,
    const float* h_TARGz,
    const float* h_s_loc,
    const float* h_TXSx,
    const float* h_TXSy,
    const float* h_TXSz,
    const float* h_RX0x,
    const float* h_RX0y,
    const float* h_RX0z,
    const float* h_f,
    float c,
    int ntarg,
    int nf,
    int ntx,
    cuFloatComplex* h_sigf   // output [nf * ntx]
)
{
    // --- allocate device memory (omitting error checks for brevity) ---
    float *d_TARGx, *d_TARGy, *d_TARGz, *d_s_loc;
    float *d_TXSx, *d_TXSy, *d_TXSz;
    float *d_RX0x, *d_RX0y, *d_RX0z;
    float *d_f;
    cuFloatComplex* d_sigf;

    cudaMalloc(&d_TARGx, ntarg * sizeof(float));
    cudaMalloc(&d_TARGy, ntarg * sizeof(float));
    cudaMalloc(&d_TARGz, ntarg * sizeof(float));
    cudaMalloc(&d_s_loc, ntarg * sizeof(float));

    cudaMalloc(&d_TXSx, ntx * sizeof(float));
    cudaMalloc(&d_TXSy, ntx * sizeof(float));
    cudaMalloc(&d_TXSz, ntx * sizeof(float));

    cudaMalloc(&d_RX0x, sizeof(float));
    cudaMalloc(&d_RX0y, sizeof(float));
    cudaMalloc(&d_RX0z, sizeof(float));

    cudaMalloc(&d_f, nf * sizeof(float));
    cudaMalloc(&d_sigf, nf * ntx * sizeof(cuFloatComplex));

    cudaMemset(d_sigf, 0, nf * ntx * sizeof(cuFloatComplex));

    // --- copy inputs ---
    cudaMemcpy(d_TARGx, h_TARGx, ntarg * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TARGy, h_TARGy, ntarg * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TARGz, h_TARGz, ntarg * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_loc, h_s_loc, ntarg * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_TXSx, h_TXSx, ntx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TXSy, h_TXSy, ntx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TXSz, h_TXSz, ntx * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_RX0x, h_RX0x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RX0y, h_RX0y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RX0z, h_RX0z, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_f, h_f, nf * sizeof(float), cudaMemcpyHostToDevice);

    // --- launch kernel ---
    int threads = 256;
    int blocks_z = (ntarg + threads - 1) / threads; // over targets
    dim3 grid(nf, ntx, blocks_z);
    dim3 block(threads);

    forward_model_kernel<<<grid, block>>>(
        d_TARGx, d_TARGy, d_TARGz,
        d_s_loc,
        d_TXSx, d_TXSy, d_TXSz,
        d_RX0x, d_RX0y, d_RX0z,
        d_f,
        c,
        ntarg,
        nf,
        ntx,
        d_sigf
    );

    cudaDeviceSynchronize();

    // --- copy result back ---
    cudaMemcpy(h_sigf, d_sigf, nf * ntx * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // --- free ---
    cudaFree(d_TARGx); cudaFree(d_TARGy); cudaFree(d_TARGz); cudaFree(d_s_loc);
    cudaFree(d_TXSx);  cudaFree(d_TXSy);  cudaFree(d_TXSz);
    cudaFree(d_RX0x);  cudaFree(d_RX0y);  cudaFree(d_RX0z);
    cudaFree(d_f);     cudaFree(d_sigf);
}
