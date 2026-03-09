#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduceSum(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int N = 1<<28; // 256 Million elements
    size_t size = N * sizeof(float);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t out_size = blocks * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(out_size);
    for(int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    printf("Launching Reduction 500 times...\n");
    for(int iter = 0; iter < 500; iter++) {
        reduceSum<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, N);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);
    printf("Reduction Block 0 Sum: %f\n", h_out[0]);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    return 0;
}
