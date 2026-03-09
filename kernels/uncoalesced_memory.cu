#include <stdio.h>
#include <cuda_runtime.h>

__global__ void uncoalescedCopy(float *input, float *output, int N) {
    // STRIDED ACCESS: Instead of adjacent threads reading adjacent memory,
    // they read elements far apart, forcing 32 separate memory transactions.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // This math ensures every thread gets a unique index, 
        // but they are spread out (strided) to break coalescing.
        int stride = 32;
        int bad_idx = (idx % stride) * (N / stride) + (idx / stride);
        
        if (bad_idx < N) {
            output[bad_idx] = input[idx] * 2.0f;
        }
    }
}

int main() {
    int N = 1<<28; // 256 Million elements (~1GB)
    size_t size = N * sizeof(float);
    float *d_in, *d_out;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("Launching Uncoalesced Memory Stress Test (500 iterations)...\n");
    for(int iter = 0; iter < 500; iter++) {
        uncoalescedCopy<<<blocks, threads>>>(d_in, d_out, N);
    }
    
    cudaDeviceSynchronize();
    cudaFree(d_in); 
    cudaFree(d_out);
    return 0;
}
