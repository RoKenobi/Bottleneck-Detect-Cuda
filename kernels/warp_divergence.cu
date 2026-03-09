#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warpDivergence(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        if (idx % 2 == 0) {
            data[idx] = data[idx] * 2.0f;
        } else {
            data[idx] = data[idx] / 2.0f;
        }
    }
}

int main() {
    int N = 1<<28; // 256 Million elements
    size_t size = N * sizeof(float);
    float *d_data;

    cudaMalloc(&d_data, size);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("Launching Warp Divergence 500 times...\n");
    for(int iter = 0; iter < 500; iter++) {
        warpDivergence<<<blocks, threads>>>(d_data, N);
    }
    cudaDeviceSynchronize();

    cudaFree(d_data);
    return 0;
}
