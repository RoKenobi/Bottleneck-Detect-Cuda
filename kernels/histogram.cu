#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BINS 256

__global__ void histogram(int *data, int *bins, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int bin = data[i] % NUM_BINS; 
        atomicAdd(&bins[bin], 1);
    }
}

int main() {
    int N = 1<<28; // 256 Million elements
    size_t data_size = N * sizeof(int);
    size_t bin_size = NUM_BINS * sizeof(int);

    int *h_data = (int*)malloc(data_size);
    int *h_bins = (int*)malloc(bin_size);
    
    for(int i = 0; i < N; i++) h_data[i] = i; 
    for(int i = 0; i < NUM_BINS; i++) h_bins[i] = 0; 

    int *d_data, *d_bins;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_bins, bin_size);

    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, bin_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("Launching Histogram 500 times...\n");
    for(int iter = 0; iter < 500; iter++) {
        histogram<<<blocks, threads>>>(d_data, d_bins, N);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_bins, d_bins, bin_size, cudaMemcpyDeviceToHost);
    printf("Histogram Bin 10 count: %d\n", h_bins[10]);

    cudaFree(d_data); cudaFree(d_bins);
    free(h_data); free(h_bins);

    return 0;
}
