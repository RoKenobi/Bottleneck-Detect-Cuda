#include <stdio.h>
#include <cuda_runtime.h>

#define MASK_WIDTH 5
__constant__ float d_Mask[MASK_WIDTH];

__global__ void convolution1D(float *N, float *P, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int start_point = i - (MASK_WIDTH / 2);

    for (int j = 0; j < MASK_WIDTH; j++) {
        if (start_point + j >= 0 && start_point + j < width) {
            Pvalue += N[start_point + j] * d_Mask[j];
        }
    }
    if (i < width) P[i] = Pvalue;
}

int main() {
    int width = 1<<28; // 256 Million elements
    size_t size = width * sizeof(float);
    
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);
    float h_Mask[MASK_WIDTH] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};

    for(int i = 0; i < width; i++) h_N[i] = 1.0f;

    float *d_N, *d_P;
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Mask, h_Mask, MASK_WIDTH * sizeof(float));

    int threads = 256;
    int blocks = (width + threads - 1) / threads;

    printf("Launching Convolution 500 times...\n");
    for(int iter = 0; iter < 500; iter++) {
        convolution1D<<<blocks, threads>>>(d_N, d_P, width);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    printf("Convolution Result sample: %f\n", h_P[2]);

    cudaFree(d_N); cudaFree(d_P);
    free(h_N); free(h_P);

    return 0;
}
