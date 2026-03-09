#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1<<28; // 256 Million elements (~1GB per array)
    size_t size = N * sizeof(float);

    printf("Allocating %zu bytes per array...\n", size);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for(int i=0;i<N;i++){
        h_A[i]=1.0f;
        h_B[i]=2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int blockSize=256;
    int gridSize=(N+blockSize-1)/blockSize;

    printf("Launching Vector Add 500 times...\n");
    
    // SUSTAINED LOAD FIX
    for(int iter = 0; iter < 500; iter++) {
        vectorAdd<<<gridSize,blockSize>>>(d_A,d_B,d_C,N);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    printf("Result sample: %f\n",h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
