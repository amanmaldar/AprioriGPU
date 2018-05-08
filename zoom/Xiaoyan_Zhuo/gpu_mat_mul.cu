#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void gpu_matrix_mult(int *A,int *B, int *C, int m, int n, int k)
{ 
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x; //col
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y; //row

    int sum = 0;
    if( tid_x < k && tid_y < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += A[tid_y * n + i] * B[i * k + tid_x];
        }
        C[tid_y * k + tid_x] = sum;
    }
} 

int main(int argc, char const *argv[])
{
    int m, n, k;
    /* seed for illustration */
    srand(3333);
    printf("please enter m n and k\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM
    int *A, *B, *C;
    cudaMallocHost((void **) &A, sizeof(int)*m*n);
    cudaMallocHost((void **) &B, sizeof(int)*n*k);
    cudaMallocHost((void **) &C, sizeof(int)*m*k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            B[i * k + j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    // Allocate memory space on the device 
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, A, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    // Launch kernel
    gpu_matrix_mult<<<256, 256>>>(d_a, d_b, d_c, m, n, k);  
    // Transefr results from device to host 
    cudaMemcpy(C, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    // printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
    printf("GPU time use on matrix multiplication is : %f ms\n", gpu_elapsed_time_ms);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    return 0;
}

//ref: https://ivanlife.wordpress.com/2011/05/09/time-cuda/ (time measure cuda)
