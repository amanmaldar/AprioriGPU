#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void gpu_matrix_vec_mult(int *A,int *b, int *C, int m, int n)
{ 
    
    __shared__ int smem[256];
    int myrow = blockIdx.x;
    while(myrow<m)
    {
    	smem[threadIdx.x] = A[myrow*n+threadIdx.x];
    	smem[threadIdx.x] *= b[threadIdx.x];
    	//parallel reduction
    	__syncthreads();
    	for ( int i = blockDim.x/2; i > 0; i = i /2)
    	{
    		if (threadIdx.x < i)
    		{
    			int temp = smem[ threadIdx.x] + smem[threadIdx.x + i];
    			smem[threadIdx.x] = temp; 
    		}
    		__syncthreads();
    	}
    	if(threadIdx.x==0)
    		C[myrow] = smem[0];
    	//after 128 blocks execution, "move" to next 128 rows for 128 blocks
    	myrow+=128;
    }
}

void cpu_matrix_vec_mult(int *A, int *b, int *result, int m, int n) {
    for (int i = 0; i < m; ++i) 
    {
        int tmp = 0;
        for (int h = 0; h < n; ++h) 
        {
        	tmp += A[i * n + h] * b[h];
        }
        result[i] = tmp;
    }
}

int main(int argc, char const *argv[])
{
    int m, n;
    /* seed for illustration */
    srand(3333);
    printf("please enter m n\n");
    scanf("%d %d", &m, &n);

    // allocate memory in host RAM, h_c is used to store CPU result
    int *A, *b, *C, *h_c;
    cudaMallocHost((void **) &A, sizeof(int)*m*n);
    cudaMallocHost((void **) &b, sizeof(int)*n*1);
    cudaMallocHost((void **) &C, sizeof(int)*m*1);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*1);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        b[i] = rand() % 1024;
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    // Allocate memory space on the device 
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*1);
    cudaMalloc((void **) &d_c, sizeof(int)*m*1);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, A, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int)*n*1, cudaMemcpyHostToDevice);

    // Launch kernel
    gpu_matrix_vec_mult<<<128, 256>>>(d_a, d_b, d_c, m, n);  
    // Transefr results from device to host 
    cudaMemcpy(C, d_c, sizeof(int)*m*1, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    // printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
    printf("GPU time use on matrix multiplication is : %f ms\n", gpu_elapsed_time_ms);

    // start the CPU version
    cudaEventRecord(start, 0);
    cpu_matrix_vec_mult(A, b, h_c, m, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("CPU time use on matrix multiplication is : %f ms",cpu_elapsed_time_ms);


    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_c[i*k + j], i, j, h_c[i*k + j]);
        if(C[i] != h_c[i])
        {
            all_ok = 0;
        }
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(A);
    cudaFreeHost(b);
    cudaFreeHost(C);
    return 0;
}






