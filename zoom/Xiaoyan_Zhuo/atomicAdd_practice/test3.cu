#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// #define num_size 16000000



__global__ void gpu_add_atom(int *a, int *res, int num_size)
{

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // while(idx < num_size)
    // {
    // 	for(int i = 0; i < 128 * 128; i++)
    // 	{
    // 		atomicAdd(res, a[i]);
    // 	}
    // 	idx += 128*128;

    // }

    for (unsigned int i = idx; i < num_size; i += blockDim.x*gridDim.x)
    {
        atomicAdd(res, a[i]);
    }

}



	// int i = atomicAdd(index,1);

	// int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// sum[i] = idx;

//CPU version. Mainly for validation
void cpu_add(int *a, int *result, int num_size)
{
    for (int i = 0; i < num_size; i++) 
    {
    	result[0] += a[i];

    }
}


int main()
{
	int *a;
	int *res;
	int *h_res;
	int num_size = 160000000; 
	cudaMallocHost((void **) &a, sizeof(int)*num_size);
	cudaMallocHost((void **) &res, sizeof(int)*1);
	cudaMallocHost((void **) &h_res, sizeof(int)*1);

    for (int i = 0; i < num_size; i++) {
        a[i] = rand() % 5;
    }

    printf("a[0] in main: %d\n", a[0]);
    res[0] = 0;
    h_res[0] = 0;


    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

    int *d_a;
    int *d_res;
    cudaMalloc((void **) &d_a, sizeof(int)*num_size);
    cudaMalloc((void **) &d_res, sizeof(int)*1);

    cudaMemcpy(d_a, a, sizeof(int)*num_size, cudaMemcpyHostToDevice);

    gpu_add_atom<<<128, 128>>>(d_a, d_res, num_size);

    cudaMemcpy(res, d_res, sizeof(int)*1, cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    // printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
    printf("GPU time use on atomicAdd is : %f ms\n", gpu_elapsed_time_ms);

    cpu_add(a, h_res, num_size);

    printf("gpu result: %d\n", res[0]);
    printf("cpu result: %d\n", h_res[0]);

    if(res[0] == h_res[0]) printf("correct result!\n");

    cudaFree(d_a);
    cudaFree(d_res);
    cudaFreeHost(a);
    cudaFreeHost(res);
    cudaFreeHost(h_res);
    return 0;


}
