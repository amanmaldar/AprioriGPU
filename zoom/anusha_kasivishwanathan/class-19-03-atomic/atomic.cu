#include <iostream>
#include <stdio.h>
#include <chrono>
#include <random>

using namespace std;

const int num_blks = 128;
const int num_threads = 128;
const int array_size = 16000000;

// __device__ int sum[1];
// int sum=0;

__global__ void atomic_add(int *array, int *sum)
{
	int bid = blockIdx.x;
	int tid=threadIdx.x;

	
	while(bid < array_size/num_threads)
	{
		int this_idx = bid*num_threads + tid;
		atomicAdd(&sum[0], array[this_idx]);
		// printf("%d %d \n", array[this_idx], sum[0]);
		__syncthreads();
		bid += gridDim.x;
	}

}

__global__ void par_red_add(int *array, int *sum)
{
	__shared__ int local_arr[128];
    int bid=blockIdx.x;
    int tid=threadIdx.x;
	while(bid < array_size/num_threads)
	{
		int arr_idx = bid*num_threads + tid;
		int local_idx = tid;
		local_arr[local_idx] = array[arr_idx];
		__syncthreads();
		for(int i=num_threads/2;i>0;i=i/2)
		{
			if(tid<i)
                local_arr[local_idx] += local_arr[local_idx+i];
            __syncthreads();
		}
        __syncthreads();
        if(tid==0)
            atomicAdd(&sum[0],local_arr[0]);

        bid += gridDim.x;

	}

}


__global__ void par_red_no_atom(int *array, int *sum)
{
    __shared__ int local_arr[128];
     int bid=blockIdx.x;
     int tid=threadIdx.x;
 	while(bid < array_size/num_threads)
 	{
 		int arr_idx = bid*num_threads + tid;
 		int local_idx = tid;
 		local_arr[local_idx] = array[arr_idx];
 		__syncthreads();
 		for(int i=num_threads/2;i>0;i=i/2)
 		{
 			if(tid<i)
                 local_arr[local_idx] += local_arr[local_idx+i];
             __syncthreads();
 		}
         __syncthreads();
         if(tid==0)
             sum[bid] += local_arr[0];
 
         bid += gridDim.x;
 	}
}




int main(int argc, char **argv)
{
	int *huge_array = (int*)malloc(sizeof(int)*array_size);
	for(int i=0;i<array_size;i++)
		huge_array[i] = (int)rand()%5;
    
    auto start = chrono::high_resolution_clock::now();
	int s=0;
	for(int i=0;i<array_size;i++)
		s += huge_array[i];
	printf("CPU SUM\t\t\t\t\t\t\t\t\t%d\n", s);
    auto end = chrono:: high_resolution_clock::now();

    auto time_ms = chrono::duration_cast<chrono::milliseconds>(end-start).count();
    auto time_us = chrono::duration_cast<chrono::microseconds>(end-start).count();

    cout<<"Time to add in the CPU\t\t\t\t\t\t\t"<<time_ms<<" ms, "<<time_us<<" us\n\n";

    start = chrono::high_resolution_clock::now();
	int *huge_array_dev;
	cudaMalloc((void**)&huge_array_dev, sizeof(int)*array_size);

	cudaMemcpy(huge_array_dev, huge_array, sizeof(int)*array_size, cudaMemcpyHostToDevice); 
    
	int *sum = (int*)malloc(sizeof(int)*1);
	for(int i=0;i<1;i++)
		sum[i]=0;
	int *sum_d;
	

	cudaMalloc((void**)&sum_d, sizeof(int));
	cudaMemcpy(sum_d, sum, sizeof(int), cudaMemcpyHostToDevice);

	atomic_add<<<num_blks, num_threads>>>(huge_array_dev, sum_d);
	
	
	cudaMemcpy(sum,sum_d,sizeof(int),cudaMemcpyDeviceToHost);
	printf("GPU Full atomic SUM\t\t\t\t\t\t\t%d\n", sum[0]);

    end = chrono::high_resolution_clock::now();
    
    time_ms = chrono::duration_cast<chrono::milliseconds>(end-start).count();
    time_us = chrono::duration_cast<chrono::microseconds>(end-start).count();
    cout<<"Time to add n GPU, atomically in every thread\t\t\t\t"<<time_ms<<" ms, "<<time_us<<" us\n\n";

	//int a=0;
    // int *sum = (int*)malloc(sizeof(int)*1);

    start=chrono::high_resolution_clock::now();
    for(int i=0;i<1;i++)
        sum[i]=0;
    // int *sum_d;
    cudaMemcpy(sum_d, sum, sizeof(int), cudaMemcpyHostToDevice);
	par_red_add<<<num_blks, num_threads>>>(huge_array_dev, sum_d);
	cudaMemcpy(sum,sum_d,sizeof(int),cudaMemcpyDeviceToHost);
	printf("GPU Parallel Reduction with Atomic SUM\t\t\t\t\t%d\n",sum[0]);

    end = chrono::high_resolution_clock::now();
    time_ms = chrono::duration_cast<chrono::milliseconds>(end-start).count();
    time_us = chrono::duration_cast<chrono::microseconds>(end-start).count();
    cout<<"Time to add in GPU, with parallel reduction and atomicAdd\t\t"<<time_ms<<" ms, "<<time_us<<" us\n\n";

    start = chrono::high_resolution_clock::now();
    int *big_sum = (int*)malloc(sizeof(int)*array_size/num_threads);
    int *big_sum_d;
    cudaMalloc((void**)&big_sum_d,sizeof(int)*array_size/num_threads);
    for(int i=0;i<array_size/num_threads;i++)
         big_sum[i]=0;
    // int *sum_d;
    cudaMemcpy(big_sum_d, big_sum, sizeof(int)*array_size/num_threads, cudaMemcpyHostToDevice);
    par_red_no_atom<<<num_blks, num_threads>>>(huge_array_dev, big_sum_d);
    cudaMemcpy(big_sum,big_sum_d,sizeof(int)*array_size/num_threads,cudaMemcpyDeviceToHost);
    int tmp_sum = 0;
    for(int i=0;i<(array_size/num_threads);i++)
        tmp_sum += big_sum[i];
    printf("GPU local sums, CPU SUM\t\t\t\t\t\t\t%d\n",tmp_sum);

    end = chrono::high_resolution_clock::now();
    time_ms = chrono::duration_cast<chrono::milliseconds>(end-start).count();
    time_us = chrono::duration_cast<chrono::microseconds>(end-start).count();

    cout<<"Time to Add on both GPU (threads accum. local sums) and then on CPU\t"<<time_ms<<" ms, "<<time_us<<" us\n\n";

    cudaThreadSynchronize();
	cudaDeviceSynchronize();
	return 0;
}
