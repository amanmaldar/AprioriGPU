
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


// scenario 1: use atomicadd on GPU
__global__ void gpu_add_atom(int *a, int *res, int num_size)
{

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < num_size; i += blockDim.x*gridDim.x)
    {
        atomicAdd(res, a[i]);
    }

}

// scenario 2: use shared memory to do para reducation, copy the results to cpu to calculate sum.
__global__ void gpu_add_sharemem(int *a, int *res_tmp, int size_tmp){

    __shared__ int smem[128];

    unsigned int myblock = blockIdx.x;

    while(myblock < size_tmp){

        smem[threadIdx.x] = a[myblock*blockDim.x+threadIdx.x];
        __syncthreads();

        for ( int i = blockDim.x/2; i > 0; i = i /2)
        {
            if (threadIdx.x < i)
            {
                int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
                smem[threadIdx.x] = temp; 
            }
            __syncthreads();
        }
        //write result of this block back to global mem
        if(threadIdx.x==0)
            res_tmp[myblock] = smem[0];
        //after 128 blocks execution, "move" to next 128 rows for 128 blocks
        myblock+=128;

    }
}

// scenario 3: use shared memory to do para reducation and then do atomicadd on GPU.
__global__ void gpu_add_atom_sharemem(int *a, int *res_c, int num_size){

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // each block loads its elements into shared memory
    __shared__ int smem[128];
    smem[tid] = (i<num_size)? a[i]:0;   // last block may pad with 0’s

    //Build summation tree over elements using para reduction
    for(int s=blockDim.x/2; s>0; s=s/2)
    {
        if(tid < s) smem[tid] += smem[tid + s];
        __syncthreads(); 
    }
    // Thread 0 adds the partial sum to the total sum 
    if( tid == 0 ) atomicAdd(res_c, smem[tid]);
}


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
  int *res_c;
  int num_size = 16000000; 
  int *res_tmp;
  int size_tmp = (num_size + 127) / 128;
  int sum_b = 0;   //2. shared_mem result
  cudaMallocHost((void **) &res_tmp, sizeof(int)*size_tmp); //2. tmp of shared_mem result
  cudaMallocHost((void **) &h_res, sizeof(int)*1);  //0.cpu version result
  cudaMallocHost((void **) &res, sizeof(int)*1);   //1.atomicADD
  cudaMallocHost((void **) &res_c, sizeof(int)*1);   //3.atomicADD_shared_mem
  cudaMallocHost((void **) &a, sizeof(int)*num_size);

    for (int i = 0; i < num_size; i++) {
        a[i] = rand() % 5;
    }

    res[0] = 0;
    h_res[0] = 0;

    float cpu_elapsed_time_ms, gpu_elapsed_time_ms1, gpu_elapsed_time_ms2, gpu_elapsed_time_ms3;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start, 0);
    //caculate CPU version for varify results of GPU version
    cpu_add(a, h_res, num_size);
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("CPU time use on sum is : %f ms\n", cpu_elapsed_time_ms);
    printf("cpu result: %d\n", h_res[0]);
    printf("---------------------------------\n");


    // start to count execution time of GPU version of atomicAdd
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
    cudaEventElapsedTime(&gpu_elapsed_time_ms1, start, stop);
    printf("GPU time use on atomicAdd is : %f ms\n", gpu_elapsed_time_ms1);
    printf("gpu atomicAdd result: %d\n", res[0]);
    if(res[0] == h_res[0]) printf("gpu atomicAdd correct result!\n");
    printf("---------------------------------\n");

    // start to count execution time of shared memory version
    cudaEventRecord(start, 0);

    int *d_b;
    int *d_res_b;
    cudaMalloc((void **) &d_b, sizeof(int)*num_size);
    cudaMalloc((void **) &d_res_b, sizeof(int)*size_tmp);
    cudaMemcpy(d_b, a, sizeof(int)*num_size, cudaMemcpyHostToDevice);

    gpu_add_sharemem<<<128, 128>>>(d_b, d_res_b, size_tmp);
    cudaMemcpy(res_tmp, d_res_b, sizeof(int)*size_tmp, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // printf("final blocks sum: %d\n", res_tmp[size_tmp-1]); //check last block result
    for(int i = 0; i < size_tmp; i++)
    {
        sum_b += res_tmp[i];
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms2, start, stop);
    printf("GPU time use on Add_shared_mem is : %f ms\n", gpu_elapsed_time_ms2);
    printf("gpu Add_shared_mem result: %d\n", sum_b);
    if(sum_b == h_res[0]) printf("gpu Add_shared_mem correct result!\n");
    printf("---------------------------------\n");

    // start to count execution time of GPU version of atomicAdd with shared memory
    cudaEventRecord(start, 0);

    int *d_c;
    int *d_res_c;
    cudaMalloc((void **) &d_c, sizeof(int)*num_size);
    cudaMalloc((void **) &d_res_c, sizeof(int)*1);

    cudaMemcpy(d_c, a, sizeof(int)*num_size, cudaMemcpyHostToDevice);

    gpu_add_atom<<<128, 128>>>(d_c, d_res_c, num_size);

    cudaMemcpy(res_c, d_res_c, sizeof(int)*1, cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms3, start, stop);
    printf("GPU time use on atomicAdd_shared_mem is : %f ms\n", gpu_elapsed_time_ms3);
    printf("gpu atomicAdd_shared_mem result: %d\n", res_c[0]);
    if(res_c[0] == h_res[0]) printf("gpu atomicAdd_shared_mem correct result!\n");


    cudaFree(d_a);
    cudaFree(d_res);
    cudaFree(d_b);
    cudaFree(d_res_b);
    cudaFree(d_c);
    cudaFree(d_res_c);

    cudaFreeHost(a);
    cudaFreeHost(res);
    cudaFreeHost(h_res);
    cudaFreeHost(res_tmp);
    cudaFreeHost(res_c);
    return 0;


}


