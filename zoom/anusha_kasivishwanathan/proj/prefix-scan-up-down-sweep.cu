#include <iostream>
#include <stdio.h>
#include <chrono>
#include <math.h>

using namespace std;

int n = 128;

__global__ void up_sweep(int *x, long int *blk_sums, int n)
{
    int bid = blockIdx.x;
    __shared__ int smem[128];
    while(bid<n/blockDim.x)
    {
        int tid = threadIdx.x;       

        smem[tid] = x[blockDim.x*bid+tid];      
        __syncthreads();

        
        for(int depth = 1; pow(2.0,depth)<=blockDim.x;depth++)
        {
            double a = pow(2.0,depth);
            if((tid+1)%(int)a==0 && tid!=0)
            {
                int idx1 = tid - pow(2.0,depth-1);
                int idx2 = tid;
                if(idx1<0 || idx2<0)
                    printf("depth=%d, a=%d, tid=%d, idx1=%d, idx2=%d, smem[%d] = %d \n", depth, (int)a, tid, idx1, idx2, idx2, smem[idx2]); 
                smem[idx2] += smem[idx1];
                
            }
            __syncthreads();
        }   

        x[blockDim.x*bid+tid] = smem[tid];
        if(tid==blockDim.x-1)
        {
            // printf("tid=%d, blk_sums[%d] = %d\n", tid, bid, smem[tid]);
            blk_sums[bid] = smem[tid];
        }

        __syncthreads();

        bid += gridDim.x;  
    }
}

__global__ void down_sweep(int *x, long int *blk_sums, int n)
{
    int bid = blockIdx.x;
    __shared__ int smem[128];
    while(bid < n/blockDim.x)
    {
        int tid = threadIdx.x;

        smem[tid] = x[blockDim.x*bid+tid];
        int max_depth=0;
        for(int i=blockDim.x;i>1;i=i/2)
            max_depth++;

        smem[blockDim.x-1] = 0;
        
        __syncthreads();

        for(int depth=max_depth;pow(2.0,depth)>=2;depth--)
        {
            if((tid+1)%(int)pow(2.0, depth)==0)
            {
                int idx1 = tid - pow(2.0, depth-1);
                int idx2 = tid;

                int tmp = smem[idx2];
                smem[idx2] += smem[idx1];
                smem[idx1] = tmp;
            }
        }

        __syncthreads();

        x[blockDim.x*bid + tid] = smem[tid];

        if(bid!=0)
            x[blockDim.x*bid+tid] += blk_sums[bid-1];

        bid += gridDim.x;  

       
    }

}

// __global__ void kogge_stone(int *x, long int *blk_sums, int n)
// {
//     __shared__ int smem[256];
//     int bid = blockIdx.x;
//     while(bid<n/blockDim.x)
//     {
//         int tid = threadIdx.x;
//         int pout = 0;
//         int pin = 1;
//         if(tid==0)
//             smem[pout*blockDim.x+tid] = 0;
//         else
//             smem[pout*blockDim.x+tid] = x[blockDim.x*bid+tid-1];
    
//         __syncthreads();

//         for(int offset=1;offset<blockDim.x;offset*=2)
//         {
//             pout = 1 - pout;
//             pin = 1 - pout;
//             int idx2 = pout*blockDim.x + tid;
//             int idx1 = pin* blockDim.x + tid - offset;
//             if(tid>=offset)
//                 smem[idx2] += smem[idx1];
//             else
//                 smem[idx2] = smem[idx1];
//             __syncthreads();
//         }
//         x[blockDim.x*bid+tid] = smem[pout*blockDim.x+tid];

//         bid += gridDim.x;
//     }    

// }

int main(int argc, char **argv)
{
	int *a = (int*)malloc(sizeof(int)*n);
	int *o = (int*)malloc(sizeof(int)*n);

	for(int i=0;i<n;i++)
		a[i] = 1;

	int num_blocks = 128;
	int num_threads = 128;    
    int num_blks = n/num_threads;
	
    auto start = chrono::high_resolution_clock::now();
    int *a_d;
    long int *blk_sums;

    cudaMalloc((void**)&a_d, sizeof(int)*n);
    cudaMalloc((void**)&blk_sums, sizeof(long int)*num_threads);

    auto end = chrono::high_resolution_clock::now();

    auto time_us = chrono::duration_cast<chrono::microseconds>(end-start).count();  
    auto time_ms = chrono::duration_cast<chrono::milliseconds>(end-start).count();   
    cout<<"\nTime for cuda malloc: "<<time_us<<" microseconds, about "<<time_ms<<" milliseconds\n";

    start = chrono::high_resolution_clock::now();
    cudaMemcpy(a_d, a, sizeof(int)*n,cudaMemcpyHostToDevice);
    up_sweep<<<num_blocks, num_threads>>>(a_d, blk_sums,n);

    long int *blk_sums_c = (long int*)malloc(sizeof(long int)*num_blks);

    cudaMemcpy(blk_sums_c, blk_sums, sizeof(long int)*num_blks,cudaMemcpyDeviceToHost);
    cudaMemcpy(a, a_d, sizeof(int)*n,cudaMemcpyDeviceToHost);

    for(int i=1;i<num_blks;i++)
    {
        blk_sums_c[i] += blk_sums_c[i-1];
        // cout<<blk_sums_c[i]<<'\n';
    }

    cudaMemcpy(blk_sums, blk_sums_c, sizeof(long int)*num_blks,cudaMemcpyHostToDevice);
    
    down_sweep<<<num_blocks, num_threads>>>(a_d, blk_sums, n);

    cudaMemcpy(a, a_d, sizeof(int)*n,cudaMemcpyDeviceToHost);

    end = chrono::high_resolution_clock::now();

    time_us = chrono::duration_cast<chrono::microseconds>(end-start).count();  
    time_ms = chrono::duration_cast<chrono::milliseconds>(end-start).count();   
    cout<<"\nTime for up, down sweeps: "<<time_us<<" microseconds, about "<<time_ms<<" milliseconds\n";

    for(int i=0;i<n;i++)
    {
        cout<<a[i]<<'\t';
    }

    cout<<'\n';

    cudaDeviceSynchronize();
    cudaFree(a_d);
    cudaFree(blk_sums);
    free(a);
    // free(blk_sums_c);
    return 0;	
 }