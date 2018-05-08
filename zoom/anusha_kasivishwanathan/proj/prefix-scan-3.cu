#include <iostream>
#include <stdio.h>
#include <chrono>
#include <math.h>

using namespace std;

int n = 128;

__global__ void up_sweep(int *x, int *blk_sums, int n)
{
    int bid = blockIdx.x;
    __shared__ int smem[128];
    while(bid<n/blockDim.x)
    {
        int tid = threadIdx.x;       

        smem[tid] = x[blockDim.x*bid+tid];      
        //printf("%d\n",x[blockDim.x*bid+tid]);
        __syncthreads();

        for(int depth = 1; depth<n/2;depth++)
        {
            // __syncthreads();
   //         if(depth==1)
   //             printf("tid=%d, depth=%d, cond=%d", tid, depth, ((tid+1) % (2^depth)));

            double a = pow(depth,2.0);
            if((tid+1)%(int)a==0)
            {
                if(depth==1)
                    printf("depth=%d, tid=%d %d %d\n ", depth, tid,(tid+1), (int)a );

//                int idx1 = tid+1-2^depth;
//                int idx2 = tid;
//                smem[idx2] += smem[idx1];
//                __syncthreads();
//                printf("%d %d ", idx1, idx2);
            }

            __syncthreads();
        }
//            if(((tid+1) % 2^depth)==0)
//            {   
//                //printf("%d  ", tid);
//                int idx1 = tid-2^depth+1;
//                int idx2 = tid;
//                smem[idx2] += smem[idx1];
//                printf("%d  ", idx2);
//            }
//
//            __syncthreads();
//            printf("%d  ", tid);
//        }
//
//        __syncthreads();
//        
//        if(tid==12)
//        {
//            __syncthreads();
//            printf("%d  ", tid);
//            blk_sums[bid] = smem[tid];
//        }
//        
//        __syncthreads();
//        
        bid += gridDim.x;  

    }
}

__global__ void down_sweep()
{
}

int main(int argc, char **argv)
{
	int *a = (int*)malloc(sizeof(int)*n);
	int *o = (int*)malloc(sizeof(int)*n);

	for(int i=0;i<n;i++)
		a[i] = 1;

//    for(int i=0;i<n;i++)
//	 	printf("a[%d]=%d \n", i, a[i]);
	int num_blocks = 128;
	int num_threads = 128;
    
    int num_blks = n/num_threads;
//	// cout<<n/(num_threads*num_threads)+1<<'\n';	
    int *a_d, *blk_sums;

    cudaMalloc((void**)&a_d, sizeof(int)*n);
    cudaMalloc((void**)&blk_sums, sizeof(int)*num_threads);
    
    cudaMemcpy(a_d, a, sizeof(int)*n,cudaMemcpyHostToDevice);
    up_sweep<<<num_blocks, num_threads>>>(a_d, blk_sums,n);

    int *blk_sums_c = (int*)malloc(sizeof(int)*num_blks);

    cudaMemcpy(blk_sums_c, blk_sums, sizeof(int)*num_blks,cudaMemcpyDeviceToHost);
    
    for(int i=0;i<num_blks;i++)
        cout<<blk_sums_c[i]<<'\n';
    cudaDeviceSynchronize();
    cudaFree(a_d);
    cudaFree(blk_sums);
    free(a);
    free(blk_sums_c);
    return 0;	
 }
