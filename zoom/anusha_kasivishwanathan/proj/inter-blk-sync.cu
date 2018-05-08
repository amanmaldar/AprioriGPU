
#include <stdio.h>

__global__ void inter_blk_sync(volatile int * lock_d)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if(bid==0)
	{
		if(tid==0)
			lock_d[tid]=1;
		while(lock_d[tid]!=1);
		__syncthreads();
		if(tid==0)
			printf("All done\n");
	}
	else
	{
		if(tid==0)
		{
			printf("block %d, thread %d checked in!!\n\n", bid, tid);
			lock_d[bid]=1;

			while(lock_d[bid]!=0);
		}
		__syncthreads();
	}
}


int main(int argc, char **argv)
{
	int num_blocks = 8;
	int num_threads = 8;
	int *lock = (int*)malloc(sizeof(int)*num_blocks);

	for(int i=0;i<num_threads;i++)
		lock[i]=0;

	int *lock_d;
	cudaMalloc((void**)&lock_d, sizeof(num_blocks));
	cudaMemcpy(lock_d, lock, sizeof(int)*num_blocks, cudaMemcpyHostToDevice);

	inter_blk_sync<<<num_blocks, num_threads>>>(lock_d);

	cudaDeviceSynchronize();
	return 0;
}