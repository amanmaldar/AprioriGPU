#include <iostream>
#include <stdio.h>
#include <chrono>

using namespace std;

int n = 32000000;

__global__ void prefix_scan(int *a_d, int *blk_sums_d, int *o_d, int num_blks)
{
	int bid = blockIdx.x;

	while(bid<num_blks)
	{
		__shared__ int temp[256];  // allocated on invocation
		int thid = threadIdx.x;
		int offset = 1;
		int n = blockDim.x*2;

		
		temp[2*thid] = a_d[bid*blockDim.x*2+2*thid]; // load input into shared memory
		temp[2*thid+1] = a_d[bid*blockDim.x*2 +2*thid+1];

		// printf("block %d, thread %d, a id %d = %d\n",bid, thid,	bid*blockDim.x*2+2*thid, a_d[bid*blockDim.x*2+2*thid]);	
		
		for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
		{ 
			__syncthreads();
		   if (thid < d)
		   {
				int ai = offset*(2*thid+1)-1;
    			int bi = offset*(2*thid+2)-1;
    			temp[bi] += temp[ai];
		   }
		   offset *= 2;
		}

		if (thid == 0) 
		{ 
			blk_sums_d[bid] = temp[n-1];
			// printf("block %d, blk_sums_d[%d]=%d\n", bid, bid, blk_sums_d[bid]);
			temp[n - 1] = 0; 

		} // clear the last element
		for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
		{
		     offset >>= 1;
		     __syncthreads();
		     if (thid < d)                     
		    {
		     	int ai = offset*(2*thid+1)-1;
    			int bi = offset*(2*thid+2)-1;

    			float t = temp[ai];
				temp[ai] = temp[bi];
				temp[bi] += t; 
			}
		}
		__syncthreads();

		o_d[bid*blockDim.x*2+2*thid] = temp[2*thid]; // write results to device memory
     	o_d[bid*blockDim.x*2+2*thid+1] = temp[2*thid+1];

     	// printf("block %d, thread %d, temp[%d]=a[%d]=%d %d\n", bid, thid, 2*thid, bid*blockDim.x*2+2*thid, o_d[bid*blockDim.x*2+2*thid], o_d[bid*blockDim.x*2+2*thid+1]); 
		// printf("block %d, thread %d, o[%d]=%d,o[%d]=%d\n", bid, thid, bid*blockDim.x*2+2*thid, o_d[bid*blockDim.x*2+2*thid], bid*blockDim.x*2+2*thid+1, o_d[bid*blockDim.x*2+2*thid+1]); 
		bid += gridDim.x;
		// printf("new bid = %d\n", bid);
	}
}

__global__ void add_scan_out(int *prev_out, int prev_size, int *curr_out, int grp_size)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while(bid<prev_size/blockDim.x)
	{
		int prev_id = bid*blockDim.x+tid;
		prev_out[prev_id] += curr_out[prev_id/grp_size];
		bid += gridDim.x;
	}

}

int main(int argc, char **argv)
{
	int *a = (int*)malloc(sizeof(int)*n);
	int *o = (int*)malloc(sizeof(int)*n);

	for(int i=0;i<n;i++)
		a[i] = 1;

	// for(int i=0;i<n;i++)
	// 	printf("a[%d]=%d \n", i, a[i]);
	int num_blks = 128;
	int num_threads = 128;

	// cout<<n/(num_threads*num_threads)+1<<'\n';	

	int *a_d, *blk_sums_1_d,*blk_sums_2_d,*blk_sums_3_d,*blk_sums_4_d,*blk_sums_1_out_d,*blk_sums_2_out_d, *blk_sums_3_out_d, *o_d;

	cudaMalloc((void**)&a_d, sizeof(int)*n);
	cudaMalloc((void**)&blk_sums_1_d, sizeof(int)*n/(num_threads*2));
	cudaMalloc((void**)&blk_sums_2_d, sizeof(int)*n/(num_threads*num_threads*4));
	cudaMalloc((void**)&blk_sums_3_d, sizeof(int)*n/(num_threads*num_threads*num_threads*8));
	cudaMalloc((void**)&blk_sums_4_d, sizeof(int)*1);
	cudaMalloc((void**)&blk_sums_1_out_d, sizeof(int)*n/(num_threads*2));
	cudaMalloc((void**)&blk_sums_2_out_d, sizeof(int)*n/(num_threads*num_threads*4));

	cudaMalloc((void**)&o_d, sizeof(int)*n);

	cudaMemcpy(a_d, a, sizeof(int)*n, cudaMemcpyHostToDevice);

	printf("%d\n", n/(num_threads*2));
	prefix_scan<<<num_blks, num_threads>>>(a_d, blk_sums_1_d, o_d, n/(num_threads*2));
	prefix_scan<<<num_blks, num_threads>>>(blk_sums_1_d, blk_sums_2_d, blk_sums_1_out_d, n/(num_threads*num_threads*4));
	prefix_scan<<<num_blks, num_threads>>>(blk_sums_2_d, blk_sums_3_d, blk_sums_2_out_d, n/(num_threads*num_threads*num_threads*8));
	prefix_scan<<<num_blks, num_threads>>>(blk_sums_3_d, blk_sums_4_d, blk_sums_3_out_d, 1);

	cudaDeviceSynchronize();
	int final_sums[32]={0};
	// cudaMemcpy(final_sums, blk_sums_4_d, sizeof(int)*1, cudaMemcpyDeviceToHost);

	// cout<<"\n\n"<<final_sums[0]<<"\n\n";


	// cout<<"\n\n";
	// cout<<"Array a:\n";
	// for(int i=0;i<n;i++)
	// 	cout<<a[i]<<" ";

	// cout<<"\n\n";

	cout<<"Outputs (Scan a in groups):\n";
	cudaMemcpy(final_sums, o_d, sizeof(int)*32, cudaMemcpyDeviceToHost);
	for(int i=0;i<32;i++)
		cout<<final_sums[i]<<' ';

	cout<<"\n\n";

	// cout<<"Block sums level 1:\n";
	int blk_sums_1_size = n/(num_threads*2);
	// cudaMemcpy(final_sums, blk_sums_1_d, sizeof(int)*blk_sums_1_size, cudaMemcpyDeviceToHost);
	// for(int i=0;i<blk_sums_1_size;i++)
	// 	cout<<final_sums[i]<<'\t';

	// cout<<"\n\n";

	// cout<<"Block sums level 1 scan:\n";
	// int blk_sums_1_out_size = blk_sums_1_size;
	// cudaMemcpy(final_sums, blk_sums_1_out_d, sizeof(int)*blk_sums_1_out_size, cudaMemcpyDeviceToHost);
	// for(int i=0;i<blk_sums_1_out_size;i++)
	// 	cout<<final_sums[i]<<'\t';

	// cout<<"\n\n";

	// cout<<"Block sums level 2:\n";
	int blk_sums_2_size = n/(num_threads*num_threads*4);
	// cudaMemcpy(final_sums, blk_sums_2_d, sizeof(int)*blk_sums_2_size, cudaMemcpyDeviceToHost);
	// for(int i=0;i<blk_sums_2_size;i++)
	// 	cout<<final_sums[i]<<'\t';


	// cout<<"\n\n";

	// cout<<"Block sums level 2 scan:\n";
	// cudaMemcpy(final_sums, blk_sums_2_out_d, sizeof(int)*blk_sums_2_size, cudaMemcpyDeviceToHost);
	// for(int i=0;i<blk_sums_2_size;i++)
	// 	cout<<final_sums[i]<<'\t';


	// cout<<"\n\n";

	add_scan_out<<<num_blks, num_threads>>>(blk_sums_2_out_d, blk_sums_2_size, blk_sums_3_out_d, num_threads*2);
	add_scan_out<<<num_blks, num_threads>>>(blk_sums_1_out_d, blk_sums_1_size, blk_sums_2_out_d, num_threads*2);
	add_scan_out<<<num_blks, num_threads>>>(o_d, n, blk_sums_1_out_d, num_threads*2);
	cout<<"\n\n";

	// // cout<<"Block sums level 1 scan after adding level 2 outputs (0,120):\n";
	
	// // cudaMemcpy(final_sums, blk_sums_1_out_d, sizeof(int)*blk_sums_1_out_size, cudaMemcpyDeviceToHost);
	// // for(int i=0;i<blk_sums_1_out_size;i++)
	// // 	cout<<final_sums[i]<<'\t';


	// // cout<<"\n\n";

	// cudaMemcpy(final_sums, o_d, sizeof(int)*32, cudaMemcpyDeviceToHost);
	// cout<<o_d[n-1];
	// for(int i=0;i<32;i++)
	// 	cout<<final_sums[i]<<' ';

	// cout<<"\n\n";

	
 }