#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void scan_block(int *in, int *out, int *aux, int n)
{
	extern __shared__ int smem[];
	int bx = blockIdx.x * blockDim.x;
	int tx = threadIdx.x;
	int px = bx + tx;
	int offset = 1;

	//load data
	smem[2*tx]   = in[2*px];
	smem[2*tx+1] = in[2*px+1];

	// up sweep
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (tx < d)
		{
			int ai = offset * (2*tx+1) - 1;
			int bi = offset * (2*tx+2) - 1;
			smem[bi] += smem[ai];
		}
		offset <<= 1;
	}

    // save block sum(last element of the scaned block) and clear last element
	if (tx == 0) {
		if (aux != NULL)
			aux[blockIdx.x] = smem[n-1];
		smem[n-1] = 0;
	}

	for (int d = 1; d < n; d <<= 1)
	{
	 	offset >>= 1;
	 	__syncthreads();
	 	if (tx < d)
	 	{
	 		int ai = offset * (2*tx+1) - 1;
	 		int bi = offset * (2*tx+2) - 1;

	 		unsigned int tmp = smem[ai];
	 		smem[ai]  = smem[bi];
	 		smem[bi] += tmp;
	 	}
	}
	__syncthreads();

	out[2*px]   = smem[2*tx];
	out[2*px+1] = smem[2*tx+1];
}

__global__ void scan_sum(int *d_arr, int *d_aux2)
{
	int bx = 2 * blockDim.x * blockIdx.x;
	int tx = threadIdx.x;
	int aux_v = d_aux2[blockIdx.x];
	d_arr[bx + 2*tx]   += aux_v;
	d_arr[bx + 2*tx+1] += aux_v;
}

int main()
{
	int *in;
	int *out;
	int *aux;
	int *aux2;
	int len = 128*128;  //data_len
	int blksize = 128;
	int nblocks = 128;
	int smem = blksize * sizeof(int);
    cudaMallocHost((void **) &in, sizeof(int)*len);
    cudaMallocHost((void **) &out, sizeof(int)*len);
    cudaMallocHost((void **) &aux, sizeof(int)*nblocks); //sum of each block
    cudaMallocHost((void **) &aux2, sizeof(int)*nblocks); //
    //initialize data
    for (int i = 0; i < len; i++)
    {
        // g_in[i] = 1;
        in[i] = i+1;  //start from 1
        out[i] = 0;
    }

    int *d_in;
    cudaMalloc((void **) &d_in, sizeof(int)*len);
    int *d_out;
    cudaMalloc((void **) &d_out, sizeof(int)*len);
    int *d_aux;
    cudaMalloc((void **) &d_aux, sizeof(int)*nblocks);
    int *d_aux2;
    cudaMalloc((void **) &d_aux2, sizeof(int)*nblocks);

    //step1: scan each block first
    cudaMemcpy(d_in, in, sizeof(int)*len, cudaMemcpyHostToDevice);
    scan_block<<<nblocks, blksize/2, smem>>>(d_in, d_out, d_aux, blksize);
    cudaMemcpy(out, d_out, sizeof(int)*len, cudaMemcpyDeviceToHost);
    cudaMemcpy(aux, d_aux, sizeof(int)*nblocks, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    //step2: scan sum of each block to otain the base for next addition
    scan_block<<<1, blksize/2, smem>>>(d_aux, d_aux2, NULL, blksize);
    cudaMemcpy(aux2, d_aux2, sizeof(int)*nblocks, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    scan_sum<<<nblocks, blksize/2>>>(d_out, d_aux2);
    cudaMemcpy(out, d_out, sizeof(int)*len, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    //step3: add base(from step2) to the result from first-step scan
    printf("First 200 elements of the scan results:\n");
    for (int i = 0; i < 200; i++){
    	printf("%d\t", out[i]);
    	if (i % 10 == 0) printf("\n");
    }
    printf("\n");


    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_aux);
    cudaFree(d_aux2);


    cudaFreeHost(in);
    cudaFreeHost(out);
    cudaFreeHost(aux);
    cudaFreeHost(aux2);

    return 0;

}


