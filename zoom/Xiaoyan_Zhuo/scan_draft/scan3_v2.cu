#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void block_psum(int *g_in, int *g_out, int *g_sums, int n)
{
  extern __shared__ unsigned int smem[];
  const size_t bx = blockIdx.x * blockDim.x;
  const size_t tx = threadIdx.x;
  const size_t px = bx + tx;
  int offset = 1;

  // init
  smem[2*tx]   = g_in[2*px];
  smem[2*tx+1] = g_in[2*px+1];

  ////
  // up sweep
  ////
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

  // save block sum and clear last element
  if (tx == 0) {
    if (g_sums != NULL)
      g_sums[blockIdx.x] = smem[n-1];
    smem[n-1] = 0;
  }

  ////
  // down sweep
  ////
  for (int d = 1; d < n; d <<= 1)
  {
    offset >>= 1;
    __syncthreads();

    if (tx < d)
    {
      int ai = offset * (2*tx+1) - 1;
      int bi = offset * (2*tx+2) - 1;

      // swap
      unsigned int t = smem[ai];
      smem[ai]  = smem[bi];
      smem[bi] += t;
    }
  }
  __syncthreads();

  // save scan result
  g_out[2*px]   = smem[2*tx];
  g_out[2*px+1] = smem[2*tx+1];
}

// __global__
// void scatter_incr(      unsigned int * const d_array,
//                   const unsigned int * const d_incr)
// {
//   const size_t bx = 2 * blockDim.x * blockIdx.x;
//   const size_t tx = threadIdx.x;
//   const unsigned int u = d_incr[blockIdx.x];
//   d_array[bx + 2*tx]   += u;
//   d_array[bx + 2*tx+1] += u;
// }

int main(){
	int *g_in;
    int *g_out;
    int *g_sums;
    int *g_sums2;
    int len = 128*128;
    int blksize = 128;
    int nblocks = 128;
    int smem = blksize * sizeof(int);
    cudaMallocHost((void **) &g_in, sizeof(int)*len);
    cudaMallocHost((void **) &g_out, sizeof(int)*len);
    cudaMallocHost((void **) &g_sums, sizeof(int)*nblocks);
    cudaMallocHost((void **) &g_sums2, sizeof(int)*nblocks);

    for (int i = 0; i < len; i++) {
        g_in[i] = 1;
        g_out[i] = 0;
    }

    int *d_in;
    cudaMalloc((void **) &d_in, sizeof(int)*len);
    int *d_out;
    cudaMalloc((void **) &d_out, sizeof(int)*len);
    int *d_sums;
    cudaMalloc((void **) &d_sums, sizeof(int)*nblocks);
    int *d_sums2;
    cudaMalloc((void **) &d_sums2, sizeof(int)*nblocks);


    cudaMemcpy(d_in, g_in, sizeof(int)*len, cudaMemcpyHostToDevice);
    block_psum<<<nblocks, blksize/2, smem>>>(d_in, d_out, d_sums, blksize);
    cudaMemcpy(g_out, d_out, sizeof(int)*len, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_sums, d_sums, sizeof(int)*nblocks, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    printf("phase1: first five scan results(base):\n");
    for (int i = 126; i < 136; i++){
        printf("%d\n", g_out[i]);
    }

    printf("last element in first five blocks:\n");
    for (int i = 0; i < 5; i++){
    	printf("%d\n", g_sums[i]);
    }

    block_psum<<<1, blksize/2, smem>>>(d_sums, d_sums2, NULL, blksize);
    cudaMemcpy(g_sums2, d_sums2, sizeof(int)*nblocks, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    printf("last element in first five blocks:\n");
    for (int i = 0; i < 5; i++){
    	printf("%d\n", g_sums2[i]);
    }


    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sums);
    cudaFree(d_sums2);


    cudaFreeHost(g_in);
    cudaFreeHost(g_out);
    cudaFreeHost(g_sums);
    cudaFreeHost(g_sums2);

    return 0;

}

