

#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;
#define BLOCK_COUNT 128
#define THREAD_COUNT 256


__global__ void sum(int *A, int *B, int *C, int rowA, int colA, int rowB, int colB)
{
    __shared__ int smem[blockDim.x];
    smem[threadIdx.x] = imput_array[threadIdx.x];
    __syncthreads();

    for(int i - blockDim.x/2; i > 0; i = 1/2)
    {
        if(threadIdx.x < i)
        {
            int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
            smem[threadIdx.x] = temp;
        }

        __syncthreads();
    }
}

int main(void)
{

    int *A, *B, *C;
    int rowA = 10240;
    int rowB = 256;
    int colA = 256;
    int sizeC = colA * rowB;
    int sizeB = rowB;
    int sizeA = rowA * colA;

    A = new(nothrow) int[sizeA];
    B = new(nothrow) int[sizeB];
    C = new(nothrow) int[sizeC];

    for(int i = 0; i < sizeA; i++){A[i] = i;}
    for(int i = 0; i < sizeB; i++){B[i] = i;}

    int *A_D, *B_D, *C_D;
    cudaMalloc((void**)&A_D, sizeof(int) * sizeA); 
    cudaMalloc((void**)&B_D, sizeof(int) * sizeA);
    cudaMalloc((void**)&C_D, sizeof(int) * sizeA);

    



    return 0;
}
