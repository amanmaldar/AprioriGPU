
#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;

#define BLOCK_COUNT 256
#define THREAD_COUNT 256

__global__ void multiply(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int rowA, unsigned int colA, unsigned int rowB, unsigned int colB)
{
    __shared__ int smem[THREAD_COUNT];
    unsigned int currentRow = blockIdx.x;
	while(currentRow < rowA)
	{
        smem[threadIdx.x] = a[currentRow * blockDim.x + threadIdx.x] * b[threadIdx.x];
        __syncthreads();
        
        for(unsigned int i = blockDim.x/2; i > 0 ; i = i/2)
        {
            if(threadIdx.x < i)
            {
               smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + i];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0){c[currentRow] = smem[0];}
		currentRow += blockDim.x;
	}
}

int main(void)
{

	unsigned int *A, *B, *C;
	unsigned int B_Col = 1;
	unsigned int A_Col = 256;
	unsigned int B_Row = 256;
	unsigned int A_Row = 10240;
	unsigned int C_Size = A_Row * B_Col;
    unsigned int B_Size = B_Row * B_Col;
    unsigned int A_Size = A_Row * A_Col;
	clock_t t1, t2;

	t1 = clock();

	A = new(nothrow) unsigned int[A_Size];
	B = new(nothrow) unsigned int[B_Size];
	C = new(nothrow) unsigned int[C_Size];

	for(unsigned int i = 0; i < A_Size; i++){A[i] = i % 10;}
    for(unsigned int i = 0; i < B_Size; i++){B[i] = i % 10;}

	unsigned int *A_D, *B_D, *C_D;

	//Dynamic Memory Allocation
	cudaMalloc((void**)&A_D, sizeof(unsigned int) * A_Size);
	cudaMalloc((void**)&B_D, sizeof(unsigned int) * B_Size);
	cudaMalloc((void**)&C_D, sizeof(unsigned int) * C_Size);

	//copy the contents of matric C to corressondingly allocated matrix on the GPU
	cudaMemcpy(A_D, A, sizeof(unsigned int) * A_Size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_D, B, sizeof(unsigned int) * B_Size, cudaMemcpyHostToDevice);

	//Call GPU Kernel.
	multiply<<<BLOCK_COUNT, THREAD_COUNT>>>(A_D, B_D, C_D, A_Row, A_Col, B_Row, B_Col);
	
	//copy the values of the allocated GPU C_D array to the C array on the CPU.
	cudaMemcpy(C, C_D, sizeof(unsigned int) * C_Size, cudaMemcpyDeviceToHost);

	for(unsigned int i = 0; i < C_Size; i++){cout << " " << C[i];}	
	
	//calculate the time difference.
	t2 = clock();
	double difference = ((double)t2-(double)t1);
	double seconds = difference / CLOCKS_PER_SEC;

	cout << endl << "Run time: " << seconds << " seconds" << endl;
	
	cudaFree(A_D); cudaFree(B_D); cudaFree(C_D); //Free dynamically allocated memory

	return 0;
}	
