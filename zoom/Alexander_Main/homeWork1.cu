
#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;

#define BLOCK_COUNT 256
#define THREAD_COUNT 256

__global__ void multiply(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int rowA, unsigned int colA, unsigned int rowB, unsigned int colB)
{
	unsigned int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int sum;

	while(threadID < rowA)
	{
		for(unsigned int j = 0; j < colB; j++)
		{
			sum = 0;

			for(unsigned int i = 0; i < colA; i++)
			{
				sum += a[threadID * colA + i] * b[i*colB + j];
			}

			c[threadID * colB + j] = sum;

		}

		threadID += blockDim.x * gridDim.x;

	}
}

int main(void)
{

	//srand(time(NULL));

	unsigned int matrixSize = 1024 * 2048;
	unsigned int *A, *B, *C;
	unsigned int B_Col = 1024;
	unsigned int A_Col = 2048;
	unsigned int B_Row = 2048;
	unsigned int A_Row = 1024;
	unsigned int C_Size = A_Row * B_Col;
	clock_t t1, t2;

	t1 = clock();

	A = new(nothrow) unsigned int[matrixSize];
	B = new(nothrow) unsigned int[matrixSize];
	C = new(nothrow) unsigned int[C_Size];

	for(unsigned int i = 0; i < matrixSize; i++)
	{

		A[i] = i % 10;//rand() % 10;
		B[i] = i % 10;//rand() % 10;
	}

	unsigned int *A_D, *B_D, *C_D;

	//Dynamic Memory Allocation
	cudaMalloc((void**)&A_D, sizeof(unsigned int) * matrixSize);
	cudaMalloc((void**)&B_D, sizeof(unsigned int) * matrixSize);
	cudaMalloc((void**)&C_D, sizeof(unsigned int) * C_Size);

	//copy the contents of matric C to corressondingly allocated matrix on the GPU
	cudaMemcpy(A_D, A, sizeof(unsigned int) * matrixSize, cudaMemcpyHostToDevice);
	cudaMemcpy(B_D, B, sizeof(unsigned int) * matrixSize, cudaMemcpyHostToDevice);

	//Call GPU Kernel.
	multiply<<<BLOCK_COUNT, THREAD_COUNT>>>(A_D, B_D, C_D, A_Row, A_Col, B_Row, B_Col);
	
	//copy the values of the allocated GPU C_D array to the C array on the CPU.
	cudaMemcpy(C, C_D, sizeof(unsigned int) * C_Size, cudaMemcpyDeviceToHost);

	for(unsigned int i = 0; i < C_Size; i++){cout << " " << C[i];}	
	
	//calculate the time difference.
	t2 = clock();
	double difference = ((double)t2-(double)t1);
	double seconds = difference / CLOCKS_PER_SEC;

	cout << endl << "Rin time: " << seconds << " seconds" << endl;
	
	cudaFree(A_D); cudaFree(B_D); cudaFree(C_D); //Free dynamically allocated memory

	return 0;
}	
