#include <stdlib.h>
#include<iostream>
#include <time.h>
#include <omp.h>

using namespace std;

int main(int argc, char** argv)
{

	unsigned int matrixSize = 1024 * 2048;
	unsigned int *A, *B, *C;
	unsigned int B_Col = 1024;
	unsigned int A_Col = 2048;
	unsigned int B_Row = 2048;
	unsigned int A_Row = 1024;
	unsigned int C_Size = A_Row * B_Col;
	int threadCount = atoi(argv[1]);
	int count = 0;
	
	double startTime = omp_get_wtime();

	omp_set_num_threads(threadCount);
	
	A = new(nothrow) unsigned int[matrixSize];
	B = new(nothrow) unsigned int[matrixSize];
	C = new(nothrow) unsigned int[C_Size];

	for(unsigned int i = 0; i < matrixSize; i++)
	{
		A[i] = i % 10;
		B[i] = i % 10;
	}

	#pragma omp parallel
	{
		unsigned int sum;
		int tid = omp_get_thread_num();
		int threadNum = omp_get_num_threads();
		if(!tid){cout << "Number of threads created: " << threadNum << endl;}
		for(unsigned int k = tid; k < A_Row; k = k + threadNum)
		{
			for(unsigned int i = 0; i < B_Col; i++)
			{
				sum = 0;
				for (unsigned int j = 0; j < A_Col; j++)
				{
					sum += A[j + A_Col * k] * B[B_Col * j + i];
				}
				C[k * B_Col + i] = sum;
			}
			if(!tid){cout << "Rows Completed: " << k << "\r" << flush;}
		}
	}

	cout << endl;

	for(unsigned int i = 0; i < C_Size; i++){cout << " " << C[i];}

	double seconds = omp_get_wtime() - startTime;

	cout << endl << "Run time: " << seconds << " seconds" << endl;

	free(A); free(B); free(C);

	return 0;
}	
