

#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;

int main(void)
{

	unsigned int matrixSize = 1024 * 2048;
	unsigned int *A, *B, *C;
	unsigned int B_Col = 1024;
	unsigned int A_Col = 2048;
	unsigned int B_Row = 2048;
	unsigned int A_Row = 1024;
	clock_t t1, t2;
	unsigned int sum;
	unsigned C_Size = A_Row*B_Col;

	t1 = clock();

	A = new(nothrow) unsigned int[matrixSize];
	B = new(nothrow) unsigned int[matrixSize];
	C = new(nothrow) unsigned int[C_Size];

	for(unsigned int i =0; i < matrixSize; i++)
	{
		A[i] = i % 10;
		B[i] = i % 10;
	}

	unsigned int p = 0;

	for(unsigned int k = 0; k < A_Row; k++)
	{
		for(unsigned int i = 0; i < B_Col; i++)
		{
			sum = 0;
		
			for(unsigned int j = 0; j < A_Col; j++)
			{
				sum += A[j+ A_Col*k] *B[B_Col * j +i];
			}
			C[p++] = sum;
		}

		cout << "Rows Completed: " << k << "\r" << flush;
	}
	cout << endl;

	for(unsigned int i = 0; i < C_Size; i++){cout << " " << C[i];}

	t2 = clock();

	double difference = ((double)t2-(double)t1);
	double seconds = difference / CLOCKS_PER_SEC;

	cout << endl << "Run time: " << seconds << " seconds" << endl;

	free(A); free(B); free(C);

	return 0;

}
