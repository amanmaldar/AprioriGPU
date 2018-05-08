#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

void cpu_matrix_mult(int *A, int *B, int *result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += A[i * n + h] * B[h * k + j];
            }
            result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m, n, k;
    clock_t starts, ends; double dura;
    /* seed */
    srand(3333);
    printf("please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM

    int *A = (int *)malloc(sizeof(int)*m*n);
    int *B = (int *)malloc(sizeof(int)*n*k);
    int *C = (int *)malloc(sizeof(int)*m*k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            B[i * k + j] = rand() % 1024;
        }
    }

    //start to count time
    starts = clock();

    cpu_matrix_mult(A, B, C, m, n, k);

    //finish counting time
    ends = clock();

    dura = double((ends-starts))/ CLOCKS_PER_SEC;

    printf("CPU time use on matrix multiplication is : %f ms\n",dura*1000); //dura in sec, dura*1000 in ms.

    free(A);
    free(B);
    free(C);

}

