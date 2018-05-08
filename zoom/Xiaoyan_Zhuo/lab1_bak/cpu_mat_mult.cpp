#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m, n, k;
    clock_t starts, ends; double dura;
    /* Fixed seed for illustration */
    srand(3333);
    printf("please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM, h_cc is used to store CPU result
    //int *h_a, *h_b, *h_cc;
    // cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    // cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    // cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

    int *h_a = (int *)malloc(sizeof(int)*m*n);
    int *h_b = (int *)malloc(sizeof(int)*n*k);
    int *h_cc = (int *)malloc(sizeof(int)*m*k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }
    starts = clock();

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    ends = clock();

    // dura = double((ends-starts));
    

    dura = double((ends-starts))/ CLOCKS_PER_SEC;

    printf("Time elapsed on matrix multiplication of %f seconds.\n",dura);

    free(h_a);
    free(h_b);
    free(h_cc);

}


