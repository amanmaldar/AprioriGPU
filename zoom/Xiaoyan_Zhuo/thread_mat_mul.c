#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// #define M 3
// #define N 2
// #define K 3
int M, N, K;
int *A, *B, *C;

struct v
{
    size_t i;
    size_t j;
};

static void * threads_matrix_mult(void *arg){
    struct v *data = (struct v *)arg;

    size_t l;
    for(l=0; l < K; l++)
    {
        size_t i=(data[l]).i;
        size_t j=(data[l]).j;
        int sum=0;
        size_t d;

        for (d = 0; d < N; d++)
        {
            sum = sum + A[i*N+d]*B[d*K+j];
        }

        C[i*K+j] = sum;
    }
    return 0;
}

int main(int argc, char *argv[]) {

    float time_use = 0;
    struct timeval start;
    struct timeval end;
    if(argc != 4) 
    {
    fprintf(stderr, "Useage: A[M][N], B[N][K], please enter M, N, K\n");
    return -1; 
    }
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    A = (int*)malloc(sizeof(int)*M*N);
    B = (int*)malloc(sizeof(int)*N*K);
    C = (int*)malloc(sizeof(int)*M*K);

    // random initialize matrix A
    if ((M<10)&&(N<10)&&(K<10))         //for testing mat mul results
        printf("A:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if ((M<10)&&(N<10)&&(K<10))
            {
                A[i*N+j] = rand() % 10;
                printf("%d\t", A[i*N+j]);
            }
            else
                A[i*N+j] = rand() % 1024;          
        }
        if ((M<10)&&(N<10)&&(K<10))
            printf("\n");
    }

    // random initialize matrix B
    if ((M<10)&&(N<10)&&(K<10))
        printf("B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            if ((M<10)&&(N<10)&&(K<10))
            {
                B[i*K+j] = rand() % 10;
                printf("%d\t", B[i*K+j]);
            }
            else
                B[i*K+j] = rand() % 1024;
        }
        if ((M<10)&&(N<10)&&(K<10))
            printf("\n");
    }

    //start to count time
    gettimeofday(&start, NULL);

    pthread_t threads[M];
    size_t i, k;


    struct v **data;
    data = (struct v **)malloc(M * sizeof(struct v*));

    for(i = 0; i < M; i++)
    {
        data[i] = (struct v *)malloc(M * sizeof(struct v));

        for(k = 0; k < K; k++)
        {
            data[i][k].i = i;
            data[i][k].j = k;
        }

        pthread_create(&threads[i], NULL, threads_matrix_mult, data[i]);


    }

    for(i = 0; i < M; i++)
    {
        pthread_join(threads[i], NULL);
    }

    //showing the mat mul results C
    if ((M<10)&&(N<10)&&(K<10)){
        printf("C:\n");

        for (int i = 0; i < M; ++i) {
         for (int j = 0; j < K; ++j) {
             printf("%d\t", C[i*K+j]);
         }
         printf("\n");
        }
    }
 
    //finish counting time
    gettimeofday(&end, NULL);
    time_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    printf("Multithreads time use on matrix multiplication is : %f ms\n",time_use/1000);

    for (i = 0; i < M; i++)
    {        
        free(data[i]);
    }

    free(data);
    free(A);
    free(B);
    free(C);

    return 0;

}    




