
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;

#define BLOCK_COUNT 20
#define THREAD_COUNT 5
#define SIZE 20

__global__ void han_barrier(volatile int *barrier)
{
	unsigned int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    while(threadID < SIZE)
    {
        barrier[threadID] = 0;
        threadID += blockDim.x * gridDim.x;
    }
    
    threadID = threadIdx.x + blockIdx.x * blockDim.x;

    if((threadIdx.x == 0)&&(blockIdx.x == 0))
    {
        printf("Block 0 set \n");
        barrier[blockIdx.x + 1] = 1;
        while(barrier[blockIdx.x] == 0);
        printf("Block 0 unset \n");
    }
    else if((blockIdx.x == (SIZE - 1))&&(threadIdx.x == 0))
    {
        printf("Block %d set\n", blockIdx.x);
        while(barrier[blockIdx.x] == 0);
        barrier[blockIdx.x - 1] = 1;
        printf("Block %d unset\n", blockIdx.x);
    }
    else if(threadIdx.x == 0)
    {
        while(barrier[blockIdx.x] == 0);
        printf("Block %d set\n", blockIdx.x);
        barrier[blockIdx.x + 1] = 1;
        while(barrier[blockIdx.x] == 1);
        barrier[blockIdx.x - 1] = 1;
    }

    __syncthreads();
    if(threadIdx.x == 0){printf("BlockIdx.x: %d passed through.\n", blockIdx.x);}

}

int main(void)
{

	clock_t t1, t2;

	t1 = clock();

    volatile int *BARRIER;

	//Dynamic Memory Allocation
	cudaMalloc((void**)&BARRIER, sizeof(int) * SIZE);

	//Call GPU Kernel.
	han_barrier<<<BLOCK_COUNT, THREAD_COUNT>>>(BARRIER);	
    
    printf("\n Passed Han_Barrier \n");	
	//calculate the time difference.
	t2 = clock();
	double difference = ((double)t2-(double)t1);
	double seconds = difference / CLOCKS_PER_SEC;

	cout << endl << "Rin time: " << seconds << " seconds" << endl;
	
	//cudaFree((volatile int)BARRIER); //Free dynamically allocated memory

	return 0;
}	
