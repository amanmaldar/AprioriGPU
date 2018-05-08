
#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;

#define BLOCK_COUNT 128
#define THREAD_COUNT 128
#define TB 16384 
#define SIZE 16000000


__global__ void atomicShareAdd(int *a, int *sum, int sizeOfData)
{
    __shared__ int smem[THREAD_COUNT];
    unsigned int currentRow = blockIdx.x;
	
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    smem[threadIdx.x] = 0;

    int end = (sizeOfData/(TB) + 1);

    if(blockIdx.x == 0){sum[threadIdx.x] = 0;}
 
    for(int i = 0; i < end; i++)
    {
        if(threadID < sizeOfData);
        {
            smem[threadIdx.x] += a[threadID];
        }

        threadID += blockDim.x * gridDim.x;
    }

    __syncthreads();
        
    for(unsigned int i = blockDim.x/2; i > 0 ; i = i/2)
    {
        if(threadIdx.x < i)
        {
             smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + i];
        }
          __syncthreads();
    }

        if(threadIdx.x == 0){sum[currentRow] = smem[0];}
}


__global__ void atomicAdd(int *a, int *sum, int sizeOfData)
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int threadID_Init = threadID;
    int end = (sizeOfData/(128*128) + 1);
    
    sum[threadID_Init] = 0;

    for(int i = 0; i < end; i++)
    {
        if(threadID < sizeOfData)
        {
             sum[threadID_Init] += a[threadID];
        }

        threadID += blockDim.x * gridDim.x;
    }

}

int main(void)
{

	
	clock_t t1, t2;

	t1 = clock();
    
    int sizeOfData = SIZE;
	int *array16M;
    int SUM[TB];
    int result = 0;

    array16M = new(nothrow) int[SIZE];

    for(int i = 0; i < SIZE; i++){array16M[i] = 1;}
    
    cout << "Array initiation: starting gpu" << endl;

    int *A_D; int *sum;

	    //Dynamic Memory Allocation
	    cudaMalloc((void**)&A_D, sizeof(int) * SIZE);
        cudaMalloc((void**)&sum, sizeof(int) * TB);

	    //copy the contents of matric C to corressondingly allocated matrix on the GPU
	    cudaMemcpy(A_D, array16M, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

	    //Call GPU Kernel.
	    atomicAdd<<<BLOCK_COUNT, THREAD_COUNT>>>(A_D, sum, sizeOfData);
	
    	//copy the values of the allocated GPU C_D array to the C array on the CPU.
	    cudaMemcpy(SUM, sum, sizeof(int) * TB, cudaMemcpyDeviceToHost);

    for (int i = 0; i < TB; i++)
    {
        result += SUM[i];
    }
    
    cout << "Summation Result: " << result << endl;

	//calculate the time difference.
	t2 = clock();
	double difference = ((double)t2-(double)t1);
	double seconds = difference / CLOCKS_PER_SEC;

	cout << endl << "Run time: " << seconds << " seconds" << endl;

    /**********************SHARED TEST*************************************/

    t1 = clock();

    atomicShareAdd<<<BLOCK_COUNT, THREAD_COUNT>>>(A_D, sum, sizeOfData);
    
    //copy the values of the allocated GPU C_D array to the C array on the CPU.
    cudaMemcpy(SUM, sum, sizeof(int) * TB, cudaMemcpyDeviceToHost);

    result = 0;
    for (int i = 0; i < THREAD_COUNT; i++)
    {
        result += SUM[i]; 
    }


    cout << "Summation Result: " << result << endl;

    t2 = clock();
    difference = ((double)t2-(double)t1);
    seconds = difference / CLOCKS_PER_SEC;
    cout << endl << "Run time: " << seconds << " seconds" << endl;


	cudaFree(A_D); //Free dynamically allocated memory
    cudaFree(sum);


	return 0;
}	
