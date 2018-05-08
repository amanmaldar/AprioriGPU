
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;

#define SIZE 32000000
#define BASE 10
#define DIGIT_PLACES 3


__global__ void K_Stone(int *data, int *tempData, int *end)
{
    unsigned int threadID;
    int offset = 1;

    if((threadIdx.x == 0)&&(blockIdx.x == 0)){*end = 0;}

    while(1)
    {
        
        threadID = threadIdx.x + blockIdx.x * blockDim.x;
        if((threadIdx.x == 0)&&(blockIdx.x == 0)){if(offset > (SIZE/2)){*end = 1;}}
        __syncthreads();
        
        if(*end == 1){return;}
        
        //copy data to temp data
        while(threadID < SIZE)
        {
            if(threadID < SIZE)
            {
                tempData[threadID] = data[threadID];
            }
            threadID += blockDim.x * gridDim.x; 
        }

        threadID = threadIdx.x + blockIdx.x * blockDim.x;
        __syncthreads();
       
        //perform pass
        while((threadID + offset) < SIZE)
        {
            if((threadID + offset) < SIZE){data[threadID + offset] = tempData[threadID + offset] + tempData[threadID];}
            threadID += blockDim.x * gridDim.x;
        }

        offset *= 2;
        __syncthreads();

    }
}

int main(void)
{

	srand(time(NULL));
	clock_t t1, t2;
    
	int *data;//holds the data to be sorted.
    int blockCount = 256;
    int threadCount = 256;

    //Buffer allocation.
    data = new(nothrow) int[SIZE];
    
    //randomize data between 0 and 511.
    for(int i = 0; i < SIZE; i++){data[i] = 1;}
    
    //Print out the Unsorted Array.
    cout << "Array initiation... " << endl;
    cout << "Starting array first 100: ";
    for(int i  = 0; i < 100; i++)
    {
        cout << data[i] << " ";
    }
    cout << endl << endl;

    //Record the time before sorting. 
    t1 = clock();

    //Buffers that will be located on the GPU.
    int *DATA, *TEMPDATA, *END;

	//Dynamic Memory Allocation
    cudaMalloc((void**)&DATA, sizeof(int) * SIZE);
    cudaMalloc((void**)&TEMPDATA, sizeof(int) *SIZE);
    cudaMalloc((void**)&END, sizeof(int));

	//copy data array into GPU. Remove when sorted.  
	cudaMemcpy(DATA, data, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

	//Call GPU Kernel. Radix sort has two sort methods...Counting and Local.
	K_Stone<<<blockCount, threadCount>>>(DATA, TEMPDATA, END); 
	    
    //copy the values of the allocated GPU C_D array to the C array on the CPU.
	cudaMemcpy(data, DATA, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

	//calculate the time difference.
	t2 = clock();
	double difference = ((double)t2-(double)t1);
	double seconds = difference / CLOCKS_PER_SEC;

    cout << "Prefix Sum array first 100: ";
    for(int i = 0; i < 100; i++)
    {
        cout << data[i] << " ";
    }
    cout << endl;

	cout << endl << "Run time: " << seconds << " seconds" << endl;

	cudaFree(DATA); //Free dynamically allocated memory
    cudaFree(TEMPDATA);
	return 0;
}	
