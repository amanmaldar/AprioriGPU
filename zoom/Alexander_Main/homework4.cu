
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;

#define SIZE 32000000
#define BPI_SIZE 128


__global__ void upSweep(int *data, int *bpi)
{
    
    //unsigned int blockDataSize = (SIZE / BPI_SIZE) * 2;
    unsigned int initial_threadID = threadIdx.x * 2;
    unsigned int offsetBlockFactor = blockIdx.x *blockDim.x * 2;
    unsigned int threadID = initial_threadID;
    
    
    int end = 256;
    
    
    int offset = 1;
    int bpiOffset = blockIdx.x;
    //int bpiIndex = 1;
    
    bpi[0] = 0;

    __syncthreads();

    while(offsetBlockFactor < SIZE)
    {
        while((threadID+offset) < end)
        {
            data[threadID + offset + offsetBlockFactor] += data[threadID + offsetBlockFactor];
            if(threadID + offset == 255){bpi[bpiOffset + 1] = data[threadID + offset + offsetBlockFactor];}
            threadID = (threadID * 2) + 1;
            offset *= 2;
            __syncthreads();
        }

        __syncthreads();

        bpiOffset += blockDim.x;
        offsetBlockFactor += (blockDim.x * gridDim.x * 2);
        threadID = initial_threadID;
        offset = 1;
        
        __syncthreads();
    }
}

__global__ void downSweep(int *data, int *bpi)
{ 
    
    unsigned int initial_threadID = threadIdx.x * 2 + 1;
    unsigned int offsetBlockFactor = blockIdx.x *blockDim.x * 2;
    unsigned int threadID = initial_threadID;
    

    //Insert _bpi into the Data.
    if(threadIdx.x == 127){data[threadID + offsetBlockFactor] = bpi[blockIdx.x];} 

    //unsigned int smem_startIndex;
    unsigned int end = 0;
    

    int temp;
    __shared__ int smem_beginProcessing[BPI_SIZE];
    int offset;
   __syncthreads(); 

    while(offsetBlockFactor < SIZE)
    {
        smem_beginProcessing[threadIdx.x] = 0;

        offset = blockDim.x;

        if(threadIdx.x  == 127){smem_beginProcessing[threadIdx.x] = 1;}

        while(offset > 0)
        {
            if(smem_beginProcessing[threadIdx.x])
            {
                if((threadID + offsetBlockFactor) - offset > end)
                {
                    temp = data[threadID + offsetBlockFactor];
                    data[threadID + offsetBlockFactor] += data[(threadID + offsetBlockFactor) - offset];
                    data[(threadID + offsetBlockFactor) - offset] = temp;
                    smem_beginProcessing[threadIdx.x - (offset/2) + 1] = 1;
                }
            }

            offset /= 2;
            __syncthreads();
        }

        
        offsetBlockFactor += (blockDim.x * gridDim.x * 2);
        __syncthreads();
    }
}

int main(void)
{

	srand(time(NULL));
	clock_t t1, t2;
    
	int *data;//holds the data to be sorted.
    int *bpi;
    int blockCount = BPI_SIZE;
    int threadCount = 128;

    //Buffer allocation.
    data = new(nothrow) int[SIZE];
    bpi = new(nothrow) int[SIZE/(blockCount*2)];
    //randomize data between 0 and 511.
    for(int i = 0; i < SIZE; i++){data[i] = 1;}
    
    //Print out the Unsorted Array.
    cout << "Array initiation... " << endl;
    cout << "Starting array first 100: ";
    for(int i  = 0; i < 25; i++)
    {
        cout << data[i] << " ";
    }
    cout << endl << endl;

    //Record the time before sorting. 
    t1 = clock();

    //Buffers that will be located on the GPU.
    int *DATA, *BPI;

	//Dynamic Memory Allocation
    cudaMalloc((void**)&DATA, sizeof(int) * SIZE);
    cudaMalloc((void**)&BPI, sizeof(int) * (SIZE/(blockCount*2)));

	//copy data array into GPU. Remove when sorted.  
	cudaMemcpy(DATA, data, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    
    upSweep<<<blockCount, threadCount>>>(DATA, BPI);
    //copy the values of the allocated GPU C_D array to the C array on the CPU.
	cudaMemcpy(bpi, BPI, sizeof(int) * (SIZE/(blockCount*2)), cudaMemcpyDeviceToHost);

    cout << "BPIs: ";
    
    for(int i = 0; i < 25; i++)
    {
        cout << bpi[i] << " ";
    }
    cout << endl;
   
     cudaMemcpy(data, DATA, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    cout << "Data: ";
    for(unsigned int i = 0; i < 256; i++)
    {
        cout << data[i] << " ";
    }
    cout << endl;
  
    for(unsigned int i = 1; i < SIZE/(blockCount*2);i++)
    {   
       bpi[i] += bpi[i-1]; 
    }
    cudaMemcpy(BPI, bpi, sizeof(int) * (SIZE/(blockCount*2)), cudaMemcpyHostToDevice);
    downSweep<<<blockCount, threadCount>>>(DATA, BPI);

    cout << "BPIs: ";
    for(int i = 0; i < 25; i++)
    {
        cout << bpi[i] << " ";
    }
    cout << endl;

    cudaMemcpy(data, DATA, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    cout << "Data: ";
    for(unsigned int i = 0; i < 50; i++)
    {
        cout << data[i] << " ";
    }
    cout << endl;

	//calculate the time difference.
	t2 = clock();
	double difference = ((double)t2-(double)t1);
	double seconds = difference / CLOCKS_PER_SEC;
	cout << endl << "Run time: " << seconds << " seconds" << endl;

	cudaFree(DATA); //Free dynamically allocated memory
    cudaFree(BPI);
	return 0;
}	
