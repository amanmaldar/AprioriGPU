
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;

#define SIZE 100000000
//#define DATA_TYPE 16
#define localSortThreshold 10000
#define BASE 10
#define DIGIT_PLACES 10


__global__ void hybridRadixSort(volatile int *data, volatile int *tempData, volatile int *bucketBounds, volatile int *bucketCount, int shift, int kernelInvocationCount)
{
    __shared__ int smem_histogram[BASE];
    __shared__ int smem_prefix_sum[BASE + 1];
    __shared__ int smem_created_bucket_count;

    int bucketBoundsIndex = blockIdx.x * 2;
    int threadID;
    int numberOfBucketMembers;
    int bucketBoundEnd = bucketBounds[bucketBoundsIndex + 1];
    int bucketBoundStart = bucketBounds[bucketBoundsIndex];
    if(threadIdx.x == 0)
    {
        numberOfBucketMembers = bucketBoundEnd - bucketBoundStart;
    }

    //Initialize histogram and prefix-sum to zero.
    threadID = threadIdx.x;
    while(threadID < BASE)
    {
        smem_histogram[threadID] = 0;
        smem_prefix_sum[threadID] = 0;
        threadID += blockDim.x;
    }

    threadID = threadIdx.x + bucketBoundStart;
    
    //Must reset bucketCount at the start of each kernel invocation. 
    if(threadID == 0)
    {
        *bucketCount = 0;
        smem_created_bucket_count = 0;
        smem_prefix_sum[BASE] = 0;//prefix sum has one extra member that needs to be initialized. 
    }

    //Variables for local sort...
    int localSortEnd = DIGIT_PLACES - kernelInvocationCount;
    int localSortShift = 1;
    int tempDataIndex;
    int prefixSumIndex;

    __syncthreads();

    //If the number of keys in the bucket exceed the localSortThreshold. Performing Counting sort. 
    if(numberOfBucketMembers > localSortThreshold)
    {
        //First pass, build histogram and prefix-sum.
        while(threadID < bucketBoundEnd)
        {
            //If thread is inbounds, begin counting sort. 
            if(threadID < bucketBoundEnd)
            {
                //update histogram atomically, by incrementing everytime a match is found.
                atomicAdd(&smem_histogram[(data[threadID]/shift) % BASE], 1);
            }
            //Increment threads while the treads have not reached the end of bucket.
            threadID += blockDim.x;
        }

        //Need to convert the prefix sum to a parallized version from class!
        if(threadIdx.x == 0)
        {
            smem_prefix_sum[0] = bucketBoundStart;
            for(int i = 1; i <= BASE; i++)
            {
                //prefix sum index calculation.
                smem_prefix_sum[i] = smem_prefix_sum[i - 1] + smem_histogram[i - 1];
                //counting the number of buckets created.
                if(smem_histogram[i-1]){atomicAdd(&smem_created_bucket_count, 1);}
            } 
        }
        
        __shared__ int smem_newStartingBucketBoundsIndex;
        if(threadIdx.x == 0)
        {
            smem_newStartingBucketBoundsIndex = atomicAdd((int*)bucketCount, smem_created_bucket_count) * 2;
        }

        __syncthreads();



        /*Need to find a way to fix this attempt at paralization of bucket bounds*/
        if((smem_created_bucket_count)&&(threadIdx.x < BASE))
        {
                   
            int threadID_BB;
            int threadID_His = threadIdx.x;

            //If a newly created bucket is found...
            if(smem_histogram[threadID_His])
            {
                threadID_BB = atomicAdd(&smem_newStartingBucketBoundsIndex, 2);
                //Store the start index into the bucketBounds array. Over writing previous bounds. 
                bucketBounds[threadID_BB] =  smem_prefix_sum[threadID_His]; //+ smem_bucketBoundsOffet;
                //Store the end index into the bucketBounds array. Over writing prevoius bounds. 
                bucketBounds[threadID_BB + 1] =  smem_prefix_sum[threadID_His + 1]; //+ smem_bucketBoundsOffet;
            }
        }

        //Reinitialize the threadID for anouther
        threadID = threadIdx.x + bucketBoundStart;
        __syncthreads();

        int tempDataIndex;
        //Scatter Keys based on histogram and prefix sum.
        while(threadID < bucketBoundEnd)
        {
            if(threadID < bucketBoundEnd)
            {
                tempDataIndex = atomicAdd(&smem_prefix_sum[(data[threadID]/shift) % BASE], 1);
                tempData[tempDataIndex] = data[threadID]; 
            }
            threadID += blockDim.x;
        }
        
        threadID = threadIdx.x + bucketBoundStart;
         __syncthreads();

        /*Triple buffer will be relace with double buffering. For now. Triple buffering is 
          used to increase production time and test logic*/
        while(threadID < bucketBoundEnd)
        {
            data[threadID] = tempData[threadID]; 
            threadID += blockDim.x;
        }
    }//End of Counting Sort.
/*############################################################################*/
/****************************Local Sort****************************************/
/*############################################################################*/
    else if(numberOfBucketMembers > 1)
    {   
        for(int i = 0; i < localSortEnd; i++)
        {
            threadID = threadIdx.x + bucketBoundStart;
           // printf("thread ID: %d\n", threadID);
            __syncthreads();
            //First pass, build histogram and prefix-sum.
            while(threadID < bucketBoundEnd)
            {
                //If thread is inbounds, begin local sort. 
                if(threadID < bucketBoundEnd)
                {
                    //update histogram atomically, by incrementing everytime a match is found.
                    atomicAdd(&smem_histogram[(data[threadID]/localSortShift) % BASE], 1);
                }
                //Increment threads while the treads have not reached the end of bucket.
                threadID += blockDim.x;
            }

            //Need to convert the prefix sum to a parallized version from class!
            if(threadIdx.x == 0)
            {
                for(int i = 1; i <= BASE; i++)
                {
                    smem_prefix_sum[0] = bucketBoundStart;
                    //prefix sum index calculation.
                    smem_prefix_sum[i] = smem_prefix_sum[i - 1] + smem_histogram[i - 1];
                    //counting the number of buckets created.
                    if(smem_histogram[i-1]){atomicAdd(&smem_created_bucket_count, 1);}
                } 
            }

            //Reinitialize the threadID for anouther
            /*To scatter the keys in order, threads 1 - Base will search from the starting data member to the last
             if the thread index matches the keys it will scatter the key. Each of these threads will scatter in order as
             they search from left to right.*/
            threadID = bucketBoundStart;
            __syncthreads();   

            //Scatter Keys based on histogram and prefix sum.
            while(threadID < bucketBoundEnd)
            {
                if(threadID < bucketBoundEnd)
                {
                    prefixSumIndex = (data[threadID]/localSortShift) % BASE;
                    if(prefixSumIndex == threadIdx.x)
                    {
                        tempDataIndex = atomicAdd(&smem_prefix_sum[prefixSumIndex], 1);
                        tempData[tempDataIndex] = data[threadID];
                    }
                }
                threadID++;
            }

            threadID = threadIdx.x + bucketBoundStart;
            __syncthreads();

            /*Triple buffer will be relace with double buffering. For now. Triple buffering is 
              used to increase production time and test logic*/
            while(threadID < bucketBoundEnd)
            {
                data[threadID] = tempData[threadID];
                threadID += blockDim.x;
            }

            //up date the digit place you will sort for the next pass.
            localSortShift *= BASE;
            
            //Initialize histogram and prefix-sum to zero.
            threadID = threadIdx.x;
            while(threadID < BASE)
            {
                smem_histogram[threadID] = 0;
                smem_prefix_sum[threadID] = 0;
                threadID += blockDim.x;
            }
            if(threadIdx.x == 0){smem_prefix_sum[BASE] = 0;}

        }
     }
    
    //Sync all threads before returning from the kernel call.
     __syncthreads();
}

int main(void)
{

	srand(time(NULL));
	clock_t t1, t2;
    
	int *data;//holds the data to be sorted.
    int *bucketBounds;//holds the start and ending index of each bucket.
    int bucketBoundsSize = 1000;//Initial Allocation of bucketbounds. (500 buckets) 
    int blockCount = 1;
    int threadCount = 1024;
    int *G_bucketCount;//Total number of currently active buckets. 
    int shift = 1;//Used to index into powers of the base.
    int kernelInvocationCount = 0;

    //Buffer allocation.
    data = new(nothrow) int[SIZE];
    bucketBounds = new(nothrow) int[bucketBoundsSize];
    G_bucketCount = new(nothrow) int[1];
    
    //randomize data between 0 and 511.
    for(int i = 0; i < SIZE; i++){data[i] = rand() % 1000000000;}//512;}
    
    //Print out the Unsorted Array.
    cout << "Array initiation... " << endl;
    cout << "Starting array: ";
    for(int i  = 0; i < SIZE; i = i + 10000)
    {
        cout << data[i] << " ";
    }
    cout << endl;

    //Record the time before sorting. 
    t1 = clock();

    //Buffers that will be located on the GPU.
    volatile int *DATA, *TEMPDATA, *BUCKETBOUNDS, *G_BUCKETCOUNT;

	//Dynamic Memory Allocation
	cudaMalloc((void**)&TEMPDATA, sizeof(int) * SIZE);
    cudaMalloc((void**)&DATA, sizeof(int) * SIZE);
    cudaMalloc((void**)&BUCKETBOUNDS, sizeof(int) * bucketBoundsSize);
    cudaMalloc((void**)&G_BUCKETCOUNT, sizeof(int) * 1);

	//copy data array into GPU. Remove when sorted.  
	cudaMemcpy((void*)DATA, data, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    //Initialize bucketBounds.
    //initially one bucket
    //The bucket starts at index zero and does not exceed SIZE.
    bucketBounds[0] = 0;
    bucketBounds[1] = SIZE;
    *G_bucketCount = 1;
    
	//intially copy the bucketBounds Array to the GPU 
	cudaMemcpy((void*)BUCKETBOUNDS, bucketBounds, sizeof(int) * bucketBoundsSize, cudaMemcpyHostToDevice);
    
    //Initialie the shift for MSD.
    for(int i = 1; i < DIGIT_PLACES; i++)
    {
        shift *= BASE;
    }

    //If there are no buckets left then sorted. If sorted count == 0, Sorted. 
    while(*G_bucketCount)//for(int k = 0; k < 3; k++)//while(*G_bucketCount)
    {

	    //Call GPU Kernel. Radix sort has two sort methods...Counting and Local.
	    hybridRadixSort<<<blockCount, threadCount>>>(DATA, TEMPDATA, BUCKETBOUNDS, G_BUCKETCOUNT, shift, kernelInvocationCount); 
	    
        //copy bucket count located on GPU to CPU. This will be used to determine blockCount.
        cudaMemcpy(G_bucketCount, (void*)G_BUCKETCOUNT, sizeof(int), cudaMemcpyDeviceToHost);
        //Determine the block and thread sizes for the next kernel invocation.
        blockCount = *G_bucketCount;
        shift /= BASE;
        //records the number of kernel invocations. Used by local sort to determine the amount of unsorted bits. 
        kernelInvocationCount++;

    }
    //copy the values of the allocated GPU C_D array to the C array on the CPU.
	cudaMemcpy(data, (void*)DATA, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

	//calculate the time difference.
	t2 = clock();
	double difference = ((double)t2-(double)t1);
	double seconds = difference / CLOCKS_PER_SEC;

    cout << "Radix sort finished." << endl;
    cout << "Sorted array: ";
    for(int i = 0; i < SIZE; i = i + 10000)
    {
        cout << data[i] << " ";
    }
    cout << endl;

	cout << endl << "Run time: " << seconds << " seconds" << endl;

	cudaFree((void*)DATA); //Free dynamically allocated memory
    cudaFree((void*)BUCKETBOUNDS);
    cudaFree((void*)G_BUCKETCOUNT);
    cudaFree((void*)TEMPDATA);

	return 0;
    }
