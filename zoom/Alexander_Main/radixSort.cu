
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;

#define SIZE 25
//#define DATA_TYPE 16
#define localSortThreshold 4
#define BASE 10
#define DIGIT_PLACES 3


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
    
    //Must reset bucketCount and G_SYNC at the start of each kernel invocation. 
    if(threadID == 0)
    {
        *bucketCount = 0;
        smem_created_bucket_count = 0;
        smem_prefix_sum[BASE] = 0;//prefix sum has one extra member that needs to be initialized. 
        printf("Number of Bucket Members: %d, localSortThreshold: %d \n", numberOfBucketMembers, localSortThreshold);
    }

    //Variables for local sort...
    int localSortEnd = DIGIT_PLACES - kernelInvocationCount;
    int localSortShift = 1;
    int tempDataIndex;
    int prefixSumIndex;

    __syncthreads();

    //printf("threadID outside: %d\n", threadID);
    //If the number of keys in the bucket exceed the localSortThreshold. Performing Counting sort. 
    if(numberOfBucketMembers > localSortThreshold)
    {
        //if((blockIdx.x == 0)&&(threadIdx.x == 0)){printf("Starting histogram... \n");}
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

        //Test print histogram
       /* __syncthreads();
        if((blockIdx.x == 0)&&(threadIdx.x == 0))
        {
            printf("Histogram: ");
            for(int i = 0; i < BASE - 1; i++)
            {
                printf("%d, ", smem_histogram[i]);
            }
            printf("%d \n", smem_histogram[BASE - 1]);
        }*/

        //if((blockIdx.x == 0)&&(threadIdx.x == 0)){printf("Starting prefix-sum... \n");}
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

        //Test print prefix-sum
       /* __syncthreads();
        if((blockIdx.x == 0)&&(threadIdx.x == 0))
        {
            printf("Prefix Sum: ");
            for(int i = 0; i < BASE - 1; i++)
            {
                printf("%d, ", smem_prefix_sum[i]);
            }
            printf("%d \n", smem_prefix_sum[BASE - 1]);
        }*/
        
        __shared__ int smem_newStartingBucketBoundsIndex;
        if(threadIdx.x == 0)
        {
            smem_newStartingBucketBoundsIndex = atomicAdd((int*)bucketCount, smem_created_bucket_count) * 2;
        }

        __syncthreads();



        /*Need to find a way to fix this attempt at paralization of bucket bounds*/
        if((smem_created_bucket_count)&&(threadIdx.x < BASE))
        {
                   
            /*ThreadID will index to even locations. Even location represent staring bucket indexes
            to be entered. The Odds represent the ending bucket indexes to be entered into bucket bounds.*/
            int threadID_BB;// = (threadIdx.x * 2) + smem_newStartingBucketBoundsIndex;
            /*Since the histogram needs to be checked for a created bucket, another index is created 
             for the same thread. threadID_His will check all the indexes throughout the histogram for 
             created buckets. If found threadID_BB will index into the bucketBounds array to apply the 
             starting and ending locations which are located in the prefix-sum, which shares the 
             same index as its histogram counter part.*/
            int threadID_His = threadIdx.x;

            //End represents the end index location for the current blocks bucketBound region. 
           // int end = smem_newStartingBucketBoundsIndex + (smem_created_bucket_count * 2);

           // printf("blockID: %d | starting BB index: %d | end: %d | threadID_his %d | threadBB: %d\n", blockIdx.x,  smem_newStartingBucketBoundsIndex, end, threadID_His, threadID_BB); 
           // while((threadID_His < BASE)&&(threadID_BB < end))
           // {
                //If a newly created bucket is found...
                if(smem_histogram[threadID_His])
                {
                    threadID_BB = atomicAdd(&smem_newStartingBucketBoundsIndex, 2);
                    //Store the start index into the bucketBounds array. Over writing previous bounds. 
                    bucketBounds[threadID_BB] =  smem_prefix_sum[threadID_His]; //+ smem_bucketBoundsOffet;
                    //Store the end index into the bucketBounds array. Over writing prevoius bounds. 
                    bucketBounds[threadID_BB + 1] =  smem_prefix_sum[threadID_His + 1]; //+ smem_bucketBoundsOffet;
                }
                //Update the next histogram index to search.
           //     if(smem_created_bucket_count < blockDim.x)
           //     {
           //         threadID_His += smem_created_bucket_count;
           //     }
           //     else
           //     {
           //         threadID_His += blockDim.x; 
           //     }
           // }
        }

        

        //Test Print Bucket Bounds
       /* __syncthreads();
        if((blockIdx.x == 0)&&(threadIdx.x == 0))
        {
            printf("Bucket Bounds: ");
            for(int i = 0; i < (*bucketCount * 2) - 1; i++)
            {
                printf("%d, ", bucketBounds[i]);
            }
            printf("%d \n", bucketBounds[(*bucketCount * 2) - 1]);
        }*/


        //Reinitialize the threadID for anouther
        threadID = threadIdx.x + bucketBoundStart;
        __syncthreads();

       // if((blockIdx.x == 0)&&(threadIdx.x == 0)){printf("Starting key scattering... \n");}

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

        //Test print tempData
       /* __syncthreads();
        if((blockIdx.x == 0)&&(threadIdx.x == 0))
        {
            printf("Temp Data Array: ");
            for(int i = 0; i < SIZE - 1; i++)
            {
                printf("%d, ", tempData[i]);
            }
            printf("%d \n", tempData[SIZE - 1]);
        }*/

        
        threadID = threadIdx.x + bucketBoundStart;
         __syncthreads();

        //if((blockIdx.x == 0)&&(threadIdx.x == 0)){printf("Starting triple buffer... \n");}

        /*Triple buffer will be relace with double buffering. For now. Triple buffering is 
          used to increase production time and test logic*/
        while(threadID < bucketBoundEnd)
        {
            data[threadID] = tempData[threadID]; 
            threadID += blockDim.x;
        }

        //Test print out the data array.
       /* __syncthreads();
        if((blockIdx.x == 0)&&(threadIdx.x == 0))
        {
            printf("Data Array: ");
            for(int i = 0; i < SIZE - 1; i++)
            {
                printf("%d, ", data[i]);
            }
            printf("%d \n", data[SIZE - 1]);
        }*/
//        printf("END threadID: %d\n", threadIdx.x);

    }
/*############################################################################*/
/****************************Local Sort****************************************/
/*############################################################################*/
    else if(numberOfBucketMembers > 1)
    {
        //if(threadIdx.x == 0){printf("BlockID: %d: Entering Local Sort\n", blockIdx.x);}

        
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



            //Test print histogram
            /*__syncthreads();
            if((blockIdx.x == 0)&&(threadIdx.x == 0))
            {
                 printf("Histogram: ");
                 for(int i = 0; i < BASE - 1; i++)
                 {
                     printf("%d, ", smem_histogram[i]);
                 }
                 printf("%d \n", smem_histogram[BASE - 1]);
             }*/


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

            //Test print prefix-sum
           /* __syncthreads();
            if((blockIdx.x == 0)&&(threadIdx.x == 0))
            {
                printf("Prefix Sum: ");
                for(int i = 0; i < BASE - 1; i++)
                {
                    printf("%d, ", smem_prefix_sum[i]);
                }
                printf("%d \n", smem_prefix_sum[BASE - 1]);
            }*/

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

            
            //Test print out the data array.
           /* __syncthreads();
            if((blockIdx.x == 0)&&(threadIdx.x == 0))
            {
                printf("Data Array: ");
                for(int i = 0; i < SIZE - 1; i++)
                {
                    printf("%d, ", data[i]);
                }
                printf("%d \n", data[SIZE - 1]);
            }
            */
            
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
    int threadCount = 20;
    int *G_bucketCount;//Total number of currently active buckets. 
    //int *G_Sync = 0;
    int shift = 1;//Used to index into powers of the base.
    int kernelInvocationCount = 0;

    //Buffer allocation.
    data = new(nothrow) int[SIZE];
    bucketBounds = new(nothrow) int[bucketBoundsSize];
    G_bucketCount = new(nothrow) int[1];
    
    //randomize data between 0 and 511.
    for(int i = 0; i < SIZE; i++){data[i] = rand() % 512;}
    
    //Print out the Unsorted Array.
    cout << "Array initiation... " << endl;
    cout << "Starting array: ";
    for(int i  = 0; i < SIZE; i++)
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
    for(int k = 0; k < 3; k++)//while(*G_bucketCount)
    {

    cout << "Starting Pass Number: " << kernelInvocationCount << " | block count: " << blockCount << " | thread count: " << threadCount << endl;//test print out!!!
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
    for(int i = 0; i < SIZE; i++)
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
