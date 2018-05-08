#include <iostream>
#include <stdio.h>
#include <chrono>
using namespace std;

int dim1        = 10240;                                       //matrix dimension 1
int dim2        = 256;                                        //matrix dimension 2
int *a          = (int*)malloc(sizeof(int) * dim1 * dim2);  //allocate mem for matrix a
int *b          = (int*)malloc(sizeof(int) * dim2 );        //allocate mem for matrix b
int *pdt        = (int*)malloc(sizeof(int) * dim1 );        //mul a[1024][2048] by b[2048][1024] gives product pdt[1024][1024]


__global__ void mat_mul_cuda(int *a, int *b, int *pdt, int dim1, int dim2)
{
    __shared__ int smem[256];
    int bid=blockIdx.x;                                                                 //store the initial block ids in int var
    while(bid<dim1)                                                                     //stopping cond. - bid< num rows
    {
         smem[threadIdx.x] = a[bid*dim2+threadIdx.x] * b[threadIdx.x];                  //each thread computes pdt of a[bid][tid], b[tid]
         //printf("blk %d, smem[%d]=%d\n",blockIdx.x, threadIdx.x, smem[threadIdx.x]);
         __syncthreads();                                                               //wait until all threads in all blocks finish computing individual pdts
        
         for(int i=blockDim.x/2;i>0;i=i/2)                                              //reduce the indiv. pdts to the sum, by half each time
         {
             if(threadIdx.x<i)                                                          //only half the threads need to be used each time
             {
                 smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x+i];           //add two consecutive elements of smem
                 //printf("thread %d blk %d i=%d, added smem[%d] and smem[%d] into smem[%d]=%d\n", threadIdx.x,blockIdx.x,i, threadIdx.x, threadIdx.x + i, threadIdx.x, smem[threadIdx.x]);
             }
             __syncthreads();                                                           //wait until all threads do one half, then do the next half and so on
         }
         if(threadIdx.x==0)
         {
             pdt[bid] = smem[0];                                                        //make the 0th thread to assign this sum (in smem[0]) to pdt[bid]
             //printf("pdt[%d]=%d\n", bid, pdt[bid]);
         }
         bid += gridDim.x;                                                              //jump by gridDim number of blocks each time
    }
}


int main(int argc, char **argv)
{
	//srand(100);                                               //random seed used for testing
	cudaDeviceProp prop;                                        //used to study the properties of the device - max allowed threads per blk, max num blocks
	cudaGetDeviceProperties(&prop, 0);
	
		int num_hw_threads = prop.maxThreadsPerBlock;
		int num_hw_blocks  = prop.maxGridSize[0];
		
		for(int i=0;i<dim1;i++)                                     //outer loop = dim1
	        for(int j=0;j<dim2;j++)                         //inner loop = dim2
	             a[i*dim2+j]= rand()%10;                 //fill a with rand nums, row-wise
	
	    for(int i=0;i<dim2;i++)
	        b[i]=rand()%10;                                     //fill b

	auto start_time = chrono::high_resolution_clock::now();     //find time
	int *a_d, *b_d, *pdt_d;	
	cudaMalloc((void **)&a_d, sizeof(int)*dim1*dim2);           //allocate mem in GPU, copy input to GPU
	cudaMalloc((void **)&b_d, sizeof(int)*dim2);
	cudaMalloc((void **)&pdt_d, sizeof(int)*dim1);

	cudaMemcpy(a_d, a, sizeof(int)*dim1*dim2, cudaMemcpyHostToDevice); 
	cudaMemcpy(b_d, b, sizeof(int)*dim2, cudaMemcpyHostToDevice);
	
	auto start_time_mul = chrono::high_resolution_clock::now();
	mat_mul_cuda<<<128, 256>>>(a_d, b_d, pdt_d, dim1, dim2);      //perform computations
	auto end_time_mul = chrono::high_resolution_clock::now();
	auto time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count();	
	auto time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count();	
	cout<<"Time to mul:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
	cudaMemcpy(pdt, pdt_d, sizeof(int)*dim1, cudaMemcpyDeviceToHost);
	
	auto end_time = chrono::high_resolution_clock::now();
	auto time_to_cal_us = chrono::duration_cast<chrono::microseconds>(end_time-start_time).count();
	auto time_to_cal_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
	cout<<"Time to calc: "<<time_to_cal_us<<" microseconds, about "<<time_to_cal_ms<<" milliseconds"<<"\n";
	/*for(int i=0;i<dim1;i++)                                   //optional: print the matrices to verify
        {
                for(int j=0;j<dim2;j++)
                {
                        cout<<"a["<<i<<"]["<<j<<"],("<<i*dim2+j<<"): "<<a[i*dim2+j]<<"\t\t";
                }
                cout<<"\n";
        }
        cout<<'\n';
        for(int i=0;i<dim2;i++)
        { 
               cout<<"b["<<i<<"]: "<<b[i]<<"\t\t";
             
        }
        cout<<'\n';
        for(int i=0;i<dim1;i++)
        {
               cout<<"pdt["<<i<<"]="<<pdt[i]<<"\n";
                
        }
    */
	cout<<"threads: "<<num_hw_threads<<", Blocks: "<<num_hw_blocks<<"\n";		
	return 0;
}
