#include <iostream>
#include <stdio.h>
#include <chrono>
using namespace std;

int dim1        = 1024;                                         //matrix dimension 1
int dim2        = 2048;                                         //matrix dimension 2
int *a          = (int*)malloc(sizeof(int) * dim1 * dim2);      //allocate mem for matrix a
int *b          = (int*)malloc(sizeof(int) * dim2 * dim1);      //allocate mem for matrix b
int *pdt        = (int*)malloc(sizeof(int) * dim1 * dim1);      //mul a[1024][2048] by b[2048][1024] gives product pdt[1024][1024]


__global__ void mat_mul_cuda(int *a, int *b, int *pdt, int dim1, int dim2)
{
    __shared__ int smem[4];
    //printf("thread Idx = %d, block idx=%d\n", threadIdx.x, blockIdx.x);
    smem[threadIdx.x] = threadIdx.x;
    printf("smem[%d]=%d\n", threadIdx.x, smem[threadIdx.x]);
    __syncthreads();
    //printf("\nsync done\n");
    //for(int i=0;i<blockDim.x*gridDim.x;i++)
    //    printf("smem[%d]=%d\n", i, smem[i]);
	for(int i=blockDim.x/2;i>0;i=i/2)
    {
        printf("i=%d, thread id=%d, threadid<i=%d\n", i, threadIdx.x, threadIdx.x<i);
        if(threadIdx.x<i)
        {
            smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x+i];
            printf("I am thread %d, added %d and %d into smem[%d]=%d\n", threadIdx.x, threadIdx.x, threadIdx.x + i, threadIdx.x, smem[threadIdx.x]);
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++)
        printf("smem[%d]=%d\n", i, smem[i]);


	
//	while(tid<dim1)                                             //dim1 = num rows, one thread per row of computations
//	{
//		//printf("Old tid:%d, block ID:%d, block dim:%d, grid dim:%d\n", tid, blockIdx.x, blockDim.x, gridDim.x);
//		int j,x,sum; 
//           
//        for(j=0;j<dim1;j++)                                     //outer loop = number of entries in a row of pdt = 1024 
//        {  
//           	sum=0;
//           	for(x=0;x<dim2;x++)                                 //inner loop = number of computations per row of pdt = 2048
//              	sum += a[tid*dim2+x]*b[x*dim1+j]; 
//            pdt[tid*dim1+j]=sum;	                            //after computing each value, assign to pdt
//        }
//		tid += blockDim.x*gridDim.x;                            //update tid (jump)
//		//printf("new tid: %d\n",tid);
//	}
}

int main(int argc, char **argv)
{
	//srand(100);                                               //random seed used for testing
	cudaDeviceProp prop;                                        //used to study the properties of the device - max allowed threads per blk, max num blocks
	cudaGetDeviceProperties(&prop, 0);
	
	int num_hw_threads = prop.maxThreadsPerBlock;
	int num_hw_blocks  = prop.maxGridSize[0];
	
	for(int i=0;i<dim1;i++)                                     //outer loop = dim1
        {
                for(int j=0;j<dim2;j++)                         //inner loop = dim2
                {
                        a[i*dim2+j]= rand()%10;                 //fill a with rand nums, row-wise
                        b[i*dim2+j]= rand()%10;                 //fill b with rand nums,column-wise
                }
        }

	auto start_time = chrono::high_resolution_clock::now();     //find time
	int *a_d, *b_d, *pdt_d;	
	cudaMalloc((void **)&a_d, sizeof(int)*dim1*dim2);           //allocate mem in GPU, copy input to GPU
	cudaMalloc((void **)&b_d, sizeof(int)*dim2*dim1);
	cudaMalloc((void **)&pdt_d, sizeof(int)*dim1*dim1);

	cudaMemcpy(a_d, a, sizeof(int)*dim1*dim2, cudaMemcpyHostToDevice); 
	cudaMemcpy(b_d, b, sizeof(int)*dim2*dim1, cudaMemcpyHostToDevice);
	
	auto start_time_mul = chrono::high_resolution_clock::now();
	mat_mul_cuda<<<2, 4>>>(a_d, b_d, pdt_d, dim1, dim2);      //perform computations
	auto end_time_mul = chrono::high_resolution_clock::now();
	auto time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count();	
	auto time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count();	
	cout<<"Time to mul:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
	cudaMemcpy(pdt, pdt_d, sizeof(int)*dim1*dim1, cudaMemcpyDeviceToHost);
	
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
        for(int i=0;i<dim2;i++)
        {
                for(int j=0;j<dim1;j++)
                        cout<<"b["<<i<<"]["<<j<<"],("<<i*dim1+j<<"): "<<b[i*dim1+j]<<"\t\t";
                cout<<"\n";
        }
        for(int i=0;i<dim1;i++)
        {
                for(int j=0;j<dim1;j++)
                {
                        cout<<"pdt["<<i<<"*"<<dim1<<"+"<<j<<"]="<<pdt[i*dim1+j]<<"\t\t";
                }
                cout<<"\n";
        }*/

	cout<<"threads: "<<num_hw_threads<<", Blocks: "<<num_hw_blocks<<"\n";		
	return 0;
}
