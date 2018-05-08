#include <iostream>
#include <stdio.h>
#include <chrono>
using namespace std;

int dim1        = 16;                                         //matrix dimension 1
int dim2        = 2;                                           //matrix dimension 2
int *a          = (int*)malloc(sizeof(int) * dim1 * dim2);      //allocate mem for matrix a
int *b          = (int*)malloc(sizeof(int) * dim2);      //allocate mem for matrix b
int *pdt        = (int*)malloc(sizeof(int) * dim1);      //mul a[1024][2048] by b[2048][1024] gives product pdt[1024][1024]


__global__ void mat_vec_mul_red(int *a, int *b, int *pdt, int dim1, int dim2)
{
    int bid = blockIdx.x;
    printf("BID=%d\n", bid);
    //while(bid<dim1)                                                     //blks wait until all rows are done, each time 256 threads in each block perform computations                         
    //{
    //    __shared__ int smem[2];                                         //the threads have shared mem, of size num threads (added up using parallel reduction)
    //    smem[threadIdx.x] =  a[bid*dim2+threadIdx.x]*v[threadIdx.x];    //each thread computes smem as the product of on element each from a and v, smem[tid]=a[bid][tid]*v[tid]
    //    printf("threadidx=%d, blockidx=%d,smem[%d]=%d\n",threadIdx.x, bid, threadIdx.x, smem[threadIdx.x]);
    //    __syncthreads();                                                //wait until all 256 threads compute a product

    //	for(int i=2/2;i>0;i=i/2)                                        //now parallely reduce smem[256] using 128,64,32,16,8,4,2,1 threads every time
    //    {
    //        printf("i=%d, thread id=%d, threadid<i=%d\n", i, threadIdx.x, threadIdx.x<i);
    //        if(threadIdx.x<i)                                           //use threads 0,1,2...127 first time -> smem[0:127] hav sums, then use threads 0,1,2...63 to reduce smem to 64..
    //        {
    //            smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x+i];
    //            printf("I am thread %d blk %d, added %d and %d into smem[%d]=%d\n", threadIdx.x,bid, threadIdx.x, threadIdx.x + i, threadIdx.x, smem[threadIdx.x]);
    //        }
    //        __syncthreads();                                             //wait until all 128 threads finish adding to reduce from 128-64,...
    //    }
    //    if(threadIdx.x==0)
    //        pdt[bid] = smem[threadIdx.x];                                //make one thread to assign the sum to pdt of this row (bid=row)
    //    bid+=gridDim.x;                                                  //blocks ID jumps by the total number of blocks (0-128, 1-129...)
    //}

}

__global__ void kernel_func()
{
    printf("Hello there!!!!!");
    printf("\nmy thread ID = %d, blk Id = %d, \n");
}

int main(int argc, char **argv)
{
	srand(100);                                                 //random seed used for testing
	cudaDeviceProp prop;                                        //used to study the properties of the device - max allowed threads per blk, max num blocks
	cudaGetDeviceProperties(&prop, 0);
	
	int num_hw_threads = prop.maxThreadsPerBlock;
	int num_hw_blocks  = prop.maxGridSize[0];
	
	for(int i=0;i<dim1;i++)                                     //outer loop = dim1
        {
                for(int j=0;j<dim2;j++)                         //inner loop = dim2
                {
                        a[i*dim2+j]= rand()%10;                 //fill a with rand nums, row-wise
                       // b[i*dim2+j]= rand()%10;                 //fill b with rand nums,column-wise
                }
        }

    for(int i=0;i<dim2;i++)
        b[i] = rand()%10;

	auto start_time = chrono::high_resolution_clock::now();     //find time
	int *a_d, *b_d, *pdt_d;	
	cudaMalloc((void **)&a_d, sizeof(int)*dim1*dim2);           //allocate mem in GPU, copy input to GPU
	cudaMalloc((void **)&b_d, sizeof(int)*dim2);
	cudaMalloc((void **)&pdt_d, sizeof(int)*dim1);

	cudaMemcpy(a_d, a, sizeof(int)*dim1*dim2, cudaMemcpyHostToDevice); 
	cudaMemcpy(b_d, b, sizeof(int)*dim2, cudaMemcpyHostToDevice);
	
	auto start_time_mul = chrono::high_resolution_clock::now();
	//mat_vec_mul_red<<<2, 2>>>(a_d, b_d, pdt_d, dim1, dim2);      //perform computations
	kernel_func<<<1,2>>>();
    auto end_time_mul = chrono::high_resolution_clock::now();
	auto time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count();	
	auto time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count();	
	cout<<"Time to mul:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
	cudaMemcpy(pdt, pdt_d, sizeof(int)*dim1*dim1, cudaMemcpyDeviceToHost);
	
	auto end_time = chrono::high_resolution_clock::now();
	auto time_to_cal_us = chrono::duration_cast<chrono::microseconds>(end_time-start_time).count();
	auto time_to_cal_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
	cout<<"Time to calc: "<<time_to_cal_us<<" microseconds, about "<<time_to_cal_ms<<" milliseconds"<<"\n";
	/*for(int i=0;i<dim1;i++)                                     //optional: print the matrices to verify
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
                //for(int j=0;j<dim1;j++)
                        cout<<"b["<<i<<"]="<<b[i]<<"\t\t";
                cout<<"\n";
        }
        cout<<'\n';
        for(int i=0;i<dim1;i++)
        {
                //for(int j=0;j<dim1;j++)
                //{
                        cout<<"pdt["<<i<<"]="<<pdt[i]<<"\t\t";
                //}
                cout<<"\n";
        }
        cout<<'\n';
    */
	cout<<"threads: "<<num_hw_threads<<", Blocks: "<<num_hw_blocks<<"\n";		
	return 0;
}
