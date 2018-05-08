#include <iostream>
#include <stdio.h>
#include <chrono>
using namespace std;

int dim1        = 256;                                       //matrix dimension 1
int dim2        = 256;                                        //matrix dimension 2
int *a          = (int*)malloc(sizeof(int) * dim1 * dim2);  //allocate mem for matrix a
int *b          = (int*)malloc(sizeof(int) * dim2 * dim2);        //allocate mem for matrix b
int *pdt        = (int*)malloc(sizeof(int) * dim1 );        //mul a[1024][2048] by b[2048][1024] gives product pdt[1024][1024]

__global__ void shared_mem_coalesce(int *a, int *b, int dim1)
{
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    
       
    __shared__ int smem[1024];
    int col = tid%32 + (bid%8)*32;
    int row = tid/32 + bid/8*32;
    //printf("tid=%d, bid=%d, row=%d, col=%d\n", tid, bid, row, col);
    //printf("row=%d, col=%d\n", row, col);
    smem[col * dim1 + row] = a[row *dim1 + col];
    __syncthreads();
    b[row*dim1+col] = smem[row*dim1+col];
    
    
}

__global__ void transpose(int *a, int *b, int dim1)
{
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    
    int col = tid%32 + (bid%8)*32;
    int row = tid/32 + bid/8*32;
    //printf("tid=%d, bid=%d, row=%d, col=%d\n", tid, bid, row, col);
    //printf("row=%d, col=%d\n", row, col);
    printf("a[%d][%d] = b[%d][%d]\n", row,col, col,row);
    b[col * dim1 + row] = a[row *dim1 + col];
    //__syncthreads();
    //b[row*dim1+col] = smem[row*dim1+col];
    
    
}











int main()
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
	cudaMalloc((void **)&b_d, sizeof(int)*dim2*dim1);
	cudaMalloc((void **)&pdt_d, sizeof(int)*dim1*dim1);

	cudaMemcpy(a_d, a, sizeof(int)*dim1*dim2, cudaMemcpyHostToDevice); 
	//cudaMemcpy(b_d, b, sizeof(int)*dim2*dim1, cudaMemcpyHostToDevice);
	
	auto start_time_mul = chrono::high_resolution_clock::now();
	transpose<<<64, 1024>>>(a_d, b_d, dim1);      //perform computations
	auto end_time_mul = chrono::high_resolution_clock::now();
	auto time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count();	
	auto time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count();	
	cout<<"Time to mul:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
	cudaMemcpy(b, b_d, sizeof(int)*dim1, cudaMemcpyDeviceToHost);
	
	auto end_time = chrono::high_resolution_clock::now();
	auto time_to_cal_us = chrono::duration_cast<chrono::microseconds>(end_time-start_time).count();
	auto time_to_cal_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
	cout<<"Time to calc: "<<time_to_cal_us<<" microseconds, about "<<time_to_cal_ms<<" milliseconds"<<"\n";
    cout<<"Hey\n";	
    for(int i=0;i<dim1;i++)
    {
        for(int j=0;j<dim1;j++)
        {
            if(b[i*dim1+j]!=a[j*dim1+i])
                cout<<"Error\n";
        }
    }
    cout<<"b correct!!\n";
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
	cout<<"b correct threads: "<<num_hw_threads<<", Blocks: "<<num_hw_blocks<<"\n";		
	return 0;
}
