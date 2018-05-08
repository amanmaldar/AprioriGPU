#include <iostream>
#include <stdio.h>
#include <chrono>
using namespace std;

int dim1        = 510;                                         //matrix dimension 1
int dim2        = 510;                                         //matrix dimension 2
int *a          = (int*)malloc(sizeof(int) * dim1 * dim2);      //allocate mem for matrix a
int *b          = (int*)malloc(sizeof(int) * dim2 * dim1);      //allocate mem for matrix b

int *a_big      = (int*)malloc(sizeof(int) * (dim1)*(dim2+1));
int *b_big      = (int*)malloc(sizeof(int) * (dim1) * (dim2+2));
int *pdt        = (int*)malloc(sizeof(int) * dim1 * dim1);      //mul a[1024][2048] by b[2048][1024] gives product pdt[1024][1024]
int *verify     = (int*)malloc(sizeof(int) * dim1 * dim1);

__global__ void mat_mul_pad_tran(int *a, int *b, int *pdt, int dim1, int dim2)
{
	int bid = blockIdx.x;            //calculate row number
	
    __shared__ int smem[256];
	while(bid<dim1)                                             //dim1 = num rows, one thread per row of computations
    {	
        int sum = 0;
        int tid = threadIdx.x;
        while(tid<dim2)
        {
            smem[tid] = a[tid+bid*dim2] + b[tid+bid*dim2];
            __syncthreads();
            for(int i=blockDim.x/2;i>0;i=1/2)
            {
                if(tid<i)
                {
                    smem[tid] += smem[tid+i];
                }
                __syncthreads();
            }
            if(tid ==0 || tid%128==0)
                sum += smem[0];
            tid += blockDim.x;
        }
        if(threadIdx.x==0)
            pdt[bid*dim1+] = sum;
        bid += gridDim.x;                            //update tid (jump)
		//printf("new tid: %d\n",tid);
    }
}


__global__ void mat_mul(int *a, int *b, int *pdt, int dim1, int dim2)
{
	int bid = blockIdx.x;            //calculate row number
	
    __shared__ int smem[256];
	while(bid<dim1)                                             //dim1 = num rows, one thread per row of computations
    {	
        int sum =0;
        int tid = threadIdx.x;
        while(tid < dim2)
        {
            smem[threadIdx.x] = a[threadIdx.x+bid*dim2] + b[threadIdx.x * dim2 + bid];
            __syncthreads();
            for(int i=blockDim.x/2;i>0;i=1/2)
            {
                if(threadIdx.x<i)
                {
                    smem[threadIdx.x] += smem[threadIdx.x+i];
                }
                __syncthreads();
            }
            if(threadIdx.x== 0)
            {
                sum += smem[0];
            }
            tid += blockDim.x;
        }

        pdt[bid] = sum;
        bid += gridDim.x;                            //update tid (jump)
		    //printf("new tid: %d\n",tid);
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
        {
                for(int j=0;j<dim2;j++)                         //inner loop = dim2
                {
                    a[i*dim2+j]= rand()%10+1;                 //fill a with rand nums, row-wise
                    b[i*dim2+j]= rand()%10+1;                 //fill b with rand nums,column-wise
                }
                
        }
   
	for(int i=0;i<dim1;i++)                                     //outer loop = dim1
        {
                for(int j=0;j<dim2+2;j++)                         //inner loop = dim2
                {
                        if(j<dim2)
                        {
                            a_big[i*dim2+j]= rand()%10+1;                 //fill a with rand nums, row-wise
                            b_big[i*dim2+j]= rand()%10+1;                 //fill b with rand nums,column-wise
                        }
                        else
                        {
                            a_big[i*dim2+j] = 0;                          //add 2 zeros to every row of a and b
                            b_big[i*dim2+j] = 0;
                        }
                }
        }


    auto start_time = chrono::high_resolution_clock::now();     //find time
	int *a_d, *b_d, *abig_d, *bbig_d, *pdt_d;	
	cudaMalloc((void **)&a_d, sizeof(int)*dim1*dim2);           //allocate mem in GPU, copy input to GPU
	cudaMalloc((void **)&b_d, sizeof(int)*dim2*dim1);
	cudaMalloc((void **)&pdt_d, sizeof(int)*dim1*dim1);
    
    cudaMalloc((void **)&abig_d, sizeof(int)*dim1*(dim2+2));
    cudaMalloc((void **)&bbig_d, sizeof(int)*dim1*(dim2+2));
	
    cudaMemcpy(a_d, a, sizeof(int)*dim1*dim2, cudaMemcpyHostToDevice); 
	cudaMemcpy(b_d, b, sizeof(int)*dim2*dim1, cudaMemcpyHostToDevice);
	
	auto start_time_mul = chrono::high_resolution_clock::now();
	mat_mul_cuda<<<128, 256>>>(a_d, b_d, pdt_d, dim1, dim2);      //perform computations
	auto end_time_mul = chrono::high_resolution_clock::now();
	auto time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count();	
	auto time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count();	
	cout<<"Time to mul small:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
	cudaMemcpy(pdt, pdt_d, sizeof(int)*dim1*dim1, cudaMemcpyDeviceToHost);
	
	auto end_time = chrono::high_resolution_clock::now();
	auto time_to_cal_us = chrono::duration_cast<chrono::microseconds>(end_time-start_time).count();
	auto time_to_cal_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
	cout<<"Time to calc small: "<<time_to_cal_us<<" microseconds, about "<<time_to_cal_ms<<" milliseconds"<<"\n";
	
    cudaMemcpy(abig_d, a_big, sizeof(int)*dim1*(dim2+2), cudaMemcpyHostToDevice); 
    cudaMemcpy(bbig_d, b_big, sizeof(int)*(dim2+2)*dim1, cudaMemcpyHostToDevice);                                       
                                                                                                             
    auto start_time_mul = chrono::high_resolution_clock::now();                                              
    mat_mul_cuda<<<128, 256>>>(a_d, b_d, pdt_d, dim1, dim2);      //perform computations                     
    auto end_time_mul = chrono::high_resolution_clock::now();                                                
    auto time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count();	 
    auto time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count();	 
    cout<<"Time to mul:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";        
    cudaMemcpy(pdt, pdt_d, sizeof(int)*dim1*dim1, cudaMemcpyDeviceToHost);
    
    auto end_time = chrono::high_resolution_clock::now();
    auto time_to_cal_us = chrono::duration_cast<chrono::microseconds>(end_time-start_time).count();
    auto time_to_cal_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();        
    cout<<"Time to calc: "<<time_to_cal_us<<" microseconds, about "<<time_to_cal_ms<<" milliseconds"<<"\n";  
    
    
    
    
    for(int i=0;i<dim1;i++)
    {
        for(int j=0;j<dim2;j++)
        {
            verify[i*dim1+j] += a[i*dim2+j]*b[i*dim1+j];
        }
    }
    
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
