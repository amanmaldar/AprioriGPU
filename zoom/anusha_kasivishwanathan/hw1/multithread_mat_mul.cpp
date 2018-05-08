#include <iostream>
#include <stdlib.h>
#include <thread>
#include <chrono>

using namespace std;

int dim1	= 1024;					                    //matrix dimension 1
int dim2	= 2048;					                    //matrix dimension 2
int *a		= (int*)malloc(sizeof(int) * dim1 * dim2);	//allocate mem for matrix a
int *b		= (int*)malloc(sizeof(int) * dim2 * dim1);	//allocate mem for matrix b
int *pdt	= (int*)malloc(sizeof(int) * dim1 * dim1);	//mul a[1024][2048] by b[2048][1024] gives product pdt[1024][1024]
int batch_size	= 128;
int num_batches	= dim1/batch_size;                      //find num batches of threads
int last_batch	= dim1%batch_size;




void mat_mul_thread(int i)                              //thread function
{
	int j,x,sum;
	for(j=0;j<dim1;j++)                                 //outer loop = num rows in pdt = 1024
	{
		sum=0;
		for(x=0;x<dim2;x++)                             //inner loop = num computations per row of pdt = 2048
		{
			sum += a[i*dim2+x]*b[x*dim1+j];
		}
		pdt[i*dim1+j]=sum;                              
	}	
}

int main(int argc, char **argv)
{
	//srand(100); 
	if(last_batch!=0)                                   //if dim1%batch_size is non-zero, smaller last batch with remaining threads
		num_batches += 1;	
	
	for(int i=0;i<dim1;i++)                             //outer loop - dim1
	{
		for(int j=0;j<dim2;j++)                         //inner loop - dim2
		{
			a[i*dim2+j]= rand()%10;			            //fill a with rand nums, row-wise
			b[i*dim2+j]= rand()%10;			            //fill b with rand nums,column-wise
		}
	}
	thread t[batch_size];                               //array to hold thread IDs
 	auto start_time = chrono::high_resolution_clock::now();
	for(int b=0;b<num_batches;b++)                      //create threads in batches
	{
		if(b==num_batches-1 && last_batch!=0)
			batch_size = last_batch;

		for(int i=0;i<batch_size;i++)
			t[i] = thread(mat_mul_thread, b*batch_size + i);//pass current row number to the function
		
		for(int i=0;i<batch_size;i++)                   //create a batch, wait to join then create the next batch
			t[i].join();
				
	}

	auto end_time = chrono::high_resolution_clock::now();
	auto time_to_cal_us = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
	auto time_to_cal_s = chrono::duration_cast<chrono::seconds>(end_time-start_time).count();
	cout<<"time to multiply matrices = "<<time_to_cal_us<<" microseconds, about "<<time_to_cal_s<<" seconds"<<"\n";
	
	/*for(int i=0;i<dim1;i++)                           //optional - print matrices to verify
	{
		for(int j=0;j<dim2;j++)
		{
			cout<<"a["<<i<<"]["<<j<<"],("<<i*dim2+j<<"): "<<a[i*dim2+j]<<"\t\t";
		}
		cout<<"\n";
	}
    cout<<"\n";
    for(int i=0;i<dim2;i++)
	{
		for(int j=0;j<dim1;j++)
			cout<<"b["<<i<<"]["<<j<<"],("<<i*dim1+j<<"): "<<b[i*dim1+j]<<"\t\t";
		cout<<"\n";
	}
    cout<<"\n";
	for(int i=0;i<dim1;i++)
	{
		for(int j=0;j<dim1;j++)
		{
			cout<<"pdt["<<i<<"*"<<dim1<<"+"<<j<<"]="<<pdt[i*dim1+j]<<"\t\t";
		}
		cout<<"\n";
	}*/
	return 0;
}

	
		
