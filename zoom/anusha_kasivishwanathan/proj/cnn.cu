#include <iostream>
#include <stdio.h>
#include <chrono>
#include <string.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <cmath>
#include <random>

using namespace std;

const int NUM_BATCHES = 1875;
const int FILE_BATCH_SIZE = 32;
const int DATA_BATCH_SIZE = 32;
const int DATA_SIZE = 784;
const int DATA_ROWS = 28;
const int DATA_COLS = 28;

__global__ void conv2D_blk(double *x, double *w, double *b, double *o, int ksize, int str, int pad, int pad_size, int relu, int pool, int in_filt, int out_filt)
{
	int bid 		= blockIdx.x;											//blocks jump, each block deals with an image
	int tid 		= threadIdx.x;											//threads DON'T jump, each thread does a CONV operation (3x3 mul,add)
	int this_in_ch 	= (bid%(in_filt*out_filt))%in_filt;						//every im has infilt*outfilt channels, every blk deals with 1 infilt
	int this_out_ch = (bid%(in_filt*out_filt))/in_filt;						//every infilt blks deal with 1 outfilt
	int this_im 	= bid/(in_filt*out_filt);								//every infilt*outfilt blks deal with 1 IMAGE
	int total_im 	= DATA_BATCH_SIZE * in_filt * out_filt;
	// int total_conv 	= DATA_BATCH_SIZE * DATA_SIZE * in_filt * out_filt; 	//this block has to process ONE BATCH of images
	extern __shared__ double this_im_buf[];									//dynamic shared mem, copy input image into block's shared mem
	int datasize 	= 784;
	int datarows 	= 28;
	int datacols 	= 28;
	// printf("%d\n\n grid dim", gridDim.x);

	while(bid<total_im)
	{	
		
		__shared__ int buf_offset;
		__shared__ int buf_cols;

		// if(bid==0 && tid==0)
		// 	printf("%d %d\t", bid,total_conv);

		int padsize=1;
		buf_offset = (padsize*(datacols+ (2*padsize))) + padsize;
		buf_cols = datacols+2*pad_size;
		
		if(tid==0)															//thread 0 copies image into shared for strided reads within block
		{
			int start_idx = (this_im*in_filt*datasize) + (this_in_ch*datasize);							//offset of idx in x

			if(bid==0)
				printf("start_idx=%d ", start_idx);
			
			if(pad==1)
			{
				for(int i=0;i<((datarows+2*pad_size)*(datarows+2*pad_size));i++)
				{
					int row = i/(datarows+2*pad_size);
					int col = i%(datacols+2*pad_size);
					if(row<pad_size || row>=(pad_size+datarows) || col<pad_size || col>=(pad_size+datacols))
						this_im_buf[i] = 0.0;
					else
					{
						int x_idx = start_idx+i-(padsize*(datarows+2*padsize)+2*row-padsize);
						// if(bid==0)
						printf("x idx = %d, row=%d\n", x_idx, row);	
						if(x[x_idx]<0)	
							this_im_buf[i] = 0.0;	
						else
							this_im_buf[i] = 1.0;
					}
				} 
			}	
			else
				for(int i=0;i<datasize;i++)
				{
					if(x[start_idx+i]<0)
						this_im_buf[i] = 0.0;
					else
						this_im_buf[i] =1.0;
				}
		}
		// buf_offset = pad_size;
		
		
		__syncthreads();													//only thread 0 copies this, wait till all threads reach here to get shared data
		
		if(tid < (datasize))
		{ 
			int o_idx = (this_im*out_filt*datasize) + (this_out_ch*datasize) + tid;
			int row= (tid/datarows) + padsize;
			int this_pix = (int)tid + (padsize*(datarows+2*padsize)+(2*row)- padsize);
			
			// printf("o_idx=%d, ",(this_im*out_filt*datasize) + (this_out_ch*datasize) + tid);
			for(int i=0;i<ksize;i++)
			{
				for(int j=0;j<ksize;j++)
				{
					int tmp = this_pix + (i-pad_size)*buf_cols + (j-pad_size);
					o[o_idx] += this_im_buf[tmp] * w[i*ksize+j];
					// if(bid==0 && o_idx==756)
					// 	printf("tmp=%d, im buf = %f, i=%d, j=%d, this_pix=%d, tid=%d, row=%d\n", tmp, this_im_buf[tmp],i,j, this_pix, tid, row);
				}
			}
			
			o[o_idx] /= (ksize*ksize);

			// if(relu)
			// 	if(o[o_idx]<0)
			// 		o[o_idx]=0.0;
		}

		__syncthreads();
		// if(tid==0)
		// 	printf("bid=%d, tid=%d, buf_offset=%d, buf_cols=%d\n", bid, tid, buf_offset, buf_cols);
		bid += gridDim.x;
	}
}

class cnn
{
	public:
	double *read_batch()
	{
		double *batch = (double*)malloc(sizeof(double)*DATA_BATCH_SIZE*DATA_SIZE);
		for(int b=0;b<DATA_BATCH_SIZE/FILE_BATCH_SIZE;b++)
		{
			//int n = rand() % NUM_BATCHES;
            int n=0;
            printf("RANDOM NUMBER = %d\n\n\n\n\n\n\n\n",n);
			ifstream my_file;
			my_file.open("train_im/tr_im_"+to_string(n)+".dat");
			for(int i=0;i<FILE_BATCH_SIZE;i++)
			{
				for(int j=0;j<DATA_SIZE;j++)
				{
					my_file >> batch[b*FILE_BATCH_SIZE*DATA_SIZE + i*DATA_SIZE+j];
				}
			}
		}
		return batch;
	}

	double *pool2x2(double *x, vector <int> in_shape, int out_size)
	{
		int in_size=1;
		for(int i=0;i<in_shape.size();i++)
			in_size = in_size*in_shape[i];
		
	}

	double *conv2D(double *x, double *w, double *b, int ksize, int stride, int padding, int relu, int pool, vector <int> in_shape, vector<int> out_shape)
	{
		int in_size=1;
		int out_size=1;
		
		for(int i=0;i<in_shape.size();i++)
			in_size = in_size*in_shape[i];
		
		for(int i=0;i<out_shape.size();i++)
			out_size = out_size*out_shape[i];

		printf("total input elts = %d, output elts=%d\n", in_size, out_size);
		int total_im = DATA_BATCH_SIZE  * in_shape[3] * out_shape[3];
		printf("total im for conv=%d\n", total_im);

		double *o = (double*)malloc(sizeof(double)*out_size);
		for(int i=0;i<out_size;i++)
			o[i]=0.0;

		double *x_d, *w_d, *b_d, *o_d;
		cudaMalloc((void**)&x_d,sizeof(double)*in_size);
		cudaMalloc((void**)&w_d,sizeof(double)*ksize*ksize*out_shape[3]);
		cudaMalloc((void**)&b_d,sizeof(double)*out_shape[3]);
		cudaMalloc((void**)&o_d,sizeof(double)*out_size);

		auto start_time_copy1 = chrono::high_resolution_clock::now();
		cudaMemcpy(x_d, x, sizeof(double)*in_size,cudaMemcpyHostToDevice);
		cudaMemcpy(w_d, w, sizeof(double)*ksize*ksize*out_shape[3],cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b, sizeof(double)*out_shape[3], cudaMemcpyHostToDevice);
		auto end_time_copy1 = chrono::high_resolution_clock::now();
		auto copy_time_us1 = chrono::duration_cast<chrono::microseconds>(end_time_copy1-start_time_copy1).count();	
		auto copy_time_ms1 = chrono::duration_cast<chrono::milliseconds>(end_time_copy1-start_time_copy1).count();	
		cout<<"\nTime to copy1 : "<<copy_time_us1<<" microseconds, about "<<copy_time_ms1<<" milliseconds\n";

		int num_blocks=(int) DATA_BATCH_SIZE;
		int num_threads=1024;
		// printf("calling kernel..\n");
		int pad_size = (ksize-1)/2;								//we need the 1st pix to be in the mid of a kernel
		int datasize = (DATA_ROWS+2*pad_size) * (DATA_COLS+2*pad_size);
		auto start_time_conv = chrono::high_resolution_clock::now();
		conv2D_blk<<<num_blocks, num_threads, sizeof(double)*datasize>>>(x_d, w_d, b_d, o_d, ksize, stride, padding, pad_size, relu, pool, in_shape[3], out_shape[3]);
		cudaThreadSynchronize();
		cudaDeviceSynchronize();
		auto end_time_conv = chrono::high_resolution_clock::now();
		auto conv_time_us = chrono::duration_cast<chrono::microseconds>(end_time_conv-start_time_conv).count();	
		auto conv_time_ms = chrono::duration_cast<chrono::milliseconds>(end_time_conv-start_time_conv).count();	
		
		cout<<"\nTime to conv:"<<conv_time_us<<" microseconds, about "<<conv_time_ms<<" milliseconds\n";
		

		
		auto start_time_copy = chrono::high_resolution_clock::now();
		cudaMemcpy(o, o_d, sizeof(double)*out_size, cudaMemcpyDeviceToHost);
		auto end_time_copy = chrono::high_resolution_clock::now();
		auto copy_time_us = chrono::duration_cast<chrono::microseconds>(end_time_copy-start_time_copy).count();	
		auto copy_time_ms = chrono::duration_cast<chrono::milliseconds>(end_time_copy-start_time_copy).count();	

		cout<<"\nTime to copy:"<<copy_time_us<<" microseconds, about "<<copy_time_ms<<" milliseconds\n";


		for(int i=0;i<DATA_SIZE;i++)
		{
			if(i%DATA_ROWS==0)
				printf("\n");
			if(x[i]<0)
                printf("0 ");
            else
                printf("1 ");
		}	

		printf("\n\n\nOUTPUT:\n\n\n");	

		for(int i=0;i<DATA_SIZE;i++)
		{
			if(i%DATA_ROWS==0)
				printf("\n");
			printf("%f ", o[i]);
		}

		ofstream my_file("out_conv.dat", ios::binary);
        for(int i=0;i<DATA_BATCH_SIZE;i++)
        {
	        for(int k=0;k<784;k++)
	        {
	            if(k%28==0)
	                my_file << "\n";

	            my_file << o[i*DATA_SIZE + k] << " ";
	            
	        }
	    }
        my_file << "\n\n"; 
    	my_file.close();

		cudaFree(x_d);
		cudaFree(w_d);
		cudaFree(b_d);
		cudaFree(o_d);

		
        cudaThreadExit();

		return o;
	}
};

int main(int argc, char **argv)
{
	cnn c;
	double *batch_im = c.read_batch();
	int ksize= atoi (argv[1]);
	vector <int> num_filt;
	num_filt.push_back(1);
	num_filt.push_back(8);
	num_filt.push_back(16);
	num_filt.push_back(32);
	num_filt.push_back(16);
	num_filt.push_back(8);
	num_filt.push_back(1);

	double *x = batch_im;
	
	double *w = (double*)malloc(sizeof(double)*ksize*ksize*num_filt[0]);

	w[0] = 1;
	w[1] = 0;
	w[2] = -1;
	w[3] = 2;
	w[4] = 0;
	w[5] = -2;
	w[6] = 1;
	w[7] = 0;
	w[8] = -1;

	// w[9] = 1;
	// w[10] = 2;
	// w[11] = 1;
	// w[12] = 0;
	// w[13] = 0;
	// w[14] = 0;
	// w[15] = -1;
	// w[16] = -2;
	// w[17] = -1;

	double *b = (double*)malloc(sizeof(double)*num_filt[0]);
	for(int i=0;i<num_filt[0];i++)
		b[i]=0;

	int stride=1;
	int padding=1;
	int relu=1;
	int pool=1;

	vector <int> in_shape;
	in_shape.push_back(DATA_BATCH_SIZE);
	in_shape.push_back(28);
	in_shape.push_back(28);
	in_shape.push_back(1);

	std::vector <int> out_shape;
	out_shape.push_back(DATA_BATCH_SIZE);
	out_shape.push_back(28);
	out_shape.push_back(28);
	out_shape.push_back(num_filt[0]);

	// random_device rd{};
	// mt19937 gen{rd()};
	double *conv1_out = c.conv2D(x,w,b,ksize,stride,padding,relu, pool, in_shape,out_shape);
	// for(int i=0;i<FILE_BATCH_SIZE;i++)
	// {
	// 	for(int j=0;j<DATA_SIZE;j++)
	// 	{
	// 		if(j%28==0)
	// 			printf("\n");
	// 		printf("%f ", batch_im[i*DATA_SIZE+j]);
	// 	}
	// 	printf("\n");
	// }
	return 0;
}
