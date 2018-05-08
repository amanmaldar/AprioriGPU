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

// const int NUM_BATCHES = 1875;
const int FILE_BATCH_SIZE = 32;
const int DATA_BATCH_SIZE = 32;
const int DATA_SIZE = 784;
const int DATA_ROWS = 28;
const int DATA_COLS = 28;

__global__ void maxPool2x2_cuda(double *x, int num_im, int x_rows, int x_cols, int x_filt, double *o)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	extern __shared__ double im_buf[];	
	
	int total_im = num_im*x_filt;
	extern __shared__ double this_im_buf[];	

	while(bid<total_im)
	{
		int this_im = bid/x_filt;
		int this_ch = bid%x_filt;
		int x_idx;

		if(tid<x_rows*x_cols)
		{
			x_idx = this_im*x_rows*x_cols*x_filt + this_ch*x_rows*x_cols + tid;
			this_im_buf[tid] = x[x_idx];
		}
		__syncthreads();

		if(tid<x_rows*x_cols)
		{
			int this_row = tid/x_rows;
			int this_col = tid%x_cols;
			if(this_row%2==0 && this_col%2==0)
			{
				double max = this_im_buf[tid];
				if(this_im_buf[tid+1]>max)
					max = this_im_buf[tid+1];
				if(this_im_buf[tid+x_rows]>max)
					max = this_im_buf[tid+x_rows];
				if(this_im_buf[tid+x_rows+1]>max)
					max = this_im_buf[tid+x_rows+1];

				// printf("bid=%d, tid=%d, comparing %fMAX = %f\n", bid, tid, this_im_buf[tid], max);
				int o_row = this_row/2;
				int o_col = this_col/2;
				int o_offset = o_row*x_cols/2+o_col;
				int o_idx = this_im*x_rows/2*x_cols/2*x_filt + this_ch*x_rows/2*x_cols/2 + o_offset;

				o[o_idx] = max;
			}
		}
		__syncthreads();

		bid += gridDim.x;
	}
}

__global__ void relu_cuda(double *x, int num_im, int x_rows, int x_cols, int x_filt, double *o)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	while(bid<num_im*x_filt)
	{
		int this_im = bid/x_filt;
		int this_ch = bid%x_filt;
		if(tid<(x_rows*x_cols))
		{

			int x_idx = this_im*x_filt*x_rows*x_cols + this_ch*x_rows*x_cols + tid;
			
			if(x[x_idx]<0)
				o[x_idx] = 0.0;
			else
				o[x_idx] = x[x_idx];
		}
		__syncthreads();
		bid += gridDim.x;
	}
}

__global__ void conv2D_cuda(double *x, double *w, double *b, double *o, int *o_int, int ksize, int pad, int pad_size, int in_filt, int out_filt)
{
	int bid 		= blockIdx.x;											//blocks jump, each block deals with an image
	int tid 		= threadIdx.x;											//threads DON'T jump, each thread does a CONV operation (3x3 mul,add)
	// int this_in_ch 	= (bid%(in_filt*out_filt))%in_filt;						//every im has infilt*outfilt channels, every blk deals with 1 infilt
	// int this_out_ch = (bid%(in_filt*out_filt))/in_filt;						//every infilt blks deal with 1 outfilt
	// int this_im 	= bid/(in_filt*out_filt);								//every infilt*outfilt blks deal with 1 IMAGE
	int total_im 	= DATA_BATCH_SIZE * in_filt * out_filt;
	// int total_conv 	= DATA_BATCH_SIZE * DATA_SIZE * in_filt * out_filt; 	//this block has to process ONE BATCH of images
	extern __shared__ double this_im_buf[];									//dynamic shared mem, copy input image into block's shared mem
	int datasize 	= 784;
	int datarows 	= 28;
	int datacols 	= 28;
	// printf("%d\n\n grid dim", gridDim.x);

	while(bid<total_im)
	{	
		
		int this_in_ch 	= (bid%(in_filt*out_filt))%in_filt;						//every im has infilt*outfilt channels, every blk deals with 1 infilt
		int this_out_ch = (bid%(in_filt*out_filt))/in_filt;						//every infilt blks deal with 1 outfilt
		int this_im 	= bid/(in_filt*out_filt);								//every infilt*outfilt blks deal with 1 IMAGE
		
		// __shared__ int buf_offset;
		__shared__ int buf_cols;

		// if(bid==0 && tid==0)
		// 	printf("%d %d\t", bid,total_conv);

		int padsize=1;
		// buf_offset = (padsize*(datacols+ (2*padsize))) + padsize;
		buf_cols = datacols+2*pad_size;
		
		if(tid==0)															//thread 0 copies image into shared for strided reads within block
		{
			int start_idx = (this_im*in_filt*datasize) + (this_in_ch*datasize);							//offset of idx in x

			// if(bid==0)
			// 	printf("start_idx=%d ", start_idx);
			
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
						// printf("x idx = %d, row=%d\n", x_idx, row);	
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
		
		__syncthreads();													//only thread 0 copies this, wait till all threads reach here to get shared data
		
		if(tid < (datasize))
		{ 
			int o_idx 		= (this_im*out_filt*datasize) + (this_out_ch*datasize) + tid;
			int row 		= (tid/datarows) + padsize;
			int this_pix 	= (int)tid + (padsize*(datarows+2*padsize)+(2*row)- padsize);			
			double this_conv_value=0;

			for(int i=0;i<ksize;i++)
			{
				for(int j=0;j<ksize;j++)
				{
					int tmp = this_pix + (i-pad_size)*buf_cols + (j-pad_size);
					this_conv_value += this_im_buf[tmp]*w[this_out_ch*ksize*ksize + i*ksize+j];
				}
			}
			
			this_conv_value /= (ksize*ksize);
			int conv_value_int = (int)(this_conv_value*100000.0);
			
			atomicAdd(&o_int[o_idx], conv_value_int);
			o[o_idx] = (double)(o_int[o_idx])/100000.0;
			
		}

		__syncthreads();
		bid += gridDim.x;
	}
}



class cnn
{
	public:
	double *read_batch(double *lab)
	{
		double *batch = (double*)malloc(sizeof(double)*DATA_BATCH_SIZE*DATA_SIZE);
		int n;
		//n = rand() % NUM_BATCHES;
        	n=0;
		for(int b=0;b<DATA_BATCH_SIZE/FILE_BATCH_SIZE;b++)
		{
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
		ifstream my_file;
		my_file.open("train_lab/tr_lab_"+to_string(n)+".dat");
		for(int i=0;i<FILE_BATCH_SIZE;i++)
		{
			for(int j=0;j<10;j++)
			{
				my_file >> lab[i*10+j];
			}
		}
		return batch;
	}

	double *conv2D(double *x, double *w, double *b, int ksize, int padding, vector <int> in_shape, vector<int> out_shape)
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
		int *o_i = (int*)malloc(sizeof(int)*out_size);
		for(int i=0;i<out_size;i++)
		{
			o[i]=0.0;
			o_i[i] = 0;
		}

		double *x_d, *w_d, *b_d, *o_d;
		int *o_int;
		cudaMalloc((void**)&x_d,sizeof(double)*in_size);
		cudaMalloc((void**)&w_d,sizeof(double)*ksize*ksize*out_shape[3]);
		cudaMalloc((void**)&b_d,sizeof(double)*out_shape[3]);
		cudaMalloc((void**)&o_d,sizeof(double)*out_size);
		cudaMalloc((void**)&o_int,sizeof(int)*out_size);

		auto start_time_copy1 = chrono::high_resolution_clock::now();
		cudaMemcpy(x_d, x, sizeof(double)*in_size,cudaMemcpyHostToDevice);
		cudaMemcpy(w_d, w, sizeof(double)*ksize*ksize*out_shape[3],cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b, sizeof(double)*out_shape[3], cudaMemcpyHostToDevice);
		cudaMemcpy(o_int, o_i, sizeof(int)*out_size, cudaMemcpyHostToDevice);

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
		conv2D_cuda<<<num_blocks, num_threads, sizeof(double)*datasize>>>(x_d, w_d, b_d, o_d, o_int, ksize, padding, pad_size, in_shape[3], out_shape[3]);
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


		// for(int i=0;i<DATA_SIZE;i++)
		// {
		// 	if(i%DATA_ROWS==0)
		// 		printf("\n");
		// 	if(x[i]<0)
  //               printf("0 ");
  //           else
  //               printf("1 ");
		// }	

		// printf("\n\n\nOUTPUT:\n\n\n");	

		// for(int i=0;i<DATA_SIZE*2;i++)
		// {
		// 	if(i%DATA_ROWS==0)
		// 		printf("\n");
		// 	printf("%f ", o[i]);
		// }

		
		cudaFree(x_d);
		cudaFree(w_d);
		cudaFree(b_d);
		cudaFree(o_d);

		
        cudaThreadExit();

		return o;
	}

	double *relu(double *x, vector <int>x_size)
	{
		int total_xsize = (x_size[0]*x_size[1]*x_size[2]*x_size[3]);
		double *o = (double*)malloc(sizeof(double)*	total_xsize);

		double *x_d, *o_d;
		cudaMalloc((void**)&x_d, sizeof(double)*total_xsize);
		cudaMalloc((void**)&o_d, sizeof(double)*total_xsize);

		cudaMemcpy(x_d, x, sizeof(double)*total_xsize, cudaMemcpyHostToDevice);
		cudaMemcpy(o_d, o, sizeof(double)*total_xsize, cudaMemcpyHostToDevice);
		int num_blocks = x_size[0];
		int num_threads = 1024;

		relu_cuda<<<num_blocks, num_threads>>>(x_d, x_size[0], x_size[1], x_size[2], x_size[3], o_d);
		cudaDeviceSynchronize();
		cudaThreadSynchronize();
		cudaMemcpy(o, o_d, sizeof(double)*total_xsize, cudaMemcpyDeviceToHost);

		cudaFree(x_d);
		cudaFree(o_d);
		return o;
	}

	double *maxPool2x2(double *x, vector <int> x_size)
	{
		
		cout<<"\n\n\n\n\n";
		// for(int i=0;i<784;i++)
		// 	cout<<x[i]<<" ";
		int total_xsize = (x_size[0]*x_size[1]*x_size[2]*x_size[3]);
		int o_size = total_xsize/4;
		cout<<o_size;
		double *o = (double*)malloc(sizeof(double)*o_size);

		for(int i=0;i<o_size;i++)
			o[i]=0;

		double *x_d, *o_d;

		cudaMalloc((void**)&x_d,sizeof(double)*total_xsize);
		cudaMalloc((void**)&o_d,sizeof(double)*o_size);

		cudaMemcpy(x_d, x, sizeof(double)*total_xsize,cudaMemcpyHostToDevice);
		cudaMemcpy(o_d, o, sizeof(double)*o_size,cudaMemcpyHostToDevice);

		int num_blocks=(int) DATA_BATCH_SIZE;
		int num_threads=1024;

		int num_im = x_size[0];
		int x_rows = x_size[1];
		int x_cols = x_size[2];
		int x_filt = x_size[3];

		int buf_size = x_rows*x_cols*x_filt;

		maxPool2x2_cuda<<<num_blocks, num_threads, sizeof(double)*buf_size>>>(x_d, num_im, x_rows, x_cols, x_filt, o_d);
		cudaThreadSynchronize();
		cudaDeviceSynchronize();

		cudaMemcpy(o,o_d, sizeof(double)*o_size,cudaMemcpyDeviceToHost);

		cudaFree(x_d);
		cudaFree(o_d);

		
        cudaThreadExit();

		return o;
	}

	void write_to_file(double *x, char type, int num, int num_im, int im_size, int filt)
	{
		// ofstream my_file("/home/hpcc7110/anusha_kasivishwanathan/proj/1/"+type+to_string(num)+".dat", ios::binary);
		ofstream my_file(type+to_string(num)+".dat", ios::binary);
			
        for(int i=0;i<num_im;i++)
        {
        	for(int f=0;f<filt;f++)
        	{
	        	for(int k=0;k<im_size;k++)
	        	{
	            	if(k%28==0)
	                	my_file << "\n";
	            	my_file << x[i*filt*im_size + f*im_size + k] << " ";
	        	}
	    	}
		}
		my_file << "\n\n"; 
    	my_file.close();
	}	
};

int main(int argc, char **argv)
{
	cnn c;
	double *lab = (double*)malloc(sizeof(double)*DATA_BATCH_SIZE*10);
	double *batch_im = c.read_batch(lab);

	for(int i=0;i<320;i++)
	{
		if(i%10==0 && i>0)
			cout<<'\n';
		cout<<lab[i]<<" ";
	}
	int ksize= atoi (argv[1]);
	vector <int> num_filt;
	num_filt.push_back(4);
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

	w[9] = 1;
	w[10] = 2;
	w[11] = 1;
	w[12] = 0;
	w[13] = 0;
	w[14] = 0;
	w[15] = -1;
	w[16] = -2;
	w[17] = -1;

	w[18] = 1;
	w[19] = 0;
	w[20] = -1;
	w[21] = 2;
	w[22] = 0;
	w[23] = -2;
	w[24] = 1;
	w[25] = 0;
	w[26] = -1;

	w[27] = 1;
	w[28] = 2;
	w[29] = 1;
	w[30] = 0;
	w[31] = 0;
	w[32] = 0;
	w[33] = -1;
	w[34] = -2;
	w[35] = -1;

	double *b = (double*)malloc(sizeof(double)*num_filt[0]);
	for(int i=0;i<num_filt[0];i++)
		b[i]=0;

	int padding=1;

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

	std::vector <int> pool_shape;
	pool_shape.push_back(out_shape[0]);
	pool_shape.push_back(out_shape[1]/2);
	pool_shape.push_back(out_shape[2]/2);
	pool_shape.push_back(out_shape[3]);

	// random_device rd{};
	// mt19937 gen{rd()};
	double *conv1_out = c.conv2D(x,w,b,ksize,padding,in_shape,out_shape);
	double *relu1_out = c.relu(conv1_out, out_shape);
	double *pool1_out = c.maxPool2x2(relu1_out,out_shape);

	c.write_to_file(conv1_out, 'c',1,out_shape[0], out_shape[1]*out_shape[2], out_shape[3]);
	c.write_to_file(relu1_out, 'r',1,out_shape[0], out_shape[1]*out_shape[2], out_shape[3]);
	c.write_to_file(pool1_out, 'p',1,out_shape[0], pool_shape[1]*pool_shape[2], pool_shape[3]);
	
	// cout<<"\n\n\n";
	// cout<<out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3];
	int out_size = out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3];
	for(int i=784*64;i<(784*64+784);i++)
	{
		if(i%DATA_ROWS==0)
			printf("\n");
		cout<<conv1_out[i]<<" ";
	}

	cout<<"\n\n\n";

	for(int i=784*30;i<(784*30+784);i++)
	{
		if(i%DATA_ROWS==0)
			printf("\n");
		cout<<relu1_out[i]<<" ";
	}

	cout<<"\n\n\n";


	for(int i=784/4*30;i<(784/4*30+784/4);i++)
	{
		if(i%(DATA_ROWS/2)==0)
			printf("\n");
		cout<<pool1_out[i]<<" ";
	}

	out_shape.push_back(pool_shape[0]);
	out_shape.push_back(pool_shape[1]);
	out_shape.push_back(pool_shape[2]);
	out_shape.push_back(num_filt[1]);


	double *conv2_out = c.conv2D(pool1_out,w,b,ksize,padding,pool_shape,out_shape);

	c.write_to_file(conv2_out, 'c',2,out_shape[0], out_shape[1]*out_shape[2], out_shape[3]);

	return 0;
}
