#include <iostream>
#include <stdio.h>
#include <chrono>
#include <string.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <tuple>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>    

using namespace std;

#define BATCH_SIZE 32;

__global__ void par_eucl_dist(double *dist, int dim, int dist_size)
{
    int tid = (blockIdx.x*blockDim.x)+threadIdx.x;
    while(tid<dim*dim)
    {
        // printf("t %d ", tid);
        int this_pixel  = tid;
        int this_row    = this_pixel/dim;
        int this_col    = this_pixel%dim;
        int start_idx   = (dim*dim * this_pixel) - this_pixel*(this_pixel+1)/2;         //idx in dist array
        int num_dist    = (dim*dim - this_pixel - 1);
        int end_idx     = start_idx+num_dist;                                           //dist array idx
        int next_pixel  = this_pixel+1;
        int next_row    = next_pixel/dim;
        int next_col    = next_pixel%dim;
        
        // printf("%d\t%d\t%d\t%d\t%d\n", this_row, this_col, start_idx, end_idx, num_dist);
        for(int i=start_idx;i<end_idx;i++)  
        {                                        //i - idx in dist array
            dist[i]     = sqrt((pow((next_col-this_col),2)+pow((next_row-this_row),2)));
            // printf("%d\t%d\t%d\t%f\n", this_row, this_col, i, dist[i]);
            // printf("%d\t%f\t",i,dist[i]);
            // if(this_pixel==744 && next_pixel==783)
                // printf("dist[%d]=%f\n\n\n\n\n\n\n\n",i,dist[i]);
            next_pixel  += 1;
            next_row    = next_pixel/dim;
            next_col    = next_pixel%dim;
        }

        tid += gridDim.x*blockDim.x;
    }
}


__global__ void par_signed_dist(double *data, int num_im, int data_size, double *eucl_dist, int eucl_dist_size, double *signed_dist, int option, int *signed_dist_int )
{
    int tid = threadIdx.x;                  //threads dont jump
    int bid = blockIdx.x;                   //blocks jump, 2401 blks deal with one image
    __shared__ double tmp_dist[256];
    int tot_blks = 2401*num_im;
    while(bid<(tot_blks))
    {
        int this_im     = bid/2401;
        int cnt         = (bid%2401)*blockDim.x+tid;
        int from_pix    = cnt/data_size;
        int to_pix      = cnt%data_size;
        int tmp         = this_im*data_size;
        int from_idx    = tmp+from_pix;
        int to_idx      = tmp+to_pix;
        int dist_idx    = tmp + from_pix;


        if(data[from_idx]!=data[to_idx])                        //if both pixels are same (either digit or non-digit, set dist to large value)
        {
            if(from_pix<to_pix)                                 //interchange if greater to smaller pix, we only have 0-1, 0-2...0-783, 1-2,1-3...1-783,2-3...
            { 
                int start_idx       = data_size*from_pix - from_pix*(from_pix+1)/2; //idx of this pix in eucl_dist array
                int eucl_dist_idx   = start_idx + (to_pix - from_pix - 1);
                tmp_dist[tid]       = eucl_dist[eucl_dist_idx];     //get sucl dist btn the 2 pixs
            }
            else
            {
                int start_idx       = data_size*to_pix - to_pix*(to_pix+1)/2; //idx of this pix in eucl_dist array
                int eucl_dist_idx   = start_idx + (from_pix - to_pix - 1);
                tmp_dist[tid]       = eucl_dist[eucl_dist_idx];     //get sucl dist btn the 2 pixs  
            }
            if(data[from_idx] == 0.0)                           //if the from_pix is outside digit, make dist negative
                tmp_dist[tid]   = (-1)*tmp_dist[tid];            
        }
        else
            tmp_dist[tid] = 100000.0;   

        if(option==0)
        {
            double t;
            if(dist_idx==0 && bid>2)
                t=signed_dist[dist_idx];
            if(tmp_dist[tid] < signed_dist[dist_idx])
                signed_dist[dist_idx] = tmp_dist[tid];
            if(dist_idx==0 && bid>2)
                printf("%d %d %d %f %f TO %f\n", bid, tid, dist_idx, tmp_dist[tid], t, signed_dist[dist_idx]);
        }
        else
        {
            if(tmp_dist[tid] < signed_dist[dist_idx])
                signed_dist[dist_idx] = tmp_dist[tid];

        }
        if(signed_dist[0]<100000 && dist_idx==0)
            printf("%d %d %f %f\n", bid, tid, tmp_dist[tid], signed_dist[dist_idx]);
        bid += gridDim.x;   
    }
}

// __global__ void par_signed_dist_atomic(double *data, int num_im, int data_size, double *eucl_dist, int eucl_dist_size, int *signed_dist )
// {
//     int tid = threadIdx.x;                  //threads dont jump
//     int bid = blockIdx.x;                   //blocks jump, 2401 blks deal with one image
//     __shared__ double tmp_dist[256];
//     int tot_blks = 2401*num_im;
//     while(bid<(tot_blks))
//     {
//         int this_im     = bid/2401;
//         int cnt         = (bid%2401)*blockDim.x+tid;
//         int from_pix    = cnt/data_size;
//         int to_pix      = cnt%data_size;
//         int tmp         = this_im*data_size;
//         int from_idx    = tmp+from_pix;
//         int to_idx      = tmp+to_pix;
//         int dist_idx    = tmp + from_pix;


//         if(data[from_idx]!=data[to_idx])                        //if both pixels are same (either digit or non-digit, set dist to large value)
//         {
//             if(from_pix<to_pix)                                 //interchange if greater to smaller pix, we only have 0-1, 0-2...0-783, 1-2,1-3...1-783,2-3...
//             { 
//                 int start_idx       = data_size*from_pix - from_pix*(from_pix+1)/2; //idx of this pix in eucl_dist array
//                 int eucl_dist_idx   = start_idx + (to_pix - from_pix - 1);
//                 tmp_dist[tid]       = eucl_dist[eucl_dist_idx];     //get sucl dist btn the 2 pixs
//             }
//             else
//             {
//                 int start_idx       = data_size*to_pix - to_pix*(to_pix+1)/2; //idx of this pix in eucl_dist array
//                 int eucl_dist_idx   = start_idx + (from_pix - to_pix - 1);
//                 tmp_dist[tid]       = eucl_dist[eucl_dist_idx];     //get sucl dist btn the 2 pixs  
//             }
//             if(data[from_idx] == 0.0)                           //if the from_pix is outside digit, make dist negative
//                 tmp_dist[tid]   = (-1)*tmp_dist[tid];            
//         }
//         else
//             tmp_dist[tid] = 100000.0;   

//         int local_dist_int = (int)(tmp_dist[tid]*10000);
//         // printf("%f %d\n", tmp_dist[tid],local_dist_int);  
//         int *p = signed_dist;
//         for(int i=0;i<dist_idx;i++)
//             p++;      
//         atomicMin(p,local_dist_int);
   
//         bid += gridDim.x;   
//     }
// }

__global__ void int_to_double(int size, int *sig_dst_int, double * sig_dst_double)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    while(bid<size/blockDim.x)
    {
        int dst_idx = bid*blockDim.x + tid;
        sig_dst_double[dst_idx] = (double)sig_dst_int[dst_idx]/10000.0;
        bid += gridDim.x;
    }
}

int reverse_int(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    // printf("%d %d %d %d\n", i, i>>8, i>>16, i>>24);
    // printf("%d %d %d %d %d %d\n", i, ch1, ch2, ch3, ch4, 4&5);
    // printf("%d %d %d %d\n", ch1<<24, ch2<<16, ch3<<8, ch4);
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

class dataset
{
    public: 

        int num_blocks      = 1024;
        int num_threads     = 784*784;
        int magic_number    = 0;
        int num_im          = 0;
        int num_rows        = 0;
        int num_cols        = 0;
        
        double *data;

    tuple<double*,int,int,int> read_data(string file_path)
    {
        ifstream file (file_path, ios::binary);
               
        if(file.is_open())
        {
            printf("Opened file!!!!!!!!\n");
            file.read((char*)&magic_number,sizeof(magic_number));
            file.read((char*)&num_im,sizeof(num_im));
            file.read((char*)&num_rows,sizeof(num_rows));
            file.read((char*)&num_cols,sizeof(num_cols));

            magic_number        = reverse_int(magic_number);
            num_im              = reverse_int(num_im);
            num_rows            = reverse_int(num_rows);
            num_cols            = reverse_int(num_cols);
            int data_size       = num_im*num_rows*num_cols;
            data = (double*)malloc(sizeof(double)*data_size);
            for(int im=0;im<num_im;im++)
            {
                for(int r=0;r<num_rows;r++)
                {
                    for(int c=0;c<num_cols;c++)
                    {
                        unsigned char tmp=0;
                        file.read((char*)&tmp, sizeof(tmp));
                        if(tmp>0)
                            tmp = 1.0;
                        data[(im*num_rows*num_cols) + (r*num_cols+c)] = tmp;
                        // printf("%d ", tmp);
                    }
                    // printf("\n");
                }
            }
        }

        return make_tuple(data,num_rows,num_cols,num_im);
    }

    double *compute_signed_dist(double *data, int num_im, int data_size, double *eucl_dist, int eucl_dist_size, int option)
    {      
        double *signed_dist_d, *eucl_dist_d, *data_d, *signed_dist;
        signed_dist = (double*)(malloc(sizeof(double)*num_im*data_size));
        // cudaMallocHost((void**)&signed_dist,sizeof(double)*num_im*data_size);
        int *signed_dist_int = (int*)malloc(sizeof(int)*num_im*data_size);
        int *signed_dist_int_d;
        for(int i=0;i<num_im*data_size;i++)
        {
            signed_dist[i]=100000;
            signed_dist_int[i]=100000;
        }

        cudaMalloc((void**)&data_d, sizeof(double)*num_im*data_size);
        cudaMalloc((void**)&signed_dist_d, sizeof(double)*num_im*data_size);
        cudaMalloc((void**)&signed_dist_int_d, sizeof(int)*num_im*data_size);
        cudaMalloc((void**)&eucl_dist_d, sizeof(double)*eucl_dist_size);
 
        cudaMemcpy(data_d, data, sizeof(double)*num_im*data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(signed_dist_d, signed_dist, sizeof(double)*num_im*data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(signed_dist_int_d, signed_dist_int, sizeof(int)*num_im*data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(eucl_dist_d, eucl_dist, sizeof(double)*eucl_dist_size, cudaMemcpyHostToDevice);

        int num_blocks = 1024;
        int num_threads = 256;

        auto start_time_mul = chrono::high_resolution_clock::now();
        par_signed_dist<<<num_blocks, num_threads>>>(data_d, num_im, data_size, eucl_dist_d, eucl_dist_size, signed_dist_d, option, signed_dist_int_d);
        // par_signed_dist2<<<num_blocks, 784>>>(data_d, num_im, data_size, eucl_dist_d, eucl_dist_size, signed_dist_d);
        auto end_time_mul = chrono::high_resolution_clock::now();
        auto time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count(); 
        auto time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count(); 
        cout<<"Time to mul:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
        
        start_time_mul = chrono::high_resolution_clock::now();
        cudaMemcpy(signed_dist, signed_dist_d, sizeof(double)*num_im*data_size, cudaMemcpyDeviceToHost);
        end_time_mul = chrono::high_resolution_clock::now();
        time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count(); 
        time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count(); 
        cout<<"Time to copy:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
        
        cudaFree(data_d);
        cudaFree(signed_dist_d);
        cudaFree(eucl_dist_d);
    
        printf("signed dst 0=%f\n\n", signed_dist[0]);
        int num_batches = num_im/BATCH_SIZE;
        num_batches += num_im%BATCH_SIZE;
        printf("train batches = %d\n", num_batches);
        
        // int batchsize = BATCH_SIZE;
        // for(int i=0;i<num_batches;i++)
        // {
        //     ofstream my_file("train_im_1/tr_im_"+to_string(i)+".dat", ios::binary);
        //     for(int j=0;j<batchsize;j++)
        //     {
        //         for(int k=0;k<data_size;k++)
        //         {
        //             int this_im = (i*batchsize) + j;
        //             if(k%28==0)
        //                 my_file << "\n";
        //             my_file << signed_dist[this_im*data_size+k] << " ";
                    
        //         }
        //         my_file << "\n\n";
        //     }
        //     my_file.close();
        // }
            

        cudaDeviceSynchronize();
        cudaThreadExit();


        return signed_dist;
    }

    // double *compute_signed_dist_atomic(double *data, int num_im, int data_size, double *eucl_dist, int eucl_dist_size)
    // {      
    //     double *eucl_dist_d, *data_d;
    //     int *signed_dist_d, *signed_dist;
    //     double *signed_dist_final, *signed_dist_final_d;
    //     signed_dist = (int*)(malloc(sizeof(int)*num_im*data_size));
    //     signed_dist_final = (double*)(malloc(sizeof(double)*num_im*data_size));
    //     // cudaMallocHost((void**)&signed_dist,sizeof(double)*num_im*data_size);
    //     for(int i=0;i<num_im*data_size;i++)
    //         signed_dist[i]=100000;

    //     cudaMalloc((void**)&data_d, sizeof(double)*num_im*data_size);
    //     cudaMalloc((void**)&signed_dist_d, sizeof(int)*num_im*data_size);
    //     cudaMalloc((void**)&signed_dist_final_d, sizeof(double)*num_im*data_size);
    //     cudaMalloc((void**)&eucl_dist_d, sizeof(double)*eucl_dist_size);
 
    //     cudaMemcpy(data_d, data, sizeof(double)*num_im*data_size, cudaMemcpyHostToDevice);
    //     cudaMemcpy(signed_dist_d, signed_dist, sizeof(int)*num_im*data_size, cudaMemcpyHostToDevice);
    //     cudaMemcpy(eucl_dist_d, eucl_dist, sizeof(double)*eucl_dist_size, cudaMemcpyHostToDevice);

    //     int num_blocks = 1024;
    //     int num_threads = 256;

    //     auto start_time_mul = chrono::high_resolution_clock::now();
    //     par_signed_dist_atomic<<<num_blocks, num_threads>>>(data_d, num_im, data_size, eucl_dist_d, eucl_dist_size, signed_dist_d);
    //     // cudaDeviceSynchronize();
    //     // par_signed_dist2<<<num_blocks, 784>>>(data_d, num_im, data_size, eucl_dist_d, eucl_dist_size, signed_dist_d);
    //     auto end_time_mul = chrono::high_resolution_clock::now();
    //     auto time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count(); 
    //     auto time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count(); 
    //     cout<<"Time to mul:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
        
    //     int_to_double<<<num_blocks, num_threads>>>(num_im*data_size, signed_dist_d, signed_dist_final_d);
        
    //     start_time_mul = chrono::high_resolution_clock::now();
    //     cudaMemcpy(signed_dist_final, signed_dist_final_d, sizeof(double)*num_im*data_size, cudaMemcpyDeviceToHost);
    //     end_time_mul = chrono::high_resolution_clock::now();
    //     time_to_mul_us = chrono::duration_cast<chrono::microseconds>(end_time_mul-start_time_mul).count(); 
    //     time_to_mul_ms = chrono::duration_cast<chrono::milliseconds>(end_time_mul-start_time_mul).count(); 
    //     cout<<"Time to copy:"<<time_to_mul_us<<" microseconds, about "<<time_to_mul_ms<<" milliseconds\n";
        
    //     cudaFree(data_d);
    //     cudaFree(signed_dist_d);
    //     cudaFree(signed_dist_final_d);
    //     cudaFree(eucl_dist_d);
 
    //     int num_batches = num_im/BATCH_SIZE;
    //     num_batches += num_im%BATCH_SIZE;
    //     printf("atomic train batches = %d\n", num_batches);
        
    //     // int batchsize = BATCH_SIZE;
    //     // for(int i=0;i<num_batches;i++)
    //     // {
    //     //     ofstream my_file("train_im/tr_im_"+to_string(i)+".dat", ios::binary);
    //     //     for(int j=0;j<batchsize;j++)
    //     //     {
    //     //         for(int k=0;k<data_size;k++)
    //     //         {
    //     //             int this_im = (i*batchsize) + j;
    //     //             if(k%28==0)
    //     //                 my_file << "\n";
    //     //             my_file << signed_dist_final[this_im*data_size+k] << " ";
                    
    //     //         }
    //     //         my_file << "\n\n";
    //     //     }
    //     //     my_file.close();
    //     // }
            

    //     cudaDeviceSynchronize();
    //     cudaThreadExit();


    //     return signed_dist_final;
    // }


    double *compute_eucl_dist(int eucl_dist_size, int num_cols)
    {
        double *eucl_dist_d;
        double *eucl_dist = (double*)malloc(sizeof(double)*eucl_dist_size);
        for(int i=0;i<eucl_dist_size;i++)
            eucl_dist[i]=0.0;

        int num_blocks = 2;
        int num_threads = 256;

        cudaMalloc((void**)&eucl_dist_d, sizeof(double)*eucl_dist_size);
        // cudaMemcpy(eucl_dist_d, eucl_dist, sizeof(double)*eucl_dist_size, cudaMemcpyHostToDevice);
        
        par_eucl_dist<<<num_blocks, num_threads>>>(eucl_dist_d, num_cols, eucl_dist_size);
        
        cudaMemcpy(eucl_dist, eucl_dist_d, sizeof(double)*eucl_dist_size, cudaMemcpyDeviceToHost);
        cudaFree(eucl_dist_d);

        cudaThreadExit();
        return eucl_dist;
    }

    void disp_signed_dist(double *signed_dist, int data_size, int num_im)
    {
        printf("%f\n", signed_dist[0]);
        for(int i=0;i<10;i++)
        {
            for(int j=0;j<data_size;j++)
            {
                if(j%28==0)
                    printf("\n");
                if(signed_dist[i*data_size+j]>0)
                    printf("+ ");
                else if(signed_dist[i*data_size+j]<0)
                    printf("- ");
                else
                    printf("0 ");
                // printf("%d ", (int)signed_dist[i*data_size+j]);

            }
            printf("\n\n\n");
        }
    }

    void disp_eucl_dist(double *eucl_dist, int num_rows, int num_cols)
    {
        int start_idx=0;
        for(int i=0;i<num_rows*num_cols;i++)
        {
            for(int j=0;j<num_rows*num_cols;j++)
            {
                int r=i;
                int c=j;
                if(i%num_rows==0 || j%num_rows==0)
                    printf("\n\n\n");
                if(r==c)
                    printf("from %d to %d\t:0.0000000\n",i,j);
                else if(r>c)
                {
                    start_idx = num_rows*num_cols*j - j*(j+1)/2;
                    printf("from %d to %d\t:%f\n", i,j,eucl_dist[start_idx+(i-j)-1]);
                }
                else
                {
                    start_idx = num_rows*num_cols*i - i*(i+1)/2;
                    printf("from %d to %d\t:%f\n", i,j,eucl_dist[start_idx+j-i-1]);
                }
            }
        }
    }
};


int main()
{
    const char *train_file_path = "/home/hpcc7110/anusha_kasivishwanathan/proj/mnist/train-images.idx3-ubyte";
    dataset tr_im;
    
    tuple<double*,int,int,int> data_size_tuple = tr_im.read_data(train_file_path);

    int num_rows, num_cols, num_im;
    double * tr_data;
    tr_data     = get<0>(data_size_tuple);
    num_rows    = get<1>(data_size_tuple);
    num_cols    = get<2>(data_size_tuple);
    num_im      = get<3>(data_size_tuple);

    int eucl_dist_size  = num_rows*num_cols*(num_rows*num_cols-1)/2;
    // for(int i=0;i<num_rows*num_cols;i++)
    // {
    //     if(i%28==0)
    //         printf("\n");
    //     printf("%d ", (int)tr_data[i]);
        
    // }
    // printf("\n");
    double *eucl_dist   = tr_im.compute_eucl_dist(eucl_dist_size, num_cols);
    double *signed_dist = tr_im.compute_signed_dist(tr_data, num_im, num_rows*num_cols, eucl_dist, eucl_dist_size,0);
    
    double *signed_dist1= tr_im.compute_signed_dist(tr_data, num_im, num_rows*num_cols, eucl_dist, eucl_dist_size,0);
    // tr_im.disp_signed_dist(signed_dist, num_rows*num_cols, num_im);
    // tr_im.disp_signed_dist(signed_dist1, num_rows*num_cols, num_im);
    for(int i=0;i<784;i++)
    {
        if(i%28==0)
            printf("\n");
        printf("%d ", (int)tr_data[i]);
    }
    printf("\n\n\n %f %f\n\n\n", signed_dist[0], signed_dist1[0] );

    return 0;
}


