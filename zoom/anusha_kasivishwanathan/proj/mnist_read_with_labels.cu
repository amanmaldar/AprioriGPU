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
        int magic_number    = 0;
        int num_im          = 0;
       
        


        int *data;
        tuple<int*,int> read_data(string file_path)
        {
            printf("reading ddataaaaaa\n\n\n");
            ifstream file (file_path, ios::binary);
            
            if(file.is_open())
            {
                printf("Opened file!!!!!!!!\n");
                file.read((char*)&magic_number,sizeof(magic_number));
                file.read((char*)&num_im,sizeof(num_im));

                magic_number        = reverse_int(magic_number);
                num_im              = reverse_int(num_im);
                
                printf("num im from redea %d\n", num_im);
                data = (int*)malloc(sizeof(double)*num_im);
                for(int im=0;im<num_im;im++)
                {
                    unsigned char tmp=0;
                    file.read((char*)&tmp, sizeof(tmp));
                    // if(tmp>0)
                    //     tmp = 1.0;
                    data[im] = (int)tmp;
                    // printf("%d ", tmp);
                }
            }

            return make_tuple(data,num_im);
        }
};

   
int main()
{
    

    int num_im;
   
    const char *train_labels_file_path = "/home/hpcc7110/anusha_kasivishwanathan/proj/mnist/train-labels.idx1-ubyte";
    dataset tr_lab;

    tuple<int*,int> label_size_tuple = tr_lab.read_data(train_labels_file_path);
    int *tr_labels;
    tr_labels   = get<0>(label_size_tuple);
    
    num_im      = get<1>(label_size_tuple);

    int num_batches = num_im/(int)BATCH_SIZE;
    int tmp;
    for(int b=0;b<num_batches;b++)
    {
        ofstream my_file("train_lab/tr_lab_"+to_string(b)+".dat", ios::binary);
        for(int i=0;i<32;i++)
        {          
            for(int j=0;j<10;j++)
            {
                if(j==tr_labels[i])
                {
                    tmp=1;
                    my_file << tmp<<" ";
                    // printf("b=%d, i=%d, 1 ",b, i);
                }
                else
                {
                    tmp=0;
                    my_file << tmp<<" ";
                    // printf("b=%d, i=%d, 0 ",b,i);
                }
            }
            my_file <<'\n';
            // printf("\n");
        }
        cout<<b<<'\n';
    }

    cout<<"\n";

    return 0;
}


