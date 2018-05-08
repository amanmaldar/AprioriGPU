#include "wtime.h"
#include "reader.hpp"
#include "perceptron.hpp"

//#define DIMENSION 61 

int DIMENSION = 61;
int NUM_THREADS = 128;
int NUM_BLOCKS = 1;


float *W = (float*)malloc(sizeof(float)*DIMENSION*NUM_THREADS);

void predict(float* &W, float &b, float *data, int count, int dimension)
{
    for(int i = 0; i < count/dimension; i++)
    {
        float predict = 0;
        float expected = data[i*dimension + dimension - 1];

        for(int j = 0; j < dimension - 1; j ++)
            predict += W[j] * data[i*dimension + j] + b;

        if (predict < 0) predict = -1;
        else predict = 1;

        std::cout<<"Expect "<<expected<<", predict "<<predict<<"\n";
    }
}


__global__ perceptron(float* &W, float &b, float *data, int count, int dimension, int epoch)
{
    __shared__ float W_sh[60];
    __shared__ float W_red[128];
    int tid = threadIdx.x;
    while(tid < count/dimension)
    {
        float predicted = 0;
        for(int i=0;i<dimension-1;i++)
            predicted = W[i]*data[tid*dimension+i];
        
        if(predicted<=0)    predicted = 0;
        else                predicted = 1;

        
        tid += blockDim.x;
    }
}



int main(int args, char **argv)
{

    std::cout<<"/path/to/exe epoch\n";

    const int EPOCH = atoi(argv[1]);

    assert(args == 2);
    float *train_data;
    int train_count;

    float *test_data;
    int test_count;
    
    //float *W;
    //float b;

    reader("dataset/train_data.bin", train_data, train_count);
    reader("dataset/test_data.bin", test_data, test_count);
    
    //printer(train_dataset, train_count, DIMENSION);
    //printer(test_dataset, test_count, DIMENSION);
    float *W_d;
    float b;
    float train_data_d;

    cudaMalloc((void**)&W_d, sizeof(float)*(DIMENSION-1)*NUM_THREADS);
    cudaMemcpy(W_d, W, sizeof(float)*(DIMENSION-1)*NUM_THREADS,cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&train_data_d, sizeof(float)*train_count);
    cudaMemcpy(train_data_d, train_data, sizeof(float)*train_count, cudaMemcpyHostToDevice);

    perceptron<<<NUM_BLOCKS, NUM_THREADS>>>(W_d, b, train_data_d, train_count, DIMENSION, epoch);

//    perceptron_seq(W, b, train_data, train_count, DIMENSION, EPOCH);
    predict(W, b, test_data, test_count, DIMENSION);
    
    printer(W, DIMENSION - 1, 1);
    return 0;
}



