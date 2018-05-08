//ref:https://github.com/kainino0x/cis565-proj2-cuda-stream-compaction/blob/master/compaction.cu
//https://github.com/riskybacon/cuda_scan/blob/master/cuda_scan/scan_bc.cu
//https://stackoverflow.com/questions/30832033/is-prefix-scan-cuda-sample-code-in-gpugems3-correct
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void gpu_scan_sharemem_phaseI(int *in, int *out, int *aux, int size_tmp, int block_d){

	__shared__ int tmp[2][128];
	unsigned int myblock = blockIdx.x;
	unsigned int tid = threadIdx.x;

	while(myblock < size_tmp){

		tmp[0][tid] = in[myblock*blockDim.x+threadIdx.x];
        __syncthreads();

        int iout = 0;
        for (int d = 0; d < block_d; ++d){   //depth d
        	iout ^= 1;
        	int *tmp_out = tmp[iout];
        	int *tmp_in = tmp[iout ^ 1];
        	int t1 = tid - (1 << d);    //tid - 2^d
        	if (t1 >= 0) {              //tid > 2^d
        		tmp_out[tid] = tmp_in[tid] + tmp_in[t1];
            }
            else {
            	tmp_out[tid] = tmp_in[tid];
            }
            __syncthreads();
        }

        out[myblock*blockDim.x+threadIdx.x] = tmp[iout][tid];
        aux[myblock] = tmp[iout][127]; //last element of each block for next scan
        myblock+=128;
	}
}

//phase2 also use scan, but limited the size (if follow phase1-2-3, grid.dim*block.dim can work, here is 128*128 works)

__global__ void gpu_scan_sharemem_phaseII(int *d_aux, int *d_aux2, int size_tmp_2, int block_d){

	__shared__ int tmp[2][128];
	unsigned int myblock = blockIdx.x;
	unsigned int tid = threadIdx.x;

	while(myblock < size_tmp_2){

		tmp[0][tid] = d_aux[myblock*blockDim.x+threadIdx.x];
        __syncthreads();

        int iout = 0;
        for (int d = 0; d < block_d; ++d){   //depth d
        	iout ^= 1;
        	int *tmp_out = tmp[iout];
        	int *tmp_in = tmp[iout ^ 1];
        	int t1 = tid - (1 << d);    //tid - 2^d
        	if (t1 >= 0) {              //tid > 2^d
        		tmp_out[tid] = tmp_in[tid] + tmp_in[t1];
            }
            else {
            	tmp_out[tid] = tmp_in[tid];
            }
            __syncthreads();
        }

        d_aux2[myblock*blockDim.x+threadIdx.x] = tmp[iout][tid];
        // aux[myblock] = tmp[iout][127]; //last element of each block for next scan
        myblock+=128;
	}
}


__global__ void gpu_scan_sharemem_phaseIII(int *d_out, int *d_out2, int *d_aux2, int size_tmp){

	__shared__ int tmp[128];
	unsigned int myblock = blockIdx.x;
	unsigned int tid = threadIdx.x;

	while(myblock < size_tmp){

		tmp[tid] = d_out[myblock*blockDim.x+threadIdx.x];
        __syncthreads();

        if(myblock > 0){
        	tmp[tid] += d_aux2[myblock-1];
        	__syncthreads();
        }

        d_out2[myblock*blockDim.x+threadIdx.x] = tmp[tid];
        myblock+=128;
	}
}

int main()
{
	int *in;
	int *out;
	int *out_2;
    int *h_out;
	int *aux;
	int *aux_2;
	int num_size = 32000000;
	int size_tmp = (num_size + 127) / 128;
	int block_d = 7;   //depth of block, log2(128) = 7
	cudaMallocHost((void **) &in, sizeof(int)*num_size);
	cudaMallocHost((void **) &out, sizeof(int)*num_size);
    cudaMallocHost((void **) &h_out, sizeof(int)*num_size);
	cudaMallocHost((void **) &out_2, sizeof(int)*num_size);
	cudaMallocHost((void **) &aux, sizeof(int)*size_tmp);  //for checking itermediate values
	cudaMallocHost((void **) &aux_2, sizeof(int)*size_tmp);  //for checking the update itermediate values

	for (int i = 0; i < num_size; i++) {
		in[i] = 1;
		out[i] = 0;
    }

    for (int i = 0; i < size_tmp; i++) {
		aux[i] = 0;
    }

    int *d_in;
    int *d_out;
    int *d_aux;
    // int dszp = (num_size)*sizeof(int);
    int dszp_aux = (size_tmp)*sizeof(int);
    cudaMalloc((void **) &d_in, sizeof(int)*num_size);
    cudaMalloc((void **) &d_out, sizeof(int)*num_size);
    cudaMalloc((void **) &d_aux, sizeof(int)*size_tmp);
    // cudaMemset(d_out, 0, dszp);
    cudaMemset(d_aux, 0, dszp_aux);

// Phase1: scan original data per block and store last element of each block for later scan

    cudaMemcpy(d_in, in, sizeof(int)*num_size, cudaMemcpyHostToDevice);
    gpu_scan_sharemem_phaseI<<<128, 128>>>(d_in, d_out, d_aux, size_tmp, block_d);
    cudaMemcpy(out, d_out, sizeof(int)*num_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(aux, d_aux, sizeof(int)*size_tmp, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    printf("phase1: first five scan results(base):\n");
    for (int i = 126; i < 136; i++){
        printf("%d\n", aux[i]);
    }

    // printf("Phase1: scan each block\n");
    // printf("---------------------------------\n");


    // printf("first 10 elements of first five blocks:\n");
    // for (int i = 0; i < 513; i=i+128){
    // 	for (int j = i; j< 10+i; j++){
    // 		printf("%d  ", out[i+j]);
    // 	}
    // 	printf("\n");
    // }

    // printf("last element in first five blocks:\n");
    // for (int i = 0; i < 5; i++){
    // 	printf("%d\n", aux[i]);
    // }

// Phase2: scan the last element of each block to accumulate the sum of each block
    // printf("Phase2: scan the last element of each block\n");
    // printf("---------------------------------\n");

    // int *d_aux2;
    // int size_tmp_2 = (size_tmp + 127) / 128;
    // cudaMalloc((void **) &d_aux2, sizeof(int)*size_tmp);
    // cudaMemset(d_aux2, 0, dszp_aux);
    // gpu_scan_sharemem_phaseII<<<128, 128>>>(d_aux, d_aux2, size_tmp_2, block_d);
    // cudaMemcpy(aux_2, d_aux2, sizeof(int)*size_tmp, cudaMemcpyDeviceToHost);

    int *d_aux2;
    cudaMalloc((void **) &d_aux2, sizeof(int)*size_tmp);

    aux_2[0] = aux[0];
    for(int i = 1; i < size_tmp; i++){
        aux_2[i] = aux[i] + aux_2[i-1];
    }

    printf("first five scan results(base):\n");
    for (int i = 126; i < 136; i++){
    	printf("%d\n", aux_2[i]);
    }

    cudaMemcpy(d_aux2, aux_2, sizeof(int)*size_tmp, cudaMemcpyHostToDevice);

// Phase3: scan the last element of each block to accumulate the sum of each block
    // printf("Phase3: add the base to each block\n");
    // printf("---------------------------------\n");
    int *d_out2;
    cudaMalloc((void **) &d_out2, sizeof(int)*num_size);
    gpu_scan_sharemem_phaseIII<<<128, 128>>>(d_out, d_out2, d_aux2, size_tmp);
    cudaMemcpy(out_2, d_out2, sizeof(int)*num_size, cudaMemcpyDeviceToHost);

// check the GPU results
    printf("First 100 elements of the scan results:\n");
    for (int i = 0; i < 100; i++){
    	printf("%d\t", out_2[i]);
    	if (i % 10 == 0) printf("\n");
    }
    printf("\n");

    printf("last 100 elements of the scan results:\n");
    for (int i = num_size-101; i < num_size-1; i++){
        printf("%d\t", out_2[i]);
        if (i % 10 == 0) printf("\n");
    }
    printf("\n");

// Verify via comparasion with cpu version
    int psum = 0;
    for (int i = 0; i < num_size; i++){
        psum += in[i];
        if (psum != out_2[i]) {printf("mismatch at %d, was: %d, should be: %d\n", i, out_2[i], psum); return 1;}
    }
    printf("successfully scan!\n");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out2);
    cudaFree(d_aux);
    cudaFree(d_aux2);


    cudaFreeHost(in);
    cudaFreeHost(out);
    cudaFreeHost(out_2);
    cudaFreeHost(h_out);
    cudaFreeHost(aux);
    cudaFreeHost(aux_2);

    return 0;
}
