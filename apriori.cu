/************************************************************************
Author - Aman Maldar
Simple code - parallel version of data association.
Static value of minSupport=1. This will show all the pairs generated.
File = 6entries.txt
Limitation - Generates only till set of 4 pairs as of now.
It needs multiple changes for the data structure as well. Need to reconfigure it.

Data: (6entries.txt)
2 3 4
1 2 3 4 5
4 5 6 7
0 2
1 2 3 4
2 3 5 7 8


*************************************************************************/
#include "apriori.hcu"
#include "functions.hcu"


double minSupp = 0.001; // 0.001;





__shared__ int smem[128];
__shared__ int sum[9];

__global__ void prefix_scan_kernel (int *A_device, int *B_device , int *ans_device) {
//__global__ void prefix_scan_kernel (int *globalDataset_device, int *globalDatasetThreadIndex_device, int *ans_device) {

	int tid = threadIdx.x;
	sum[tid] = 0;
	__syncthreads(); 	//wait for all threads
    
	int index1=0;
	//int sum = 0;
	int begin = B_device[tid];
	while (tid < 9){
		printf("tid: %d begin: %d A_device[begin]: %d \n", tid, begin,A_device[begin]);
		//printf("tid: %d  A_device[begin+index1]: %d  begin: %d \n", tid, A_device[begin+index1], begin);
		while (A_device[begin+index1] != -1){
			//printf("\n A_device[begin+index1]: %d \n", A_device[begin+index1]);
			smem[index1] = A_device[begin+index1];
			__syncthreads(); 	//wait for all threads
			
			index1++;
		}
		printf("index1: %d \n", index1);
		//printf("tid: %d sum[tid]:",tid);
		for (int i=begin;i<begin+index1;i++){
			sum[tid] += smem[i];
			//printf(" %d ", smem[i]);
			__syncthreads(); 	//wait for all threads
		}
		//ans_device[threadIdx.x] = sum;
		ans_device[tid] = sum[tid];
		tid+=9;
	}
} // end kernel function




void Execute(int argc){

    	parse_database(argc);

	vector <int> A; //= globalDataset     // convert itemId_TidMapping into long array
    vector <int> B ; // = globalDatasetThreadIndex;
    //int *globalDatasetCpu = (int *) malloc (sizeof(int)* totalItems);
    int *A_cpu = (int *) malloc (sizeof(int)* totalItems);
    int *B_cpu = (int *) malloc (sizeof(int)* (maxItemID+1));    //index array lenght same as number of items
	int *ans_cpu = (int *) malloc (sizeof(int)* (maxItemID+1));
	
    int k =0;                   // global pointer for globalMap
    //globalDatasetThreadIndex.push_back(k);
	for(int i=0;i<=maxItemID;i++){
        B.push_back(k);
        B_cpu[i] = k;
        vector <int> tmp11 = itemId_TidMapping[i];    // copy entire vector
        for(int j=1;j<tmp11.size();j++){ // last item should be inclusive, first element is excluded
            A.push_back(tmp11[j]);
            A_cpu[k] = tmp11[j];
            //globalDatasetCpu[k] = globalDataset[k];
            k++;
		}

        A.push_back(-1);    // seperate mappings by -1
        A_cpu[k] = -1;
        k++;
       // globalDatasetThreadIndex.push_back(k);
	}
/*	cout << " Printing itemId_TidMapping as array: " << endl;
    for(int i =0;i<A.size();i++){
        //A_cpu[i] = A[i];
        cout << A[i] << " " ;
    }cout << endl;*/
    cout << " Printing itemId_TidMapping copy: " << totalItems << endl;
    for(int i =0;i<totalItems;i++){
        cout << A_cpu[i] << " " ;
    }cout << endl;
	cout << A_cpu[7] << " " << A_cpu[8] << " " << A_cpu[9] << " tesing \n " ;

 /*   cout << " Printing starting indexes " << endl;
    for(int i =0;i<B.size();i++){
        cout << B[i] << " " ;
    }cout << endl;*/

    cout << " Printing starting indexes " << endl;
    for(int i =0;i<B.size();i++){
        cout << B_cpu[i] << " " ;
    }cout << endl;
	
	//return;
	

	//vector <int> A; //= globalDataset     // convert itemId_TidMapping into long array
    //vector <int> B ; // = globalDatasetThreadIndex;
	//int *A_cpu = globalDataset
    //int *B_cpu = globalDatasetThreadIndex
	
	int *A_device; //device storage pointers
	int *B_device;
	int *ans_device;

	
    cudaMalloc ((void **) &A_device, sizeof (int) * totalItems);
    cudaMalloc ((void **) &B_device, sizeof (int) * 9);
    cudaMalloc ((void **) &ans_device, sizeof (int) * 9);

	
    cudaMemcpy (A_device, A_cpu, sizeof (int) * totalItems, cudaMemcpyHostToDevice);
    cudaMemcpy (B_device, B_cpu, sizeof (int) * 9, cudaMemcpyHostToDevice);
	cudaMemcpy (ans_device, ans_cpu, sizeof (int) * 9, cudaMemcpyHostToDevice);

	int numberOfBlocks = 1;
	int threadsInBlock = 100;
	
	prefix_scan_kernel <<< numberOfBlocks,threadsInBlock >>> (A_device, B_device, ans_device);
    cudaMemcpy (ans_cpu, ans_device, sizeof (int) * 9, cudaMemcpyDeviceToHost);

	cout << "answer addition is: ";
	for(int i=0;i<9;i++){
		cout << ans_cpu[i] << " ";
	} cout << endl;
 
    //cout << "two_freq_itemset:      " << two_freq_itemset << endl << "\n";

    //******************************************************************************************************************

    
    //work till pair of 2
    return;
    
   
}   // end Execute



int main(int argc, char **argv){

    auto start = chrono::high_resolution_clock::now();

    Execute(argc);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> el = end - start;
    cout<<"Execution time is:     " << el.count() * 1000 << " mS " << endl;

    return 0;
}






/*
	

	  
    L1.push_back(0);    // initialized first index with 0 as we are not using it.
    //minSupport = round(minSupp *  TID_Transactions);
    minSupport = 1;
    // Following code generates single items which have support greater than min_sup
    // compare the occurrence of the object against minSupport

    cout << "\n Support:" << minSupport << endl << "\n";
    //Generate L1 - filtered single items ? I think this should be C1, not L1.

    for (int i=0; i<= maxItemID; i++)
    {
        if(itemIDcount[i] >= minSupport){
            L1.push_back(i);     //push TID into frequentItem
            one_freq_itemset++;
            //cout << "1 Frequent Item is: (" << i << ") Freq is: " << itemIDcount[i] << endl;
        }
    }
    //cout << "one_freq_itemset:      " << one_freq_itemset << endl << "\n";
    //******************************************************************************************************************
    //Generate L2 .  Make a pair of frequent items in L1
    for (int i=0;i <= L1.size() -1 -1; i++)     //-1 is done for eliminating first entry
    {
        for (int j=i+1;j <= L1.size() -1; j++){
            twoStruct.a = L1[i];
            twoStruct.b = L1[j];
            L2.push_back(twoStruct);
            //cout << "2 Items are: (" <<L1[i]<< "," << L1[j] << ") " << endl;

        }
    }
    //******************************************************************************************************************
    //Generate C2. Prune L2 . Compare against min_support and remove less frequent items.
	



*/



