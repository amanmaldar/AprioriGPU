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


//double minSupp = 0.001; // 0.001;

/*
__shared__ int smem[128];

__global__ void addition_scan_kernel (int *A_device, int *B_device , int *ans_device) {

	int tid = threadIdx.x;
	__syncthreads(); 	
    
	int index1=0;
	int sum = 0;
	int begin = B_device[tid];
	while (tid < 9){
		printf("tid: %d begin: %d A_device[begin]: %d \n", tid, begin,A_device[begin]);
		while (A_device[begin+index1] != -1){
		// map data from A_device to smem // limitation on smem comes in picture
			smem[begin+index1] = A_device[begin+index1];
			__syncthreads(); 	
			
			index1++;
		}
		printf("index1: %d \n", index1);
		for (int i=begin;i<begin+index1;i++){
			sum += smem[i];
			__syncthreads(); 	
		}
		ans_device[tid] = sum;
		tid+=9;
	}
} // end kernel function
*/

__global__ void find2_common_kernel (int *A_device, int *B_device , int *p, int *q, int *common_device) {

int tid = threadIdx.x;
//__syncthreads(); 	
while (tid < 1) 
{	
	// p =3 , q = 5
//int len_p = 4; // B_device[p+1] - B_device[p] - 1; // = 16-11 -1 = 4 	1,2,5,6
//int len_q = 3; // B_device[q+1] - B_device[q] - 1; // = 25-21 -1 = 3   2,3,6
int len_p = B_device[*p+1] - B_device[*p] - 1; // = 16-11 -1 = 4 	1,2,5,6
int len_q = B_device[*q+1] - B_device[*q] - 1; // = 25-21 -1 = 3   2,3,6

*common_device = 0;
	
//int p_offset = 11;
//int q_offset = 21;
int p_offset = B_device[*p];
int q_offset = B_device[*q];
	
for (int i = 0; i < len_p; i++) 
{
	//int x = A_device[B_device[p]+i];
	//xtmp += i;
	int x = A_device[p_offset+i];		
	int y = 0;
		for (int j = 0; j < len_q; j++)
		{	
			y = A_device[q_offset+j];			
			if (x == y)
			{
				printf("tid: %d x: %d y: %d\n", tid, x, y );
				*common_device +=1;
			}
		} // end inner for 
} // end outer for
	//*common_device = 10;
	tid++;
} // end while
} // end kernel function




void Execute(int argc){

    parse_database(argc);

    int *A_cpu = (int *) malloc (sizeof(int)* totalItems);
    int *B_cpu = (int *) malloc (sizeof(int)* (maxItemID+1));    //index array lenght same as number of items
	int *ans_cpu = (int *) malloc (sizeof(int)* (maxItemID+1));
	
	//---------------This section processes the variables that should be transferred to GPU memory as global database--------
	// TIP - gloabal Map should contain all the itemIds, even if they are not frequent, we need them to have correct mapping
    int k =0;                  						 // global pointer for globalMap
	for(int i=0;i<=maxItemID;i++) {
        B_cpu[i] = k;
        vector <int> tmp11 = itemId_TidMapping[i];    // copy entire vector
        for(int j=1;j<tmp11.size();j++) {			  // last item should be inclusive, first element is excluded
            A_cpu[k] = tmp11[j];
            k++;
		}
		A_cpu[k] = -1;								// seperate mappings by -1
        k++;

	}

    cout << " Printing itemId_TidMapping copy A_cpu: " << totalItems << endl;
    for(int i =0;i<totalItems;i++) {
        cout << A_cpu[i] << " " ;
    } cout << endl;


    cout << " Printing starting indexes B_cpu: " << endl;
    for(int i =0;i<= maxItemID;i++) {
        cout << B_cpu[i] << " " ;
    } cout << endl;
	//-----------------------------------------------------------------------------------------------------------
	
	
	//-----------Generates single frequent items. Used later to create pairs-------------------------------------
    L1.push_back(0);    // initialized first index with 0 as we are not using it.
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
            cout << "1 Frequent Item is: (" << i << ") Freq is: " << itemIDcount[i] << endl;
        }
    }
    //cout << "one_freq_itemset:      " << one_freq_itemset << endl << "\n";
	//-----------------------------------------------------------------------------------------------------------
	
	int *pairs_cpu;
	pairs_cpu = new int[50];
	//----------------This section generates the pair of 2-------------------------------------------------------
	//Generate L2 .  Make a pair of frequent items in L1
	for (int i=0;i <= L1.size() -1 -1; i++)     //-1 is done for eliminating first entry
	{
		for (int j=i+1;j <= L1.size() -1; j++){
			twoStruct.a = L1[i];
			twoStruct.b = L1[j];
			L2.push_back(twoStruct);
			pairs[i*2] = L1[i];
			pairs[i*2 + 1] = L1[j];
			cout << "2 Items are: (" <<L1[i]<< "," << L1[j] << ") " << endl;
		}
		cout << "pairs size is " << pairs.size();
	}
	//-----------------------------------------------------------------------------------------------------------
	
	
	int *A_device; //device storage pointers
	int *B_device;
	int *ans_device;
	//int *pairs_device;
	
    cudaMalloc ((void **) &A_device, sizeof (int) * totalItems);
    cudaMalloc ((void **) &B_device, sizeof (int) * 9);
    cudaMalloc ((void **) &ans_device, sizeof (int) * 9);
	 //cudaMalloc ((void **) &pairs_device, sizeof (int) * 9);


	int *p_cpu = (int *) malloc (sizeof(int));
	int *q_cpu = (int *) malloc (sizeof(int));
	int *common_cpu = (int *) malloc (sizeof(int));
	*p_cpu = 2;
	*q_cpu = 3;
	*common_cpu = 0;
	
	int *p_device;
	int *q_device;
	int *common_device;
	cudaMalloc ((void **) &p_device, sizeof (int));
	cudaMalloc ((void **) &q_device, sizeof (int));
	cudaMalloc ((void **) &common_device, sizeof (int));
	
    cudaMemcpy (A_device, A_cpu, sizeof (int) * totalItems, cudaMemcpyHostToDevice);
    cudaMemcpy (B_device, B_cpu, sizeof (int) * 9, cudaMemcpyHostToDevice);
    //cudaMemcpy (ans_device, ans_cpu, sizeof (int) * 9, cudaMemcpyHostToDevice);
	cudaMemcpy (p_device, p_cpu, sizeof (int) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy (q_device, q_cpu, sizeof (int) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy (common_device, common_cpu, sizeof (int) * 1, cudaMemcpyHostToDevice);

	int numberOfBlocks = 1;
	int threadsInBlock = 2;
	
	find2_common_kernel <<< numberOfBlocks,threadsInBlock >>> (A_device, B_device, p_device, q_device, common_device);

    cudaMemcpy (common_cpu, common_device, sizeof (int), cudaMemcpyDeviceToHost);
	
	cout << "total common elements are: " << *common_cpu << endl; 

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



