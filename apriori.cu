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

//----------------------------------------------------------------------------
chrono::duration<double> parse_el;

__global__ void find4_common_kernel (int *A_device, int *B_device , int *pairs_device, int *pairs_device_count) {

int tid = blockIdx.x;
__shared__ int smem1[128];  
__shared__ int smem2[128];

while (tid < 13) 	//36
{	
pairs_device_count[tid] = 0;
int p = pairs_device[tid*4];
int q = pairs_device[tid*4+1];
int r = pairs_device[tid*4+2]; 
int s = pairs_device[tid*4+3]; 
int len_p = B_device[p+1] - B_device[p] - 1; // = 16-11 -1 = 4 	1,2,5,6
int len_q = B_device[q+1] - B_device[q] - 1; // = 25-21 -1 = 3   2,3,6
int len_r = B_device[r+1] - B_device[r] - 1; // = 25-21 -1 = 3   2,3,6
int len_s = B_device[s+1] - B_device[s] - 1; // = 25-21 -1 = 3   2,3,6

	
//int p_offset = 11;
//int q_offset = 21;
int p_offset = B_device[p];
int q_offset = B_device[q];
int r_offset = B_device[r];
int s_offset = B_device[s];

int k1,k2 = 0;
for (int i = 0; i < len_p; i++) 
{
	int x = A_device[p_offset+i];		// without shared memory	
	int y = 0;
		for (int j = 0; j < len_q; j++)
		{	
			y = A_device[q_offset+j];		// without shared memory		
			if (x == y)
			{	
				smem1[k1] = x;
				k1 += 1;
			}
		} // end inner for 
} // end outer for

for (int i = 0; i < len_r; i++) 
{
	int x = A_device[r_offset+i];		// without shared memory	
	int y = 0;
		for (int j = 0; j < len_s; j++)
		{	
			y = A_device[s_offset+j];		// without shared memory		
			if (x == y)
			{	
				//pairs_device_count[tid] += 1;	
				smem2[k2] = x;
				k2 += 1;
			}
		} // end inner for 
} // end outer for

	
for (int i = 0; i < k1; i++) 
{
	int x = smem1[i];		// without shared memory	
	int y = 0;
		for (int j = 0; j < k2; j++)
		{	
			y = smem2[j];		// without shared memory		
			if (x == y)
			{	
				pairs_device_count[tid] += 1;	
				//smem1[k] = x;
			}
		} // end inner for 
} // end outer for

	

	tid += 13;
} // end while
} // end kernel function

//----------------------------------------------------------------------------
__global__ void find3_common_kernel (int *A_device, int *B_device , int *pairs_device, int *pairs_device_count) {

int tid = blockIdx.x;
__shared__ int smem[128];  

while (tid < 28) 	//36
{	
pairs_device_count[tid] = 0;
int p = pairs_device[tid*3];
int q = pairs_device[tid*3+1];
int r = pairs_device[tid*3+2]; 
int len_p = B_device[p+1] - B_device[p] - 1; // = 16-11 -1 = 4 	1,2,5,6
int len_q = B_device[q+1] - B_device[q] - 1; // = 25-21 -1 = 3   2,3,6
int len_r = B_device[r+1] - B_device[r] - 1; // = 25-21 -1 = 3   2,3,6
	
//int p_offset = 11;
//int q_offset = 21;
int p_offset = B_device[p];
int q_offset = B_device[q];
int r_offset = B_device[r];

int k = 0;
for (int i = 0; i < len_p; i++) 
{
	int x = A_device[p_offset+i];		// without shared memory	
	int y = 0;
		for (int j = 0; j < len_q; j++)
		{	
			y = A_device[q_offset+j];		// without shared memory		
			if (x == y)
			{	
				smem[k] = x;
				k += 1;
			}
		} // end inner for 
} // end outer for

for (int i = 0; i < len_r; i++) 
{
	int x = A_device[r_offset+i];		// without shared memory	
	int y = 0;
		for (int j = 0; j < k; j++)
		{	
			y = smem[j];		// without shared memory		
			if (x == y)
			{	
				pairs_device_count[tid] += 1;	
			}
		} // end inner for 
} // end outer for

	tid += 28;
} // end while
} // end kernel function

//----------------------------------------------------------------------------

__global__ void find2_common_kernel (int *A_device, int *B_device , int *pairs_device, int *pairs_device_count) {

int tid = blockIdx.x;
//__shared__ int smem[128];  


//__syncthreads(); 	
while (tid < 36) 	//36
{	
int p = pairs_device[tid*2];
int q = pairs_device[tid*2+1];
int len_p = B_device[p+1] - B_device[p] - 1; // = 16-11 -1 = 4 	1,2,5,6
int len_q = B_device[q+1] - B_device[q] - 1; // = 25-21 -1 = 3   2,3,6

	
//int p_offset = 11;
//int q_offset = 21;
int p_offset = B_device[p];
int q_offset = B_device[q];

/*//--------------- copy data into shared memory--------------------------
for (int i =0; i <len_p; i++)
{
	smem[i] = A_device[p_offset+i];
	__syncthreads();
}

for (int i =0; i <len_q; i++)
{
	smem[len_p+i] = A_device[q_offset+i];
	__syncthreads();
}
//-------------------------------------------*/
	
for (int i = 0; i < len_p; i++) 
{
	//int x = A_device[B_device[p]+i];
	//xtmp += i;
	int x = A_device[p_offset+i];		// without shared memory	
	//int x = smem[i];			// with shared memory
	int y = 0;
		for (int j = 0; j < len_q; j++)
		{	
			y = A_device[q_offset+j];		// without shared memory		
			//y = smem[len_p+j];			// with shared memory
			if (x == y)
			{	//if (tid == 20 || tid == 32 || tid == 35)
				//{ printf("tid: %d x: %d y: %d\n", tid, x, y );}
			
				pairs_device_count[tid] += 1;	
			}
		} // end inner for 
} // end outer for

	tid += 36;
} // end while
} // end kernel function




void Execute(int argc){

	printing = argv[1]; //decide printing actions
	// Generate C1. Parsing the database generates C1.
	auto parse_start = chrono::high_resolution_clock::now();
  	parse_database(argc);
    auto parse_end = chrono::high_resolution_clock::now();
     parse_el = parse_end - parse_start;
  	

    int *A_cpu = (int *) malloc (sizeof(int)* totalItems);
    int *B_cpu = (int *) malloc (sizeof(int)* (maxItemID+1+1));    //index array lenght same as number of items // add an ending index
	
	//---------------This section processes the variables that should be transferred to GPU memory as global database--------
	// TIP - gloabal Map should contain all the itemIds, even if they are not frequent, we need them to have correct mapping
	int globalDbIdx = 0;							// globalDbIdx = C1_Index			
	for(int i = 0; i <= maxItemID; i++) {
        B_cpu[i] = globalDbIdx;
        vector <int> tmp = itemId_TidMapping[i];    	// copy entire vector
        for(int j = 1; j < tmp.size(); j++) {			// last item should be inclusive, first element is excluded
            A_cpu[globalDbIdx] = tmp[j];
            globalDbIdx++;
		}
		A_cpu[globalDbIdx] = -1;						// seperate mappings by -1
		
        globalDbIdx++;
		if (i == maxItemID) {
			 B_cpu[i+1] = globalDbIdx;	//add last index
		}
	}

	if (printing == 1) {
		cout << " Printing itemId_TidMapping A_cpu: (C1) " << totalItems << endl;
		for(int i =0;i<totalItems;i++) {
			cout << A_cpu[i] << " " ;
		} cout << endl;
	}

	if (printing == 1) {
		cout << " Printing starting indexes B_cpu: (C1Index)" << endl;
		for(int i =0;i<= maxItemID+1;i++) {
			cout << B_cpu[i] << " " ;
		} cout << endl;
	}
//-----------------------------------------------------------------------------------------------------------
	
	
//-----------Generates L1. Single frequent items. Used later to create pairs.-------------------------------------
    L1.push_back(0);    // initialized first index with 0 as we are not using it. For loop later starts at i=1 [1]
    minSupport = 1;
    // Following code generates single items which have support greater than minSupport
    // compare the occurrence of the object against minSupport

    if (printing == 1) { cout << "\n Support:" << minSupport << endl << "\n"; }
	
    for (int i=0; i<= maxItemID; i++)
    {
        if(C1[i] >= minSupport){		//itemIDcount = C1
            L1.push_back(i);     		//push TID into frequentItem
            one_freq_itemset++;			
        }
    }
	// printL1();			// we will print L1 through C1 only.
//-----------------------------------------------------------------------------------------------------------
	

//----------------This section generates the pair of 2-------------------------------------------------------
//Generate C2 .  Make a pair of frequent items in L1
	int *pairs_cpu, *pairs_cpu_count;
	pairs_cpu = new int[150];		// 72 this is large in size but we copy only required size of bytes
	pairs_cpu_count = new int[150];		// 36 this is large in size but we copy only required size of bytes
	int k1 = 0;
	for (int i = 1; i < L1.size() -1; i++)     //-1 is done for eliminating first entry from L1 [1]
	{
		for (int j = i+1; j < L1.size(); j++) {
			twoStruct.a = L1[i];
			twoStruct.b = L1[j];
			C2.push_back(twoStruct);
			k1+=2;
		}
	}
	//cout << "pairs size is: " << sizeof(pairs_cpu) << " k1: " << k1 <<endl;

	//pairs_cpu = new int[k1];		// 72 this is large in size but we copy only required size of bytes
	//pairs_cpu_count = new int[k1/2];		// 36 this is large in size but we copy only required size of bytes
	k1 = 0;
	for (auto i = C2.begin(); i < C2.end(); i++) {
			pairs_cpu[k1] = i -> a;
			pairs_cpu[k1+1] = i -> b;
			pairs_cpu_count[k1/2] = 0;	//initizlize with zero
			//cout << "2 Items are: (" <<pairs_cpu[k1]<< "," << pairs_cpu[k1+1] << ") " << endl;
			k1+=2;
	}
	 //printC2();
	
	//-----------------------------------------------------------------------------------------------------------
	
	// next - PASS THIS ARRAY TO GPU AND LET DIFFERENT THREADS WORK ON DIFFERENT PAIRS
	int *A_device; //device storage pointers
	int *B_device;
	int *ans_device;
	int *pairs_device;
	int *pairs_device_count;
	
    cudaMalloc ((void **) &A_device, sizeof (int) * totalItems);
    cudaMalloc ((void **) &B_device, sizeof (int) * 10); // maxItemID+1+1 = 10
    cudaMalloc ((void **) &ans_device, sizeof (int) * 9);
    cudaMalloc ((void **) &pairs_device, sizeof (int) * 150);		// 72 this is large in size but we copy only required size of bytes
    cudaMalloc ((void **) &pairs_device_count, sizeof (int) * 150);	// 36 // this is large in size but we copy only required size of bytes

    cudaMemcpy (A_device, A_cpu, sizeof (int) * totalItems, cudaMemcpyHostToDevice);
    cudaMemcpy (B_device, B_cpu, sizeof (int) * 10, cudaMemcpyHostToDevice);	//maxItemID+1+1 =10
	cudaMemcpy (pairs_device, pairs_cpu, sizeof (int) * 72, cudaMemcpyHostToDevice);	// COPY PAIRS

	
	int numberOfBlocks = 36;
	int threadsInBlock = 1;
	
	find2_common_kernel <<< numberOfBlocks,threadsInBlock >>> (A_device, B_device, pairs_device, pairs_device_count );
	
    cudaMemcpy (pairs_cpu_count, pairs_device_count, sizeof (int)*36, cudaMemcpyDeviceToHost);
	
	for (int i =0 ; i < 36; i++){
		if (pairs_cpu_count[i] >= 1) {
			//cout << "2 Frequent Items are: (" << pairs_cpu[i*2] << "," << pairs_cpu[i*2+1] <<") Freq is: " <<  pairs_cpu_count[i] << endl;
			twoStruct.a = pairs_cpu[i*2];
		    twoStruct.b = pairs_cpu[i*2+1];
		    twoStruct.freq = pairs_cpu_count[i];
		    L2.push_back(twoStruct);
		    two_freq_itemset++;
		}
	}
	    //cout << "two_freq_itemset:      " << two_freq_itemset << endl;
	//printL2();
	
    //---------------------------------------------------------------------
	
    //Generate C3
    int delta=1;
    // FOLLOWING 2 FOR LOOPS GENERATE SET OF 3 ITEMS
	k1 = 0;
    for (auto it = L2.begin(); it != L2.end(); it++,delta++ ) {     //delta is stride
        int base = it->a;

        auto it1 = L2.begin();                     // assign second iterator to same set *imp
        for (int k = 0; k < delta; k++) { it1++; }   //add a offset to second iterator and iterate over same set
	
        for (it1 = it1; it1 != L2.end(); it1++) {  //iterating over same set.
            if (base == it1->a) {
                    threeStruct.a = it ->a;
                    threeStruct.b = it ->b;
                    threeStruct.c = it1->b;
                    threeStruct.freq = 0;
                    C3.push_back(threeStruct);
            }	// if end
	 		else
               break;  // break internal for loop once base is not same as first entry in next pair. Increment *it
           } // internal for
    } // external for

	
		//pairs_cpu = new int[k1];		// 72 this is large in size but we copy only required size of bytes
	//pairs_cpu_count = new int[k1/3];		// 36 this is large in size but we copy only required size of bytes
	k1 = 0;
	for (auto i = C3.begin(); i < C3.end(); i++) {
			pairs_cpu[k1] = i -> a;
			pairs_cpu[k1+1] = i -> b;
			pairs_cpu[k1+2] = i -> c;
		    pairs_cpu_count[k1/3] = 0;	// initialize with zero
		    
            //cout << "3 Items are: (" <<pairs_cpu[k1] << "," << pairs_cpu[k1+1] << "," << pairs_cpu[k1+3]<< ") "  << endl;	// 28 total
     		k1 +=3;
	}
	//printC3();
	
	
	numberOfBlocks = 28;
	threadsInBlock = 1;
	cudaMemcpy (pairs_device, pairs_cpu, sizeof (int) * 84, cudaMemcpyHostToDevice);	//28*3 pairs
	find3_common_kernel <<< numberOfBlocks,threadsInBlock >>> (A_device, B_device, pairs_device, pairs_device_count );
        cudaMemcpy (pairs_cpu_count, pairs_device_count, sizeof (int)*28, cudaMemcpyDeviceToHost);

	for (int i =0 ; i < 28; i++){
	if (pairs_cpu_count[i] >= 1) {
          //  cout << "3 Frequent Items are: (" <<pairs_cpu[i*3] << "," << pairs_cpu[i*3+1] << "," << pairs_cpu[i*3+2]<< ") " << "Freq is: " <<pairs_cpu_count[i] << endl;
	    three_freq_itemset++;
	    threeStruct.a = pairs_cpu[i*3];
            threeStruct.b = pairs_cpu[i*3+1];
            threeStruct.c = pairs_cpu[i*3+2];
            threeStruct.freq = pairs_cpu_count[i];
            L3.push_back(threeStruct);
	}
	}
	//cout << "three_freq_itemset:    " << three_freq_itemset << endl << "\n";
	
	//printL3();
    //******************************************************************************************************************

//----------------------------------------------------------------------------------
	    //Generate C4
    delta= 1;
	k1=0;
    for(auto it2 = L3.begin(); it2 != L3.end(); it2++,delta++)
    {
        int c,d;
        auto it3 = L3.begin();                          // assign second iterator to same set *imp
        for (int k = 0; k < delta; k++) { it3++; }       //add a offset to second iterator and iterate over same set

        c = it2->a;
        d = it2->b;

        for (it3 = it3; it3 != L3.end(); it3++) {    //iterating over same set.
              if (c == it3->a && d == it3->b) {
                  fourStruct.a = it2->a;
                  fourStruct.b = it2->b;
                  fourStruct.c = it2->c;
                  fourStruct.d = it3->c;
                  fourStruct.freq =0;
                  C4.push_back(fourStruct);
		      
              } // end if
        } // end inner for
    }// end outer for
	

	//pairs_cpu = new int[k1];		// 72 this is large in size but we copy only required size of bytes
	//pairs_cpu_count = new int[k1/3];		// 36 this is large in size but we copy only required size of bytes
	k1 = 0;
	for (auto i = C4.begin(); i < C4.end(); i++) {
			       pairs_cpu[k1] = i->a;
			pairs_cpu[k1+1] = i->b;
			pairs_cpu[k1+2] = i->c;
		      pairs_cpu[k1+3] = i->d;
		    pairs_cpu_count[k1/4] = 0;	// initialize with zero
		      
                 // cout << "4 Items are: (" <<pairs_cpu[k1] << "," << pairs_cpu[k1+1] << "," << pairs_cpu[k1+2]<< "," << pairs_cpu[k1+3] << ") "  << endl;
			k1 +=4;
	}
	//printC4();
	
	
	numberOfBlocks = 13;
	threadsInBlock = 1;
	cudaMemcpy (pairs_device, pairs_cpu, sizeof (int) * 52, cudaMemcpyHostToDevice);	//13*4 pairs
	find4_common_kernel <<< numberOfBlocks,threadsInBlock >>> (A_device, B_device, pairs_device, pairs_device_count );
        cudaMemcpy (pairs_cpu_count, pairs_device_count, sizeof (int)*13, cudaMemcpyDeviceToHost);	// 13 pairs

	for (int i =0 ; i < 13; i++){
	if (pairs_cpu_count[i] >= 1) {
		four_freq_itemset++;
         //cout << "4 Frequent Items are: (" <<pairs_cpu[i*4] << "," <<pairs_cpu[i*4+1] << "," << pairs_cpu[i*4+2]<< "," << pairs_cpu[i*4+3] << ") " << "Freq is: " <<pairs_cpu_count[i] << endl;
		// L4 .push back
		four_freq_itemset++;
	    fourStruct.a = pairs_cpu[i*4];
            fourStruct.b = pairs_cpu[i*4+1];
            fourStruct.c = pairs_cpu[i*4+2];
            fourStruct.freq = pairs_cpu_count[i];
            L4.push_back(fourStruct);
	}
	}
	//cout << "four_freq_itemset:    " << four_freq_itemset << endl << "\n";
	//printL4();
//---------------------------------------------------------------------------------
	
	
	
    return;
    
   
}   // end Execute



int main(int argc, char **argv){

    auto start = chrono::high_resolution_clock::now();

    Execute(argc);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> el = end - start;
	
	if (printing == 1){
	printL1();
	printC2();
	printL2();
	printL3();
	printC3();
	printL4();
	printC4();
	}
	
	cout<<"Database Parsing time:    " << parse_el.count() * 1000 << " mS " << endl;
    cout<<"Total execution time:     " << el.count() * 1000 << " mS " << endl;

	
	
    return 0;
}









