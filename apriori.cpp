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

void Execute(int argc){

    	parse_database(argc);

    //return;
	vector <int> globalDataset;     // convert itemId_TidMapping into long array
    vector <int> globalDatasetThreadIndex;
    int k =0;                   // global pointer for globalMap
    //globalDatasetThreadIndex.push_back(k);
	for(int i=0;i<=maxItemID;i++){
        globalDatasetThreadIndex.push_back(k);

        vector <int> tmp11 = itemId_TidMapping[i];    // copy entire vector
        //tmp11 = {1,2,3};
        for(int j=1;j<tmp11.size();j++){ // last item should be inclusive, first element is excluded
            globalDataset.push_back(tmp11[j]);
			k++;
		}

        globalDataset.push_back(-1);    // seperate mappings by -1
        k++;
       // globalDatasetThreadIndex.push_back(k);
	}
	cout << " Printing itemId_TidMapping as array: " << endl;
	for(int i =0;i<globalDataset.size();i++){
		cout << globalDataset[i] << " " ;
	}cout << endl;
    cout << " Printing starting indexes " << endl;
    for(int i =0;i<globalDatasetThreadIndex.size();i++){
        cout << globalDatasetThreadIndex[i] << " " ;
    }cout << endl;
	
	
	//int numberOfBlocks = 1;
	//int threadsInBlock = 100;
	  
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
            cout << "1 Frequent Item is: (" << i << ") Freq is: " << itemIDcount[i] << endl;
        }
    }
    cout << "one_freq_itemset:      " << one_freq_itemset << endl << "\n";
    //******************************************************************************************************************
    //Generate L2 .  Make a pair of frequent items in L1
    for (int i=0;i <= L1.size() -1 -1; i++)     //-1 is done for eliminating first entry
    {
        for (int j=i+1;j <= L1.size() -1; j++){
            twoStruct.a = L1[i];
            twoStruct.b = L1[j];
            L2.push_back(twoStruct);
            cout << "2 Items are: (" <<L1[i]<< "," << L1[j] << ") " << endl;

        }
    }
    //******************************************************************************************************************
    //Generate C2. Prune L2 . Compare against min_support and remove less frequent items.
 
	//vector <int> *globalDataset_device; //device storage pointers
    //cudaMalloc ((void **) &globalDataset_device, sizeof (globalDataset));
	//cudaMemcpy (a_d, itemId_TidMapping, sizeof (itemId_TidMapping), cudaMemcpyHostToDevice);

	//prefix_scan_kernel <<< numberOfBlocks,threadsInBlock >>> (a_d,8);

 
    cout << "two_freq_itemset:      " << two_freq_itemset << endl << "\n";

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
__shared__ int smem[128];

__global__ void prefix_scan_kernel (int *b_d, int *a_d, int n, int depth) {
while (tid < n) {
        smem[threadIdx.x] = a_d[tid];       // each thread copy data to shared memory
        __syncthreads();                    // wait for all threads

        //if (tid%16384 == 0 ) {   smem[tid] += res; __syncthreads();  } // result are written at the end*

        offset = 1;                 //1->2->4->8
        for (d =0; d < depth ; d++) {

            if (threadIdx.x >= offset) {
                smem[threadIdx.x] += smem[threadIdx.x-offset] ;           //after writing to smem do synchronize
                __syncthreads();
            } // end if

            offset *=2;
        } // end for loop

        b_d[tid] = smem[threadIdx.x];        // *write the result to array b_d[tid] location
        __syncthreads();                    // wait for all threads to write results

        //if ((tid + 1) % 16384 == 0) { inc++; printf("\n incremented %d times\n", inc);}
        tid += 16384;               //there are no actual grid present, we just increment the tid to fetch next elemennts from input array.

    } // end while (tid < n)
} // end kernel function

*/



