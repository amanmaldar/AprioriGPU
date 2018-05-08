
#include <iostream>
#include <stdlib.h>
#include <time.h>


#define SIZE 2250000
#define DIGITS 9
#define BASE 10

using namespace std;

void count(int data[], int shift)
{
    int hist[BASE] = {0};
    int prefixSum[BASE] = {0};
    int temp[SIZE] = {0};


    for(unsigned int j = 0; j < SIZE; j++)
    {
            hist[(data[j]/shift) % BASE]++;
    }

    for(unsigned int j = 0; j < SIZE; j++)
    {
            prefixSum[j] = hist[j];
    }

                            
    for(unsigned int j = 1; j < BASE; j++)
    {
            prefixSum[j] = prefixSum[j] + prefixSum[j - 1];
    }


    for(int j = SIZE - 1; j >= 0; j--)
    {
        temp[prefixSum[(data[j]/shift) % BASE] - 1] = data[j];
        prefixSum[(data[j]/shift) % BASE]--;
    }


    for(unsigned int j = 0; j < SIZE; j++)
    {
        data[j] = temp[j];
    }

}

int main(void)
{
    srand(time(NULL));

    int *data;
    int shift = 1;
                    
    data = new(nothrow) int[SIZE];

    cout << "Initializing Array..." << endl << endl;

    for(int i = 0; i < SIZE; i++)
    {
            data[i] = rand() % 1000000000;//(SIZE - i) % 100;
    }
    /*
    *  cout << "Unsorted Array: " << endl << endl;
    *
    *      for(unsigned int i = 0; i < SIZE - 1; i++)
    *          {
    *                  cout << data[i] << ", ";
    *          }
    *          cout << data[SIZE - 1] << endl;
    *   */
    cout << "Radix sort started..." << endl << endl;
    clock_t t1, t2;
    t1 = clock();

    for(unsigned int i = 0; i < DIGITS; i++)
    {
                                                            
        count(data, shift);
        shift *= BASE;
    }

    t2 = clock();
    double difference = ((double)t2-(double)t1);
    double seconds = difference / CLOCKS_PER_SEC;

    cout << endl << "Run time: " << seconds << " seconds" << endl;


    return 0;
}



