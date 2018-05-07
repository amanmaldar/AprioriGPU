//new
#include <iostream>
#include <cstdlib>
#include <set>
#include <ctime>
using namespace std;

int main()
{
	int i = 0;
	int lines = 100000;
	int count;
	set <int> numbers;
	set <int> :: iterator it, it1;
	for (int j=0 ; j< lines; j++){
	    //cout << "looping in" << endl;
	    int a=0,b=0,c=0;
	    int iteration;
    	int m = (rand() % 3) + 2;
    	while(i++ < m) {
    		int r = (rand() % 10);

    		count = numbers.count(r);
    		if (count == 0){
    		    cout << r << " ";
    		    numbers.insert(r);
    		}
    		else{
    		    i--;
    		}

    	}
    	numbers.clear();
    	i= 0;
    	cout << endl;
	}
	return 0;
}
