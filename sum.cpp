#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;
int main() {

	int num1[] = {1,2,5,6,7};
	int num2[]= {2,3,5,6,7};

	cout << endl;

	cout << "common elements" << endl;
	for (int i = 0; i < 5; i++) 
	{
		{
			for (int j = 0; j < 5; i++)
			{
				if (num1[i] < num2[j])
				{
					i++;
				}

				else if (num2[j] < num1[i])
				{
					j++;
				}

				else if (num1[i] == num2[j])
				{

					cout << " " << num1[i];
					i++;
					j++;

				}
			}
		}

	}



	return 0;
}
