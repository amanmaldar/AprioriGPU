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
			for (int j = 0; j < 5; j++)
			{
/*				if (num1[i] < num2[j])
				{
					i++;
                    continue;
				}

				else if (num2[j] < num1[i])
				{
					j++;
                    continue;
				}
*/
				if (num1[i] == num2[j])
				{

					cout << " " << num1[i];
					//i++;
					//j++;
                    //continue;

				}
			}
		}

	}



	return 0;
}
