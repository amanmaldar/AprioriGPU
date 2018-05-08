#include <iostream>
#include <stdlib.h>

using namespace std;

void mat_mul(int *a, int *b, int *pdt, int dim1, int dim2)
{
	int sum, tmp;
	cout<<"Dim1\t:"<<dim1<<"\tDim2: "<<dim2<<"\n";
	for(int i=0;i<dim1;i++)
	{
		sum = 0;
		cout<<"\n--------------------------------------------------------------------------------------------------------------------------------------------\n";
		cout<<"i\t: "<<i<<"\n";
		for(int j=0;j<dim1;j++)
		{	
			sum=0;
			cout<<"j\t: "<<j<<"\n";
			cout<<"--------------------------------------------------------------------------------------------------------------------------------------------\n";
			for(int x=0;x<dim2;x++)
			{
				sum+=a[i*dim2+x]*b[x*dim1+j];
				cout<<"x\t: "<<x<<"\t sum += a["<<i<<"]["<<x<<"],("<<i*dim2+x<<")="<<a[i*dim2+x]<<" \t*\t b["<<x<<"]["<<j<<"],("<<x*dim1+j<<") = "<<b[x*dim1+j]<<", \t cum. sum="<<sum<<"\n";
			}
			pdt[i*dim1+j]=sum;
			cout<<"pdt["<<i*dim1+j<<"]\t: "<<pdt[i*dim1+j]<<"\n";
			cout<<"--------------------------------------------------------------------------------------------------------------------------------------------\n";	
		}
	}
}

int main(int argc, char **argv)
{
	srand(100);
	int dim1	= 3;					//matrix dimension 1
	int dim2	= 2;					//matrix dimension 2
	int *a		= (int*)malloc(sizeof(int) * dim1 * dim2);	//allocate mem for matrix a
	int *b		= (int*)malloc(sizeof(int) * dim2 * dim1);	//allocate mem for matrix b
	int *pdt	= (int*)malloc(sizeof(int) * dim1 * dim1);	//mul a[1024][2048] by b[2048][1024] gives product pdt[1024][1024]
	for(int i=0;i<dim1;i++)
	{
		for(int j=0;j<dim2;j++)
		{
			a[i*dim2+j]= rand()%10;			//fill a with rand nums, row-wise
			b[i*dim2+j]= rand()%10;			//fill b with rand nums,column-wise
		}
	}
	mat_mul(a,b,pdt,dim1,dim2);
	
	for(int i=0;i<dim1;i++)
	{
		for(int j=0;j<dim2;j++)
		{
			cout<<"a["<<i<<"]["<<j<<"],("<<i*dim2+j<<"): "<<a[i*dim2+j]<<"\t\t";
		}
		cout<<"\n";
	}
	for(int i=0;i<dim2;i++)
	{
		for(int j=0;j<dim1;j++)
			cout<<"b["<<i<<"]["<<j<<"],("<<i*dim1+j<<"): "<<b[i*dim1+j]<<"\t\t";
		cout<<"\n";
	}
	for(int i=0;i<dim1;i++)
	{
		for(int j=0;j<dim1;j++)
		{
			cout<<"pdt["<<i<<"*"<<dim1<<"+"<<j<<"]="<<pdt[i*dim1+j]<<"\t\t";
		}
		cout<<"\n";
	}
	return 0;
}

	
		
