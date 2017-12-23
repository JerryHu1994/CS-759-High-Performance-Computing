#include <stdio.h>
#include <iostream>

using namespace std;

int main()
{
	int x[2][3];
	int i,j;
	for(i=0;i<2;i++){
		for(j=0;j<3;j++){
			cout<<(void*)&x[i][j]<<" ";
		}
		cout<<endl;
	}
	
	return 0;
}
