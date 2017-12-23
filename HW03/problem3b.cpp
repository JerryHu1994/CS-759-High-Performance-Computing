#include <stdio.h>
#include <iostream>
#include <iomanip>
#include "stopwatch.hpp"

int main()
{
	const int cols = 96, rows = 209;
	float x[cols][rows];
	float y[cols][rows];
	float *z;
	const int totalSize = cols*rows;
	if((z=(float *)malloc(totalSize*sizeof(float))) == NULL){
		std::cout <<"Malloc failed" <<std::endl;
		return 1;
	}
	float *xstart = &x[0][0];
	float *ystart = &y[0][0];
	int i;
	stopwatch<std::milli, float> sw;
	sw.start();
	for(i=0;i<totalSize;i++){
		z[i] = xstart[i] + ystart[i];
	}	
	sw.stop();
	//std::cout << std::fixed << std::setprecision(6) << " " << sw.count() << '\n';
	free(z);
	return 0;
}

