/* CS 759 HW 4-Problem 1
 * Author: Jieru Hu
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "omp.h"
#include <sys/utsname.h>

// Define the f(x)
double f(double x)
{
	double base = 2.71828182845904523536;
	double result = pow(base, sin(x))*cos(x/(double)40.0);
	return result;
}

int main(int argc, char *argv[])
{ 
	if(argc != 2){
		fprintf(stderr, "The input format is : ./problem1 N\n");
		exit(1);
	}
	int numOfThreads = atoi(argv[1]);
	const int n = 1000000;
	const double h = 0.0001;

	double *xlist;
	double *parameterList;

	if((xlist=(double *)malloc((n+1)*sizeof(double))) == NULL){
		fprintf(stderr,"Malloc Error !\n");
		exit(1);
	}
	if((parameterList= (double *)malloc((n+1)*sizeof(double))) == NULL){
		fprintf(stderr, "Malloc Error !\n");
		exit(1);
	}


	parameterList[0] = 17.0;
	parameterList[1] = 59.0;
	parameterList[2] = 43.0;
	parameterList[3] = 49.0;
	parameterList[n] = 17.0;
	parameterList[n-1] = 59.0;
	parameterList[n-2] = 43.0;
	parameterList[n-3] = 49.0;
	for(int k=0; k<=n; k++){
		xlist[k] = k*h;
	}
	for(int i=4; i<n-3; i++){
		parameterList[i] = 48.0;
	}

	double sum = 0.0;

	double start = omp_get_wtime();
#pragma omp parallel num_threads(numOfThreads)
{
#pragma omp for reduction(+:sum) 
	for(int j=0; j<=n;j++){
		sum = sum + parameterList[j]*f(xlist[j]);
	}
}	
	double end = omp_get_wtime();
	sum = h*sum/(double)48.0;	

	//printing the results
	printf("%d\n",numOfThreads);
	printf("%.16g\n", 1000.0*(end-start));
	struct utsname unameData;
	uname(&unameData);
	printf("%s\n", unameData.nodename);	
	printf("%f\n", sum);
	return 0;
}
