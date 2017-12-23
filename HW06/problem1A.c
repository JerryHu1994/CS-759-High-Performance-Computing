#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

int main( int argc, char *argv[])
{

	if(argc!=2)
	{
		printf("Invalid argument Usage: ./problem1A  N");
		return 0;
	}
	int N = atoi(argv[1]);
	const int length = 1024;
	const int totalSize = length*length;
	FILE *fpA,*fpB;
	double *ptA, *ptB, *ptC;
	ptA = (double *)malloc(sizeof(double)*totalSize);
	ptB = (double *)malloc(sizeof(double)*totalSize);
	ptC = (double *)malloc(sizeof(double)*totalSize);
	//reading files
	fpA = fopen("inputA.inp", "r");
	fpB= fopen("inputB.inp", "r");
	for (int i=0;i<totalSize;i++){
		fscanf(fpA, "%lf",&ptA[i]);
	}
	for (int i=0;i<totalSize;i++){
		fscanf(fpB, "%lf",&ptB[i]);
	}
	double start = omp_get_wtime();

	#pragma omp parallel num_threads(N) 
	{
	#pragma omp for collapse(2) 
	for(int i=0;i<length;i++){
		for(int j=0;j<length;j++){
			double sum = 0.0;
			for(int k=0;k<length;k++){
				sum += ptA[i*length+k]*ptB[k*length+j];
			}
			ptC[i*length + j] = sum;
		}
	}
	}
	double end = omp_get_wtime();
	
	// print the results
	printf("%f\n", ptC[totalSize-1]);
	printf("%.16g\n", 1000.0*(end-start));
	printf("%d\n", N);
	fclose(fpA);
	fclose(fpB);
	return 0;
}
