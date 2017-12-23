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
	const int subsize = 64;
	int numtiles = length/subsize;
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
	for(int a=0;a<numtiles;a++){ //block row
		for(int b=0;b<numtiles;b++){//block col
			double A[subsize*subsize];
			double B[subsize*subsize];
			//loop through all the tiles
			for(int i=0;i<numtiles;i++){ 
				//load a tile into stack
				int idx = 0;	
				for(int j=0;j<subsize;j++){//sub col
					for(int k=0;k<subsize;k++){//sub row
						A[idx] = ptA[(a*subsize+j)*length + i*subsize + k]; 
						B[idx] = ptB[(i*subsize+j)*length + b*subsize + k];
						idx++;
					}
				}
				for(int l=0;l<subsize;l++){
					for(int m=0;m<subsize;m++){
						double sum = 0.0;
						for(int n=0;n<subsize;n++){
							sum += A[l*subsize+n]*B[n*subsize+m];
						}	
						ptC[(a*subsize+l)*length + b*subsize + m] += sum;	
					}
				}
			}
		}
	}
	}
	double end = omp_get_wtime();
	
	// print the results
	printf("%f\n", ptC[totalSize-1]);
	printf("%.16g\n", 1000.0*(end-start));
	printf("%d\n", N);
	return 0;
}
