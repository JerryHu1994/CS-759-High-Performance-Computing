/* CS 759 HW 4-Problem 2
 * Author: Jieru Hu
 */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "omp.h"
#include <sys/utsname.h>

// allocate the memory for the matrix
int* allocateMatrix(int size)
{
	//allocate the pointer memory
	int *matrix;
	
	if((matrix = (int *)malloc((size_t)(size*size)*(int)sizeof(int))) == NULL){
		fprintf(stderr,"Malloc Failed\n");
		exit(1);
	}
	return matrix;
}

// for debug use
void printMatrix(int *matrix, int size, char *type)
{
	int i,j;
	printf("Printing the %s matrix:\n", type);
	for(i=0;i<size;i++){
		for(j=0;j<size;j++){
			int index = i*size + j;
			printf("%d ",matrix[index]);
		}
		printf("\n");
	}
}

int* flipMatrix(int *matrix, int size)
{
	int i;
	int *flipedMatrix;
	if((flipedMatrix = (int *)malloc((size_t)(size*size)*(int)sizeof(int))) == NULL) {
		fprintf(stderr,"Malloc Failed\n");
		exit(1);
	}
	for(i=0;i<size;i++){
		int srcIdx = i*size;
		int dstIdx = (size-1-i)*size;
		memcpy(&flipedMatrix[dstIdx], &matrix[srcIdx], (size_t)size*sizeof(int));
	}
	return flipedMatrix;
}

// function implements the pattern matching for the part c
void patternMatching(int *imageMatrix, int *featureMatrix, int imageSize, int featureSize, int threadSize)
{
	
	double start = omp_get_wtime();
	// flip the matrix
	int *flipedMatrix = flipMatrix(imageMatrix, imageSize);
	
	//printMatrix(flipedMatrix, imageSize, "result");
	//printMatrix(featureMatrix, featureSize, "result");
	int resultSize = imageSize-featureSize+1;
	int *resultMatrix;
	if((resultMatrix= (int *)malloc((size_t)(resultSize*resultSize)*(int)sizeof(int *))) == NULL){
		fprintf(stderr,"Malloc Failed\n");
		exit(1);
	}
	int i;

#pragma omp parallel num_threads(threadSize)
{
#pragma omp for schedule(dynamic, threadSize)
	for(i=0;i<resultSize*resultSize;i++){
		//printf("The thread ID is %d\n", omp_get_thread_num());	
		int y = i%resultSize;
		int x = i/resultSize;
		//compute the cross correlation for entry(x,y)
		int sum = 0;
		int k;
		for(k=0;k<featureSize*featureSize;k++){
			int a = k/featureSize;
			int b = k%featureSize;
			int imageIdx = (x+a)*imageSize + (y+b);
			int featureIdx = featureSize*a + b;
			sum += flipedMatrix[imageIdx]*featureMatrix[featureIdx];
		}
		resultMatrix[i] = sum;
	}
}
	//find the best match
	int bestx = 0, besty = 0, bestValue = -1;
	for(i=0;i<resultSize*resultSize;i++){
		int ycoor = i%resultSize;
		int xcoor = i/resultSize;
		if(resultMatrix[i] > bestValue){
			bestValue = resultMatrix[i];
			besty = ycoor;
			bestx = xcoor;
		} else if (resultMatrix[i] == bestValue) {
			if(xcoor+ycoor < bestx+besty){
				besty = ycoor;
				bestx = xcoor;
			}
		}
	}
	double end = omp_get_wtime();
	printf("%.16g\n", 1000.0*(end-start));
	struct utsname unameData;
	uname(&unameData);
	printf("%s\n", unameData.nodename);
	printf("%d\n",bestx);
	printf("%d\n",besty);
	printf("%d\n", bestValue);
	free(resultMatrix);
	resultMatrix = NULL;
	free(flipedMatrix);
	flipedMatrix=NULL;
}

int main(int argc, char*argv[])
{
	// Read the input matrix from the file
	if(argc !=4){
		fprintf(stderr,"The input format is : ./problem2 N sizeImage sizeFeature\n");
		return 1;
	}
	int threadSize = atoi(argv[1]);
	int imageSize = atoi(argv[2]);
	int featureSize = atoi(argv[3]);

	printf("%d\n", threadSize);
	printf("%d\n", imageSize);
	printf("%d\n", featureSize);
	

	// allocate the matrix
	int *imageMatrix = allocateMatrix(imageSize);
	int *featureMatrix = allocateMatrix(featureSize);
	
	// read the image and feature from problem2.dat
	FILE *fp;
	if((fp=fopen("problem2.dat", "r")) == NULL) {
		fprintf(stderr,"Failed to read from the problem3.dat\n");
		exit(1);
	}
	
	int i;
	int number;
	for(i=0;i<imageSize*imageSize;i++){
		fscanf(fp, "%d", &number);
		imageMatrix[i] = number;
	}
	for(i=0; i<featureSize*featureSize;i++){
		fscanf(fp, "%d", &number);
		featureMatrix[i] = number;
	}
	
	patternMatching(imageMatrix, featureMatrix, imageSize, featureSize, threadSize);
	
	free(imageMatrix);
	imageMatrix = NULL;
	free(featureMatrix);
	imageMatrix = NULL;
	return 0;
	
}
