#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>


// fill the matrix with random 1 and -1
void fillMaxtrix(int*matrix, int size)
{
	int i,j;
	for(i=0;i<size;i++){
		for(j=0;j<size;j++){
			int index = i*size+j;
			matrix[index] = 2*(rand()%2) - 1;
		}
	}
}

// allocate the memory for the matrix
int* allocateMatrix(int size)
{
	//allocate the pointer memory
	int *matrix;

	if((matrix = malloc((size_t)(size*size)*(int)sizeof(int))) == NULL){
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

//finds the best match entry in the result matrix
void findBestMatch(int *resultMatrix, int size)
{
    int bestx = 0, besty = 0, bestValue = -1;
    int i,j;
    for(i=0;i<size;i++){
        for(j=0;j<size;j++){
			int idx = i*size + j;
            if(resultMatrix[idx] > bestValue){
                bestValue = resultMatrix[idx];
                besty = j;
                bestx = i;
            } else if (resultMatrix[idx] == bestValue) {
                if(i+j < bestx+besty){
                    besty = j;
                    bestx = i;
                }
            }
        }
    }
    printf("[%d,%d]\n", bestx, besty);
}

// function implements the pattern matching for the part c
void patternMatching(int *imageMatrix, int *featureMatrix, int imageSize, int featureSize)
{
    int resultSize = imageSize-featureSize+1;
    int *resultMatrix;
    if((resultMatrix= (int *)malloc((size_t)(resultSize*resultSize)*(int)sizeof(int *))) == NULL){
		fprintf(stderr,"Malloc Failed\n");
		exit(1);
	}
	int i,j,a,b;
	for(i=0;i<resultSize;i++){
        for(j=0;j<resultSize;j++){
            //compute the cross correlation for entry(i,j)
			int sum = 0;
            for(a=0;a<featureSize;a++){
                for(b=0;b<featureSize;b++){
                    int imageIdx = (i+a)*imageSize + (b+j);
					int featureIdx = featureSize*a + b;
					sum += imageMatrix[imageIdx]*featureMatrix[featureIdx];
                }
            }
            resultMatrix[i*resultSize+j] = sum;
        }
    }
    //printMatrix(resultMatrix, resultSize, "result");   
    findBestMatch(resultMatrix, resultSize);
	free(resultMatrix);
	resultMatrix = NULL;	
}

void printtoOutput(int *matrix, FILE* fp, int size)
{
    int i,j;
    for(i=0; i<size;i++) {
		for(j=0;j<size;j++) {
			int index = i*size + j;
			fprintf(fp, "%d ", matrix[index]);
		}
		fprintf(fp, "\n");
	}
}

int main(int argc, char*argv[])
{
	// Read the input matrix from the file
	if(argc !=3){
		fprintf(stderr,"Please two numbers for the image matrix and feature matrix\n");
		return 1;
	}
	int imageSize = atoi(argv[1]);
	int featureSize = atoi(argv[2]);
	
	int *imageMatrix = allocateMatrix(imageSize);
	int *featureMatrix = allocateMatrix(featureSize);
    
	FILE *fp;
	if((fp=fopen("problem3.dat", "w+")) == NULL) {
		fprintf(stderr,"Failed to print the problem3.dat\n");
		exit(1);
	}
	
	//part a
	
	srand(time(NULL));
	fillMaxtrix(imageMatrix, imageSize);
	fillMaxtrix(featureMatrix, featureSize);
	
	//printMatrix(imageMatrix, imageSize, "image");
    //printMatrix(featureMatrix, featureSize, "feature");
	
	printtoOutput(imageMatrix, fp, imageSize);
	printtoOutput(featureMatrix, fp, featureSize);
	fclose(fp);
	//printMatrix(imageMatrix, imageSize, "image");
    //printMatrix(featureMatrix, featureSize, "feature");
	

    //part b
    //int *flipedMatrix = flipMatrix(imageMatrix, imageSize);
    //printMatrix(flipedMatrix, imageSize, "fliped");

    //part c
    //patternMatching(flipedMatrix, featureMatrix, imageSize, featureSize);
   
	free(imageMatrix);
	imageMatrix = NULL;
	free(featureMatrix);
	featureMatrix = NULL; 
    return 0;

}

