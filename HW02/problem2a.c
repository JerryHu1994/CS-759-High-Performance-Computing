#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#define MAXLINE 8

void exclusiveScan(int *arr, int size, int printNum)
{
    int i;
    int currSum = 0;
	
    int *tempArr;
	if ((tempArr = malloc((size_t)size*(int)sizeof(int)))==NULL) {
		fprintf(stderr,"Malloc Failed\n");
		exit(1);
	}
    for(i=1;i<size;i++){
        currSum += arr[i-1];
        tempArr[i] = currSum;
	//	printf("%d\n",currSum);
    }
    tempArr[0] = 0;
	if (printNum){
    	printf("%d\n", size);
		printf("%d\n", tempArr[size-1]);
	}
	free(tempArr);
	tempArr = NULL;
}

int main(int argc, char *argv[])
{
	
    FILE *fp;
	int printNum = 1;
	if((argc == 3) && !strcmp(argv[1], "-f")){
		printNum = 0;
		if((fp =fopen(argv[2], "r")) == NULL){
			fprintf(stderr,"File not found\n");
			exit(1);
		}	
	} else if (argc == 2) {
		if((fp = fopen(argv[1],"r")) == NULL){
        	fprintf(stderr, "Cannot open the file\n");
        	return 1;
    	}
	} else {
		fprintf(stderr,"Please enter a file name\n");
        return 1;
	}

	int *intList = NULL;
	int totalIntSize;
	char currChar[MAXLINE];
	int firstNumRead = 1; //indicator on whether it is reading the first number
	int ind = 0;
	while (fgets(currChar, MAXLINE, fp) != NULL) {
		if(firstNumRead){
			//read the integer size and malloc the integer list
			firstNumRead = 0;
			totalIntSize = atoi(currChar);
			if((intList = malloc((size_t)totalIntSize*(int)sizeof(int)))==NULL){
				fprintf(stderr,"Malloc Failed\n");
				exit(1);
			}
			continue;
		}
		intList[ind] = atoi(currChar);
		ind++;
	}

    totalIntSize = ind;
	
	//start timing
	struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);

	if(intList)	exclusiveScan(intList, totalIntSize, printNum);
        
	clock_gettime(CLOCK_REALTIME, &stop);
 	double calTime = (double)(stop.tv_sec - start.tv_sec)*1000.0+(stop.tv_nsec - start.tv_nsec)/1000000.0; 
	printf("%f\n",calTime);
	
	if(intList){
		free(intList);
		intList = NULL;
    }
	fclose(fp);
	fp = NULL;
    return 0;
}
