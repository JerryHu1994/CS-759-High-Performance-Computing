#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define MAXLINE 20 //maximum length of a single line in the input file

//Compare function for the qsort
int intCompare(const void *a, const void *b){
    const int *inta = (const int *)a;
    const int *intb = (const int *)b;
    return *inta - *intb;
}

//Function outputs the sorting results to the problem1.out file
void output(int *results, int size)
{
	FILE *outFile;
	if ((outFile = fopen("problem1.out", "w+")) == NULL) {
		fprintf(stderr,"Cannot create the output file\n");
		exit(1);
	}
	int i;
	for(i=0;i<size;i++){
		fprintf(outFile,"%d\n", results[i]);
	}
	fclose(outFile);
	outFile = NULL;
}

//starts the sorting, timing and printing
void launchSort(int *intList, int size, int printNum)
{
	int **copy = malloc((size_t)sizeof(int *));
	*copy = malloc((size_t)size*(int)sizeof(int));
	
	memcpy(*copy,intList,(size_t)size*(int)sizeof(int));
	struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	
	//starting timing the sort
	
	//start the qsort
	qsort(intList,(size_t)size,(size_t)sizeof(int),intCompare);
	
	clock_gettime(CLOCK_REALTIME, &stop);
	//output the results to the problem1.out
	output(intList, size);
	
	if(printNum)	printf("%d\n", size);
	double calTime = (double)(stop.tv_sec - start.tv_sec)*1000.0+(stop.tv_nsec - start.tv_nsec)/1000000.0; 
	printf("%f\n",calTime);
	
	free(*copy);
	*copy = NULL;
	free(copy);
	copy = NULL;
}


int main(int argc, char* argv[])
{

  	FILE *fp;
	int totalIntSize;
	int randn = 0; //indicator for random mode
	int *intList = NULL;
  	int printNum = 1;
	if (argc == 1) {
		//open the input file
		if((fp =fopen("problem1.in", "r")) == NULL ){
			fprintf(stderr,"File not found\n");
			exit(1);
		}
	} else if (argc == 2) {
		randn = 1;
		totalIntSize = atoi(argv[1]);
	} else if((argc == 3) && !strcmp(argv[1], "-f")){
		printNum = 0;
		if((fp =fopen(argv[2], "r")) == NULL){
			fprintf(stderr,"File not found\n");
			exit(1);
		}	
	} else {
		fprintf(stderr,"The program needs either zero or one input parameter\n");
		exit(1);
	}

	if (randn) {
		//randomly generate numbers
		if((intList = malloc((size_t)totalIntSize*(int)sizeof(int)))==NULL){
			fprintf(stderr,"Malloc Failed");
			exit(1);
		}
		int i;
		srand(time(NULL));
		for(i=0;i<totalIntSize;i++)	intList[i] = rand();
		
		//sort the integers using myMergeSort
		launchSort(intList, totalIntSize, printNum);
	
	} else {
		//read from the file and sort
		char currChar[MAXLINE];
		int firstNumRead = 1; //indicator on whether it is reading the first number
		int ind = 0;
		while (fgets(currChar, MAXLINE, fp) != NULL) {
			if(firstNumRead){
				//read the integer size and malloc the integer list
				firstNumRead = 0;
				totalIntSize = atoi(currChar);
				if((intList = malloc((size_t)totalIntSize*(int)sizeof(int)))==NULL){
					fprintf(stderr,"Malloc Failed");
					exit(1);
				}
				continue;
			}
			intList[ind] = atoi(currChar);
			ind++;
		}
		totalIntSize = ind;
		launchSort(intList, totalIntSize, printNum);
	}
	
	//free the memory	
	if(intList)	{
		free(intList);
		intList = NULL;
	}
  	if(!randn)	{
		fclose(fp);
		fp = NULL;
	}
  	return 0;    
}
