#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define MAXLINE 20 //maximum length of a single line in the input file

//Function for merging two lists
void merging(int **ptolist, int left, int mid, int right)
{
    
    int* list = *ptolist;
    int * temp = malloc((size_t)(right-left+1)*(int)sizeof(int));
    if(temp == NULL){
        fprintf(stderr,"Malloc failed");
        exit(1);
    }
    int l1, l2, i;
    for(l1 = left, l2 = mid +1,i = 0;(l1<=mid) & (l2<=right);i++)
    {
        if(list[l1] <= list[l2]){
            temp[i] = list[l1++];   
        }else{
            temp[i] = list[l2++];
        }
    }
    while(l1 <= mid)    temp[i++] = list[l1++];
    while(l2 <= right)    temp[i++] = list[l2++];
    for(i=left;i<=right;i++)    list[i] = temp[i-left]; 
    free(temp);
	temp = NULL;
}

//Function implements the Merge sort
void myMergeSort(int **ptolist,int left, int right)
{
    int mid;
    if(left < right){
        mid = (left + right)/2;
        myMergeSort(ptolist, left,mid);
        myMergeSort(ptolist,mid+1,right);
        merging(ptolist,left, mid, right);
    }else{
        return;
    }
    
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
}

//starts the sorting, timing and printing
void launchSort(int *intList, int size, int printNum)
{
	int **copy = malloc((size_t)sizeof(int *));
	*copy = malloc((size_t)size*(int)sizeof(int));
	
	memcpy(*copy,intList,(size_t)size*(int)sizeof(int));
	//starting timing the sort
	struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	myMergeSort(copy,0, size-1);
	clock_gettime(CLOCK_REALTIME, &stop);
	//output the results to the problem1.out
	output(*copy, size);
	
	if(printNum)	printf("%d\n", size);
//	printf("%f\n", ((double)(time_b - time_a))*1000.0/CLOCKS_PER_SEC);
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
	} else	{
		fprintf(stderr,"The program needs either zero or one input parameter\n");
		exit(1);
	}

	if (randn) {
		//randomly generate numbers
		if((intList = malloc((size_t)totalIntSize*(int)sizeof(int)))==NULL){
			fprintf(stderr,"Malloc Failed");
			exit(1);
		}
		srand(time(NULL));
		int i;
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
