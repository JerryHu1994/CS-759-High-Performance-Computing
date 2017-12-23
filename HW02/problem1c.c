//run a scale analysis on different sizes of input
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main() {
	// generate the large random input files
	int start = 10, end = 20;
	int i;
	srand(time(NULL));
	for(i=start;i<end;i++){
		FILE *outFile;
		char fileName[20];
		sprintf(fileName, "problem1_2^%d.txt", i);
		if ((outFile = fopen(fileName, "w+")) == NULL) {
			fprintf(stderr,"Cannot create the output file\n");
			exit(1);
		}
		int randSize = pow(2.0, i);
		int j;
		fprintf(outFile, "%d\n", randSize);	
		for(j=0;j<randSize;j++){
			fprintf(outFile, "%d\n", rand());	
		}			
		fclose(outFile);
	}

	for(i=10;i<20;i++){
		char cmd[100];
		sprintf(cmd, "./problem1a.exe -f problem1_2^%d.txt", i);
		system(cmd);
	}
	for(i=10;i<20;i++){
		char cmd[100];
		sprintf(cmd, "./problem1b.exe -f problem1_2^%d.txt", i);
		system(cmd);
	}
	return 0;
}
