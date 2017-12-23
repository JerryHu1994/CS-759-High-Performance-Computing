//run a scale analysis on different sizes of input
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main() {
	// generate the large random input files
	int start = 5, end = 13;
	int i;
	srand(time(NULL));
	for(i=start;i<end;i++){
		FILE *outFile;
		char fileName[20];
		sprintf(fileName, "problem2_2^%d.txt", i);
		if ((outFile = fopen(fileName, "w+")) == NULL) {
			fprintf(stderr,"Cannot create the output file\n");
			exit(1);
		}
		int randSize = pow(2.0, i);
		int j;
		fprintf(outFile, "%d\n", randSize);	
		for(j=0;j<randSize;j++){
			fprintf(outFile, "%d\n", rand()%20-10);	
		}			
		fclose(outFile);
	}
	for(i=5;i<13;i++){
		char cmd[100];
		sprintf(cmd, "./problem2a.exe -f problem2_2^%d.txt", i);
		system(cmd);
	}
	return 0;
}
