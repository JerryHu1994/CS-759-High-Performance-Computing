#include <stdlib.h>
#include <stdio.h>
#include <float.h>

float* read_array(const char* filename, int len) {
	float *x = (float*) malloc(len * sizeof(float));
	FILE *fp = fopen(filename, "r");
        for( int i=0; i<len; i++){
		int r=fscanf(fp,"%f",&x[i]);		
		if(r == EOF){
			rewind(fp);
		}
		//x[i]-=5;
         }
	fclose(fp);
	return x;
}

void computeSum( float* reference, float* idata, const unsigned int len) 
{
  reference[0] = 0;
  double total_sum = 0;
  unsigned int i;
  for( i = 0; i < len; ++i) 
  {
      total_sum += idata[i];
  }
  *reference = total_sum;
}

int main( int argc, char** argv) 
{
	int num_elements = atoi(argv[1]);

	float* h_data=read_array("problem1.inp",num_elements);
	
	float reference = 1.0f;  
	computeSum(&reference , h_data, num_elements);

	printf("The result is %f\n", reference);
	return 1;
}
