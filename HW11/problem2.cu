#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>


float* read_array(const char* filename, int len) {
	float *x = (float*) malloc(len * sizeof(float));
	FILE *fp = fopen(filename, "r");
        for( int i=0; i<len; i++){
		int r=fscanf(fp,"%f",&x[i]);		
		if(r == EOF){
			rewind(fp);
		}
		x[i]-=5;
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
	if(argc != 2) {
		fprintf(stderr, "usage: ./problem2 N\n");
		exit(1);
	}
	int num_elements = atoi(argv[1]);

	float* h_data=read_array("problem1.inp",num_elements);
	
	float reference = 1.0f;  
	computeSum(&reference , h_data, num_elements);

	//start inclusive timing
	float time;
	cudaEvent_t startIn,stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);	

	int size = num_elements*sizeof(float);
	float *d_in;
	assert(cudaSuccess == cudaMalloc((void**)&d_in, size));

	//copy the memory to device
	assert(cudaSuccess == cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice));
	
	//set up the pointer
    thrust::device_ptr<float> dev_ptr(d_in);
	
	//perform the thrust reduction
	double result = thrust::reduce(dev_ptr,dev_ptr + num_elements, (double) 0.0,thrust::plus<float>());
	
	//stop inclusive timing
	cudaEventRecord(stopIn, 0);     
	cudaEventSynchronize(stopIn);
	cudaEventElapsedTime(&time, startIn, stopIn);     
	cudaEventDestroy(startIn); 
	cudaEventDestroy(stopIn);

	// Run accuracy test
	float epsilon = 0.3f;
	unsigned int result_regtest = (abs(result - reference) <= epsilon);

	if(!result_regtest)	printf("Test failed device: %f  host: %f\n",result,reference);
	//print the outputs
	printf("%d\n%f\n%f\n",num_elements, result, time);
	//printf("%f\n", time);
	// cleanup memory
	cudaFree(d_in);  
	//cudaFree(d_out);
	free( h_data);
	return 0;
}
