#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <math.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>


int checkResults(float*res, float* cudaRes,int length)
{
	int nDiffs=0;
	const float smallVal = 0.2f; // Keeping this extra high as we have repetitive addition and sequence matters
	for(int i=0; i<length; i++)
		if(fabs(cudaRes[i]-res[i])>smallVal){
			nDiffs++;
		}
	return nDiffs;
}

void initializeArray(FILE* fp,float* arr, int nElements)
{
	for( int i=0; i<nElements; i++){
		int r=fscanf(fp,"%f",&arr[i]);
		if(r == EOF){
			rewind(fp);
		}
		arr[i]-=5; // This is to make the data zero mean. Otherwise we reach large numbers and lose precision
	}
}

void inclusiveScan_SEQ(float *in, float *out,int length) {
	float sum=0.f;
	for (int i =0; i < length; i++) {
		sum+=in[i];
		out[i]=sum;
	}
}


int main(int argc, char* argv[]) {
	if(argc!=2){
		printf("Usage %s N\n",argv[0]);
		return 1;
	}
	int N=atoi(argv[1]);
	FILE *fp = fopen("problem1.inp","r");
	int size = N * sizeof(float); 
	//allocate resources
	float *h_in      = (float *)malloc(size);
	float *h_out     = (float *)malloc(size); 
	float *cuda_out= (float *)malloc(size);
	float time = 0.f;
	initializeArray(fp,h_in, N);

	//start inclusive timing
	cudaEvent_t startIn,stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

	float *d_in;
	//float *d_out;
	cudaMalloc(&d_in, size);
	//cudaMalloc(&d_out, size);
	
	//copy the memory to device
	assert(cudaSuccess == cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
	
	//set up the pointer
    thrust::device_ptr<float> dev_ptr(d_in);
    
    //perform in-place inclusive scan
	thrust::inclusive_scan(dev_ptr,dev_ptr + N, dev_ptr);
		
	cudaMemcpy(cuda_out, d_in, size, cudaMemcpyDeviceToHost);

	//stop inclusive timing
	cudaEventRecord(stopIn, 0);     
	cudaEventSynchronize(stopIn);
	cudaEventElapsedTime(&time, startIn, stopIn);     
	cudaEventDestroy(startIn); 
	cudaEventDestroy(stopIn);


	inclusiveScan_SEQ(h_in, h_out,N);
	int nDiffs = checkResults(h_out, cuda_out,N);
	if(nDiffs)printf("Test Failed\n"); // This should never print
	printf("%d\n%f\n%f\n",N,cuda_out[N-1],time);
	//printf("%f\n", time);


	//free resources 
	free(h_in); free(h_out); free(cuda_out);
	return 0;
}
