#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <math.h>


#define BLOCK_SIZE 1024


// Kernel for the first iteration of parallel scan
__global__ void parallelScan(float *d_out, float *d_in, int length) {
  volatile extern __shared__ double sharedData[];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int threadx = threadIdx.x;
  // load the data into the shared memory
  sharedData[threadx] = d_in[tid];
  __syncthreads();
  
  int pout = 0; int pin = 1;
  
  if (tid < length) {
	for (int offset = 1; offset < blockDim.x; offset <<= 1) {
		pout = 1 - pout;
		pin = 1 - pin;

		if (threadx >= offset) {
			sharedData[pout * blockDim.x + threadx] = sharedData[pin * blockDim.x + threadx] + sharedData[pin * blockDim.x + threadx - offset];
		} else {
			sharedData[pout * blockDim.x + threadx] = sharedData[pin * blockDim.x + threadx];
		}
	__syncthreads();
  }

  d_out[tid] = sharedData[pout * blockDim.x + threadx];
  //save the sum of the block back to d_in
  if (threadx == blockDim.x - 1) d_in[tid] = sharedData[pout * blockDim.x + threadx];
  }
}


__global__ void addPreSums(float *d_out, float *d_in)
{
	volatile extern __shared__ double sharedData[];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int bid = blockIdx.x + 1;
	int threadx = threadIdx.x;
	int bDim = blockDim.x;
	
	//load seperatedly scanned blocks (with one block offset)
	sharedData[threadx] = d_out[tid + bDim];
	
	//each thread adds the previous sums stored in the d_in
	for (int i=0; i<bid; i++) {
		sharedData[threadx] += d_in[i*bDim + bDim - 1];
	}
	__syncthreads();
	//store the sums back to the correct position
	d_out[tid + bDim] = sharedData[threadx];
}

// The function starts the prefix scan on values from d_in
void prefixScan(float * d_in, float *d_out, int num)
{	
	int blockx = (num + BLOCK_SIZE - 1)/BLOCK_SIZE; //blockx < 2^14, not exceeding the maximum
	dim3 dimGrid(blockx, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	
	parallelScan<<<dimGrid, dimBlock, sizeof(double)*BLOCK_SIZE*2>>>(d_out, d_in, num);
	
	//when there is one more block
	if (blockx > 1) {
		//launch with one less block since we do not need to add previous sum for the very first block
		addPreSums<<<blockx - 1, BLOCK_SIZE, sizeof(double)*(BLOCK_SIZE)>>>(d_out,d_in);
	}
	
}


int checkResults(float*res, float* cudaRes,int length)
{
	int nDiffs=0;
	const float smallVal = 0.3f; // Keeping this extra high as we have repetitive addition and sequence matters
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
	float *in      = (float *)malloc(size);
	float *out     = (float *)malloc(size); 
	float *cuda_out= (float *)malloc(size);
	float time = 0.f;
	initializeArray(fp,in, N);

	//start inclusive timing
	cudaEvent_t startIn,stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

	float *d_in;
	float *d_out;
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, size);
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	
	prefixScan(d_in, d_out, N);

	cudaMemcpy(cuda_out, d_out, size, cudaMemcpyDeviceToHost);

	//stop inclusive timing
	cudaEventRecord(stopIn, 0);     
	cudaEventSynchronize(stopIn);
	cudaEventElapsedTime(&time, startIn, stopIn);     
	cudaEventDestroy(startIn); 
	cudaEventDestroy(stopIn);


	inclusiveScan_SEQ(in, out,N);
	int nDiffs = checkResults(out, cuda_out,N);
	if(nDiffs)printf("Test Failed\n"); // This should never print
	printf("%d\n%f\n%f\n",N,cuda_out[N-1],time);
	//printf("%f\n", time);


	//free resources 
	cudaFree(d_in);cudaFree(d_out);free(in); free(out); free(cuda_out);
	return 0;
}
