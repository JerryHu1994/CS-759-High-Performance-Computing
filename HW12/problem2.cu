#include <stdio.h>
#include <cuda.h>
#define BLOCK_SIZE 1024
#define CUDA_CHECK(value, label) {              \
   cudaError_t c = (value);                     \
   if (c != cudaSuccess) {                      \
   fprintf(stderr,                              \
     "Error: '%s' at line %d in %s\n",          \
     cudaGetErrorString(c),__LINE__,__FILE__);  \
   goto label;                                  \
   } }

// Kernel for the first iteration of parallel scan
static __global__ void parallelScan(float *d_out, float *d_in, int length) {
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


static __global__ void addPreSums(float *d_out, float *d_in)
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

static __global__ void addPartialSum(float *d_in, float preSum, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){
		d_in[tid] = d_in[tid] + preSum;
	}
}

void addFirstHalf(float *in, float preSum, int size){
	float *d_in=0;
	cudaMalloc(&d_in, size * sizeof(float));
	
	cudaMemcpy(d_in, in, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int blockx = (size + BLOCK_SIZE - 1)/BLOCK_SIZE;
	dim3 dimGrid(blockx, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	addPartialSum<<<dimGrid, dimBlock>>>(d_in, preSum, size);
	
	cudaMemcpy(in, d_in, size * sizeof(float), cudaMemcpyDeviceToHost);
	if(d_in)	cudaFree(d_in);
}

void prefix_scan(float *in, float *out, int size) {
	float *d_in=0, *d_out=0;
	cudaMalloc(&d_in, size * sizeof(float));
	cudaMalloc(&d_out, size * sizeof(float));
	
	cudaMemcpy(d_in, in, size * sizeof(float), cudaMemcpyHostToDevice);
	//prefix_scan_device<<<128, 1>>>(d_in, d_out, size);
	int blockx = (size + BLOCK_SIZE - 1)/BLOCK_SIZE; //blockx < 2^14, not exceeding the maximum
	dim3 dimGrid(blockx, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	
	parallelScan<<<dimGrid, dimBlock, sizeof(double)*BLOCK_SIZE*2>>>(d_out, d_in, size);
	
	//when there is one more block
	if (blockx > 1) {
		//launch with one less block since we do not need to add previous sum for the very first block
		addPreSums<<<blockx - 1, BLOCK_SIZE, sizeof(double)*(BLOCK_SIZE)>>>(d_out,d_in);
	}
	cudaMemcpy(out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

	if(d_in) cudaFree(d_in);
	if(d_out) cudaFree(d_out);
}
