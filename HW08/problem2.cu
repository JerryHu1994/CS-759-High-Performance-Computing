#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// includes, project

// includes, kernels
#define WIDTH 1024
#define MATRIXSIZE 1024*1024
#define SHAREDSIZE 32


// Kernel function for multiplying the tiles
__global__ void MatrixMulKernel(double* A, double* B, double* C, int width)
{
	int BLOCK_SIZE = blockDim.x;
	// Block index
	int bx = blockIdx.x;  
	int by = blockIdx.y;	
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y; 
   
	// Index of the first sub-matrix of A processed by the block
	int aBegin = width * BLOCK_SIZE * by;
	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + width - 1;
	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;
	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;
	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * width;
	// The element of the block sub-matrix that is computed
	// by the thread
	float Csub = 0;

    //create the shared memory for two sub-blocks in A and B respectively
    __shared__ volatile float As[SHAREDSIZE][SHAREDSIZE];
    __shared__ volatile float Bs[SHAREDSIZE][SHAREDSIZE];
    
    for (int a = aBegin, b = bBegin;
    	a <= aEnd;
      	a += aStep, b += bStep) {

    	// Load the tiles from global memory into shared memory;
        // each thread loads one element of the two tiles from A & B
        As[ty][tx] = A[a + width * ty + tx];
        Bs[ty][tx] = B[b + width * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();
       // __threadfence_block();

        // Each thread in this block computes one element 
        // of the block sub-matrix (tile).  Thread with indexes
        // ty and tx computes in this tile the entry [ty][tx] .  
        for (int k = 0; k < BLOCK_SIZE; ++k)
        	Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
   		}
   // Write the block sub-matrix to global memory;
   // each thread writes one element
   int c = width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
   C[c + width * ty + tx] = Csub;


}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward

double* read_array(const char* filename, int len) {
	double *x = (double*) malloc(len * sizeof(double));
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < len; i++) {
		fscanf(fp, "%lf", &x[i]);
	}
	fclose(fp);
	return x;
}

void computeOnDevice(double* hA,double* hB, double* hC, int nRows, int tileSize, float* incTime );

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, char** argv) 
{
	if(argc!=2)
	{
		printf("Usage: ./problem2 N\n");
		return 0;
	}
	int nRows = 1024;
	int num_elements = nRows*nRows;
	int tileSize = atoi(argv[1]);  //change this for scaling analysis
	double* hA = read_array("inputA.inp",num_elements);
	double* hB = read_array("inputB.inp",num_elements);
	double* hC = (double*) malloc(num_elements * sizeof(double));

	float incTime=0; // Time for GPU
	// **===-------- Modify the body of this function -----------===**
	computeOnDevice( hA, hB,hC, nRows, tileSize, &incTime);
	// **===-----------------------------------------------------------===**

	printf("%f\n%f\n%d\n",hC[num_elements-1],incTime,tileSize);
	// cleanup memory
	free(hA);
	free(hB);
	free(hC);

	return 0;
}



void computeOnDevice(double* hA,double* hB, double* hC, int nRows, int TileSize, float* incTime)
{
	//start inclusive timing
	cudaEvent_t startIn,stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);
	
	//allocate the device memory
	double *dA, *dB, *dC;
	cudaMalloc((void **)&dA, sizeof(double)*MATRIXSIZE);
	cudaMalloc((void **)&dB, sizeof(double)*MATRIXSIZE);
	cudaMalloc((void **)&dC, sizeof(double)*MATRIXSIZE);
	
	//copy from host to device
	cudaMemcpy(dA, hA, sizeof(double)*MATRIXSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double)*MATRIXSIZE, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(WIDTH/TileSize,WIDTH/TileSize);
	dim3 dimBlock(TileSize,TileSize);
	
	MatrixMulKernel<<<dimGrid, dimBlock>>>(dA, dB, dC, WIDTH);
	
	cudaMemcpy(hC, dC, sizeof(double)*MATRIXSIZE,cudaMemcpyDeviceToHost);
	//stop inclusive timing
	cudaEventRecord(stopIn, 0);     
	cudaEventSynchronize(stopIn);
	cudaEventElapsedTime(incTime, startIn, stopIn);     
	cudaEventDestroy(startIn); 
	cudaEventDestroy(stopIn); 

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	
	return;//Placeholder
}


