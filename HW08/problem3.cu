#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define SHAREDSIZE 32

__global__ void MatrixMulKernel(double* A, double* B, double* C, int wA, int wB, int tileSize)
{
	int BLOCK_SIZE = tileSize;
	// Block index
	int bx = blockIdx.x;  
	int by = blockIdx.y;	
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y; 
   
	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;
	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;
	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;
	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;
	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;
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
         As[ty][tx] = A[a + wA * ty + tx];
         Bs[ty][tx] = B[b + wB * ty + tx];

         // Synchronize to make sure the matrices are loaded
         __syncthreads();
         __threadfence_block();

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
   int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
   C[c + wB * ty + tx] = Csub;
}

double* read_array(const char* filename, int len) {
	double *x = (double*) malloc(len * sizeof(double));
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < len; i++) {
		fscanf(fp, "%lf", &x[i]);
	}
	fclose(fp);
	return x;
}

void computeOnDevice(double* hA,double* hB, double* hC, int nRows,
	int nInnerDimension,int nCols, int tileSize, float* incTime );

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, char** argv) 
{
	if(argc!=5)
        {
                printf("Usage: ./problem3 i j k N\n");
                return 0;
        }


	int nRows = atoi(argv[1]);
	int nInnerDimension = atoi(argv[2]);
	int nCols = atoi(argv[3]);
	int num_elementsA= nRows*nInnerDimension;
	int num_elementsB=nInnerDimension*nCols;
	int num_elementsC= nRows*nCols;
	int tileSize = atoi(argv[4]);  //change this for scaling analysis
	float incTime=0; // Time for GPU
	double* hA = read_array("problem3.inp",num_elementsA);
	double* hB = read_array("problem3.inp",num_elementsB);
	double* hC = (double*) malloc(num_elementsC * sizeof(double));

	// **===-------- Modify the body of this function -----------===**
	computeOnDevice( hA, hB,hC, nRows, nInnerDimension, nCols, tileSize, &incTime);
	// **===-----------------------------------------------------------===**

	//cpu calculation check
	/*double check = 0.0;
	for(int i=0;i<nInnerDimension;i++){
		check += hA[(nRows-1)*nInnerDimension+i]*hB[i*nCols+nCols-1];
	}
	printf("%f\n", check);
	*/
	printf("%f\n%f\n%d\n%d\n%d\n",hC[num_elementsC-1],incTime,tileSize,nRows,nCols);
	// cleanup memory
	free(hA);
	free(hB);
	free(hC);

	return 0;
}

//ZeroPadthe matrix so that it could be exactly devided by Tile Size in both row and col
double * zeroPadMatrix(double *unpadded, int row, int col, int paddedRow, int paddedCol, int TileSize, int copy)
{
	double *paddedMatrix = (double *)calloc(paddedRow*paddedCol, sizeof(double));
	
	//Copy the values from unpadded matrix to padded matrix
	if(copy){
		for (int i=0;i<row;i++) {
			memcpy(&paddedMatrix[i*paddedCol], &unpadded[i*col], col*sizeof(double));
		}
	}
	return paddedMatrix;
}

void extractPaddedMaxtrix(double *unpadded, double *padded, int row, int col, int paddedRow, int PaddedCol, int TileSize)
{
	for(int i=0;i<row; i++){
		memcpy(&unpadded[i*col], &padded[i*PaddedCol], col*sizeof(double));
	}
}

//for debug use
void printMatrix(double *matrix, int row, int col)
{
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			printf("%f ", matrix[i*col + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void computeOnDevice(double* hA,double* hB, double* hC, int nRows, int nInnerDimension, int nCols, int TileSize, float* incTime)
{
	//calculate the size needed for padding
	int tempRow = (nRows-1)/TileSize + 1;
	int paddednRows = tempRow*TileSize;
	int tempnInnerDimension = (nInnerDimension-1)/TileSize + 1;
	int paddedtempnInnerDimension = tempnInnerDimension*TileSize;
	int tempCol = (nCols-1)/TileSize + 1;
	int paddednCols = tempCol*TileSize;
	//zero paddding
	double *paddedA = zeroPadMatrix(hA, nRows, nInnerDimension, paddednRows, paddedtempnInnerDimension, TileSize, 1);
	double *paddedB = zeroPadMatrix(hB, nInnerDimension, nCols, paddedtempnInnerDimension, paddednCols, TileSize, 1);
	double *paddedC = zeroPadMatrix(hB, nRows, nCols, paddednRows, paddednCols, TileSize, 0);
	//printMatrix(paddedA, paddednRows, paddedtempnInnerDimension);
	//printMatrix(paddedB, paddedtempnInnerDimension, paddednCols);
	
	//start inclusive timing
	cudaEvent_t startIn,stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);
	
	//allocate the device memory
	double *dA, *dB, *dC;
	cudaMalloc((void **)&dA, sizeof(double)*paddednRows*paddedtempnInnerDimension);
	cudaMalloc((void **)&dB, sizeof(double)*paddedtempnInnerDimension*paddednCols);
	cudaMalloc((void **)&dC, sizeof(double)*paddednRows*paddednCols);
	
	//copy from host to device
	cudaMemcpy(dA, paddedA, sizeof(double)*paddednRows*paddedtempnInnerDimension, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, paddedB, sizeof(double)*paddedtempnInnerDimension*paddednCols, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(paddednCols/TileSize, paddednRows/TileSize);
	dim3 dimBlock(TileSize,TileSize);
	
	MatrixMulKernel<<<dimGrid, dimBlock>>>(dA, dB, dC, paddedtempnInnerDimension, paddednCols, TileSize);
	
	cudaMemcpy(paddedC, dC, sizeof(double)*paddednRows*paddednCols,cudaMemcpyDeviceToHost);
	extractPaddedMaxtrix(hC, paddedC, nRows, nCols, paddednRows, paddednCols, TileSize);
	//stop inclusive timing
	cudaEventRecord(stopIn, 0);     
	cudaEventSynchronize(stopIn);
	cudaEventElapsedTime(incTime, startIn, stopIn);     
	cudaEventDestroy(startIn); 
	cudaEventDestroy(stopIn); 
	
	return;//Placeholder
}


