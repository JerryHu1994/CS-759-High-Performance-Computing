#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

int* read_array(const char* filename, int len) {
	int *x = (int*) malloc(len * sizeof(int));
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < len; i++) {
		fscanf(fp, "%d", &x[i]);
	}
	fclose(fp);
	return x;
}

// The cuda kernel for multiply the matrix
__global__ void multiplyKernel(int* dA, int* dB, int* dC, int row, int col)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int loc = index/row;
	int local = dA[index]*dB[index%row];
	atomicAdd(&dC[loc],local);
}

int main(int argc, char *argv[]) {
	if (argc != 1) {
		printf("Invalid argument Usage: ./problem1");
		return -1;
	}

	const int rowWidth=32;
	const int colWidth=16;	
	int *hA = read_array("inputA.inp",rowWidth*colWidth );
	int *hB = read_array("inputB.inp", rowWidth);
	int *hC = (int*) malloc(colWidth * sizeof(int));
	int *refC;
	// TODO - allocate host memory for refC (you have to figure out how much)
	// The skeleton currently segfaults because refC is accessed without allocation
	refC = (int*)malloc(colWidth * sizeof(int));
	assert(hC != NULL);
	assert(refC != NULL);
	// TODO do a reference host implementation (Ch) here. ie populate answer in refC
	int i,j;
	for(i=0;i<colWidth;i++){
		int sum=0;
		for(j=0;j<rowWidth;j++){
			sum += hA[i*rowWidth + j]*hB[j];
		}
		refC[i] = sum;
	}


	int *dA, *dB, *dC;
	// TODO allocate device memory for dA,dB and dC
	cudaMalloc((void **)&dA, sizeof(int)*rowWidth*colWidth);
	cudaMalloc((void **)&dB, sizeof(int)*rowWidth);
	cudaMalloc((void **)&dC, sizeof(int)*colWidth);
  	cudaMemset(dC, 0, sizeof(int)*colWidth);

	// TODO copy data from host to GPU 
	cudaMemcpy(dA, hA, sizeof(int)*rowWidth*colWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(int)*rowWidth, cudaMemcpyHostToDevice); 

	// TODO call your kernel
	multiplyKernel<<<colWidth, rowWidth>>>(dA, dB, dC, rowWidth, colWidth);

	// TODO copyback results
	cudaMemcpy(hC, dC, sizeof(int)*colWidth, cudaMemcpyDeviceToHost);	

	float Error=0;

	for(int i=0;i<colWidth;i++)
		Error+=(hC[i]-refC[i])*(hC[i]-refC[i]);
	printf("%f\n%d",sqrt(Error),hC[colWidth-1]);

	free(refC);
	free(hB);
	free(hA);
	free(hC);

	return 0;
}
