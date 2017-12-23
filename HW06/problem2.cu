#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

//A kernel calculates the sum of the threadidx and blockidx
__global__ void sumKernel(int* data, int size)
{
    int threadi = threadIdx.x;
    int blocki = blockIdx.x;
    int value = blocki*blockDim.x + threadi;
    if(value < size){
        data[value] = threadi+blocki;
    }

}


int main(int argc, char* argv[])
{
    const int totalSize = 16;
    const int blockSize = 2;
    const int threadSize  = 8;

    int hostArr[totalSize];

    //allocate the memory
    int *dArray;
    cudaMalloc((void **)&dArray,sizeof(int)*totalSize);

    sumKernel<<<blockSize, threadSize>>>(dArray, totalSize);
    
    cudaMemcpy(&hostArr, dArray, sizeof(int)*totalSize, cudaMemcpyDeviceToHost); 

    //print the output
    int i;
    for(i=0; i<totalSize;i++){
        printf("%d\n", hostArr[i]);
    }

    //clean up the memory
    cudaFree(dArray);
    return 0;
}
