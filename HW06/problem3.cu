#include<iostream>
#include<stdio.h>
#include<cuda.h>

__global__ void sumKernel(double* dA, double* dB, double* dC, int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < size){
        dC[index] = dA[index] + dB[index];
    }
}


int main( int argc, char *argv[])
{

	if(argc!=3)
	{
		printf("Invalid argument Usage: ./problem3 N M");
		return 0;
	}

	FILE *fpA,*fpB;
	int N = atoi(argv[1]);
	int M = atoi(argv[2]);
	int threadSize = M;
	int blockSize = N/threadSize + 1;
	double *hA= new double[N];
	double *hB= new double[N];
	double *hC=  new double[N];
	double *refC=  new double[N]; // Used to verify functional correctness
	double *dA,*dB,*dC;  // You may use these to allocate memory on gpu
	//defining variables for timing
	cudaEvent_t startEvent_inc, stopEvent_inc, startEvent_exc, stopEvent_exc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventCreate(&startEvent_exc);
	cudaEventCreate(&stopEvent_exc);
	float elapsedTime_inc, elapsedTime_exc;

	//reading files
	fpA = fopen("inputA.inp", "r");
	fpB= fopen("inputB.inp", "r");


	for (int i=0;i<N;i++){    
		fscanf(fpA, "%lf",&hA[i]);
	}
	for (int i=0;i<N;i++){
		fscanf(fpB, "%lf",&hB[i]);
	}

      for(int i=0;i<N;i++)
        refC[i]=hA[i]+hB[i];

    cudaMalloc((void**) &dA,sizeof(double)*N);
    cudaMalloc((void**) &dB,sizeof(double)*N);
    cudaMalloc((void**) &dC,sizeof(double)*N);
	cudaEventRecord(startEvent_inc,0); // starting timing for inclusive
	// TODO allocate memory for arrays and copay array A and B
	//copy the hA into dA, hB into dB
    cudaMemcpy(dA, hA, sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(double)*N, cudaMemcpyHostToDevice);

	cudaEventRecord(startEvent_exc,0); // staring timing for exclusive
	//launch the kernel to compute the sum
    sumKernel<<<blockSize, threadSize>>>(dA,dB,dC,N);

	cudaEventRecord(stopEvent_exc,0);  // ending timing for exclusive
	cudaEventSynchronize(stopEvent_exc);   
	cudaEventElapsedTime(&elapsedTime_exc, startEvent_exc, stopEvent_exc);
	
	// TODO copy data back
	cudaMemcpy(hC, dC, sizeof(double)*N, cudaMemcpyDeviceToHost);

	cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
	cudaEventSynchronize(stopEvent_inc);   
	cudaEventElapsedTime(&elapsedTime_inc, startEvent_inc, stopEvent_inc);


	//verification
	int count=0;
	for(int i=0;i<N;i++)
	{
		if(hC[i]!=refC[i])
		{
			count++;
		}
	}
	if(count!=0) // This should never be printed in correct code
		std::cout<<"Error at "<< count<<" locations\n";
	std::cout<<N<<"\n"<<M<<"\n"<<elapsedTime_exc<<"\n"<<elapsedTime_inc<<"\n"<<hC[N-1]<<"\n";
	//std::cout<<elapsedTime_exc<<"\n"<<elapsedTime_inc<<"\n";
	//freeing memory
	delete[] hA,hB,hC,refC;     

	// TODO free CUDA memory allocated
	if(dA)  cudaFree(dA);
    if(dB)  cudaFree(dB);
    if(dC)  cudaFree(dC);
    cudaEventDestroy(startEvent_inc);
    cudaEventDestroy(stopEvent_inc);
    cudaEventDestroy(startEvent_exc);
    cudaEventDestroy(stopEvent_exc);

	return 0;
}
