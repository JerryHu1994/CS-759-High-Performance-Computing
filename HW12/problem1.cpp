#include <iostream>
#include <omp.h>
#define OMPI_SKIP_MPICXX  /* Don't use OpenMPI's C++ bindings (they are deprecated) */
#include <mpi.h>

namespace mpi {
	class context {
		int m_rank, m_size;
	public:
		context(int *argc, char **argv[]) : m_rank { -1 } {
			if (MPI_Init(argc, argv) == MPI_SUCCESS) {
				MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
				MPI_Comm_size(MPI_COMM_WORLD, &m_size);
			}
		}
		~context() {
			if(m_rank >= 0) {
				MPI_Finalize();
			}
		}
		explicit operator bool() const {
			return m_rank >= 0;
		}
		int rank() const noexcept { return m_rank; }
		int size() const noexcept { return m_size; }
	};
}


float calculateSum(float *in, int size)
{
	float sum = 0.0;
	int i;
#pragma omp parallel for reduction (+:sum)
	for(i=0;i<size;i++){
		sum = sum + in[i];
	}
	return sum;
}

void initializeArray(FILE* fp,float* arr, int nElements)
{
	for( int i=0; i<nElements; i++){
		int r=fscanf(fp,"%f",&arr[i]);
		if(r == EOF){
			rewind(fp);
		}
	}
}
int main(int argc, char *argv[]) {
	mpi::context ctx(&argc, &argv);

	if(!ctx) {
		std::cerr << "MPI Initialization failed\n";
		return -1;
	}

	if(argc!=2){
		printf("Usage %s N\n",argv[0]);
		return 1;
	}
	int N=atoi(argv[1]);
	FILE *fp = fopen("problem1.inp","r");
	int size = N * sizeof(float); 
	float *in = (float *)malloc(size);
	initializeArray(fp,in, N);
	double start=0.0;
	if(ctx.rank() == 0)	start= omp_get_wtime();	
	float localSum = calculateSum(in, N);

	if(ctx.rank() == 0) {
		float x=0.0;
		constexpr int source_rank = 1;  // We expect a message from Task 1
		MPI_Status status;
		MPI_Recv(&x, 1, MPI_FLOAT, source_rank, 0, MPI_COMM_WORLD, &status);
		float result = localSum + x;
		double end = omp_get_wtime();	
		//std::cout << "Received x = " << x << " on root task.\n";
		printf("%d\n%f\n%f\n",N,1000.0*(end - start), result);
	} else {
		const float i=localSum;
		constexpr int dest_rank = 0;  // We send a message to Task 0
		MPI_Send(&i, 1, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD);
	}
	return 0;
}
