#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <iterator>
#include <algorithm>
#include "problem2.hu"
#include <omp.h>
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

template <typename T>
auto read_file(char const* name, int size) {
	std::ifstream fin{name};
	std::vector<T> x;
	x.reserve(static_cast<size_t>(size));
	std::copy_n(std::istream_iterator<T>(fin), size, std::back_inserter(x));
	return x;
}

int main(int argc, char* argv[]) {
	mpi::context ctx(&argc, &argv);
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " N\n";
		return -1;
	}

	const auto N = std::atoi(argv[1]);
	auto x = read_file<float>("problem2.inp", N);

	std::vector<float> sum(static_cast<size_t>(N));
	double start=0.0;
	if(ctx.rank() == 0)	start = omp_get_wtime();
	prefix_scan(x.data(), sum.data(), N);

	if(ctx.rank() == 0) {
		float presum = sum.data()[N-1];
		MPI_Send(&presum, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
		MPI_Status status;
		float result=0.0;
		constexpr int source_rank = 1;  // We expect a message from Task 1
		MPI_Recv(&result, 1, MPI_FLOAT, source_rank, 0, MPI_COMM_WORLD, &status);
		//std::cout << "Received x = " << result << " on root task.\n";
		double end = omp_get_wtime();
		printf("%d\n%f\n%f\n", N, 1000.0*(end-start), result);
	} else {
		MPI_Status status;
		float presum = 0.0;
		constexpr int dest_rank = 0;  // We send a message to Task 0

		MPI_Recv(&presum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		//call kernel to add partial sum
		addFirstHalf(sum.data(), presum, N);
		float result = sum.data()[N-1];
		MPI_Send(&result, 1, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD);
	}

	return 0;
}
