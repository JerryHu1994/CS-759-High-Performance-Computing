#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <vector>
#include "stopwatch.hpp"
#include "math.hpp"
#include <omp.h>

#define HEIGHT 1800
#define WIDTH 1200

int main(int argc, char* argv[])
{

	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " N\n";
	}
	
	const auto num_threads = std::atoi(argv[1]);
	//omp_set_num_threads(num_threads);

	int totalSize = HEIGHT*WIDTH;

	int *matrix;
	if ((matrix = (int *)malloc(sizeof(int)*totalSize)) == NULL) {
		std::cerr << "Malloc Error" << std::endl;
		exit(1);
	}	

	// File input and read matrix
	FILE *fp = fopen("picture.inp", "r");
	if (fp == NULL) {
		std::cerr << "Failed to open picture.inp" << std::endl;
		exit(1);
	}

	int i;
	for(i=0;i<totalSize;i++){
		fscanf(fp, "%d\n", &matrix[i]);
	}
	constexpr size_t num_iters = 1;
	std::vector<float> timings;
	timings.reserve(num_iters);
	stopwatch<std::milli, float> sw;

	int global_result[7] = {0};
	// Do it a few times to get rid of timing jitter
	for (size_t m = 0; m < num_iters; ++m) {
		// clean the result array after each 
		for(int z=0;z<7;z++)	global_result[z] = 0;
		sw.start();	
		
		int local_result[7] = {0};
	#pragma omp parallel num_threads(num_threads) firstprivate(local_result)
	{
	#pragma omp for schedule(static)
		for(int i=0; i<totalSize;i++){
			local_result[matrix[i]]++;
		}
	// merge the local results into the global result
	#pragma omp critical
	{
		for(int a=0;a<7;a++){
			global_result[a] += local_result[a]; 
			}
	}
	}
		sw.stop();
		timings.push_back(sw.count());	
	}	
	for(int j=0;j<7;j++)	std::cout << global_result[j] << std::endl;
	const auto min_time = *std::min_element(timings.begin(), timings.end());
	std::cout << num_threads << std::endl;
	std::cout << min_time <<std::endl;
	
	return 0;
} 
