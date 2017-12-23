#include <vector>
#include <iostream>
#include <fstream>
#include <list>
#include <algorithm>
#include <iomanip>
#include "stopwatch.hpp"

/*
 * For the vector, you may want to look into the following member functions
 *
 * 	insert - http://en.cppreference.com/w/cpp/container/vector/insert
 * 	push_back - http://en.cppreference.com/w/cpp/container/vector/push_back
 * 	begin - http://en.cppreference.com/w/cpp/container/vector/begin
 *
 * For the list, these may be helpful.
 *
 * 	push_front - http://en.cppreference.com/w/cpp/container/list/push_front
 * 	insert - http://en.cppreference.com/w/cpp/container/list/insert
 * 	push_back - http://en.cppreference.com/w/cpp/container/list/push_back
 *
 */

int main() {
	// for saving the timings
	std::vector<float> timings;

	// a stopwatch in milliseconds
	stopwatch<std::milli, float> sw;

	// The number of integers in each test
	constexpr static size_t sizes[] = { 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13};

	// vector 
	std::ofstream vectorFile;
	vectorFile.open("problem1_array.out");

	// part 1
	// Insert to the front
	for (auto size : sizes) {
		// std::vector is the "dynamically-allocated array" for C++
		std::vector<int> data;

		// Do 10 iterations to remove jitter from the timings
		for (size_t iter = 0; iter < 10; iter++) {

			// Start the timer
			sw.start();
			for (size_t n = 0; n < size; n++) {
				auto pos = data.begin();
				data.insert(pos, n);
			}

			// stop the timer
			sw.stop();

			// save the wall time for the current iteration
			timings.push_back(sw.count());

			// empty the vector
			data.clear();
		}

		// Calculate the minimum wall time
		const auto min_time = *std::min_element(timings.begin(), timings.end());

		// Report the size and minimum time
		//std::cout << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';
		vectorFile << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';

		// clear the saved timings
		timings.clear();
	}

	// part 2
	// Insert to the end
	for (auto size : sizes) {
		// std::vector is the "dynamically-allocated array" for C++
		std::vector<int> data;

		// Do 10 iterations to remove jitter from the timings
		for (size_t iter = 0; iter < 10; iter++) {

			// Start the timer
			sw.start();
			for (size_t n = 0; n < size; n++) {
				// Part C: insert into middle
				auto pos = data.begin();
				std::advance(pos, data.size());
				data.insert(pos, n);
			}

			// stop the timer
			sw.stop();

			// save the wall time for the current iteration
			timings.push_back(sw.count());

			// empty the vector
			data.clear();
		}

		// Calculate the minimum wall time
		const auto min_time = *std::min_element(timings.begin(), timings.end());

		// Report the size and minimum time
		//std::cout << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';
		vectorFile << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';
			

		// clear the saved timings
		timings.clear();
	}
	
	// part 3
	// Insert to the middle
	for (auto size : sizes) {
		// std::vector is the "dynamically-allocated array" for C++
		std::vector<int> data;

		// Do 10 iterations to remove jitter from the timings
		for (size_t iter = 0; iter < 10; iter++) {

			// Start the timer
			sw.start();
			for (size_t n = 0; n < size; n++) {
				// Part C: insert into middle
				auto pos = data.begin();
				std::advance(pos, data.size() / 2UL);
				data.insert(pos, n);
			}

			// stop the timer
			sw.stop();

			// save the wall time for the current iteration
			timings.push_back(sw.count());

			// empty the vector
			data.clear();
		}

		// Calculate the minimum wall time
		const auto min_time = *std::min_element(timings.begin(), timings.end());

		// Report the size and minimum time
		//std::cout << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';
		vectorFile << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';

		// clear the saved timings
		timings.clear();
	}
	vectorFile.close();

	// List
	// Part 1
	// Insert to the middle
	std::ofstream listFile;
	listFile.open("problem1_list.out");
	for (auto size : sizes) {
		// std::vector is the "dynamically-allocated array" for C++
		std::list<int> data;

		// Do 10 iterations to remove jitter from the timings
		for (size_t iter = 0; iter < 10; iter++) {

			// Start the timer
			sw.start();
			for (size_t n = 0; n < size; n++) {
				auto pos = data.begin();
				data.insert(pos, n);
			}

			// stop the timer
			sw.stop();

			// save the wall time for the current iteration
			timings.push_back(sw.count());

			// empty the vector
			data.clear();
		}

		// Calculate the minimum wall time
		const auto min_time = *std::min_element(timings.begin(), timings.end());

		// Report the size and minimum time
		//std::cout << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';
		listFile << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';

		// clear the saved timings
		timings.clear();
	}
	
	// Part 2
	// Insert to the end
	for (auto size : sizes) {
		// std::vector is the "dynamically-allocated array" for C++
		std::list<int> data;

		// Do 10 iterations to remove jitter from the timings
		for (size_t iter = 0; iter < 10; iter++) {

			// Start the timer
			sw.start();
			for (size_t n = 0; n < size; n++) {
				// Part C: insert into middle
				auto pos = data.begin();
				std::advance(pos, data.size());
				data.insert(pos, n);
			}

			// stop the timer
			sw.stop();

			// save the wall time for the current iteration
			timings.push_back(sw.count());

			// empty the vector
			data.clear();
		}

		// Calculate the minimum wall time
		const auto min_time = *std::min_element(timings.begin(), timings.end());

		// Report the size and minimum time
		//std::cout << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';
		listFile << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';
			

		// clear the saved timings
		timings.clear();
	}
	
	// part 3
	// Insert to the middle
	for (auto size : sizes) {
		// std::vector is the "dynamically-allocated array" for C++
		std::list<int> data;

		// Do 10 iterations to remove jitter from the timings
		for (size_t iter = 0; iter < 10; iter++) {

			// Start the timer
			sw.start();
			for (size_t n = 0; n < size; n++) {
				// Part C: insert into middle
				auto pos = data.begin();
				std::advance(pos, data.size() / 2UL);
				data.insert(pos, n);
			}

			// stop the timer
			sw.stop();

			// save the wall time for the current iteration
			timings.push_back(sw.count());

			// empty the vector
			data.clear();
		}

		// Calculate the minimum wall time
		const auto min_time = *std::min_element(timings.begin(), timings.end());

		// Report the size and minimum time
		//std::cout << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';
		listFile << std::fixed << std::setprecision(6) << size << " " << min_time << '\n';

		// clear the saved timings
		timings.clear();
	}
	listFile.close();


}
