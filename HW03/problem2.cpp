#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <limits>
#include <iomanip>
#include "stopwatch.hpp"


struct point3D_A {
	float x, y, z;
};

struct point3D_B {
	float x, y, z;
	char c;
};

struct point3D_C {
	float x, y, z;
	char c[52];
};
int main() {
	std::ofstream fout{"problem2.out"};

	auto min_time = [&fout](array_view<float> const& x) {
		fout << *std::min_element(x.begin(), x.end()) << '\n';
	};

	constexpr auto size = 1'000'000UL;

	stopwatch<std::milli, float> sw;
	// Part A
	std::vector<point3D_A> x(size);

	/*
	 * Note that this is really just a dummy variable we are using
	 * to prevent the compiler from optimizing away our calculations.
	 */
	volatile float sum{};
	auto f_A = [&x, &sum]() {
		sum = std::accumulate(x.begin(), x.end(), 0.0f, [](float total, point3D_A const& p){ return total + p.x;});
	};
	sw.time_it(10UL, f_A, min_time);
	
	// Part B
	std::vector<point3D_B> y(size);
	auto f_B = [&y, &sum]() {
		sum = std::accumulate(y.begin(), y.end(), 0.0f, [](float total, point3D_B const& p){ return total + p.x;});
	};
	sw.time_it(10UL, f_B, min_time);

	std::vector<point3D_C> z(size);
	auto f_C = [&z, &sum]() {
		sum = std::accumulate(z.begin(), z.end(), 0.0f, [](float total, point3D_C const& p){ return total + p.x;});
	};
	sw.time_it(10UL, f_C, min_time);

//	std::cout<<sizeof(point3D)<<std::endl;	
}
