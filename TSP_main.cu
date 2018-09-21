#include <iostream>
#include <cstdio>
#include <vector>
#include <ctime>
#include "TSP_city.h"
#include "TSP_serial.h"
#include "TSP_parallel.h"

using namespace std;

int main(int argc, char* argv[])
{
	int num_starts, num_cities;
	cout << "Enter number of restarts : ";
	cin >> num_starts;
	freopen(argv[1], "r", stdin);
	cin >> num_cities;

	TSP_Solver_Parallel parallel_solver(num_cities);
	TSP_Solver_Serial serial_solver(num_cities);

	vector<city> cities_vector, opt_sequence;
	city* cities_array = new city[num_cities];
	for(int i = 0; i < num_cities; i++)
	{
		int id;
		float x_coord, y_coord;
		cin >> id >> x_coord >> y_coord;
		cities_vector.push_back(city(id, x_coord, y_coord));
		cities_array[i] = city(id, x_coord, y_coord);
	}
	for(int i = 0; i < num_cities; i++)
	{
		int id;
		cin >> id;
		opt_sequence.push_back(cities_vector[id - 1]);
	}
	float opt_cost = serial_solver.compute_cost(opt_sequence);

	clock_t start = clock();
	float min_cost = serial_solver.solve(cities_vector, num_starts, num_cities);
	clock_t stop = clock();
	cout << "\nApproximation Ratio : " << min_cost / opt_cost;
	double time_serial = (double)(stop - start) / CLOCKS_PER_SEC;
   	printf("\nTime taken: %.2fs\n\n", time_serial);

   	int grid_dim = 256, block_dim = 256;
	start = clock();
	min_cost = parallel_solver.solve(cities_array, num_starts, num_cities, grid_dim, block_dim);
	stop = clock();
	cout << "\nApproximation Ratio : " << min_cost / opt_cost;
	double time_parallel = (double)(stop - start) / CLOCKS_PER_SEC;
   	printf("\nTime taken: %.2fs\n\n", time_parallel);
	delete cities_array;
	return 0;
}
