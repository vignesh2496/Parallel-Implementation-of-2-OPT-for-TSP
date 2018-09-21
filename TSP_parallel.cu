#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include "TSP_city.h"
#include "TSP_parallel.h"
using namespace std;

__device__ void random_sequence(city* cities, city* sequence, int num_cities)
{
	curandState_t state;
	curand_init(blockDim.x * blockIdx.x + threadIdx.x, 0, 0, &state);
	int random_shuffle[100];
	for(int i = 0; i < num_cities; i++)
		random_shuffle[i] = i;
	for(int i = num_cities - 1; i > 0; i--)
	{
		int j = curand(&state) % (i + 1);
		int temp = random_shuffle[i];
		random_shuffle[i] = random_shuffle[j];
		random_shuffle[j] = temp;
	}
	for(int i = 0; i < num_cities; i++)
	{
		sequence[i] = cities[random_shuffle[i]];
	}
}

__device__ float dist(city i, city j)
{
	return sqrt((i.x - j.x) * (i.x - j.x) + (i.y - j.y) * (i.y - j.y));
}

__device__ float compute_cost(city* sequence, int num_cities)
{
	float cost = 0;
	for(int i = 0; i < num_cities; i++)
		cost += dist(sequence[i], sequence[(i + 1) % num_cities]);
	return cost;
}

__device__ bool check_2_opt_move(city* sequence, int i, int j, int num_cities)
{
	return dist(sequence[(i - 1 + num_cities) % num_cities], sequence[i]) + dist(sequence[j], sequence[j + 1])
						> dist(sequence[(i - 1 + num_cities) % num_cities], sequence[j]) + dist(sequence[i], sequence[j + 1]);
}

__device__ void my_reverse(city* cities, int i, int j)
{
	city temp;
	for(int k = i; k <= (j - i) / 2 ; k++)
	{
		temp = cities[k];
		cities[k] = cities[j - k + i];
		cities[j - k + i] = temp;
	}
}

__device__ float start_climber(city* sequence, int num_cities)
{
	for(int i = 0; i <= num_cities - 3; i++)
		for(int j = i + 1; j <= num_cities - 2; j++)
		{
			bool is_better = check_2_opt_move(sequence, i, j, num_cities);
			if(is_better)
			{
				my_reverse(sequence, i, j);
				return start_climber(sequence, num_cities);
			}
		}
	return compute_cost(sequence, num_cities);
}

__global__ void call_climbers(city* cities, int* num_starts, int num_cities, float* min_cost_array)
{
	int num_threads = blockDim.x;
	__shared__ float thread_costs[256];
	city start_sequence[100];
	thread_costs[threadIdx.x] = FLT_MAX;
	while(*num_starts > 0)
	{
		atomicSub(num_starts, 1);
		random_sequence(cities, start_sequence, num_cities);
		float thread_cost = start_climber(start_sequence, num_cities);

		if(thread_cost < thread_costs[threadIdx.x])
			thread_costs[threadIdx.x] = thread_cost;
	}

	__syncthreads();

	while(threadIdx.x < num_threads && num_threads > 1)
    	{
        	if(threadIdx.x < num_threads / 2)
        		thread_costs[threadIdx.x] = (thread_costs[threadIdx.x] < thread_costs[threadIdx.x + num_threads / 2]) ? thread_costs[threadIdx.x] : thread_costs[threadIdx.x + num_threads / 2];
        	num_threads = num_threads >> 1;
        	__syncthreads();
    	}

	if(threadIdx.x == 0)
		min_cost_array[blockIdx.x] = thread_costs[0];
}

float TSP_Solver_Parallel::solve(city* cities, int num_starts, int num_cities, int grid_dim, int block_dim)
{
	int *dev_num_starts;
	cudaMalloc(&dev_num_starts, sizeof(int));
	cudaMemcpy(dev_num_starts, &num_starts, sizeof(int), cudaMemcpyHostToDevice);
	float *dev_min_cost_array, *min_cost_array = new float[grid_dim];
	cudaMalloc(&dev_min_cost_array, sizeof(float) * grid_dim);
	city *dev_cities;
	cudaMalloc(&dev_cities, sizeof(city) * num_cities);
	cudaMemcpy(dev_cities, cities, sizeof(city) * num_cities, cudaMemcpyHostToDevice);
	call_climbers<<<grid_dim, block_dim>>>(dev_cities, dev_num_starts, num_cities, dev_min_cost_array);
	cudaMemcpy(min_cost_array, dev_min_cost_array, sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < grid_dim; i++)
	{	if(min_cost_array[i] < min_cost)
			min_cost = min_cost_array[i];
		//cout << min_cost_array[i] << "  ";
	}
	cudaFree(dev_num_starts);
	cudaFree(dev_min_cost_array);
	cudaFree(dev_cities);
	delete min_cost_array;
	return min_cost;
}
