#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include "TSP_city.h"
#include "TSP_serial.h"
using namespace std;

void TSP_Solver_Serial::random_sequence(vector<city> cities, vector<city>& sequence)
{
	srand(time(NULL));
	int num_cities = sequence.size();
	int random_shuffle[num_cities];
	for(int i = 0; i < num_cities; i++)
		random_shuffle[i] = i;
	for(int i = num_cities - 1; i > 0; i--)
	{
		int j = rand() % (i + 1);
		swap(random_shuffle[i], random_shuffle[j]);
	}
	for(int i = 0; i < num_cities; i++)
		sequence[i] = cities[random_shuffle[i]];
}

float TSP_Solver_Serial::dist(city i, city j)
{
	return sqrt((i.x - j.x) * (i.x - j.x) + (i.y - j.y) * (i.y - j.y));
}

float TSP_Solver_Serial::compute_cost(vector<city> sequence)
{
	float cost = 0; 
	int num_cities = sequence.size();
	for(int i = 0; i < num_cities; i++)
		cost += dist(sequence[i], sequence[(i + 1) % num_cities]);
	return cost;      
}

bool TSP_Solver_Serial::check_2_opt_move(vector<city> sequence, int i, int j)
{
	int num_cities = sequence.size();
	return dist(sequence[(i - 1 + num_cities) % num_cities], sequence[i]) + dist(sequence[j], sequence[j + 1])
					> dist(sequence[(i - 1 + num_cities) % num_cities], sequence[j]) + dist(sequence[i], sequence[j + 1]);
}

float TSP_Solver_Serial::start_climber(vector<city>& sequence)
{
	int num_cities = sequence.size();
	for(int i = 0; i <= num_cities - 3; i++)
		for(int j = i + 1; j <= num_cities - 2; j++)
		{
			bool is_better = check_2_opt_move(sequence, i, j);
			if(is_better)
			{
				reverse(sequence.begin() + i, sequence.begin() + j + 1);
				return start_climber(sequence);
			}
		}
	return compute_cost(sequence);
}

float TSP_Solver_Serial::solve(vector<city> cities, int num_starts, int num_cities)
{
	vector<city> start_sequence(num_cities);
	for(int i = 0; i < num_starts; i++)
	{
		random_sequence(cities, start_sequence);
		float cost = start_climber(start_sequence);
		if(cost < min_cost)
			min_cost = cost;
	}
	return min_cost;
}
