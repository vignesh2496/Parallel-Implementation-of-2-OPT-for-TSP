#ifndef _TSP_SOLVER_SERIAL_H
#define _TSP_SOLVER_SERIAL_H
#include <vector>
#include <cfloat>
#include "TSP_city.h"
using namespace std;

class TSP_Solver_Serial
{
	float min_cost;
	public:
		TSP_Solver_Serial(int num_cities)
		{
			min_cost = FLT_MAX;
		}
		float solve(vector<city>, int, int);
		void random_sequence(vector<city>, vector<city>&);
		float dist(city, city);
		float compute_cost(vector<city>);
		bool check_2_opt_move(vector<city>, int, int);
		float start_climber(vector<city>&);	
};

#endif
