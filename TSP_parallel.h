#ifndef _TSP_SOLVER_PARALLEL_CUDA_H
#define _TSP_SOLVER_PARALLEL_CUDA_H
#include <cfloat>
#include "TSP_city.h"
using namespace std;

class TSP_Solver_Parallel
{
        float min_cost;
        public:
                __device__ __host__ TSP_Solver_Parallel(int num_cities)
                {
                        min_cost = FLT_MAX;
                }
                float solve(city*, int, int, int, int);
};

#endif
