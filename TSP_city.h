#ifndef _TSP_SOLVER_CITY_H
#define _TSP_SOLVER_CITY_H
#include <cfloat>

struct city
{
        int idx;
        float x, y;
        __device__ __host__ city(int id = -1, float x_coord = FLT_MIN, float y_coord = FLT_MIN) : idx(id), x(x_coord), y(y_coord) {}
};

#endif
