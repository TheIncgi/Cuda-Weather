#pragma once
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

float __constant__ PI     = 3.14159654;
float __constant__ E      = 2.71828182846;
float __constant__ HALF_C = 3.14159654/180;
float __constant__ sqrt2  = 1.41421356237;
float __constant__ sqrt3  = 1.73205080757;

float __constant__ PLANET_RADIUS = 6371; //km
float __constant__ PLANET_MASS   = 5.972e24; //kg ...that's 5 septillion
float __constant__ PLANET_TILT   = 23.5;
float __constant__ GRAVITATIONAL_CONSTANT = 6.67E-11;
float __constant__ STEFAN_BOLTZMANN_CONSTANT = 5.670374E-8;

int __constant__ SQ_M_IN_SQ_KM = 1000000;

int __constant__ SAND     = 0;
int __constant__ DIRT     = 1;
int __constant__ OCEAN    = 2;
int __constant__ GRASS    = 3; //promoted from dirt in low rain, dry climates
int __constant__ STONE    = 4;
int __constant__ ICE      = 5; //ocean, but past 75 degrees, arctic circles are about 66, but that's seasonally related
int __constant__ FOREST   = 6; //promoted from dirt, humid climates
int __constant__ LAKE     = 7; //local minima of rainy areas

__device__ const char * biomeName(int groundType) {
	switch(groundType) {
		case 0: return "SAND";
		case 1: return "DIRT";
		case 2: return "OCEAN";
		case 3: return "GRASS";
		case 4: return "STONE";
		case 5: return "ICE";
		case 6: return "FOREST";
		case 7: return "LAKE";
		default: return "???";
	}
}