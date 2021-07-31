#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include "math.h"
#include "constants.cu"
#include "vectors.cu"
#include "common.cu"
#include "perlin.cu"

////////////// Host functions ////////////////


extern "C"
__global__ void genTerrain(int* worldSize, int** groundType, float** elevation){
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1]; //ground only ( * worldSize[2])

	if (i<n) {
		dim3 pos = getWorldCoords(i, worldSize); //x is latitude in return result
		//mostly random numbers

		double s = (worldSize[0]) / 3.0;
		double baseScale = s;
		double p =       perlin(pos.x, pos.y, s, 0, 0, worldSize) / 2 + .5;
			   s /= 2;
			   p +=  s/baseScale * perlin(pos.x, pos.y, s, 5, 82, worldSize);
			   s /= 2;
			   p +=  s/baseScale * perlin(pos.x, pos.y, s, 3, 44, worldSize);
			   s /= 2;
			   p +=  s/baseScale * perlin(pos.x, pos.y, s, 6, 22, worldSize);
			   s /= 2;
			   p +=  s/baseScale * perlin(pos.x, pos.y, s, 15, 325, worldSize);
			   s /= 2;
			   p +=  s/baseScale * perlin(pos.x, pos.y, s, 15, 325, worldSize);

		      p*=1.5;
		      p-=.2;
		//https://www.desmos.com/calculator/uuvyruqbyk
		float latitude = map(pos.x, 0, worldSize[0], 90, -90);
		float deadZone = 12;
		float iceyP = p* (zone(abs(latitude), 90, 6, 1.4));//(p * ( 1-zone(abs(latitude), 90, 2, 1.4)));
		p *= 1- zone(abs(latitude), 90, deadZone, 4);
//		p *= 1 - 1 / ( 1 + pow(abs(90/deadZone-abs(latitude)/deadZone), 4) );

		//p will range from 0 to a little less than 2
		int gt = 0;
		float e = 0;
		if(p<.6) {
			if(/*abs(latitude) > 78 &&*/ iceyP>.4) {
				gt = ICE;
				e = map(p, 0, .6, 0,1);
			}else {
				gt = LAKE; //about 71% of the earth's surface is water
				e = map(p, 0, .6, -.5, -.01);
			}
		}
		else if(p<.65){
			gt = SAND;
			e = map(p, .6, .65, 0, .05);
		}else if(p<.9) {
			gt = DIRT;
			e = map(p, .65, .9, .05, 3/2);
		}else {
			gt = STONE;
			e = map(p, .9, 1, 3/2, 6/2);
		}

		groundType[pos.x][pos.y] = gt;




		e*= 1000;
		elevation[pos.x][pos.y] = e;
	}
}

/**
 * A second attempt to create a way to detect enclosed bodies of water with minimal memory usage.
 * */
extern "C"
__global__ void convertLakes2(int* worldSize, int** groundType,  int* changed){
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1]; //ground only

	if (i<n) {
		dim3 pos = getWorldCoords(i, worldSize);
		int ground = groundType[pos.x][pos.y];
		float lat = map(pos.x, 0, worldSize[0], -90,90);
		if(ground != LAKE) return;
		if(lat < -80 || lat > 80) { //https://www.desmos.com/calculator/uuvyruqbyk area forced to be ocean or ice
			if(ground != OCEAN) {
				groundType[pos.x][pos.y] = OCEAN;
				*changed = 1;
			}
		}else{
			dim2 q = wrapCoords(pos.x+1, pos.y  , worldSize);
			if(groundType[q.x][q.y] == OCEAN) {
				groundType[pos.x][pos.y] = OCEAN;
				*changed = 2;
			}
			q = wrapCoords(pos.x-1, pos.y  , worldSize);
			if(groundType[q.x][q.y] == OCEAN) {
				 groundType[pos.x][pos.y] = OCEAN;
				*changed = 3;
			}
			q = wrapCoords(pos.x  , pos.y+1, worldSize);
			if(groundType[q.x][q.y] == OCEAN) {
				groundType[pos.x][pos.y] = OCEAN;
				*changed = 4;
			}
			q = wrapCoords(pos.x  , pos.y-1, worldSize);
			if(groundType[q.x][q.y] == OCEAN) {
				groundType[pos.x][pos.y] = OCEAN;
				*changed = 5;
			}
		}


	}
}
