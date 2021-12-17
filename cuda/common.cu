#pragma once
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include "cuda.h"
#include "vectors.cu"
#include "math.h"
#include "constants.cu"

//cuda related
__device__ int getThreadsPerBlock(){
	return blockDim.x * blockDim.y * blockDim.z;
}
__device__ int getBlockID(){
	return    blockIdx.x +
			  blockIdx.y * gridDim.x +
			  blockIdx.z * gridDim.x * gridDim.y;
}
__device__ int getLocalThreadID() {
	return threadIdx.x +
		   threadIdx.y * blockDim.x +
		   threadIdx.z * blockDim.x * blockDim.y;
}
__device__ int getGlobalThreadID(){
	return getLocalThreadID() + getBlockID() * getThreadsPerBlock();
}

__device__ dim3 getWorldCoords(int gThreadID, int* worldSize){
	int latitude =  gThreadID % worldSize[0];
	gThreadID = gThreadID / worldSize[0];
	int longitude = gThreadID % worldSize[1];
	gThreadID = gThreadID / worldSize[1];
	int altitude  = gThreadID % worldSize[2];
	return dim3(latitude, longitude, altitude);
}

__device__ dim2 wrapCoords(int x, int y, int* worldSize){
	int la = x;
	int lo = y % worldSize[1];
	if(la < 0){
		lo = (lo + worldSize[1] / 2) % worldSize[1];
		la = -la -1;
	}else if(la >= worldSize[0]){
		lo = (lo + worldSize[1] / 2) % worldSize[1];
		la = worldSize[0]-(la-worldSize[0]+1);
	}
	la = (la + worldSize[0]) % worldSize[0];
	lo = (lo + worldSize[1]) % worldSize[1];
	dim2 out = {la, lo};
	return out;
}

__device__ float map(float x, float inMin, float inMax, float outMin, float outMax) {
	return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

__device__ float clamp(float x, float low, float high){
	if(high < low){
		int t = low;
		low = high;
		high = t;
	}
	return x < low? low : (x > high ? high : x);
}
__device__ int sign(float val) {
    return ((0) < val) - (val < (0));
}

__device__ float distance(vec3 a, vec3 b){
	float dx = b.x-a.x;
	float dy = b.y-a.y;
	float dz = b.z-a.z;
	return sqrt( dx*dx + dy*dy + dz*dz );
}
__device__ float distance(float a, float b, float x, float y){
	float dx = x-a;
	float dy = y-b;
	return sqrt( dx*dx + dy*dy );
}
__device__ float zone(float x, float target, float width, float sharpness){
	return 1 / ( 1 + pow(abs(target/width - x/width), sharpness) );
}

__device__ void rotVec3AboutZ(vec3 &vec, float yaw){
	double nx = vec.x*cos(yaw) - vec.y*sin(yaw);
	double ny = vec.x*sin(yaw) + vec.y*cos(yaw);
	vec.x = (float) nx;
	vec.y = (float) ny;
}
__device__ void rotateVec3AboutY(vec3 &vec, float pitch) {
	double nx = vec.x*cos(pitch) + vec.z*sin(pitch);
	double nz = -vec.x*sin(pitch) + vec.z*cos(pitch);
	vec.x = (float) nx;
	vec.z = (float) nz;
}

//you've got a pocket full of it?
/*Calculates the brightness of sunshine in an area on a scale of 0 to 1*/
__device__ float sunshine(float lat, float lon, int* worldSize, float* worldTime){
	vec3 sun = {1, 0, 0};
	float yaw = (float) (-360.0 * (lon ) / worldSize[1]);
	float pitch = (float) (180.0 * (lat ) / worldSize[0] -90.0);
	yaw += worldTime[0] * -360; //daily rotation
	yaw += worldTime[1] * -360; //if the earth didn't rotate, it'd have 1 day every year
	vec3 p = {1, 0, 0};
	rotateVec3AboutY(p, pitch * HALF_C);
	rotVec3AboutZ(p, yaw * HALF_C);
	rotateVec3AboutY(p, PLANET_TILT * HALF_C * sin( worldTime[1]*360 * HALF_C ));

	return clamp(dot(sun,p), 0, 1); //praise the sun
}
///////////// Perlin ///////////////// //TODO make this into a sep file, share with Terrain gen





__device__ float interpolate(float x, float y,float f){
	return clamp( x * (1-f) + y * f,   x, y);
}
__device__ float smoothstep(float a0, float a1, float w){
    float value = w*w*w*(w*(w*6 - 15) + 10);
    return a0 + value*(a1 - a0);
}

/////////////////////////////////////////////
//bit flag util

__device__ bool hasFlag(int value, int test){
	return value & test > 0;
}