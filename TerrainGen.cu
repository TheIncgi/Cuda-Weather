#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include "math.h"

float __constant__ PI     = 3.14159654;
float __constant__ E      = 2.71828182846;
float __constant__ HALF_C = 3.14159654/180;
float __constant__ sqrt2  = 1.41421356237;
float __constant__ sqrt3  = 1.73205080757;

struct vec2{
	float x, y;
};
struct vec3{
	float x, y,z;
};

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

__device__ float distance(vec3 a, vec3 b){
	float dx = b.x-a.x;
	float dy = b.y-a.y;
	float dz = b.z-a.z;
	return sqrt( dx*dx + dy*dy + dz*dz );
}
///////////// Perlin /////////////////

__device__ float interpolate(float x, float y,float f){
	return clamp( x * (1-f) + y * f,   x, y);
}
//from the wiki https://en.wikipedia.org/wiki/Perlin_noise
__device__ vec2 randomGradient(int ix, int iy) {
    // Random float. No precomputed gradients mean this works for any number of grid coordinates
    float random = 2920.f * sin(ix * 21942.f + iy * 171324.f + 8912.f) * cos(ix * 23157.f * iy * 217832.f + 9758.f);
    vec2 out = {cos(random), sin(random) };
    return out;
}

// Computes the dot product of the distance and gradient vectors.
__device__ float dotGridGradient(int ix, int iy, float x, float y) {
    // Get gradient from integer coordinates
    vec2 gradient = randomGradient(ix, iy);

    // Compute the distance vector
    float dx = x - ix;
    float dy = y - iy;

    // Compute the dot-product
    return (dx*gradient.x + dy*gradient.y);
}

// Compute Perlin noise at coordinates x, y
//world size added for texture wrapping
__device__ float perlin(float x, float y, bool wrapX, bool wrapY) {
    // Determine grid cell coordinates
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    // Determine interpolation weights
    // Could also use higher order polynomial/s-curve here
    float sx = x - (float)x0;
    float sy = y - (float)y0;

    float xm = wrapX? 0 : x;
    float ym = wrapY? 0 : y;
          x1 = wrapX? 0 : x1;
          y1 = wrapY? 0 : y1;


    // Interpolate between grid point gradients
    float n0, n1, ix0, ix1, value;

    n0 = dotGridGradient(x0, y0, x, y);
    n1 = dotGridGradient(x1, y0, xm, y);
    ix0 = interpolate(n0, n1, sx);

    n0 = dotGridGradient(x0, y1, x, ym);
    n1 = dotGridGradient(x1, y1, xm, ym);
    ix1 = interpolate(n0, n1, sx);

    value = interpolate(ix0, ix1, sy);
    return value;
}

__device__ float perlin(int x, int y, float scale, float offsetX, float offsetY,float f1, float f2, int* worldSize){
	float dx = (x +  f1/(scale*2)   +offsetX)/scale;
	float dy = (y +  f2/(scale*2)   +offsetY)/scale;
	bool wrapX = x==3 || x==(int)(worldSize[0]-1)/scale;
	bool wrapY = y==(int)(worldSize[1]-1)/scale;
	return perlin(dx, dy, wrapX, wrapY);
}

////////////// Host functions ////////////////
int __constant__ SAND     = 0;
int __constant__ DIRT     = 1;
int __constant__ OCEAN    = 2;
int __constant__ GRASS    = 3; //promoted from dirt in low rain, dry climates
int __constant__ STONE    = 4;
int __constant__ ICE      = 5; //ocean, but past 75 degrees, arctic circles are about 66, but that's seasonally related
int __constant__ FOREST   = 6; //promoted from dirt, humid climates
int __constant__ LAKE     = 7; //local minima of rainy areas

extern "C"
__global__ void genTerrain(int* worldSize, int** groundType, float** elevation){
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1] * worldSize[2];

	if (i<n) {
		dim3 pos = getWorldCoords(i, worldSize); //x is latitude in return result
		//mostly random numbers

		float s = 48;
		float   p =       perlin(pos.x, pos.y, s, 0, 0, 1, 1, worldSize) / 2 + .5;
//				s /= 2;
//				p += s/48 * perlin(pos.x, pos.y, s, 55147, 887412, 1.5, 1.3, worldSize);
//				s /= 2;
//				p += s/48 * perlin(pos.x, pos.y, s, 575347, 83412, 1.6, 1.74, worldSize);
//				s /= 2;
//				p += s/48 * perlin(pos.x, pos.y, s, 25747, 286492, 1.7, 1.6, worldSize);
//				s /= 2;
//				p += s/48 * perlin(pos.x, pos.y, s, 37747, 876484, 1.437, 1.18563, worldSize);
//				s /= 2;

		//float   p =  map(perlin((pos.x + 1/(s*2))/s       , (pos.y+(1/(s*2)))/s        , worldSize), -1, 1, 0, 1 );
//				s /= 2;
//			    p +=   s/48 *  perlin((pos.x + 1.5/(s*2) +55147)/s, (pos.y+(1.3/(s*2)))/s +887412 , worldSize);
//				s /= 2;
//			    p +=   s/48 *  perlin((pos.x + 1.6/(s*2) +55347)/s, (pos.y+(1.74/(s*2)))/s +877412, worldSize);
//				s /= 2;
//			    p +=   s/48 *  perlin((pos.x + 1.7/(s*2) +57747)/s, (pos.y+(1.6/(s*2)))/s +886492, worldSize);
		      p*=1.5;
		      p-=.2;
		int gt = 0;
		if(p<.5) {
			if(abs(map(pos.x, 0, worldSize[0], -90, 90)) > 80) {
				gt = ICE;
			}else {
				gt = OCEAN; //about 71% of the earth's surface is water
			}
		}
		else if(p<.55) gt = SAND;
		else if(p<.9) gt = DIRT;
		else           gt = STONE;

		groundType[pos.x][pos.y] = gt;

		float e = 0;
		if(p<.5)
			e = map(p, 0, .71, -.5, 0);
		else if(p<.52)
			e = map(p, .71, .85, 0, 1);
		else if(p<.68)
			e = map(p, .85, .95, 1, 3);
		else
			e = map(p, .95, 1, 3, 6);
		e*= 1000;
		elevation[pos.x][pos.y] = p*1000;
	}
}
