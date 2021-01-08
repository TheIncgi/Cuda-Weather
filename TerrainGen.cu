#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include "math.h"
#include "List.h"

float __constant__ PI     = 3.14159654;
float __constant__ E      = 2.71828182846;
float __constant__ HALF_C = 3.14159654/180;
float __constant__ sqrt2  = 1.41421356237;
float __constant__ sqrt3  = 1.73205080757;

int __constant__ SAND     = 0;
int __constant__ DIRT     = 1;
int __constant__ OCEAN    = 2;
int __constant__ GRASS    = 3; //promoted from dirt in low rain, dry climates
int __constant__ STONE    = 4;
int __constant__ ICE      = 5; //ocean, but past 75 degrees, arctic circles are about 66, but that's seasonally related
int __constant__ FOREST   = 6; //promoted from dirt, humid climates
int __constant__ LAKE     = 7; //local minima of rainy areas

struct vec2{
	float x, y;
};
struct vec3{
	float x, y,z;
};
struct dim2{
	int x,y;
};
__device__ bool operator==(const dim2& lhs, const dim2& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

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

__device__ float distance(vec3 a, vec3 b){
	float dx = b.x-a.x;
	float dy = b.y-a.y;
	float dz = b.z-a.z;
	return sqrt( dx*dx + dy*dy + dz*dz );
}
__device__ float zone(float x, float target, float width, float sharpness){
	return 1 / ( 1 + pow(abs(target/width - x/width), sharpness) );
}
///////////// Perlin /////////////////

__device__ float interpolate(float x, float y,float f){
	return clamp( x * (1-f) + y * f,   x, y);
}
__device__ float smoothstep(float a0, float a1, float w){
    float value = w*w*w*(w*(w*6 - 15) + 10);
    return a0 + value*(a1 - a0);
}

//based on this https://en.wikipedia.org/wiki/Perlin_noise
//modified to create wrapping while still allowing offsets for seed
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
__device__ float perlin(double x, double y, bool wrapTop, bool wrapX1, bool wrapY, double scale) {
//    if(wrapY) y -= scale-1;
	// Determine grid cell coordinates
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    // ya, x       y1(+1), x
    // yb, x+1     y2(+1), x+1



    //if(x0%2==1 && y0 %2 == 1) return 1;
    //if(wrapY) return 1 ;
    // Determine interpolation weights
    // Could also use higher order polynomial/s-curve here
    double sx = x - x0;
    double sy = y - y0;

    double back = scale;//scale;
//    wrapY = false;
    y1       = wrapY? y1 - back : y1;
    double ym = wrapY? y  - back : y;




    // Interpolate between grid point gradients
    float n0, n1, ix0, ix1, value;

    n0 = dotGridGradient(x0, y0, x, y);
    n1 = dotGridGradient(x1, y0, x, y);
    ix0 = smoothstep(n0, n1, sx);

    n0 = dotGridGradient(x0, y1, x, ym);
    n1 = dotGridGradient(x1, y1, x, ym);
    ix1 = smoothstep(n0, n1, sx);

    value = smoothstep(ix0, ix1, sy);
    return value;
}

__device__ float perlin(int x, int y, double scale, float offsetX, float offsetY, int* worldSize){
	double dx = (x + offsetX*scale)/scale;//(x /*+  f1/(scale*2)*/   +offsetX)/scale;
	double dy = (y + offsetY*scale)/scale;//(y /*+  f2/(scale*2)*/   +offsetY)/scale;
	bool wrapTop = ((int)(x / scale)) == 0;
	bool wrapBottom = ((int)(x / scale)) == ((int)(worldSize[0] / scale))-1;
	bool wrapWidth = ((int)(y / scale)) == ((int)(worldSize[1] / scale))-1;
	return perlin(dx, dy, wrapTop, wrapBottom, wrapWidth, 2*worldSize[0]/scale);
}

__device__ int countLakeR(int lat, int lon, int dx, int dy, int* worldSize, int depth, int maxDepth, int**groundType){
	if(depth >= maxDepth) return maxDepth;
	dim2 pos = wrapCoords(lat, lon, worldSize);
	int type = OCEAN; //groundType[pos.x][pos.y];

	if(type != OCEAN && type != LAKE) return depth-1;

	//if(depth>15) return depth-1;

	dim2 posNext= {pos.x, pos.y};
	int nextType = type;
	while(nextType != OCEAN && nextType != LAKE){
		pos = {posNext.x, posNext.y};
		posNext = wrapCoords(posNext.x+dx, posNext.y+dy, worldSize);
		type = OCEAN; //groundType[posNext.x][posNext.y];
		depth++;
		if(depth >= maxDepth) return maxDepth;
	}


	int m = 0;
	if(dx==1)
		m = max(m, countLakeR(pos.x+1, pos.y,   1, dy, worldSize, depth+1, maxDepth, groundType));
	if(dx==-1)
		m = max(m, countLakeR(pos.x-1, pos.y,  -1, dy, worldSize, depth+1, maxDepth, groundType));

	if(dy==1)
		m = max(m, countLakeR(pos.x  , pos.y+1, dx, 1, worldSize, depth+1, maxDepth, groundType));
	if(dy==-1)
		m = max(m, countLakeR(pos.x  , pos.y-1, dx, -1, worldSize, depth+1, maxDepth, groundType));

	return m;
}
__device__ int countLake(int lat, int lon, int* worldSize, int depth, int** groundType){
	int m = 0;
	List<dim2> open(depth+4);
	List<dim2> closed(depth+4);
	dim2 pos = {lat, lon};
	dim2 pos2;
	open.push(pos);
	while(open.size() > 0){
		pos = open.pop();
		closed.push(pos);
		int gt = groundType[pos.x][pos.y];
		if(gt==LAKE || gt==OCEAN)
			m++;
		else
			continue;
		pos2 = wrapCoords(pos.x+1, pos.y, worldSize);
		if(!closed.contains(pos2))
			open.push(pos2);
		pos2 = wrapCoords(pos.x-1, pos.y, worldSize);
		if(!closed.contains(pos2))
			open.push(pos2);
		pos2 = wrapCoords(pos.x, pos.y+1, worldSize);
		if(!closed.contains(pos2))
			open.push(pos2);
		pos2 = wrapCoords(pos.x, pos.y-1, worldSize);
		if(!closed.contains(pos2))
			open.push(pos2);
		if(m>depth) return depth;
	}
	return m;
	//return test[0];
	//return m+countLake(lat, lon, worldSize, depth, groundType);
//	m = max(m, countLakeR(lat, lon  ,  1, 0, worldSize, 0, depth, groundType));
//	m = max(m, countLakeR(lat, lon  , -1, 0, worldSize, 0, depth, groundType));
//	m = max(m, countLakeR(lat,   lon,  0, 1, worldSize, 0, depth, groundType));
//	m = max(m, countLakeR(lat,   lon,  0,-1, worldSize, 0, depth, groundType));
}

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


/*
 * if an ocean consists of less than Ceil(2.5% of the maps height)^2 tiles, it becomes a lake */
extern "C"
__global__ void convertLakes(int* worldSize, int** groundType, float** elevation){
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1]; //ground only

	if (i<n) {
		int thresh = ceil(pow(ceil(worldSize[0] * 0.15),2));

		dim3 pos = getWorldCoords(i, worldSize); //x is latitude in return result
		if(groundType[pos.x][pos.y]==OCEAN){
			int matches = countLake(pos.x, pos.y, worldSize, thresh+1, groundType);//countNeighbors(pos.x, pos.y, 0, 0, worldSize, groundType, LAKE, OCEAN, thresh+1);
			//elevation[pos.x][pos.y] = matches;
			if(matches < thresh)
				groundType[pos.x][pos.y] = LAKE;
		}
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
