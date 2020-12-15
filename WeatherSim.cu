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
float __constant__ PLANET_RADIUS = 6371; //km
float __constant__ PLANET_MASS   = 5.972e24; //kg ...that's 5 septillion
float __constant__ PLANET_TILT   = 23.5;
float __constant__ GRAVITATIONAL_CONSTANT = 6.67E-11;

struct vec3{
	float x, y,z;
};

//Notes about functions:
//temperature is in fahrenheit
//mass is in kg except for maxWaterHeld which is in grams per cubic meter
//wind speed is in meters per second
//wind speed is used to calculate momentum


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

//math
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

__device__ dim3 getWorldCoords(int gThreadID, int* worldSize){
	int latitude =  gThreadID % worldSize[0];
	gThreadID = gThreadID / worldSize[0];
	int longitude = gThreadID % worldSize[1];
	gThreadID = gThreadID / worldSize[1];
	int altitude  = gThreadID % worldSize[2];
	return dim3(latitude, longitude, altitude);
}

__device__ dim3 wrapCoords(int x, int y, int* worldSize){
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
	dim3 out(la, lo, 0);
	return out;
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

/**altitude in KM*/
__device__ float altitudeOfIndex(int index, int* worldSize){
	return map(index, 0, worldSize[2], -.5, 12);
}
__device__ float distance(vec3 a, vec3 b){
	float dx = b.x-a.x;
	float dy = b.y-a.y;
	float dz = b.z-a.z;
	return sqrt( dx*dx + dy*dy + dz*dz );
}
__device__ float dot(vec3 a, vec3 b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
//polar to cartesian
__device__ vec3 mapTo3D(int *worldSize, int latitude, int longitude, int altitude, bool center){
	float yaw = (float) (360.0 * (longitude + 0.5) / worldSize[1]);
	float pitch = (float) (180.0 * (latitude + 0.5) / worldSize[0] -90.0);
	vec3 p = {PLANET_RADIUS + altitudeOfIndex(altitude, worldSize), 0, 0};
	rotateVec3AboutY(p, pitch * HALF_C);
	rotVec3AboutZ(p, yaw * HALF_C);
	return p;
}
__device__ vec3 mapTo3D(int *worldSize, int latitude, int longitude, int altitude){
	return mapTo3D(worldSize, latitude, longitude, altitude, false);
}
/**Distance between the center of two grid points*/
__device__ float surfaceDistance(int lat1, int lon1, int alt1, int lat2, int lon2, int alt2, int* worldSize){
	vec3 a = mapTo3D(worldSize, lat1, lon1, alt1, true);
	vec3 b = mapTo3D(worldSize, lat2, lon2, alt2, true);
	return distance(a,b);
}
__device__ float surfaceAreaAt(int* worldSize, int latitude, int altitude){
	vec3 a = mapTo3D(worldSize, latitude,   0, altitude);
	vec3 b = mapTo3D(worldSize, latitude,   1, altitude);
	vec3 c = mapTo3D(worldSize, latitude+1, 0, altitude);
	vec3 d = mapTo3D(worldSize, latitude+1, 1, altitude);
	return (distance(a, b) + distance(c, d)) / 2 * distance(a, c);
}
__device__ float volumeAt(int* worldSize, int lat, int alt){
	float a = surfaceAreaAt(worldSize, lat, alt);
	float b = surfaceAreaAt(worldSize, lat, alt+1);
	float h = altitudeOfIndex(alt+1, worldSize) - altitudeOfIndex(alt, worldSize);
	return (a+b)/2 * h;
}

__device__ float CtoF(float celcius){
	return celcius * 9 / 5 + 32;
}

/**aprox air density at temp kg per meter cubed*/
__device__ float airDensity(float tempF){
	return tempF * -.0026 + 1.38;
}

//https://www.desmos.com/calculator/gzqcdksdhs
//https://www.engineeringtoolbox.com/maximum-moisture-content-air-d_1403.html
/**returns the maximum grams of water held by 1 cubic meter of air at some given temp*/
__device__ float maxWaterHeld(float temp){
	temp = CtoF(temp);
	float s = 486647.468932;
	float k = 139.7;
	float p = 44.210324391;
	return s * ( 1 / (p * sqrt(2 * PI * p))) * pow(E, (
			- ( ((temp-k)*(temp-k)) / (2*p*p)   )
			) ) ;
}

//altitude provided as index
__device__ float gravityAccel(int altitude, int* worldSize){
	float r =  altitudeOfIndex(altitude, worldSize) + PLANET_RADIUS * 1000; //in meters

	return GRAVITATIONAL_CONSTANT * ( PLANET_MASS ) / (r*r);
}

/**
 * Calculates the mass of an air volume in kg
 * */
__device__ float airMass(float kmCubed, float temp, float relHumid){
	float mass = airDensity(temp) * kmCubed; //mass of air
	//1 billion cubic meters in 1 cubic km
	//divide 1000 because maxWater is returned in grams
	//operations cancled out to * 1M
	mass += maxWaterHeld(temp) * 1'000'000 * (relHumid>1?1:relHumid)  * kmCubed; //mass of water in form of clouds
	return mass;
}

//you've got a pocket full of it?
/*Calculates the brightness of sunshine in an area on a scale of 0 to 1*/
__device__ float sunshine(int lat, int lon, int* worldSize, float* worldTime){
	vec3 sun = {1, 0, 0};
	float yaw = (float) (-360.0 * (lon ) / worldSize[1]);
	float pitch = (float) (180.0 * (lat ) / worldSize[0] -90.0);
	yaw += worldTime[0] * -360; //daily rotation
	yaw += worldTime[1] * -360; //if the earth didn't rotate, it'd have 1 day every year
	vec3 p = {1, 0, 0};
	rotateVec3AboutY(p, pitch * HALF_C);
	rotVec3AboutZ(p, yaw * HALF_C);
	rotateVec3AboutY(p, PLANET_TILT * HALF_C * sin( worldTime[1]*360 * HALF_C ));

	return clamp(dot(sun,p), 0, 1);
}

//Host accessable functions


/**
 * Setup a world pressure and temperature
 * */
extern "C"
__global__ void initAtmosphere(
	int* worldSize,
	float*** pressureOut,
	float*** tempOut
){
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1] * worldSize[2];
	dim3 pos = getWorldCoords(i, worldSize);
	if (i<n) {
		float alt = map(pos.z, 0, worldSize[2], 0, 1);
		pressureOut[pos.x][pos.y][pos.z] = map(alt, 0, 1, 1.02, 0.197385);
		float lat = 180-(180*pos.x / (float)worldSize[0]) -90;
		float maxTemp = map(alt,0, 1, 95, -50);
		float minTemp = map(alt,0, 1, -40, -50);
		tempOut[pos.x][pos.y][pos.z] = map(cos(abs(lat) * HALF_C), 0,1, minTemp, maxTemp);
	}
}

extern "C" //test function
__global__ void copy(
	//static
	int*     worldSize,
	float*   worldSpeed,
	float**  elevation,
	int**    groundType,

	//inputs
	float*   worldTimeIn,

	float**  groundMoistureIn,
	float**  snowCoverIn,

	float*** temperatureIn,
	float*** pressureIn,
	float*** humidityIn,
	float*** cloudCoverIn,
	float*** windSpeedIn,
	
	//outputs
	float*   worldTimeOut,

	float**  groundMoistureOut,
	float**  snowCoverOut,

	float*** temperatureOut,
	float*** pressureOut,
	float*** humidityOut,
	float*** cloudCoverOut,
	float*** windSpeedOut
	) { //end of args...
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1] * worldSize[2];
	dim3 pos = getWorldCoords(i, worldSize);
    if (i<n) {
    	if(i==0){
    		worldTimeOut[0] += worldSpeed[0];
    		worldTimeOut[1] += worldSpeed[1];
    	}
    	if(pos.z==0){
    		groundMoistureOut[pos.x][pos.y] = groundMoistureIn[pos.x][pos.y];
    		snowCoverOut[pos.x][pos.y]      = snowCoverIn[pos.x][pos.y];
    	}
        temperatureOut[pos.x][pos.y][pos.z] = temperatureIn[pos.x][pos.y][pos.z];
        pressureOut[pos.x][pos.y][pos.z]    = pressureIn[pos.x][pos.y][pos.z];
        humidityOut[pos.x][pos.y][pos.z]    = humidityIn[pos.x][pos.y][pos.z];
        cloudCoverOut[pos.x][pos.y][pos.z]  = cloudCoverIn[pos.x][pos.y][pos.z];
        windSpeedOut[pos.x][pos.y][pos.z*3]   = windSpeedIn[pos.x][pos.y][pos.z*3];
        windSpeedOut[pos.x][pos.y][pos.z*3+1]   = windSpeedIn[pos.x][pos.y][pos.z*3+1];
        windSpeedOut[pos.x][pos.y][pos.z*3+2]   = windSpeedIn[pos.x][pos.y][pos.z*3+2];
    }
}

//http://zebu.uoregon.edu/disted/ph162/images/greenbalance.gif
//thanks google
extern "C"
__global__ void solarHeating() {}
extern "C"
__global__ void infraredCooling() {}
extern "C"
__global__ void calcWind(
		//static
		int*     worldSize,     float*   worldSpeed,        float**  elevation,
		//inputs
		float*   worldTimeIn,   float*** temperatureIn,     float*** pressureIn,        float*** humidityIn,
		float*** windSpeedIn,
		//outputs
		float*   worldTimeOut, float*** windSpeedOut
	) { //end of args...
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1] * worldSize[2];

	if (i<n) {
		dim3 pos = getWorldCoords(i, worldSize); //x is latitude in return result
//		if(i==0){
//			worldTimeOut[0] = worldTimeIn[0] + worldSpeed[0];
//			worldTimeOut[1] = worldTimeIn[1] + worldSpeed[1];
//		}
		//temperatureIn[pos.x][pos.y][pos.z];
		//temperatureIn[pos.x][pos.y][pos.z];
		//humidityIn[pos.x][pos.y][pos.z];
		float  curVol  = volumeAt(worldSize, pos.x, pos.z);
		float  curPres = pressureIn[pos.x][pos.y][pos.z];
		//float* curWind = windSpeedIn[pos.x][pos.y][pos.z];
		float  curTemp  = temperatureIn[pos.x][pos.y][pos.z];
		float  curHumid = humidityIn[pos.x][pos.y][pos.z];

		vec3 target {0,0,0};
		for(int dlat = -1; dlat<=1; dlat++){
			for(int dlon = -1; dlon<=1; dlon++){
				for(int dalt = -1; dalt<=1; dalt++){
					int manhtDist = abs(dlat)+abs(dlon)+abs(dalt);
					if(manhtDist==0) continue;

					int al = pos.z+dalt;
					if(al < 0) continue;
					if(al >= worldSize[2]) continue;


					int la = pos.x+dlat;
					int lo = pos.y+dlon;
					dim3 wrapped = wrapCoords(la, lo, worldSize);
					la = wrapped.x;
					lo = wrapped.y;

					float distanceFactor = manhtDist==1? 1 :
							 (manhtDist==2? sqrt2 : sqrt3) *surfaceDistance(pos.x, pos.y, pos.z, la, lo, al, worldSize) * 1000; //to meters

					float presDiff = curPres - pressureIn[la][lo][al];
					float volFactor = 1;//curVol / volumeAt(worldSize, la, al); //seemed like a good idea at the time, causes small amounts of force in a balanced system
					float tempDiff = curTemp - temperatureIn[la][lo][al];

					float forceMag = presDiff * volFactor * worldSpeed[2];
					forceMag /= distanceFactor;
					target.x += dlat * forceMag * worldSpeed[2];
					target.y += dlon * forceMag * worldSpeed[2];
					target.z += (dalt * forceMag + tempDiff) * worldSpeed[2];//warm air above
					//TODO forces from other pushing / pulling pressures
					//pressure at altitudes
					//ground type friction

				}
			}
		}
		float mass = airMass(curVol, curTemp, curHumid);
		//f = ma
		//a = f/m

		windSpeedOut[pos.x][pos.y][pos.z*3  ] = target.x;//windSpeedIn[pos.x][pos.y][pos.z][0] + target.x / mass;
		windSpeedOut[pos.x][pos.y][pos.z*3+1] = target.y;//windSpeedIn[pos.x][pos.y][pos.z][1] + target.y / mass;
		windSpeedOut[pos.x][pos.y][pos.z*3+2] = target.z;//windSpeedIn[pos.x][pos.y][pos.z*3+2] + target.z / mass;
	}
}
