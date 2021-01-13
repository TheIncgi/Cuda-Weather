#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

//TODO make this a shared header

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
//
//
//

extern "C"
__global__ void render(
		//static
			int*     worldSize,
			//float*   worldSpeed,
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

			int*   imageSize,
			int* imageOut
			) {
	int n = getGlobalThreadID();
	if(n > imageSize[0]*imageSize[1]) return;
	int x = n % imageSize[0];
	int y = n / imageSize[1];

	int cx = imageSize[0]/2;
	int cy = imageSize[1]/2;

	float d = pow( (double)(cx-x)*(cx-x) + (cy-y)*(cy-y), .5 );
	int fragColor = ((int)clamp(d/40, 0, 1) * 255) | 0xFF000000;

	imageOut[n] = fragColor;

}
