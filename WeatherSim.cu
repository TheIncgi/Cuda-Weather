#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

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

extern "C"
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
        pressureOut[pos.x][pos.y][pos.z]    = temperatureIn[pos.x][pos.y][pos.z];
        humidityOut[pos.x][pos.y][pos.z]    = humidityIn[pos.x][pos.y][pos.z];
        cloudCoverOut[pos.x][pos.y][pos.z]  = cloudCoverIn[pos.x][pos.y][pos.z];
        windSpeedOut[pos.x][pos.y][pos.z]   = windSpeedIn[pos.x][pos.y][pos.z];
    }
}
