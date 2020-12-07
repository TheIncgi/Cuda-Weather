extern "C"
__global__ void add(
	int* worldSizeIn,
	float* worldTimeIn, 
	float* groundMoistureIn, 
	float* snowCoverIn,
	float* elevationIn,
	float* temperatureIn,
	float* pressureIn,
	float* humidityIn,
	float* cloudCoverIn,
	float* windSpeedIn,
	
	float* groundMoistureIn, 
	float* snowCoverIn,
	float* elevationIn,
	float* temperatureIn,
	float* pressureIn,
	float* humidityIn,
	float* cloudCoverIn,
	float* windSpeed
	)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        sum[i] = a[i] + b[i];
    }
}