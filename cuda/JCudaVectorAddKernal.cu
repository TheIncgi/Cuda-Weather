extern "C"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

__global__ void add(int n, float *a, float *b, float *sum)
{
	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	int blockID = blockIdx.x +
				  blockIdx.y * gridDim.x +
				  blockIdx.z * gridDim.x * gridDim.y;
	int threadInBlock = threadIdx.x +
						threadIdx.y * blockDim.x +
						threadIdx.z * blockDim.x * blockDim.y;
    int i = threadInBlock + blockID * threadsPerBlock;
    if (i<n)
    {
        sum[i] = a[i] + b[i];
    }
}
