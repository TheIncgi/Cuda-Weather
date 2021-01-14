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
struct vec4{
	/**R*/
	float x;
	/**G*/
	float y;
	/**B*/
	float z;
	/**A*/
	float w;
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
///////////// Perlin ///////////////// //TODO make this into a sep file, share with Terrain gen

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
//
// |~~~~~~~~~~~~~~~~~~~~~
// |    Color utils
// |~~~~~~~~~~~~~~~~~~~~~
__device__ vec4 hsvToArgb(vec4 in) {
	//h s v
	//r g b
	//x y z
	float c = in.z * in.y;
	float x = c * (1 - abs(fmod(in.x / 60.0 , 2.0) -1));
	float m = in.z - c;
	vec4 out;
	out.w = in.w;

	if(in.x < 60) {
		out.x = c;
		out.y = x;
	}else if(in.x < 120){
		out.x = x;
		out.y = c;
	}else if(in.x <180){
		out.y = c;
		out.z = x;
	}else if(in.x < 240){
		out.y = x;
		out.z = c;
	}else if(in.x < 300){
		out.x = x;
		out.z = c;
	}else{
		out.x = c;
		out.z = x;
	}
	return out;
}
__device__ vec4 argbToHsv(vec4 in){
	vec4 out;
	float cmax = max(max(in.x, in.y), in.z);
	float cmin = min(min(in.x, in.y), in.z);
	float delta = cmax - cmin;
	float h = 0;
	//rgb
	//xyz
	if( delta==0 ){
		h = 0;
	}else if( cmax == in.x ){
		h = fmod(60.0 * ((in.y - in.z) / delta) , 6.0);
	}else if( cmax == in.y ){
		h = 60 * ((in.z - in.x) / delta) + 2;
	}else if( cmax == in.z ){
		h = 60 * ((in.x - in.y) / delta) + 4;
	}

	out.x = h;
	if(cmax==0)
		out.y = 0;
	else
		out.y = delta/cmax;
	out.z = cmax;
	out.w = in.w;
	return out;
}
__device__ int argbToHex(vec4 in){
	return
		(((int)(in.w * 255)) << 24) |
		(((int)(in.x * 255)) << 16) |
		(((int)(in.y * 255)) <<  8) |
		(((int)(in.z * 255))      ) ;
}
__device__ vec4 hexToRgb(int hex){
	vec4 out;
	out.w = ( hex >> 24 ) & 0xFF;
	out.x = ( hex >> 16 ) & 0xFF;
	out.y = ( hex >>  8 ) & 0xFF;
	out.z = ( hex       ) & 0xFF;
	return out;
}
__device__ int mixColors(int a, int b, float bAmount) {
	vec4 i = hexToRgb(a);
	vec4 j = hexToRgb(b);
	vec4 t;
	float aAmount = 1-bAmount;
	//https://youtu.be/LKnqECcg6Gw?t=185
	t.x = sqrt( ((i.x * i.x) * aAmount) + ((j.x * j.x) * bAmount));
	t.y = sqrt( ((i.y * i.y) * aAmount) + ((j.y * j.y) * bAmount));
	t.z = sqrt( ((i.z * i.z) * aAmount) + ((j.z * j.z) * bAmount));
	t.w = sqrt( ((i.w * i.w) * aAmount) + ((j.w * j.w) * bAmount));
	return argbToHex(t);
}
// |~~~~~~~~~~~~~~~~~~~~~
// |     Render code
// |~~~~~~~~~~~~~~~~~~~~~

__device__ int colorAt(int lat, int lon, int** groundType, float** groundMoistureIn){
	int fragColor = 0xFF000000;
	int gType = groundType[lat][lon];
	if(gType==ICE){
				fragColor = 0xFFFFFFFF;
			}else if(gType==SAND){
				fragColor = 0xFFFFEBCD;
			}else if(gType==DIRT){
				fragColor = 0xFFBC8F8F;
			}else if(gType==OCEAN){
				fragColor = 0xFF4169E1;
			}else if(gType==GRASS){
				fragColor = 0xFF90EE90;
			}else if(gType==STONE){
				fragColor = 0xFFD3D3D3;
			}else if(gType==FOREST){
				fragColor = 0xFF228B22;
			}else if(gType==LAKE){
				fragColor = 0xFF00FFFF;
			}

			vec4 hsvColor = argbToHsv(hexToRgb(fragColor));
			hsvColor.z *= map( clamp(groundMoistureIn[lat][lon],0,1), 0 ,1, 1, .75 );

			return fragColor;
}

//star background
__device__ int  backgroundColor(int n, int*imgSize){
	return 0;
}
__device__ void renderFlat(
		int i,
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
		int* imageOut){

		int x = i % imageSize[0];
		int y = i / imageSize[0];

//		int cx = imageSize[0]/2;
//		int cy = imageSize[1]/2;
		int lat = y * (worldSize[0]) / (imageSize[1]);
		int lon = x * worldSize[1] / imageSize[0];
		float latf = y * (float)worldSize[0] / imageSize[1] + .5;
		float lonf = x * (float)worldSize[1] / imageSize[0] + .5;

		float offX = perlin(latf, lonf, .5, 999, 45, worldSize);
		float offY = perlin(latf, lonf, .5, 27, 98, worldSize);

		int theColor = colorAt(lat, lon, groundType, groundMoistureIn);
		int blendColor = colorAt(
				(lat + sign(offX)) % worldSize[0],
				(lon + sign(offY)) % worldSize[1],
				groundType,
				groundMoistureIn
		);

		//FIXME awesome blending
		imageOut[i] = theColor;//mixColors(theColor, blendColor, distance((float)lat, (float)lon, latf, lonf));
//		if(lat == 47)
//			fragColor = 0xFFFF00FF;
//		fragColor = 0xFF000000 | ((int)(lat/((float)worldSize[0])*255));
//		fragColor |= ((int)(lon/((float)worldSize[1])*255)) << 8;

//		if(lat < 0 )
//			fragColor = 0xFFFF0000;
//		if(lat >= worldSize[0])
//			fragColor = 0xFFFF0000;





		//imageOut[i] = fragColor;//, 0, 0xFFFFFFFF);
}

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
	int i = getGlobalThreadID();
	if(i>= imageSize[0]*imageSize[1]) return;

	renderFlat(i, worldSize, elevation, groundType, worldTimeIn, groundMoistureIn, snowCoverIn, temperatureIn, pressureIn, humidityIn, cloudCoverIn, windSpeedIn, imageSize, imageOut);

}
