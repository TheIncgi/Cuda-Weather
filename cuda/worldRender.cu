#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include "common.cu"
#include "vectors.cu"
#include "perlin.cu"

__device__ vec4 sunshineColorArgb( float lat, float lon, int* worldSize, float* worldTime ) {
	float s = sunshine(lat, lon, worldSize, worldTime);
	vec4 in;
	in.x = 35;//h
	in.y = 1-s*s; //s
	in.z = max(s, .3); //v
	in.w = 1; //a
	return hsvToArgb(in);
};
// |~~~~~~~~~~~~~~~~~~~~~
// |     Render code
// |~~~~~~~~~~~~~~~~~~~~~

__device__ float distanceToEdge(int a, int b, float af, float bf){
	float da = af - a;
	float db = bf - b;
	float thresh = .6;
	return min(1.0, max( 0.0, max( abs(thresh-da), abs(thresh-db) ) ));
}
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

			vec4 hsvColor = argbToHsv(hexToArgb(fragColor));
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
		int lat    = y *         worldSize[0]  / imageSize[1];
		int lon    = x *         worldSize[1]  / imageSize[0];
		float latf = y * ((float)worldSize[0]) / imageSize[1] + .5;
		float lonf = x * ((float)worldSize[1]) / imageSize[0] + .5;

		float tileWidth  = imageSize[0] / (float)worldSize[1];
		float tileHeight = imageSize[1] / (float)worldSize[0];

		float offX = perlin(latf, lonf, .25, 999, 45, worldSize);
		float offY = perlin(latf, lonf, .25, 27, 98, worldSize);

		int theColor = colorAt(lat, lon, groundType, groundMoistureIn);
		int blendColor = colorAt(
				((int)(latf + offX)) % worldSize[0],
				((int)(lonf + offY)) % worldSize[1],
				groundType,
				groundMoistureIn
		);
		offX = perlin(latf, lonf, .1, 7798, 45, worldSize);
		offY = perlin(latf, lonf, .1, 991, 98, worldSize);

		blendColor = mixColors_hex(blendColor, colorAt(
				((int)(latf + offX)) % worldSize[0],
				((int)(lonf + offY)) % worldSize[1],
				groundType,
				groundMoistureIn), .5);

		int PERL = ((int)( (offX+1)*127 )) + 0xFF000000 ;

		vec4 sunColor = sunshineColorArgb(latf, lonf, worldSize, worldTimeIn);
		theColor = argbToHex(multipyColor(hexToArgb(theColor), sunColor));
		imageOut[i] = theColor;//mixColors(theColor, blendColor, .5*distanceToEdge(lat, lon, latf, lonf));
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
