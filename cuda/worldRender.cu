#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include "common.cu"
#include "vectors.cu"
#include "perlin.cu"
#include "colorBlend.cu"
#include<stdint.h>
#include<string>
#include "font.cu"

//[x bits - map mode][2 bits - effects/layers]
int __constant__          LIGHT_EFFECT =   1;
int __constant__         CLOUD_OVERLAY =   2;
int __constant__          SNOW_OVERLAY =   4;
int __constant__       THERMAL_OVERLAY =   8;
int __constant__      HUMIDITY_OVERLAY =  16;
int __constant__          WIND_OVERLAY =  32;
int __constant__    SNOW_COVER_OVERLAY =  64;
int __constant__ PERCIPITATION_OVERLAY = 128;
int __constant__     ELEVATION_OVERLAY = 256;
//wind, humidity, snow, pressure, rain, ground moisture, precipitation type



__device__ vec4 sunshineColorArgb( float lat, float lon, int* worldSize, float* worldTime ) {
	float s = sunshine(lat, lon, worldSize, worldTime);
	vec4 in;
	in.x = 240;//h
	in.y = clamp(1-s,0,.5); //s
	in.z = max(s, .3); //v
	in.w = 1; //a
	return hsvToArgb(in);
};

__device__ vec4 thermalColor( float degreesF, float opacity ) {
	//hue 66% deep blue
	//50% bright blue
	//33% green
	//15% yellow
	// 0%  red
	vec4 in;
	
	in.y = 1; //s
	in.z = 1; //v

	if(isnan(degreesF) || isinf(degreesF)) {
		in.x = 310;
	}else{
		degreesF = clamp(degreesF, -40, 120);

		if( degreesF < 50)
			in.x = map( degreesF, -40, 50, 241,180);
		else if( degreesF < 72 )
			in.x = map( degreesF, 50, 71, 180, 124);
		else if( degreesF < 85)
			in.x = map( degreesF, 71, 85, 124, 60);
		else
			in.x = map( degreesF, 85, 100, 60, 0);
		if( 100 < degreesF )
			in.y = map( degreesF, 100, 120, 1, .5);
	}
	
	in.w = opacity;
	return hsvToArgb(in);
}

__device__ vec4 humidityColor( float humidity, float opacity ) {
	// hue 183 degrees - sky blue
	// hue 241 degrees - deep blue
	vec4 color;
	color.x = clampedMap( humidity, 0, 1, 183, 241 );
	color.y = clamp( humidity, 0, 1);
	color.z = 1;
	color.w = opacity;
	return color;
}

//in meters
__device__ vec4 elevationColor( float elevation, float opacity ) {
	vec4 color;
	color.x = 0;
	color.y = 0;
	color.z = clampedMap( elevation, -500, 4000, 0 ,1 );
	return color;
}
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
__device__ void applyThermalOverlay(
	int  i,
	int lat, int lon,
	int* worldSize,
	float*** temperatureIn,
	int*   imageSize,
	int* imageOut,
	int elevation
	) {
	vec4 color = thermalColor(temperatureIn[lat][lon][elevation], .6f);
	imageOut[i] = argbToHex( mixColors( hexToArgb(imageOut[i]), color ) );
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
		int* imageOut,
		int overlayFlags,
		FontData &fontData,
		int* mousePos){

		int x = i % imageSize[0];
		int y = i / imageSize[0];
		char strBuf[100];
		char strBuf2[100];

//		int cx = imageSize[0]/2;
//		int cy = imageSize[1]/2;
		int lat    = y *         worldSize[0]  / imageSize[1];
		int lon    = x *         worldSize[1]  / imageSize[0];
		float latf = y * ((float)worldSize[0]) / imageSize[1] + .5;
		float lonf = x * ((float)worldSize[1]) / imageSize[0] + .5;

		int   mouseLat  =         mousePos[1]  * worldSize[0] / imageSize[1];
		int   mouseLon  =         mousePos[0]  * worldSize[1] / imageSize[0];
		float mouseLatf = ((float)mousePos[1]) * worldSize[0] / imageSize[1];
		float mouseLonf = ((float)mousePos[0]) * worldSize[1] / imageSize[0];

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
		
		bool useOverlayValue = false;
		float overlayValue = 0;
		int snapLat = ((int)(lat/3))*3;
		int snapLon = ((int)(lon/3))*3;
		dim2 snapped = wrapCoords(snapLat+2, snapLon + 2, worldSize);
		{
			vec4 v4c  = hexToArgb(theColor);
			vec4 overlayColor = hexToArgb(0);
			overlayColor.w = 0;

			if(hasFlag(overlayFlags, THERMAL_OVERLAY)){
				overlayValue = temperatureIn[snapped.x][snapped.y][0];
				overlayColor = thermalColor( overlayValue, .6 );
				useOverlayValue = true;

			}else if(hasFlag(overlayFlags, HUMIDITY_OVERLAY)) {
				overlayValue = humidityIn[lat][lon][0];
				overlayColor = humidityColor( overlayValue, .6 );
				useOverlayValue = true;

			}else if(hasFlag(overlayFlags, WIND_OVERLAY)) {

			}else if(hasFlag(overlayFlags,SNOW_COVER_OVERLAY)){

			}else if(hasFlag(overlayFlags, PERCIPITATION_OVERLAY)){

			}else if(hasFlag(overlayFlags, ELEVATION_OVERLAY)) {
				overlayValue = elevation[lat][lon];
				overlayColor = elevationColor( overlayValue, .6 );
				useOverlayValue =- true;
			}
			v4c = mixColors( v4c, overlayColor );
			theColor = argbToHex(v4c);
		

			if(useOverlayValue){
				int groupX = x - snapLon * tileWidth;
				int groupY = y - snapLat * tileHeight;
				floatToStr( overlayValue, strBuf, 2 );
				// strBuf[0] = 'G';
				// strBuf[1] = '\0';
				if(getFontStringPixel(fontData, strBuf, (int)(groupX*1.5), groupY) >= 0)
					v4c = mixColors( v4c,  hexToArgb(0x88000000) );
				theColor = argbToHex(v4c);
			}
		}
		if(hasFlag(overlayFlags, LIGHT_EFFECT)) //sunshine
			theColor = argbToHex(multipyColor(hexToArgb(theColor), sunColor));
		else{
			float f = sunshine(latf, lonf, worldSize, worldTimeIn);
			if(0.2 <= f && f <=.225)
				theColor = argbToHex(mixColors( hexToArgb(theColor),hexToArgb(0x88FF0000)));
		}

		const char* str = "It's a good day if we render some text :D";
		if(getFontStringPixel(fontData, str, x-1,y-1) >= 0)
			theColor = 0xFFFFFFFF;
		if(getFontStringPixel(fontData, str, x+1,y+1) >= 0)
			theColor = 0xFFFFFFFF;
		if(getFontStringPixel(fontData, str, x+1,y-1) >= 0)
			theColor = 0xFFFFFFFF;
		if(getFontStringPixel(fontData, str, x-1,y+1) >= 0)
			theColor = 0xFFFFFFFF;
		if(getFontStringPixel(fontData, str, x,y) >= 0)
			theColor = 0xFF000000;

		
		
		intToStr( overlayFlags , strBuf, 16 );
		const char* label = "Flags: ";
		concat( label, strBuf, strBuf2, 100);
		if(getFontStringPixel(fontData, strBuf2, x,y-32) >= 0)
			theColor = 0xFFFF0000;

		label = "Rev: ";
		floatToStr( worldTimeIn[1], strBuf, 5 ); //rev
		concat( label, strBuf, strBuf2, 100);
		if(getFontStringPixel(fontData, strBuf2, x,y-64) >= 0)
			theColor = 0xFFFF0000;

		label = "Rot: ";
		floatToStr( worldTimeIn[0], strBuf, 5 ); //rot
		concat( label, strBuf, strBuf2, 100);
		if(getFontStringPixel(fontData, strBuf2, x,y-96) >= 0)
			theColor = 0xFFFF0000;

		{
			label = "Biome: ";
			concat( label, strBuf, strBuf2, 100);
			if(getFontStringPixel(fontData, biomeName( groundType[mouseLat][mouseLon] ), x,y-128) >= 0)
				theColor = 0xFFFF0000;
		}

		{
			label = "Sun: ";
			floatToStr( sunshine(mouseLat, mouseLon, worldSize, worldTimeIn), strBuf, 5 ); //rot
			concat( label, strBuf, strBuf2, 100);
			if(getFontStringPixel(fontData, strBuf2, x,y-160) >= 0)
				theColor = 0xFFFF0000;
		}

		{
			label = "Elevation: ";
			floatToStr( elevation[mouseLat][mouseLon], strBuf, 5 ); //rot
			concat( label, strBuf, strBuf2, 100);
			if(getFontStringPixel(fontData, strBuf2, x,y-192) >= 0)
				theColor = 0xFFFF0000;
		}
		if(distance(mousePos[0],mousePos[1], x,y) < 3){
			theColor = mixColors_hex( theColor, 0xFFFFFFFF, .3 );
			// theColor = 0xFFFFFFFF;
		}

		//
		//imageOut[i] = 0xFF00FFFF;
		imageOut[i] = theColor;//mixColors(theColor, blendColor, .5*distanceToEdge(lat, lon, latf, lonf));
//		if(lat == 47)
//			fragColor = 0xFFFF00FF;
//		fragColor = 0xFF000000 | ((int)(lat/((float)worldSize[0])*255));
//		fragColor |= ((int)(lon/((float)worldSize[1])*255)) << 8;

//		if(lat < 0 )
//			fragColor = 0xFFFF0000;
//		if(lat >= worldSize[0])
//			fragColor = 0xFFFF0000;

//		applyThermalOverlay( i,
//			lat, lon,
//			worldSize,
//			temperatureIn,
//			imageSize,
//			imageOut,
//			0 //min elevation
//			);



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
			int* imageOut,
			int* overlayFlags,
			uint8_t* font,
			int* mousePos
			) {
	int i = getGlobalThreadID();
	if(i>= imageSize[0]*imageSize[1]) return;

	FontData fontData;
	loadFont(font, fontData);

	renderFlat(i, worldSize, elevation, groundType, worldTimeIn, groundMoistureIn, snowCoverIn, temperatureIn, pressureIn, humidityIn, cloudCoverIn, windSpeedIn, imageSize, imageOut, *overlayFlags, fontData, mousePos);
}