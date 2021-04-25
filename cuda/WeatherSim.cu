#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include "math.h"
#include "constants.cu"
#include "common.cu"
#include "vectors.cu"

//Notes about functions:
//temperature is in fahrenheit
//mass is in kg except for maxWaterHeld which is in grams per cubic meter
//wind speed is in meters per second
//wind speed is used to calculate momentum




//math

__device__ dim3 getWorldCoords(int gThreadID, int* worldSize){
	int latitude =  gThreadID % worldSize[0];
	gThreadID = gThreadID / worldSize[0];
	int longitude = gThreadID % worldSize[1];
	gThreadID = gThreadID / worldSize[1];
	int altitude  = gThreadID % worldSize[2];
	return dim3(latitude, longitude, altitude);
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
__device__ float FtoC(float f){
	return (f-32) * 5/9;
}

/**aprox air density at temp kg per meter cubed
 * https://www.desmos.com/calculator/5ziykdrgdq
 * */
__device__ float airDensity(float tempF, float relHumid, float pressureAtmos){
	float tempC = FtoC(tempF);
	float tempK = tempC + 273.15;
	float pSat = 10 * 610.78 * pow(10.0, (7.5*tempC)/(tempC+237.3) ); //desmos #10
	float pv = relHumid - pSat;
	float pd = 101325 * pressureAtmos - pv; //to Pa

	return clamp( (pd * 0.0289654 + pv * 0.018016) / (8.314 * tempK), 0.5, 1.5); //formula is only accurate in -10 to 50 c, clamping before values get crazy
}

//https://www.desmos.com/calculator/gzqcdksdhs
//https://www.engineeringtoolbox.com/maximum-moisture-content-air-d_1403.html
/**returns the maximum grams of water held by 1 cubic meter of air at some given temp*/
__device__ float maxWaterHeld(float temp){
	temp = FtoC(temp); //TODO this had CtoF, switched to FtoC, checkme
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
__device__ float airMass(float kmCubed, float temp, float relHumid, float pressureAtmos){
	float mass = airDensity(temp, relHumid, pressureAtmos) * kmCubed; //mass of air
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

	return clamp(dot(sun,p), 0, 1); //praise the sun
}

//https://en.wikipedia.org/wiki/Albedo#/media/File:Albedo-e_hg.svg
__device__ float groundReflectance(int groundType, float moisture){
	float r = .3; //default value
	switch(groundType){
	case 0:  /*sand  */   r=.45;  break;
	case 1:  /*dirt  */   r=.30;  break;
	case 2:  /*ocean */   r=.75;  break;
	case 3:  /*grass */   r=.15;  break;
	case 4:  /*stone */   r=.20;  break; //not in resource, best guess
	case 5:  /*ice   */   r=.35;  break;
	case 6:  /*forest*/   r=.10;  break;
	case 7:  /*lake  */  r=.85;  break; //made this slightly brighter than the ocean
	}
	return map(clamp(moisture,0,1), 0, 1, r, LAKE);//fully/over saturated areas form reflective puddles/flood zones
}
__device__ float specificHeatAir(float temp) {
	return map(temp, CtoF(0), 70,.7171,  1); //TODO verify, not really a proper formula
}
__device__ float specificHeatTerrain(int ground, int moisture){
	switch(ground){
		case 0:   /*sand  */  return map(clamp(moisture, 0, 1), 0, 1, .2, 3);
		case 1:   /*dirt  */  return map(clamp(moisture, 0, 1), 0, 1, .8, 2.52); //dry soil v wet mud
		case 2:   /*ocean */  return 4.3; //overestimating for ocean depth or something
		case 3:   /*grass */  return map(clamp(moisture, 0, 1), 0, 1, 1.8, 4); //
		case 4:   /*stone */  return .8;  //sandstone or something https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html
		case 5:   /*ice   */  return 2.04;
		case 6:   /*forest*/  return map(clamp(moisture, 0, 1), 0, 1, 2.4, 4);  //birch wood, plants have lots of water, kinda have to guess a bit
		case 7:   /*lake  */  return 4.18;
	}
	return 1;
}
__device__ float biomeMass( int ground, int moisture){
	//surface area * .2m depth //another unclear thing
	//TODO verify/alter mass used for heat storage
	float kgPerCm = 1;
	switch(ground){
		case 0:  /*sand  */  kgPerCm = map(clamp(moisture, 0, 1), 0, 1, 1500, 1900);
		case 1:  /*dirt  */  kgPerCm = map(clamp(moisture, 0, 1), 0, 1, 1200, 1700); //dry soil v wet mud
		case 2:  /*ocean */  kgPerCm = 1000; //overestimating for ocean depth or something
		case 3:  /*grass */  kgPerCm = map(clamp(moisture, 0, 1), 0, 1, 1400, 1700); // more guestimates
		case 4:  /*stone */  kgPerCm = 2560;  //limestone https://www.engineeringtoolbox.com/dirt-mud-densities-d_1727.html
		case 5:  /*ice   */  kgPerCm = 920;
		case 6:  /*forest*/  kgPerCm = map(clamp(moisture, 0, 1), 0, 1, 1700, 1800);  //guestimates,
		case 7:  /*lake  */  kgPerCm = 1000;
	}
	return kgPerCm;
}
//https://www.desmos.com/calculator/jut6bbsuw7
//https://www.eng-tips.com/viewthread.cfm?qid=260422
//temp is provided as F
//cp is provided in J/KG C
//
__device__ float tempChange(float currentTemp, float seconds, float watts, float mass, float cp){
	currentTemp = FtoC(currentTemp);
	return CtoF( (seconds*watts)/(mass*cp) + currentTemp );
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
__global__ void solarHeating(int* worldSize, float* worldTime, float* worldSpeed, float*** pressure, float*** cloudCover, float*** humidity, int** groundType, float** elevation, float** snowCover, float** groundMoisture, float*** temperatureIn, float*** temperatureOut) {
	//1,360 watts per square meter

	//TODO consider changing rates based on air pressure/density
	// 8% backscatter from air                  30%      27%
	// 19% absorbed by humidity dust and o3     70%

	//  4% absorbed by clouds                   20%      21% avg cloud cover 67%
	// 17% reflected from clouds                80%
	//

	// 46% absorbed by surface                  88.5%    52%
	//  6% reflected by surface                 11.5%
	//if(true) return;
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1] * worldSize[2];
	dim3 pos = getWorldCoords(i, worldSize);
	if (i>=n) return;
	int lat = pos.x;
	int lon = pos.y;


	const float SUN_WATTS = 1360;
	float sun = sunshine(lat, lon, worldSize, worldTime);
	//float rawSunshine = sun; //debug value
	float scatterLoss = pow(.92, 1/(worldSize[2])); //8% loss over any world size after all altitude steps
	float humidityAbsorbtion = pow(.81, 1/worldSize[2]);
	for(int alt = worldSize[2]-1; alt>=0; alt--){
		float appliedWatts = 0;
		float volume = volumeAt(worldSize, lat, alt);
		float surface = surfaceAreaAt(worldSize, lat, alt);
		float depth  = altitudeOfIndex(alt+1, worldSize) - altitudeOfIndex(alt, worldSize);
		float cloud = cloudCover[lat][lon][alt];
		float mass = airMass(volume, temperatureIn[lat][lon][alt], humidity[lat][lon][alt], pressure[lat][lon][alt]);

		//.5 is a guesstimate, no easily located data on cloud light absorbtion per km of depth
		//value is on about a 1/3 of light making it to the ground if under a storm cloud and such clouds being around a KM thick
		//value may change as simulations results differ

		sun *= scatterLoss; //after 12km this should be about 8%

		float humidityAbsorbtionFactor =  1-((1-humidityAbsorbtion) * humidity[lat][lon][alt] * (1-cloud)); //not doubling up on absorbtion of clouds
		//when humidity    is 0, this factor is 1,
		//when cloud cover is 1, this factor is 1
		sun *= humidityAbsorbtionFactor;

		//TODO heat air based on 1360 watts * humidityAbsorbtion
		appliedWatts += (1-humidityAbsorbtionFactor) * SUN_WATTS;

		float cloudPassthru = 1-( (1-pow((float).5, depth)) * cloud); //light still sent down further
		float cloudRedirection = 1-cloudPassthru;    // bounced or absorbed
		float cloudAlbedo = map(clamp(altitudeOfIndex(alt, worldSize), 2, 6),2,6, .5, .8);
		float cloudAbsorbtion = sun * (1-cloudAlbedo) * cloudRedirection;        //TODO checkme: adjusted based on altitude https://en.wikipedia.org/wiki/Albedo#/media/File:Albedo-e_hg.svg
		//float cloudReflection = sun * cloudAlbedo * cloudRedirection; unused reflection amount
		sun *= cloudPassthru;

		///TODO heat clouds based on 1360 watts * cloudAbsorbtion
		appliedWatts += cloudAbsorbtion  * SUN_WATTS;
		float cp = specificHeatAir(temperatureIn[lat][lon][alt]);
		bool brk = false;
		if(alt==0 || altitudeOfIndex(alt, worldSize)*1000 <= elevation[lat][lon]){ //altitude is in KM, elevation is in meters!
			int ground = groundType[lat][lon];
			float snow = snowCover[lat][lon];
			float moisture = groundMoisture[lat][lon];
			float reflect = map(clamp(snow,0,1),0,1, groundReflectance(ground, moisture), .8);
			float groundAbsorbtion = 1-reflect;

			//TODO heat ground/low atmosphere by groundAbsorbtion * 1360 watts
			appliedWatts += sun * groundAbsorbtion * SUN_WATTS;
			mass += biomeMass(ground, moisture);
			cp = specificHeatTerrain(ground, moisture);
			brk = true;
		}

		if(mass < 0)
			temperatureOut[lat][lon][alt] = -95525;
		else
			temperatureOut[lat][lon][alt] = tempChange(temperatureIn[lat][lon][alt], worldSpeed[2], appliedWatts, mass, cp);

		if(brk){
			for(int g=alt; g>=0; g--)
				temperatureOut[lat][lon][g] = temperatureOut[lat][lon][alt]; //set inground temp for ease of viewing in UI
			break;
		}
	}

}
extern "C"
__global__ void infraredCooling(int* worldSize, float* worldTime, float* worldSpeed, float*** pressure, int** groundType, float **elevation, float** groundMoisture, float*** humidity, float*** tempIn, float*** tempOut, float*** cloudCover, float** snowCover) {
	int i = getGlobalThreadID();
	int n = worldSize[0] * worldSize[1] * worldSize[2];
	dim3 pos = getWorldCoords(i, worldSize);
	if (i>=n) return;
	int lat = pos.x;
	int lon = pos.y;
	int alt = pos.z;

	const float SUN_WATTS = 1360;
	//float GROUND_IR    = SUN_WATTS * .09;
	float AIR_IR = .60 / worldSize[2];//pow(.6, 1/(worldSize[2]));

	float vol = volumeAt(worldSize, lat, alt);
	float mass = airMass(vol, tempIn[lat][lon][alt], humidity[lat][lon][alt], pressure[lat][lon][alt]);
	float wattsIR = AIR_IR * SUN_WATTS;
	float cp = specificHeatAir(tempIn[lat][lon][alt]);
	if(alt==0 || altitudeOfIndex(alt, worldSize)*1000 <= elevation[lat][lon]){
		mass = biomeMass(groundType[lat][lon], groundMoisture[lat][lon]);
		wattsIR += SUN_WATTS * .09;
		cp = specificHeatTerrain(groundType[lat][lon], groundMoisture[lat][lon]);
	}

	tempOut[lat][lon][alt] = tempChange(tempIn[lat][lon][alt], worldSpeed[2], -wattsIR, mass, cp);
}
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
				for(int dalt = -1; dalt<=1; dalt++){ //TODO ignore altitudes that are literally in the ground
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
		float mass = airMass(curVol, curTemp, curHumid, curPres);
		//f = ma
		//a = f/m

		windSpeedOut[pos.x][pos.y][pos.z*3  ] = target.x;//windSpeedIn[pos.x][pos.y][pos.z][0] + target.x / mass;
		windSpeedOut[pos.x][pos.y][pos.z*3+1] = target.y;//windSpeedIn[pos.x][pos.y][pos.z][1] + target.y / mass;
		windSpeedOut[pos.x][pos.y][pos.z*3+2] = target.z;//windSpeedIn[pos.x][pos.y][pos.z*3+2] + target.z / mass;
	}
}
