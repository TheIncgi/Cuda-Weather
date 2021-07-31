package app;

import java.io.Serializable;
import java.util.Optional;
import java.util.Random;

import app.util.MathUtils;
import app.util.Vector3f;

public class GlobeData implements Serializable {
	/**
	 * generated value 
	 */
	private static final long serialVersionUID = 4823288204349570589L;
	
	private boolean initalized = false;
	/**Time is stored as {world rotation, world revolution}<br>
	 * where {1,2} is 1 day after the second year*/
	public final float[] time;
	/**[Latitude][longitude][altitude]*/
	public final float[][][] temp, humidity, cloudCover, pressure;
	/**[Latitude][longitude][altitude]*/
	public final float[][][] windSpeed;
	/**[Latitude][longitude]*/
	public final float[][]    snowCover, groundMoisture, elevation;
	/**[Latitude][longitude]<br>
	 * Matches the {@link GlobeData.GroundType} enum*/
	public final int[][] groundType;
	/**[Latitude][longitude] - only calculated for ground level*/
	public final float[][] percipitation;
	/**[Latitude][longitude] - only calculated for ground level<br>
	 * matches the {@link GlobeData.PrecipitationType} enum*/
	public final int[][]   precipitationType;
	
	public final transient float worldTilt = 23.5f;
	
	public final int LATITUDE_DIVISIONS;
	public final int LONGITUDE_DIVISIONS;
	public final int ALTITUDE_DIVISIONS;
	
	public static transient final float SIMULATION_HEIGHT_KM = 12;
	public static transient final float SIMULATION_DEPTH_KM  = -.5f;
	public static transient final float MAX_RAIN_CLOUD_HEIGHT_KM = 8;
	public static transient final float EARTH_RADIUS_KM = 6371;
	
	
	
	/**
	 * Example: 12587, 25174, 50
	 * 5700, 2857,
	 * longitude should be 2x latitude, 180 degrees vs 360
	 * */
	public GlobeData(int latitudeDivisions, int longitudeDivisions, int altitudeDivisions) {
		this.LATITUDE_DIVISIONS = latitudeDivisions;
		this.LONGITUDE_DIVISIONS = longitudeDivisions;
		this.ALTITUDE_DIVISIONS = altitudeDivisions;
		time = new float[] {0,0};
		temp = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions];
		pressure = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions];
		humidity = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions];
		cloudCover = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions];
		windSpeed = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions*3];
		snowCover = new float[latitudeDivisions][longitudeDivisions];
		groundMoisture = new float[latitudeDivisions][longitudeDivisions];
		groundType = new int[latitudeDivisions][longitudeDivisions];
		elevation = new float[latitudeDivisions][longitudeDivisions];
		percipitation = new float[latitudeDivisions][longitudeDivisions];
		precipitationType = new int[latitudeDivisions][longitudeDivisions];
	}
	public GlobeData(GlobeData copySize) {
		this(copySize.LATITUDE_DIVISIONS, copySize.LONGITUDE_DIVISIONS, copySize.ALTITUDE_DIVISIONS);
	}
	
	/**Return the lower bound for this altitude index*/
	public float altitudeLowerBound(int index) {
		return index/(float)ALTITUDE_DIVISIONS * (SIMULATION_HEIGHT_KM - SIMULATION_DEPTH_KM) - SIMULATION_DEPTH_KM;
	}
	/**Return the lower bound for this altitude index*/
	public float altitudeUpperBound(int index) {
		return (index+1)/(float)ALTITUDE_DIVISIONS * (SIMULATION_HEIGHT_KM - SIMULATION_DEPTH_KM) - SIMULATION_DEPTH_KM;
	}
	
	
	public Optional<Integer> indexOfAltitude(float km) {
		if(km < SIMULATION_DEPTH_KM) return Optional.empty();
		if(km >= SIMULATION_HEIGHT_KM) return Optional.empty();
		return Optional.of(indexOfAltitudeDirect(km));
	}
	public int indexOfAltitudeDirect(float km) {
		return (int) MathUtils.map(km, SIMULATION_DEPTH_KM, SIMULATION_HEIGHT_KM, 0, ALTITUDE_DIVISIONS);
	}
	
	/**Corner of region*/
	private Vector3f pointOf(int lat, int lon, int alt) {
		float yaw = 360f * lon / LONGITUDE_DIVISIONS;
		float pitch = 180f * lat / LATITUDE_DIVISIONS;
		Vector3f p = new Vector3f(EARTH_RADIUS_KM + altitudeLowerBound(alt), 0, 0);
		p.rotateAboutY((float) Math.toRadians(pitch));
		p.rotateAboutZ((float) Math.toRadians(yaw));
		return p;
	}
	/**
	 * Returns area in squared KM<br>
	 * Area is identical at all longitudes
	 * */
	public float areaAtLowerBound(int latitude, int altitude) {
		Vector3f a = pointOf(latitude, 0, altitude);
		Vector3f b = pointOf(latitude, 1, altitude);
		Vector3f c = pointOf(latitude+1, 0, altitude);
		Vector3f d = pointOf(latitude+1, 1, altitude);
		return MathUtils.areaTrapazoid(a.distance(b), c.distance(d), a.distance(c)/*Aprox*/);
	}
	/**
	 * Returns volume in KM cubed for a single grid with a height of 1 altitude division<br>
	 * Volume is identical at all longitudes
	 * */
	public float volumeAt(int latitude, int altitude) {
		return volumeAt(latitude, altitude, altitude+1);
	}
	/**
	 * Returns volume in KM cubed<br>
	 * Volume is identical at all longitudes
	 * */
	public float volumeAt(int latitude, int altitudeLow, int altHi) {
		float a = areaAtLowerBound(latitude, altitudeLow);
		float b = areaAtLowerBound(latitude, altHi);
		float h = altitudeLowerBound(altHi) - altitudeLowerBound(altitudeLow);
		return (a+b)/2 * h;
	}
	/**
	 * Returns volume in KM cubed<br>
	 * Volume is identical at all longitudes<br>
	 * This returns the the volume for all altitude divisions summed
	 * */
	public float volumeAt(int latitude) {
		return volumeAt(latitude, 0, ALTITUDE_DIVISIONS);
	}
	/**
	 * {@link Deprecated} this calculation will be completed on the GPU for consistant results on
	 * FX viewer and Headless simulation.
	 * */
	@Deprecated
	public float precipitationChanceAt(int latitude, int longitude) {
		int altLimit = indexOfAltitudeDirect(GlobeData.MAX_RAIN_CLOUD_HEIGHT_KM);
		altLimit = MathUtils.clamp(altLimit, 0, ALTITUDE_DIVISIONS);
		float chance = 1;
		for (int alti = 0; alti < altLimit; alti++) {
			float c = cloudCover[latitude][longitude][alti]*(humidity[latitude][longitude][alti]>.99?1:0); //more than 99% humidity
			chance *= 1-c;
		}
		return 1-chance;
	}
	
	public float getSnowCoveragePercent(int latitude, int longitude) {
		return Math.min(1, snowCover[latitude][longitude]/2 ); //2cm of snow is 100%
	}
	public float getSnowDepthCM(int latitude, int longitude) {
		return snowCover[latitude][longitude];
	}
	
	public GlobeData random() {return random(new Random().nextLong());}
	@Deprecated
	public GlobeData random(long seed) {
		Random r = new Random(seed);
		for(int lat = 0; lat<LATITUDE_DIVISIONS; lat++) {
			for(int lon = 0; lon<LONGITUDE_DIVISIONS; lon++) {
				groundType[lat][lon] = r.nextInt(GroundType.values().length);
				snowCover[lat][lon] = r.nextFloat()>.75? r.nextFloat()*5 : 0;
				groundMoisture[lat][lon] = r.nextFloat();
				elevation[lat][lon] = r.nextFloat() * 3000; //meters
				
				for(int al = 0; al<ALTITUDE_DIVISIONS; al++) {
					temp[lat][lon][al] = r.nextFloat()*95;
					humidity[lat][lon][al] = r.nextFloat();
					cloudCover[lat][lon][al] = r.nextFloat()>.5? r.nextFloat() : 0;
					pressure[lat][lon][al] = (r.nextFloat()-.5f)*.2f+1;
					windSpeed[lat][lon][al*3] = r.nextFloat()*r.nextFloat()*.1f -.05f ;
					windSpeed[lat][lon][al*3+1] = r.nextFloat()*r.nextFloat()*.1f -.05f;
					windSpeed[lat][lon][al*3+2] = r.nextFloat()*r.nextFloat()*.05f -.025f;
					
					
				}
				percipitation[lat][lon] = precipitationChanceAt(lat, lon);
				precipitationType[lat][lon] = PrecipitationType.NONE.ordinal();
			}
		}
		return this;
	}
	
	public String getTime(float longitude) {
		float x = (time[0] + MathUtils.clamp(longitude, 0, 360)/360) * 24 * 60 ;
		int min  = (int) x % 60;
		x-= min;
		x /= 60;
		int hour = (int) x % 24;
		x-= hour;
		x/=24;
		boolean isAM = hour < 12;
		hour -= isAM? 0 : 12;
		int day = (int) x;
		
		float ofYear = time[1] % 1;
		int year = (int) time[1];
		
		if(hour==0) hour = 12;
		return String.format("%2d:%02d %s, Day: %d (%4.1f%%) Year: ", hour, min, isAM?"AM":"PM", day, ofYear, year);
	}
	
	public GlobeData generateTerrain(long seed) {
		return random(); //TODO
	}
	
	public enum GroundType{
		SAND, //0
		DIRT, //1
		OCEAN,//2
		GRASS,//3
		STONE,//4
		ICE,  //5
		FOREST,//6
		LAKE  //7
	}
	
	public enum PrecipitationType {
		NONE,
		RAIN,
		THUNDER,
		SLEET,
		SNOW,
		HAIL;
	}

	public long groundCells() {
		return LATITUDE_DIVISIONS * (long)LONGITUDE_DIVISIONS;
	}

	public long totalCells() {
		return groundCells() * ALTITUDE_DIVISIONS;
	}
	
	public static long byteSizeIf(int lon, int lat, int alt) {
		return ((long)lon)*lat*alt*9*4;
	}
	
	public boolean isInitalized() {
		return initalized;
	}
	public void markInitalized() {
		this.initalized = true;
	}
}
