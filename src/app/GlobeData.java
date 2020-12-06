package app;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Optional;
import java.util.Random;

import app.util.MathUtils;
import app.util.Pair;
import app.util.Vector3f;
import jcuda.driver.CUdeviceptr;

public class GlobeData {
	/**[Latitude][logitude][altitude]*/
	public final float[][][] temp, humidity, cloudCover, pressure;
	/**[Latitude][logitude][altitude]*/
	public final float[][][][] windSpeed;
	/**[Latitude][logitude]*/
	public final float[][]    snowCover, groundMoisture, elevation;
	/**[Latitude][logitude]*/
	public final int[][] groundType;
	
	public final int latitudeDivisions;
	public final int longitudeDivisions;
	public final int altitudeDivisions;
	
	public static final float SIMULATION_HEIGHT_KM = 12;
	public static final float SIMULATION_DEPTH_KM  = -.5f;
	public static final float MAX_RAIN_CLOUD_HEIGHT_KM = 8;
	public static final float EARTH_RADIUS_KM = 6371;
	
	public GlobeData(int latitudeDivisions, int longitudeDivisions, int altitudeDivisions) {
		this.latitudeDivisions = latitudeDivisions;
		this.longitudeDivisions = longitudeDivisions;
		this.altitudeDivisions = altitudeDivisions;
		temp = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions];
		pressure = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions];
		humidity = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions];
		cloudCover = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions];
		windSpeed = new float[latitudeDivisions][longitudeDivisions][altitudeDivisions][3];
		snowCover = new float[latitudeDivisions][longitudeDivisions];
		groundMoisture = new float[latitudeDivisions][longitudeDivisions];
		groundType = new int[latitudeDivisions][longitudeDivisions];
		elevation = new float[latitudeDivisions][longitudeDivisions];
	}
	
	/**Return the lower bound for this altitude index*/
	public float altitudeLowerBound(int index) {
		return index/(float)altitudeDivisions * (SIMULATION_HEIGHT_KM - SIMULATION_DEPTH_KM) - SIMULATION_DEPTH_KM;
	}
	/**Return the lower bound for this altitude index*/
	public float altitudeUpperBound(int index) {
		return (index+1)/(float)altitudeDivisions * (SIMULATION_HEIGHT_KM - SIMULATION_DEPTH_KM) - SIMULATION_DEPTH_KM;
	}
	
	
	public Optional<Integer> indexOfAltitude(float km) {
		if(km < SIMULATION_DEPTH_KM) return Optional.empty();
		if(km >= SIMULATION_HEIGHT_KM) return Optional.empty();
		return Optional.of(indexOfAltitudeDirect(km));
	}
	public int indexOfAltitudeDirect(float km) {
		return (int) MathUtils.map(km, SIMULATION_DEPTH_KM, SIMULATION_HEIGHT_KM, 0, altitudeDivisions);
	}
	
	/**Corner of region*/
	private Vector3f pointOf(int lat, int lon, int alt) {
		float yaw = 360f * lon / longitudeDivisions;
		float pitch = 180f * lat / latitudeDivisions;
		Vector3f p = new Vector3f(EARTH_RADIUS_KM + altitudeLowerBound(alt), 0, 0);
		p.rotateAboutY((float) Math.toRadians(pitch));
		p.rotateAboutZ((float) Math.toRadians(yaw));
		return p;
	}
	
	public float areaAtLowerBound(int latitude, int altitude) {
		Vector3f a = pointOf(latitude, 0, altitude);
		Vector3f b = pointOf(latitude, 1, altitude);
		Vector3f c = pointOf(latitude+1, 0, altitude);
		Vector3f d = pointOf(latitude+1, 1, altitude);
		return MathUtils.areaTrapazoid(a.distance(b), c.distance(d), a.distance(c)/*Aprox*/);
	}
	public float volumeAt(int latitude, int altitude) {
		return volumeAt(latitude, altitude, altitude+1);
	}
	public float volumeAt(int latitude, int altitudeLow, int altHi) {
		float a = areaAtLowerBound(latitude, altitudeLow);
		float b = areaAtLowerBound(latitude, altHi);
		float h = altitudeLowerBound(altHi) - altitudeLowerBound(altitudeLow);
		return (a+b)/2 * h;
	}
	public float volumeAt(int latitude) {
		return volumeAt(latitude, 0, altitudeDivisions);
	}
	public float percipitationChanceAt(int latitude, int longitude) {
		int altLimit = indexOfAltitudeDirect(GlobeData.MAX_RAIN_CLOUD_HEIGHT_KM);
		altLimit = MathUtils.clamp(altLimit, 0, altitudeDivisions);
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
	public GlobeData random(long seed) {
		Random r = new Random(seed);
		for(int lat = 0; lat<latitudeDivisions; lat++) {
			for(int lon = 0; lon<longitudeDivisions; lon++) {
				groundType[lat][lon] = r.nextInt(GroundType.values().length);
				snowCover[lat][lon] = r.nextFloat()>.75? r.nextFloat()*5 : 0;
				groundMoisture[lat][lon] = r.nextFloat();
				elevation[lat][lon] = r.nextFloat() * 3000; //meters
				for(int al = 0; al<altitudeDivisions; al++) {
					temp[lat][lon][al] = r.nextFloat()*95;
					humidity[lat][lon][al] = r.nextFloat();
					cloudCover[lat][lon][al] = r.nextFloat()>.5? r.nextFloat() : 0;
					pressure[lat][lon][al] = r.nextFloat()+1;
					windSpeed[lat][lon][al][0] = r.nextFloat()*r.nextFloat()*110 -55 ;
					windSpeed[lat][lon][al][1] = r.nextFloat()*r.nextFloat()*110 -55;
					windSpeed[lat][lon][al][2] = r.nextFloat()*r.nextFloat()*10 -5;
					
					
				}
			}
		}
		return this;
	}
	
	public GlobeData generateTerrain(long seed) {
		return random(); //TODO
	}
	
	public enum GroundType{
		SAND,
		DIRT,
		OCEAN,
		GRASS,
		STONE,
		ICE,
		FOREST,
		LAKE
	}
}
