package app;

import jcuda.driver.CUdeviceptr;

public class GlobeData {
	public final float[/*latitude*/][/*longitude*/][/*altitude*/] temp, humidity, cloudCover, pressure;
	public final float[/*latitude*/][/*longitude*/][/*altitude*/][/*xyz*/] windSpeed;
	public final float[/*latitude*/][/*longitude*/]               snowCover, groundMoisture;
	public final GroundType[][] groundType;
	public final int latitudeDivisions;
	public final int longitudeDivisions;
	public final int altitudeDivisions;
	
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
		groundType = new GroundType[latitudeDivisions][longitudeDivisions];
	}
	
	enum GroundType{
		SAND,
		DIRT,
		OCEAN,
		GRASS,
		STONE,
		ICE,
		FOREST
	}
}
