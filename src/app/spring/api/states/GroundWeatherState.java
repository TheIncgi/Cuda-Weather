package app.spring.api.states;

import app.GlobeData.GroundType;
import app.GlobeData.PrecipitationType;

/**
 * This class is used to store the weather state at a single point in the world (for use with Spring API)
 * */
public class GroundWeatherState {
	float temperature;
	float humidity;
	/**
	 * Cloud coverage is provided for all altitudes 
	 * */
	float[] cloudCover;
	float pressure;
	float[] windSpeed;
	float snowCover;
	float groundMoisture;
	float percipitationChance;
	PrecipitationType percipitationType;
	GroundType groundType; //can change based on climate
	
	/**time is at longitude 0*/
	float globalTime;
	float localTime;
	float year;
	
	
	//generated constructor
	public GroundWeatherState(float temperature, float humidity, float[] cloudCover, float pressure,
			float[] windSpeed, float snowCover, float groundMoisture, float percipitationChance, 
			PrecipitationType percipitationType, GroundType groundType, float globalTime, float localTime, float year) {
		super();
		this.temperature 			= temperature;
		this.humidity 				= humidity;
		this.cloudCover 			= cloudCover;
		this.pressure 				= pressure;
		this.windSpeed 				= windSpeed;
		this.snowCover 				= snowCover;
		this.groundMoisture 		= groundMoisture;
		this.percipitationChance 	= percipitationChance;
		this.percipitationType  	= percipitationType;
		this.groundType 			= groundType;
		this.globalTime 			= globalTime;
		this.localTime 				= localTime;
		this.year 					= year;
	}
	
	//generated getters
	public float getTemperature() {
		return temperature;
	}
	public float getHumidity() {
		return humidity;
	}
	public float[] getCloudCover() {
		return cloudCover;
	}
	public float getPressure() {
		return pressure;
	}
	public float[] getWindSpeed() {
		return windSpeed;
	}
	public float getSnowCover() {
		return snowCover;
	}
	public float getGroundMoisture() {
		return groundMoisture;
	}
	public GroundType getGroundType() {
		return groundType;
	}
	public float getPercipitationChance() {
		return percipitationChance;
	}
	public PrecipitationType getPercipitationType() {
		return percipitationType;
	}
	
	public float getGlobalTime() {
		return globalTime;
	}
	public float getLocalTime() {
		return localTime;
	}
	public float getYear() {
		return year;
	}
}
