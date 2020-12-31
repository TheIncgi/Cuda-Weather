package app.spring.api;

import static app.spring.SpringAPI.headlessSimulation;

import java.util.Arrays;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import app.GlobeData;
import app.GlobeData.GroundType;
import app.GlobeData.PercipitationType;
import app.spring.api.states.GroundWeatherState;

@RestController
@RequestMapping("api")
public class Status {
	
	//TODO cache results between simulation batches to reduce read-lock time
	
	@GetMapping("worldSize")
	public int[] worldSize() {
		GlobeData gd = headlessSimulation().getData(); //no synchronization because these values are final
		return new int[] { gd.LATITUDE_DIVISIONS, gd.LONGITUDE_DIVISIONS, gd.ALTITUDE_DIVISIONS };
	}
	
	/**
	 * Returns the world time at 0 degrees latitude<br>
	 * [0] - % of day completed
	 * [1] - % of year completed
	 * */
	@GetMapping("worldTime")
	public float[] worldTime() {
		GlobeData gd = headlessSimulation().getData();
		headlessSimulation().readLock();
		float[] out = Arrays.copyOf(gd.time, gd.time.length);
		headlessSimulation().readUnlock();
		return out;
	}
	
	//it's expected that this method should have some 'hotspot' locations where towns may be located
	//so only a small number of locations should need to be saved
	/**Supplies all the weather information needed for rendering of an area*/
	@GetMapping("weatherAt")
	public GroundWeatherState weatherAt( 
			@RequestParam(name = "lat", required = true                     ) int lat, 
			@RequestParam(name = "lon", required = true                     ) int lon,
			@RequestParam(name = "alt", required = false, defaultValue = "0") int alt) {
		GlobeData gd = headlessSimulation().getData();
		
		headlessSimulation().readLock();
		
		float temperature 		= gd.temp[lat][lon][alt];
		float humidity    		= gd.humidity[lat][lon][alt];
		float[] totalCloudCover = gd.cloudCover[lat][lon];
		float pressure 			= gd.pressure[lat][lon][alt];  
		float[] windSpeed 		= new float[] {    //data was interleaved for faster transfer
				gd.windSpeed[lat][lon][alt * 3    ],
				gd.windSpeed[lat][lon][alt * 3 + 1],
				gd.windSpeed[lat][lon][alt * 3 + 2],
		};
		float snowCover						= gd.snowCover[lat][lon];     //values under 2 indicate % coverage (out of 2), this value is also the depth in cm
		float groundMoisture    			= gd.groundMoisture[lat][lon];//values over 1 indicate flooding
		float percipitationChance 			= gd.percipitation[lat][lon];
		PercipitationType percipitationType = PercipitationType.values()[gd.percipitationType[lat][lon]]; //uses ordinal
		GroundType groundType    			= GroundType.values()[gd.groundType[lat][lon]];
		
		headlessSimulation().readUnlock();
		
		GroundWeatherState gws = new GroundWeatherState(temperature, humidity, totalCloudCover, pressure, windSpeed, 
				snowCover, groundMoisture, percipitationChance, percipitationType, groundType);
		return gws;
	}
}
