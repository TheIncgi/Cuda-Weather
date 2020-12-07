package app.util;

import java.util.function.Consumer;

import app.CudaUtils;
import app.GlobeData;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import static app.CudaUtils.*;

public class Simulator {
	GlobeData in, out;
	Consumer<GlobeData> onStepComplete;
	private float[] worldTime;
	
	private CUfunction[] funcs;
	private CUdeviceptr worldSizePtr;
	private CUdeviceptr worldTimePtr;
	//private CUdeviceptr[][] argPointers; //kernal, ordinal
	private CudaInt2[]   groundType;
	private CudaFloat2[] groundMoisture, snowCover, elevation;
	private CudaFloat3[] temperature, pressure, humidity, cloudCover;
	private CudaFloat4[] windSpeed;
	private Pointer[] kernalParams;
	private int activeKernal = 0;
	private boolean dataLoaded = false;
	
	public Simulator(GlobeData world, float planetRot, float planetRev) {
		in = world;
		
		CUmodule module = loadModule("WeatherSim.ptx");
		funcs = new CUfunction[] {
				getFunction(module, "identity")
		};
		
		worldSizePtr = loadToGPU( new int[] { //constant
				world.latitudeDivisions, 
				world.longitudeDivisions, 
				world.altitudeDivisions
		});
		
		
		worldTime = new float[] {planetRot, planetRev};
		worldTimePtr = new CUdeviceptr();
		
		kernalParams = new Pointer[2];
		
		
		CUdeviceptr[] paramsA = new CUdeviceptr[10];
		CUdeviceptr[] paramsB = new CUdeviceptr[10];
		int i = 0;
		
		paramsA[i] = paramsB[i++] = worldSizePtr;
		
		paramsA[i] = paramsB[i++] = worldTimePtr;
		
		paramsA[i]   = groundType[0].getThePointer();
		paramsB[i++] = groundType[1].getThePointer();
		
		paramsA[i]   = groundMoisture[0].getThePointer();
		paramsB[i++] = groundMoisture[1].getThePointer();
		
		paramsA[i]   = snowCover[0].getThePointer();
		paramsB[i++] = snowCover[1].getThePointer();
		
		paramsA[i]   = elevation[0].getThePointer();
		paramsB[i++] = elevation[1].getThePointer();
		
		paramsA[i]   = temperature[0].getThePointer();
		paramsB[i++] = temperature[1].getThePointer();
		
		paramsA[i]   = pressure[0].getThePointer();
		paramsB[i++] = pressure[1].getThePointer();
		
		paramsA[i]   = humidity[0].getThePointer();
		paramsB[i++] = humidity[1].getThePointer();
		
		paramsA[i]   = cloudCover[0].getThePointer();
		paramsB[i++] = cloudCover[1].getThePointer();
		
		paramsA[i]   = windSpeed[0].getThePointer();
		paramsB[i++] = windSpeed[1].getThePointer();
		
		
//		for(int i =0; i<argPointers[0].length; i++) {
//			paramsA[2+i] = argPointers[0][i];
//			paramsB[2+i] = argPointers[1][i];
//		}
		
		kernalParams[0] = Pointer.to(paramsA);
		kernalParams[1] = Pointer.to(paramsB);
		
	}
	
	/**
	 * Loads input globe data into active param pointers
	 * */
	private void pushData() {
		groundType		[activeKernal].push( in.groundType		);
		groundMoisture	[activeKernal].push( in.groundMoisture  );
		snowCover		[activeKernal].push( in.snowCover		);
		elevation		[activeKernal].push( in.elevation		);
		temperature		[activeKernal].push( in.temp			);
		pressure		[activeKernal].push( in.pressure		);
		humidity		[activeKernal].push( in.humidity		);
		cloudCover		[activeKernal].push( in.cloudCover		);
		windSpeed		[activeKernal].push( in.windSpeed		);
	}
	
	private void pullResult() {
		groundType		[1-activeKernal].pull( out.groundType		);
		groundMoisture	[1-activeKernal].pull( out.groundMoisture   );
		snowCover		[1-activeKernal].pull( out.snowCover		);
		elevation		[1-activeKernal].pull( out.elevation		);
		temperature		[1-activeKernal].pull( out.temp				);
		pressure		[1-activeKernal].pull( out.pressure			);
		humidity		[1-activeKernal].pull( out.humidity			);
		cloudCover		[1-activeKernal].pull( out.cloudCover		);
		windSpeed		[1-activeKernal].pull( out.windSpeed		);
	}
	
	public void timeStep() {
		if(!dataLoaded) {
			pushData();
			dataLoaded = true;
		}
		Pointer kernal = kernalParams[activeKernal];
		
		//sun warming
		//infrared radiative cooling
		//rates should be balanced
		//pressure changes->wind
		//fronts -> temp change -> percipitation
		//moisture -> humdidity
		//rainfall
		//water accumulation -> lakes/rivers
		//terrain alterations? dry: grass->desert,  warm: ice->water
		
		
	}
	
	public void setOnStepComplete(Consumer<GlobeData> onStepComplete) {
		this.onStepComplete = onStepComplete;
	}
	
	
	/*
	 * init:
	 * load world into pointer set a
	 * 
	 * timestep
	 * b is calculated
	 * 
	 * on complete run with results from b
	 * next time step may be run imediatly 
	 * pointers are swapped for input output each step
	 * 
	 * while results are read to the application they can also be used to calculate
	 * the next state
	*/
}
