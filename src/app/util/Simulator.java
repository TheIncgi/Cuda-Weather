package app.util;

import java.util.function.Consumer;

import app.CudaUtils;
import app.GlobeData;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

import static app.CudaUtils.*;

public class Simulator {
	GlobeData in;
	Runnable onStepComplete, onResultReady;
	
	float timeStepSizeSeconds          = 5;
	float worldRotationRatePerStep     = timeStepSizeSeconds / (24*60*60) ;
	float worldRevolutionRatePerStep   = timeStepSizeSeconds / (24*60*60*365);
	private Triplet<CUfunction, Boolean, Pointer[]>[] funcs;
	//constant
	private CUdeviceptr worldSizePtr;
	private CudaFloat1   worldSpeed; //not changed by CUfunction
	private CudaFloat1[] worldTimePtr = new CudaFloat1[2];
	private CudaInt2   groundType;
	private CudaFloat2 elevation;    //not changed
	private CudaFloat2[] groundMoisture = new CudaFloat2[2],
			             snowCover      = new CudaFloat2[2];
	private CudaFloat3[] temperature = new CudaFloat3[2], 
			             pressure    = new CudaFloat3[2], 
			             humidity    = new CudaFloat3[2], 
			             cloudCover  = new CudaFloat3[2];
	private CudaFloat4[] windSpeed = new CudaFloat4[2];
	
	//private Pointer[] kernalParams = new Pointer[2];
	private int activeKernal = 0;
	private boolean dataLoaded = false;
	
	public Simulator(GlobeData world) {
		in = world;
		//out = new GlobeData(in);
		initPtrs();
		
		setupFuncs();
	}
	
	@SuppressWarnings("unchecked")
	private void setupFuncs() {
		CUmodule module = loadModule("WeatherSim.ptx");
		Pointer worldSize = Pointer.to(worldSizePtr);
		funcs = new Triplet[] {
				new Triplet<>(getFunction(module, "copy"), true, new Pointer[] {
						Pointer.to(
							worldSize,
							worldSpeed       .getArgPointer(),
							elevation        .getArgPointer(),
							groundType       .getArgPointer(),
							worldTimePtr  [0].getArgPointer(),
							groundMoisture[0].getArgPointer(),
							snowCover     [0].getArgPointer(),
							temperature   [0].getArgPointer(),
							pressure      [0].getArgPointer(),
							humidity      [0].getArgPointer(),
							cloudCover    [0].getArgPointer(),
							windSpeed     [0].getArgPointer(),
							worldTimePtr  [1].getArgPointer(),
							groundMoisture[1].getArgPointer(),
							snowCover     [1].getArgPointer(),
							temperature   [1].getArgPointer(),
							pressure      [1].getArgPointer(),
							humidity      [1].getArgPointer(),
							cloudCover    [1].getArgPointer(),
							windSpeed     [1].getArgPointer()
						),Pointer.to(
								worldSize,
								worldSpeed       .getArgPointer(),
								elevation        .getArgPointer(),
								groundType       .getArgPointer(),
								worldTimePtr  [1].getArgPointer(),
								groundMoisture[1].getArgPointer(),
								snowCover     [1].getArgPointer(),
								temperature   [1].getArgPointer(),
								pressure      [1].getArgPointer(),
								humidity      [1].getArgPointer(),
								cloudCover    [1].getArgPointer(),
								windSpeed     [1].getArgPointer(),
								worldTimePtr  [0].getArgPointer(),
								groundMoisture[0].getArgPointer(),
								snowCover     [0].getArgPointer(),
								temperature   [0].getArgPointer(),
								pressure      [0].getArgPointer(),
								humidity      [0].getArgPointer(),
								cloudCover    [0].getArgPointer(),
								windSpeed     [0].getArgPointer()
								)
				}),
				new Triplet<>(getFunction(module, "calcWind"), true, new Pointer[] {
					Pointer.to(new Pointer[] {
							worldSize,
							worldSpeed     .getArgPointer(),
							elevation      .getArgPointer(),
							
							worldTimePtr[0].getArgPointer(),
							temperature [0].getArgPointer(),
							pressure    [0].getArgPointer(),
							humidity    [0].getArgPointer(),
							windSpeed   [0].getArgPointer(),
							
							worldTimePtr[1].getArgPointer(),
							windSpeed   [1].getArgPointer()
					}),Pointer.to(new Pointer[] {
							worldSize,
							worldSpeed.getArgPointer(),
							elevation.getArgPointer(),
							
							worldTimePtr[1].getArgPointer(),
							temperature [1].getArgPointer(),
							pressure    [1].getArgPointer(),
							humidity    [1].getArgPointer(),
							windSpeed   [1].getArgPointer(),
							
							worldTimePtr[0].getArgPointer(),
							windSpeed   [0].getArgPointer()
					})
				})
		};
	}

	private void initPtrs() {
		worldSizePtr = loadToGPU( new int[] { //constant
				in.latitudeDivisions, 
				in.longitudeDivisions, 
				in.altitudeDivisions
		});
		worldSpeed = new CudaFloat1(3);
		elevation = new CudaFloat2(in.latitudeDivisions, in.longitudeDivisions);
		groundType = new CudaInt2(in.latitudeDivisions, in.longitudeDivisions);
		
		worldTimePtr 		= new CudaFloat1[] { new CudaFloat1(2), new CudaFloat1(2) };
		groundMoisture[0] 	= new CudaFloat2(in.latitudeDivisions, in.longitudeDivisions);
		groundMoisture[1] 	= new CudaFloat2(in.latitudeDivisions, in.longitudeDivisions);
		snowCover[0] 		= new CudaFloat2(in.latitudeDivisions, in.longitudeDivisions);
		snowCover[1] 		= new CudaFloat2(in.latitudeDivisions, in.longitudeDivisions);
		temperature[0] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions);
		temperature[1] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions);
		pressure[0] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions);
		pressure[1] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions);
		humidity[0] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions);
		humidity[1] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions);
		cloudCover[0] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions);
		cloudCover[1] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions);
		windSpeed[0] 		= new CudaFloat4(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions, 3);
		windSpeed[1] 		= new CudaFloat4(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions, 3);
		
	}

	/**
	 * Loads input globe data into active param pointers
	 * */
	private synchronized void pushData() {
		worldSpeed                    .push(new float[] {
										worldRotationRatePerStep, 
										worldRevolutionRatePerStep,
										timeStepSizeSeconds
		});
		elevation		              .push( in.elevation		);
		groundType		              .push( in.groundType		);
		worldTimePtr    [activeKernal].push( in.time            );
		groundMoisture	[activeKernal].push( in.groundMoisture  );
		snowCover		[activeKernal].push( in.snowCover		);
		
		temperature		[activeKernal].push( in.temp			);
		pressure		[activeKernal].push( in.pressure		);
		humidity		[activeKernal].push( in.humidity		);
		cloudCover		[activeKernal].push( in.cloudCover		);
		windSpeed		[activeKernal].push( in.windSpeed		);
	}
	
	private void pullResult() {
		worldTimePtr    [1-activeKernal].pull( in.time             );
		groundMoisture	[1-activeKernal].pull( in.groundMoisture   );
		snowCover		[1-activeKernal].pull( in.snowCover		);
		temperature		[1-activeKernal].pull( in.temp				);
		pressure		[1-activeKernal].pull( in.pressure			);
		humidity		[1-activeKernal].pull( in.humidity			);
		cloudCover		[1-activeKernal].pull( in.cloudCover		);
		windSpeed		[1-activeKernal].pull( in.windSpeed		);
	}
	
	public synchronized void setWorldSpeed(float secondsPerTimeStep, float hoursInDay, float daysInYear) {
		timeStepSizeSeconds          = 5;
		worldRotationRatePerStep     = timeStepSizeSeconds / (24*60*60) ;
		worldRevolutionRatePerStep   = timeStepSizeSeconds / (24*60*60*365);
		worldSpeed                    .push(new float[] {
				worldRotationRatePerStep, 
				worldRevolutionRatePerStep
		});
	}
	
	/**Begins calculation of next timestep<br>
	 * Results will be ready before this emthod unblocks
	 * returns number of milliseconds elapsed
	 * */
	public synchronized long timeStep() {
		long start = System.currentTimeMillis();
		if(!dataLoaded) {
			pushData();
			dataLoaded = true;
		}
		
		//sun warming
		//infrared radiative cooling
		//rates should be balanced
		//pressure changes->wind
		//fronts -> temp change -> percipitation
		//moisture -> humdidity
		//rainfall
		//water accumulation -> lakes/rivers
		//terrain alterations? dry: grass->desert,  warm: ice->water
		int blockSizeX = 256;
		long gridSizeX_withAtmosphere = (long)Math.ceil((double)(in.totalCells()) / blockSizeX);
		long gridSizeX_groundOnly     = (long)Math.ceil((double)(in.groundCells()) / blockSizeX);
		for (Triplet<CUfunction, Boolean, Pointer[]> step : funcs) {
			CUfunction f = step.a;
			double blocksNeeded = step.b? gridSizeX_withAtmosphere : gridSizeX_groundOnly;
			
			int dimLimit = 65535;
			int dim = (int) Math.ceil(Math.pow(blocksNeeded, 1/3d));
			if(dim > dimLimit) throw new RuntimeException("Too many blocks required for simulation ("+dim+"^3 vs limit 65535^3)");
			JCudaDriver.cuLaunchKernel(f,       
				//CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
			    dim,  dim, dim,      // Grid dimension 
			    blockSizeX, 1, 1,      // Block dimension
			    0, null,               // Shared memory size and stream 
			    step.c[activeKernal], null // Kernel- and extra parameters
			); 
			
			JCudaDriver.cuCtxSynchronize();
			
		}
		pullResult(); //read from input side while also computing
		onResultReady.run();
		
		activeKernal = 1-activeKernal;
		
		if(onStepComplete!=null)
			onStepComplete.run();
		long end = System.currentTimeMillis();
		return end-start;
	}
	
	/**
	 * Called when the timestep has completed computation for the next result<br>
	 * if waiting for a result, instead use onResultReady
	 * */
	public void setOnStepComplete(Runnable onStepComplete) {
		this.onStepComplete = onStepComplete;
	}
	/**
	 * Called when the result from the timestep is available<br>
	 * This may be triggered while the next timestep is being calculated
	 * */
	public void setOnResultReady(Runnable onResultReady) {
		this.onResultReady = onResultReady;
	}
	/**It's really the current input, but the old output becomes the new input*/
	public GlobeData getStepResult() {
		return in;
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
	 * 
	 * 
	 * Pointer[] paramsA = new Pointer[20];
		Pointer[] paramsB = new Pointer[20];
		int i = 0;
		
		//static
		paramsA[i] = paramsB[i++] = Pointer.to(worldSizePtr);
		paramsA[i] = paramsB[i++] = Pointer.to(worldSpeed.getThePointer());
		paramsA[i] = paramsB[i++] = Pointer.to(elevation.getThePointer());
		paramsA[i] = paramsB[i++] = Pointer.to(groundType.getThePointer());

		//in
		paramsA[i]   = Pointer.to(worldTimePtr[0].getThePointer());
		paramsB[i++] = Pointer.to(worldTimePtr[1].getThePointer());
		
		paramsA[i]   = Pointer.to(groundMoisture[0].getThePointer());
		paramsB[i++] = Pointer.to(groundMoisture[1].getThePointer());
		
		paramsA[i]   = Pointer.to(snowCover[0].getThePointer());
		paramsB[i++] = Pointer.to(snowCover[1].getThePointer());
		
		paramsA[i]   = Pointer.to(temperature[0].getThePointer());
		paramsB[i++] = Pointer.to(temperature[1].getThePointer());
		
		paramsA[i]   = Pointer.to(pressure[0].getThePointer());
		paramsB[i++] = Pointer.to(pressure[1].getThePointer());
		
		paramsA[i]   = Pointer.to(humidity[0].getThePointer());
		paramsB[i++] = Pointer.to(humidity[1].getThePointer());
		
		paramsA[i]   = Pointer.to(cloudCover[0].getThePointer());
		paramsB[i++] = Pointer.to(cloudCover[1].getThePointer());
		
		paramsA[i]   = Pointer.to(windSpeed[0].getThePointer());
		paramsB[i++] = Pointer.to(windSpeed[1].getThePointer());
		
		//out
		paramsA[i]   = Pointer.to(worldTimePtr[1].getThePointer());
		paramsB[i++] = Pointer.to(worldTimePtr[0].getThePointer());
		
		paramsA[i]   = Pointer.to(groundMoisture[1].getThePointer());
		paramsB[i++] = Pointer.to(groundMoisture[0].getThePointer());
		
		paramsA[i]   = Pointer.to(snowCover[1].getThePointer());
		paramsB[i++] = Pointer.to(snowCover[0].getThePointer());
				
		paramsA[i]   = Pointer.to(temperature[1].getThePointer());
		paramsB[i++] = Pointer.to(temperature[0].getThePointer());
		
		paramsA[i]   = Pointer.to(pressure[1].getThePointer());
		paramsB[i++] = Pointer.to(pressure[0].getThePointer());
		
		paramsA[i]   = Pointer.to(humidity[1].getThePointer());
		paramsB[i++] = Pointer.to(humidity[0].getThePointer());
		
		paramsA[i]   = Pointer.to(cloudCover[1].getThePointer());
		paramsB[i++] = Pointer.to(cloudCover[0].getThePointer());
		
		paramsA[i]   = Pointer.to(windSpeed[1].getThePointer());
		paramsB[i++] = Pointer.to(windSpeed[0].getThePointer());
	*/
}