package app.util;

import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import app.CudaUtils;
import app.GlobeData;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

import static app.CudaUtils.*;

public class Simulator implements AutoCloseable{
	GlobeData in;
	Runnable onStepComplete, onResultReady;
	
	float timeStepSizeSeconds          = 5;
	float worldRotationRatePerStep     = timeStepSizeSeconds / (24*60*60) ;
	float worldRevolutionRatePerStep   = timeStepSizeSeconds / (24*60*60*365);
	private Triplet<CUfunction, Boolean, Pointer> atmosphereInit;
	private Step[] funcs;
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
	private CudaFloat3[] windSpeed = new CudaFloat3[2];
	
	//private Pointer[] kernalParams = new Pointer[2];
	private int activeKernal = 0;
	private boolean dataLoaded = false;
	private CUmodule module;
	private Optional<BiConsumer<Double,String>> progressListener = Optional.empty();
	
	public Simulator(GlobeData world) {
		in = world;
		//out = new GlobeData(in);
		System.out.println("Simulation - Creating pointers");
		initPtrs();
		System.out.println("Simulation - Setting up functions");
		setupFuncs();
	}
	
	@SuppressWarnings("unchecked")
	private void setupFuncs() {
		module = loadModule("WeatherSim.ptx");
		Pointer worldSize = Pointer.to(worldSizePtr);
		
		atmosphereInit = new Triplet<CUfunction, Boolean, Pointer>(
				getFunction(module, "initAtmosphere"),
				true,
				Pointer.to(
						worldSize,
						pressure[0].getArgPointer(),
						temperature[0].getArgPointer()
						)
		);
		funcs = new Step[] {
				new Step("Copy",getFunction(module, "copy"), true, new Pointer[] {
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
				new Step("Calcuate Wind",getFunction(module, "calcWind"), true, new Pointer[] {
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
				}),new Step("Solar Heating",getFunction(module, "solarHeating"), false, new Pointer[] {
						Pointer.to(new Pointer[] {
								worldSize,
								worldTimePtr[0].getArgPointer(),
								cloudCover[0].getArgPointer(),
								humidity[0].getArgPointer(),
								groundType.getArgPointer(),
								elevation.getArgPointer(),
								snowCover[0].getArgPointer(),
								groundMoisture[0].getArgPointer(),
								temperature[0].getArgPointer(),
								temperature[1].getArgPointer()
						}),
						Pointer.to(new Pointer[] {
								worldSize,
								worldTimePtr[1].getArgPointer(),
								cloudCover[1].getArgPointer(),
								humidity[1].getArgPointer(),
								groundType.getArgPointer(),
								elevation.getArgPointer(),
								snowCover[1].getArgPointer(),
								groundMoisture[1].getArgPointer(),
								temperature[1].getArgPointer(),
								temperature[0].getArgPointer()
						})
				})
				,new Step("Infared Cooling",getFunction(module, "infraredCooling"), true, new Pointer[] {
						Pointer.to(new Pointer[] {
								worldSize,
								worldTimePtr[0].getArgPointer(),
								cloudCover[0].getArgPointer(),
								humidity[0].getArgPointer(),
								groundType.getArgPointer(),
								elevation.getArgPointer(),
								snowCover[0].getArgPointer(),
								groundMoisture[0].getArgPointer(),
								temperature[0].getArgPointer(),
								temperature[1].getArgPointer()
						}),
						Pointer.to(new Pointer[] {
								worldSize,
								worldTimePtr[1].getArgPointer(),
								cloudCover[1].getArgPointer(),
								humidity[1].getArgPointer(),
								groundType.getArgPointer(),
								elevation.getArgPointer(),
								snowCover[1].getArgPointer(),
								groundMoisture[1].getArgPointer(),
								temperature[1].getArgPointer(),
								temperature[0].getArgPointer()
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
		windSpeed[0] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions*3);
		windSpeed[1] 		= new CudaFloat3(in.latitudeDivisions, in.longitudeDivisions, in.altitudeDivisions*3);
		
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
	
	/**
	 * Use on newly generated worlds to add air
	 * */
	public void initAtmosphere() {
		int blockSizeX = 256;
		long gridSizeX_withAtmosphere = (long)Math.ceil((double)(in.totalCells()) / blockSizeX);
		Triplet<CUfunction, Boolean, Pointer> step = atmosphereInit;
		CUfunction f = step.a;
		double blocksNeeded = gridSizeX_withAtmosphere;
		
		int dimLimit = 65535;
		int dim = (int) Math.ceil(Math.pow(blocksNeeded, 1/3d));
		if(dim > dimLimit) throw new RuntimeException("Too many blocks required for simulation ("+dim+"^3 vs limit 65535^3)");
		JCudaDriver.cuLaunchKernel(f,       
			//CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
		    dim,  dim, dim,      // Grid dimension 
		    blockSizeX, 1, 1,      // Block dimension
		    0, null,               // Shared memory size and stream 
		    step.c, null // Kernel- and extra parameters
		); 
		
		JCudaDriver.cuCtxSynchronize();
	}
	
	/**Begins calculation of next timestep<br>
	 * Results will be ready before this emthod unblocks
	 * returns number of milliseconds elapsed
	 * */
	public synchronized long timeStep(boolean pullResult) {
		System.out.println("Begining timestep..");
		long start = System.currentTimeMillis();
		if(!dataLoaded) {
			progress(0, "Pushing data");
			pushData();
			JCudaDriver.cuCtxSynchronize();
			dataLoaded = true;
		}
		
		if(!in.isInitalized()) {
			progress(0, "Initalizing atmosphere");
			initAtmosphere();
			in.markInitalized();
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
		for(int i = 0; i<funcs.length; i++) {
			Step step = funcs[i];
			progress((i+1) / (float)funcs.length, "Step "+(i+1)+" of "+funcs.length + " - " + step.stepName);
			CUfunction f = step.function;
			double blocksNeeded = step.is3DSpace? gridSizeX_withAtmosphere : gridSizeX_groundOnly;
			
			int dimLimit = 65535;
			int dim = (int) Math.ceil(Math.pow(blocksNeeded, 1/3d));
			if(dim > dimLimit) throw new RuntimeException("Too many blocks required for simulation ("+dim+"^3 vs limit 65535^3)");
			JCudaDriver.cuLaunchKernel(f,       
				//CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
			    dim,  dim, dim,      // Grid dimension 
			    blockSizeX, 1, 1,      // Block dimension
			    0, null,               // Shared memory size and stream 
			    step.args[activeKernal], null // Kernel- and extra parameters
			); 
			
			JCudaDriver.cuCtxSynchronize();
			
		}
		if(pullResult) {
			progress(1, "Pulling result");
			pullResult(); //read from input side while also computing
			progress(1, "Waiting for result");
			JCudaDriver.cuCtxSynchronize();
			onResultReady.run();
		}
		activeKernal = 1-activeKernal;
		
		if(onStepComplete!=null)
			onStepComplete.run();
		long end = System.currentTimeMillis();
		progress(-1, "Ready");
		return end-start;
	}
	
	
	
	public void setProgressListener(BiConsumer<Double,String> progressListener) {
		this.progressListener = Optional.ofNullable(progressListener);
	}
	private void progress(double i, String status) {
		System.out.printf("Simulation - %f%% - %s\n", i ,status);
		progressListener.ifPresent(p->p.accept(i, status));
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
	
	@Override
	public void close() {
		JCudaDriver.cuMemFree(worldSizePtr);
		worldSpeed.close();
		worldTimePtr[0].close();
		worldTimePtr[1].close();
		groundType.close();
		elevation.close();
		groundMoisture[0].close();
		groundMoisture[1].close();
		snowCover[0].close();
		snowCover[1].close();
		temperature[0].close();
		temperature[1].close();
		pressure[0].close();
		pressure[1].close();
		humidity[0].close();
		humidity[1].close();
		cloudCover[0].close();
		cloudCover[1].close();
		windSpeed[0].close();
		windSpeed[1].close();
		JCudaDriver.cuModuleUnload(module);
	}
	
	private class Step {
		String stepName;
		CUfunction function;
		Pointer[] args;
		boolean is3DSpace;
		
		public Step(String step, CUfunction function, boolean is3DSpace, Pointer[] argSets) {
			this.stepName = step;
			this.function = function;
			this.is3DSpace = is3DSpace;
			this.args = argSets;
		}
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
