package app.util;

import static app.CudaUtils.getFunction;
import static app.CudaUtils.loadModule;
import static app.CudaUtils.loadToGPU;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Optional;
import java.util.function.BiConsumer;

import app.CudaUtils;
import app.CudaUtils.CuWrappedFunction;
import app.CudaUtils.CuWrappedModule;
import app.GlobeData;
import app.view.GpuGlobeRenderView;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class Simulator implements AutoCloseable{
	GlobeData in;
	Runnable onStepComplete, onResultReady;
	
	float timeStepSizeSeconds          = 5;
	float worldRotationRatePerStep     = timeStepSizeSeconds / (24*60*60) ;
	float worldRevolutionRatePerStep   = timeStepSizeSeconds / (24*60*60*365);
	private Step[] funcs;
	//constant
	private CUdeviceptr worldSizePtr;
	private CudaFloat1   worldSpeed; //not changed by CUfunction
	private CudaFloat1[] worldTimePtr = new CudaFloat1[2];
	private CudaInt1 imageSize;
	private CudaInt1 imageOut;
	private CudaByte1 rawFontData;
	private CudaInt0 overlayFlags;
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
	private CuWrappedModule module, renderModule;
	private Optional<BiConsumer<Double,String>> progressListener = Optional.empty();
//	private Triplet<CUfunction, Boolean, Pointer> atmosphereInit;
	private Step renderStep, copyOver, atmosphereInit;
	
	public Simulator(GlobeData world) {
		in = world;
		//out = new GlobeData(in);
		System.out.println("Simulation - Creating pointers");
		initPtrs();
		System.out.println("Simulation - Setting up functions");
		setupFuncs();
	}
	
	private void setupFuncs() {
		module = new CuWrappedModule("cuda/ptx/WeatherSim.ptx");
		renderModule = new CuWrappedModule("cuda/ptx/worldRender.ptx");
		Pointer worldSize = Pointer.to(worldSizePtr);
		
		var atmoArgs = Pointer.to(
				worldSize,
				pressure[0].getArgPointer(),
				temperature[0].getArgPointer()
				);
		atmosphereInit = new Step(
				"Init Atmosphere",
				module.getFunction("initAtmosphere"),
				true,
				new Pointer[] {atmoArgs, atmoArgs}
		);
		
		
		
		funcs = new Step[] {
				copyOver = new Step("Copy",module.getFunction("copy"), true, new Pointer[] {
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
				new Step("Calcuate Wind",module.getFunction("calcWind"), true, new Pointer[] {
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
				}),
				copyBack3Step("Wind", windSpeed, worldSize),
				new Step("Solar Heating",module.getFunction("solarHeating"), false, new Pointer[] {
						Pointer.to(new Pointer[] {
								worldSize,
								worldTimePtr[0].getArgPointer(),
								worldSpeed.getArgPointer(),
								pressure[0].getArgPointer(),
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
								worldSpeed.getArgPointer(),
								pressure[1].getArgPointer(),
								cloudCover[1].getArgPointer(),
								humidity[1].getArgPointer(),
								groundType.getArgPointer(),
								elevation.getArgPointer(),
								snowCover[1].getArgPointer(),
								groundMoisture[1].getArgPointer(),
								temperature[1].getArgPointer(),
								temperature[0].getArgPointer()
						})
				}),
				copyBackStep("Temp", temperature, worldSize),
				new Step("Infared Cooling",module.getFunction("infraredCooling"), true, new Pointer[] {
						Pointer.to(new Pointer[] {
								worldSize,
								worldTimePtr[0].getArgPointer(),
								worldSpeed.getArgPointer(),
								pressure[0].getArgPointer(),
								groundType.getArgPointer(),
								elevation.getArgPointer(),
								groundMoisture[0].getArgPointer(),
								humidity[0].getArgPointer(),
								temperature[0].getArgPointer(),
								temperature[1].getArgPointer(),
								
								cloudCover[0].getArgPointer(),
								
								
								snowCover[0].getArgPointer()
								
								
						}),
						Pointer.to(new Pointer[] {
								worldSize,
								worldTimePtr[1].getArgPointer(),
								worldSpeed.getArgPointer(),
								pressure[1].getArgPointer(),
								groundType.getArgPointer(),
								elevation.getArgPointer(),
								groundMoisture[1].getArgPointer(),
								humidity[1].getArgPointer(),
								temperature[1].getArgPointer(),
								temperature[0].getArgPointer(),
								
								cloudCover[1].getArgPointer(),
								
								
								snowCover[1].getArgPointer(),
								
								
						})
				})	
		};
		
		renderStep = new Step("Render", renderModule.getFunction("render"), false, new Pointer[] {
				Pointer.to(
						worldSize,
						elevation.getArgPointer(),
						groundType.getArgPointer(),
						worldTimePtr[0].getArgPointer(),
						groundMoisture[0].getArgPointer(),
						snowCover[0].getArgPointer(),
						temperature[0].getArgPointer(),
						pressure[0].getArgPointer(),
						humidity[0].getArgPointer(),
						cloudCover[0].getArgPointer(),
						windSpeed[0].getArgPointer(),
						imageSize.getArgPointer(),
						imageOut.getArgPointer(),
						overlayFlags.getArgPointer(),
						rawFontData.getArgPointer()
				),Pointer.to(
						worldSize,
						elevation.getArgPointer(),
						groundType.getArgPointer(),
						worldTimePtr[1].getArgPointer(),
						groundMoisture[1].getArgPointer(),
						snowCover[1].getArgPointer(),
						temperature[1].getArgPointer(),
						pressure[1].getArgPointer(),
						humidity[1].getArgPointer(),
						cloudCover[1].getArgPointer(),
						windSpeed[1].getArgPointer(),
						imageSize.getArgPointer(),
						imageOut.getArgPointer(),
						overlayFlags.getArgPointer(),
						rawFontData.getArgPointer()
				)
		});
		
	}

	private void initPtrs() {
		byte[] fd;
		try {
			FileInputStream fis = new FileInputStream("epilogue.cuFont");
			fd = fis.readAllBytes();
			fis.close();
		}catch(IOException ioe){
			ioe.printStackTrace();
			fd=new byte[8 + (127-32+4)*4]; 
		}
		worldSizePtr = loadToGPU( new int[] { //constant
				in.LATITUDE_DIVISIONS, 
				in.LONGITUDE_DIVISIONS, 
				in.ALTITUDE_DIVISIONS
		});
		worldSpeed = new CudaFloat1(3);
		elevation = new CudaFloat2(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS);
		groundType = new CudaInt2(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS);
		
		imageSize           = new CudaInt1(2);
		imageOut            = new CudaInt1(GpuGlobeRenderView.IMAGE_HEIGHT * GpuGlobeRenderView.IMAGE_WIDTH);
		overlayFlags        = new CudaInt0();
		rawFontData			= new CudaByte1(fd.length);
		rawFontData.push(fd);
		
		worldTimePtr 		= new CudaFloat1[] { new CudaFloat1(2), new CudaFloat1(2) };
		groundMoisture[0] 	= new CudaFloat2(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS);
		groundMoisture[1] 	= new CudaFloat2(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS);
		snowCover[0] 		= new CudaFloat2(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS);
		snowCover[1] 		= new CudaFloat2(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS);
		temperature[0] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS);
		temperature[1] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS);
		pressure[0] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS);
		pressure[1] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS);
		humidity[0] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS);
		humidity[1] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS);
		cloudCover[0] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS);
		cloudCover[1] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS);
		windSpeed[0] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS*3);
		windSpeed[1] 		= new CudaFloat3(in.LATITUDE_DIVISIONS, in.LONGITUDE_DIVISIONS, in.ALTITUDE_DIVISIONS*3);
		
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
		imageSize					  .push(new int[] {GpuGlobeRenderView.IMAGE_WIDTH, GpuGlobeRenderView.IMAGE_HEIGHT});
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
	
	public synchronized void setOverlayFlags( int flags ) {
		overlayFlags.push(flags);
	}
	
	public void initAtmosphere(boolean force) {
		if(!force && in.isInitalized()) return;
		
		int blockSizeX = 256;
		long gridSizeX_withAtmosphere = (long)Math.ceil((double)(in.totalCells()) / blockSizeX);
		
		Step[] steps = new Step[] {atmosphereInit, copyOver};
		for(Step step:steps) {
			CUfunction f = step.function.getFunction();
			double blocksNeeded = gridSizeX_withAtmosphere;
			
			int dimLimit = 65535;
			int dim = (int) Math.ceil(Math.pow(blocksNeeded, 1/3d));
			if(dim > dimLimit) throw new RuntimeException("Too many blocks required for simulation ("+dim+"^3 vs limit 65535^3)");
			JCudaDriver.cuLaunchKernel(f,       
				//CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
			    dim,  dim, dim,      // Grid dimension 
			    blockSizeX, 1, 1,      // Block dimension
			    0, null,               // Shared memory size and stream 
			    step.args[0], null // Kernel- and extra parameters
			); 
		}
		JCudaDriver.cuCtxSynchronize();
	}
	/**
	 * Use on newly generated worlds to add air
	 * */
	public void initAtmosphere() {
		initAtmosphere(false);
	}
	
	/**Begins calculation of next timestep<br>
	 * Results will be ready before this emthod unblocks
	 * returns number of milliseconds elapsed
	 * */
	public synchronized long timeStep(boolean pullResult) {
		//System.out.println("Begining timestep..");
		long start = System.currentTimeMillis();
		
		if(module.reload()) {
			System.out.println(module.getModuleFile() + " has been automaticly reloaded");
//			dataLoaded = false; //needed?
		}
		
		if(!dataLoaded) {
			progress(0, "Pushing data", false);
			pushData();
			JCudaDriver.cuCtxSynchronize();
			dataLoaded = true;
		}
		
		if(!in.isInitalized()) {
			progress(0, "Initalizing atmosphere", false);
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
			try {
				progress((i+1) / (float)funcs.length, "Step "+(i+1)+" of "+funcs.length + " - " + step.stepName, !pullResult);
				CUfunction f = step.function.getFunction();
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
			}catch(CudaException ce) {
				System.err.println("Cuda exception on step: "+step.stepName);
				throw ce;
			}
		}
		if(pullResult) {
			JCudaDriver.cuCtxSynchronize();
			progress(1, "Pulling result", !pullResult);
			//pullResult(); //read from input side while also computing
			progress(1, "Waiting for result", !pullResult);
			JCudaDriver.cuCtxSynchronize();
			onResultReady.run();
		}
		activeKernal = 1-activeKernal;
		
		if(onStepComplete!=null)
			onStepComplete.run();
		long end = System.currentTimeMillis();
		progress(-1, "Ready", !pullResult);
		return end-start;
	}
	
	
	
	public void setProgressListener(BiConsumer<Double,String> progressListener) {
		this.progressListener = Optional.ofNullable(progressListener);
	}
	private void progress(double i, String status, boolean quiet) {
		if(quiet) return;
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
		module.close();
		renderModule.close();
	}
	
	private class Step {
		String stepName;
		CuWrappedFunction function;
		Pointer[] args;
		boolean is3DSpace;
		
		public Step(String step, CuWrappedFunction function, boolean is3DSpace, Pointer[] argSets) {
			this.stepName = step;
			this.function = function;
			this.is3DSpace = is3DSpace;
			this.args = argSets;
		}
	}

	public float getTimestepSize() {
		return timeStepSizeSeconds;
	}
	
	public void render(int[] buffer) {
		
		
		if(renderModule.reload()) {
			System.out.println("Render module has been automaticly reloaded.");
			dataLoaded = false; //TODO is this needed here?
		}
		if(!dataLoaded) {
			progress(0, "Pushing data", false);
			pushData();
			JCudaDriver.cuCtxSynchronize();
			dataLoaded = true;
		}
		
		int k = 1-activeKernal;
		//renderStep.
		
		int blockSizeX = 256;
		long pixelCount = (long) Math.ceil(GpuGlobeRenderView.IMAGE_WIDTH * GpuGlobeRenderView.IMAGE_HEIGHT / (float)blockSizeX);
		Step step = renderStep;
		CUfunction f = step.function.getFunction();
		double blocksNeeded = pixelCount;
			
		int dimLimit = 65535;
		int dim = (int) Math.ceil(Math.pow(blocksNeeded, 1/3d));
		if(dim > dimLimit) throw new RuntimeException("Too many blocks required for simulation ("+dim+"^3 vs limit 65535^3)");
		JCudaDriver.cuLaunchKernel(f,       
			//CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
		    dim,  dim, dim,      // Grid dimension 
		    blockSizeX, 1, 1,      // Block dimension
		    0, null,               // Shared memory size and stream 
		    step.args[k], null // Kernel- and extra parameters
		); 
		
	
			
		
		imageOut.pull(buffer);
		JCudaDriver.cuCtxSynchronize();
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
	
	
	private Step copyBackStep(String varName, CudaFloat2[] buf, Pointer worldSize) {
		return new Step("Copy back "+varName, module.getFunction("copyBack2f"), false, new Pointer[] {
				Pointer.to(
						worldSize,
						buf[0].getArgPointer(),
						buf[1].getArgPointer()
				),
				Pointer.to(
						worldSize,
						buf[1].getArgPointer(),
						buf[0].getArgPointer()
				)
		});
	}
	private Step copyBackStep(String varName, CudaFloat3[] buf, Pointer worldSize) {
		return new Step("Copy back "+varName, module.getFunction("copyBack3f"), false, new Pointer[] {
				Pointer.to(
						worldSize,
						buf[0].getArgPointer(),
						buf[1].getArgPointer()
				),
				Pointer.to(
						worldSize,
						buf[1].getArgPointer(),
						buf[0].getArgPointer()
				)
		});
	}
	
	//for wind
	private Step copyBack3Step(String varName, CudaFloat3[] buf, Pointer worldSize) {
		return new Step("Copy back "+varName, module.getFunction("copyBack3f3"), false, new Pointer[] {
				Pointer.to(
						worldSize,
						buf[0].getArgPointer(),
						buf[1].getArgPointer()
						),
				Pointer.to(
						worldSize,
						buf[1].getArgPointer(),
						buf[0].getArgPointer()
						)
		});
	}
	
}
