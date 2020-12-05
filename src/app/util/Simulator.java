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
	
	private CUfunction func;
	private CUdeviceptr[] inputPointers;
	private CUdeviceptr[] outputPointers;
	private Pointer kernalParams;
	private float[] worldTime;
	
	public Simulator(GlobeData world, float planetRot, float planetRev) {
		in = world;
		
		CUmodule module = loadModule("WeatherSim.ptx");
		func = getFunction(module, "timeStep");
		
		inputPointers = new CUdeviceptr[8];
		outputPointers = new CUdeviceptr[inputPointers.length];
		
		for(int i = 0; i<inputPointers.length; i++) {
			inputPointers[i] = new CUdeviceptr();
			outputPointers[i] = new CUdeviceptr();
		}
		
		worldTime = new float[] {planetRot, planetRev};
		
		int ip = 0;
		int op = 0;
		kernalParams = Pointer.to(
			Pointer.to(worldTime),
			Pointer.to(inputPointers[ip++]),
			Pointer.to(inputPointers[ip++]),
			Pointer.to(inputPointers[ip++]),
			Pointer.to(inputPointers[ip++]),
			Pointer.to(inputPointers[ip++]),
			Pointer.to(inputPointers[ip++]),
			Pointer.to(inputPointers[ip++]),
			Pointer.to(inputPointers[ip++]),
			Pointer.to(outputPointers[op++]),
			Pointer.to(outputPointers[op++]),
			Pointer.to(outputPointers[op++]),
			Pointer.to(outputPointers[op++]),
			Pointer.to(outputPointers[op++]),
			Pointer.to(outputPointers[op++]),
			Pointer.to(outputPointers[op++]),
			Pointer.to(outputPointers[op++])
		);
	}
	
	public void timeStep() {
		
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
