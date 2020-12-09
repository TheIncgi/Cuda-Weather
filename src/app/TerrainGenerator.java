package app;

import static app.CudaUtils.loadModule;

import java.io.Closeable;
import java.io.IOException;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import app.util.CudaFloat1;
import app.util.CudaFloat2;
import app.util.CudaInt2;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class TerrainGenerator {
	
	Optional<BiConsumer<String, Float>> stepListener = Optional.empty();
	
	public TerrainGenerator() {
	}
	
	public void generate(GlobeData globe) {
		tell("Loading Terrain Generator...",.1f);
		CUmodule module = loadModule("TerrainGen.ptx");
		CUfunction groundFunc = CudaUtils.getFunction(module, "genTerrain");
		
		try(
				CudaFloat1 worldSize = new CudaFloat1(3);
				CudaInt2   groundType= new CudaInt2(globe.latitudeDivisions, globe.longitudeDivisions);
				CudaFloat2 elevation = new CudaFloat2(globe.latitudeDivisions, globe.longitudeDivisions);
		){
		
			Pointer kernel = Pointer.to(
					worldSize.getArgPointer(),
					groundType.getArgPointer(),
					elevation.getArgPointer()
			);
			
			int blockSizeX = 256;
			long gridSizeX_groundOnly     = (long)Math.ceil((double)(globe.groundCells()) / blockSizeX);
			double blocksNeeded = gridSizeX_groundOnly;
			
			int dimLimit = 65535;
			int dim = (int) Math.ceil(Math.pow(blocksNeeded, 1/3d));
			if(dim > dimLimit) throw new RuntimeException("Too many blocks required for simulation ("+dim+"^3 vs limit 65535^3)");
			tell("Generating ground...", .2f);
			JCudaDriver.cuLaunchKernel(groundFunc,
					dim, dim, dim, blockSizeX, 1, 1, 0, null, kernel, null);
			JCudaDriver.cuCtxSynchronize();
			groundType.pull(globe.groundType);
			elevation.pull(globe.elevation);
		}
	}
	
	private void tell(String msg, float progress) {
		System.out.println(msg + " ("+progress+")");
		stepListener.ifPresent(c->c.accept(msg, progress));
	}
	
	public void setStepListener(BiConsumer<String, Float> stepListener) {
		this.stepListener = Optional.ofNullable(stepListener);
	}
	
}
