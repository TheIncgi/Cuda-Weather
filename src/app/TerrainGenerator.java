package app;

import static app.CudaUtils.loadModule;
import static app.CudaUtils.loadToGPU;

import java.io.Closeable;
import java.io.IOException;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import app.util.CudaFloat1;
import app.util.CudaFloat2;
import app.util.CudaInt2;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class TerrainGenerator {
	
	Optional<BiConsumer<String, Float>> stepListener = Optional.empty();
	
	public TerrainGenerator() {
		CudaUtils.init();
	}
	
	public void generate(GlobeData globe) {
		tell("Loading Terrain Generator...",.1f);
		CUmodule module = loadModule("TerrainGen.ptx");
		CUfunction groundFunc = CudaUtils.getFunction(module, "genTerrain");
		CUfunction lakeFunc = CudaUtils.getFunction(module, "convertLakes");
		
		CUdeviceptr worldSizePtr = loadToGPU( new int[] { //constant
				globe.latitudeDivisions, 
				globe.longitudeDivisions, 
				globe.altitudeDivisions
		});
		try(
				CudaInt2   groundType= new CudaInt2(  globe.latitudeDivisions, globe.longitudeDivisions);
				CudaFloat2 elevation = new CudaFloat2(globe.latitudeDivisions, globe.longitudeDivisions);
		){
			
			Pointer kernel = Pointer.to(
					Pointer.to(worldSizePtr),
					groundType.getArgPointer(),
					elevation.getArgPointer()
			);
			
			int blockSizeX1 = 256;
			int blockSizeX2 = 4;
			long gridSizeX_groundOnly1     = (long)Math.ceil((double)(globe.groundCells()) / blockSizeX1);
			long gridSizeX_groundOnly2     = (long)Math.ceil((double)(globe.groundCells()) / blockSizeX2);
			double blocksNeeded1 = gridSizeX_groundOnly1;
			double blocksNeeded2 = gridSizeX_groundOnly2;
			
			int dimLimit = 65535;
			int dim1 = (int) Math.ceil(Math.pow(blocksNeeded1, 1/3d));
			int dim2 = (int) Math.ceil(Math.pow(blocksNeeded2, 1/3d));
			if(dim1 > dimLimit) throw new RuntimeException("Too many blocks required for simulation ("+dim1+"^3 vs limit 65535^3)");
			if(dim2 > dimLimit) throw new RuntimeException("Too many blocks required for simulation ("+dim2+"^3 vs limit 65535^3)");
			tell("Generating ground...", -1);
			JCudaDriver.cuCtxSynchronize();
			JCudaDriver.cuLaunchKernel(groundFunc,
					dim1, dim1, dim1, blockSizeX1, 1, 1, 0, null, kernel, null);
			JCudaDriver.cuCtxSynchronize();
			JCudaDriver.cuLaunchKernel(lakeFunc,
					dim2, dim2, dim2, blockSizeX2, 1, 1, 0, null, kernel, null);
			JCudaDriver.cuCtxSynchronize();
			groundType.pull(globe.groundType);
			elevation.pull(globe.elevation);
			JCudaDriver.cuCtxSynchronize();
			System.out.println("Generation complete");
		}catch (Exception e) {
			e.printStackTrace();
		}finally {
			JCudaDriver.cuMemFree(worldSizePtr);
			JCudaDriver.cuModuleUnload(module);
		}
		tell("Generation complete", 1);
	}
	
	private void tell(String msg, float progress) {
		System.out.println(msg + " ("+progress+")");
		stepListener.ifPresent(c->c.accept(msg, progress));
	}
	
	public void setStepListener(BiConsumer<String, Float> stepListener) {
		this.stepListener = Optional.ofNullable(stepListener);
	}
	
}
