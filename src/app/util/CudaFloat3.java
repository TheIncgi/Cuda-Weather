package app.util;

import java.io.Closeable;
import java.io.IOException;

import app.CudaUtils;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class CudaFloat3 implements Closeable{
	private CUdeviceptr thePointer;
	private CUdeviceptr[] pointerArray;
	private CudaFloat2[] inner;
	private final int dim1;
	
	public CudaFloat3(int dim1, int dim2, int dim3) {
		this.dim1 = dim1;
		
		pointerArray = new CUdeviceptr[dim1];
		inner = new CudaFloat2[dim1];
		for(int i = 0; i<pointerArray.length; i++) {
			inner[i] = new CudaFloat2(dim2, dim3);
			pointerArray[i] = inner[i].getThePointer();
		}
		thePointer = CudaUtils.pointerArray(pointerArray);
	}
	
	/**Host to device*/
	public void push(float[][][] data) {
		if(data.length!=dim1) throw new IllegalArgumentException("Dimension Missmatch");
		for (int i = 0; i < inner.length; i++) {
			inner[i].push(data[i]);
		}
	}
	
	/**Device to host*/
	public void pull(float[][][] data) {
		for(int i = 0; i<dim1; i++) {
			inner[i].pull(data[i]);
		}
	}
	
	public CUdeviceptr getThePointer() {
		return thePointer;
	}
	
	@Override
	public void close() throws IOException {
		for (int i = 0; i < pointerArray.length; i++) {
			inner[i].close();
			JCudaDriver.cuMemFree(pointerArray[i]);
		}
		JCudaDriver.cuMemFree(thePointer);
	}
}
