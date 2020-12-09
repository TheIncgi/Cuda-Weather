package app.util;

import java.io.Closeable;
import java.io.IOException;

import app.CudaUtils;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class CudaFloat2 implements Closeable{
	private CUdeviceptr thePointer;
	private CUdeviceptr[] pointerArray;
	private final int dim1, dim2;
	
	public CudaFloat2(int dim1, int dim2) {
		this.dim1 = dim1;
		this.dim2 = dim2;
		
		pointerArray = new CUdeviceptr[dim1];
		for(int i = 0; i<pointerArray.length; i++) {
			pointerArray[i] = CudaUtils.allocateFloatArray(dim2);
		}
		thePointer = CudaUtils.pointerArray(pointerArray);
	}
	public void push(float[][] data) {
		if(data.length!=dim1 || data[0].length!=dim2) throw new IllegalArgumentException("Dimension Missmatch");
		for(int x = 0; x<dim1; x++)
			JCudaDriver.cuMemcpyHtoD(pointerArray[x], Pointer.to(data[x]), Float.BYTES * dim2);
	}
	
	/**Device to host*/
	public void pull(float[][] data){
		for(int i = 0; i<pointerArray.length; i++) {
			JCudaDriver.cuMemcpyDtoH(Pointer.to(data[i]), pointerArray[i], dim2*Float.BYTES);
		}
	}
	
	public CUdeviceptr getThePointer() {
		return thePointer;
	}
	
	public Pointer getArgPointer() {
		return Pointer.to(thePointer);
	}
	
	@Override
	public void close() throws IOException {
		for (int i = 0; i < pointerArray.length; i++) {
			JCudaDriver.cuMemFree(pointerArray[i]);
		}
		JCudaDriver.cuMemFree(thePointer);
	}
}
