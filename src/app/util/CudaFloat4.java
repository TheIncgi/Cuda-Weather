package app.util;

import app.CudaUtils;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class CudaFloat4 implements AutoCloseable{
	private CUdeviceptr thePointer;
	private CUdeviceptr[] pointerArray;
	private CudaFloat3[] inner;
	private final int dim1;
	
	public CudaFloat4(int dim1, int dim2, int dim3, int dim4) {
		this.dim1 = dim1;
		
		pointerArray = new CUdeviceptr[dim1];
		inner = new CudaFloat3[dim1];
		for(int i = 0; i<pointerArray.length; i++) {
			inner[i] = new CudaFloat3(dim2, dim3, dim4);
			pointerArray[i] = inner[i].getThePointer();
		}
		thePointer = CudaUtils.pointerArray(pointerArray);
	}
	public void push(float[][][][] data) {
		if(data.length!=dim1) throw new IllegalArgumentException("Dimension Missmatch");
		for (int i = 0; i < inner.length; i++) {
			inner[i].push(data[i]);
		}
	}
	
	/**Device to host*/
	public void pull(float[][][][] data) {
		for(int i = 0; i<dim1; i++) {
			inner[i].pull(data[i]);
		}
	}
	
	public CUdeviceptr getThePointer() {
		return thePointer;
	}
	
	public Pointer getArgPointer() {
		return Pointer.to(thePointer);
	}
	
	@Override
	public void close() {
		for (int i = 0; i < pointerArray.length; i++) {
			inner[i].close();
			try{JCudaDriver.cuMemFree(pointerArray[i]);}catch(CudaException ce) {};
		}
		try{JCudaDriver.cuMemFree(thePointer);}catch(CudaException ce) {};
	}
}
