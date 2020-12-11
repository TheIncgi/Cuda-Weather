package app.util;

import java.io.Closeable;
import java.io.IOException;

import app.CudaUtils;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class CudaInt2 implements AutoCloseable{
	private CUdeviceptr thePointer;
	private CUdeviceptr[] pointerArray;
	private final int dim1, dim2;
	
	public CudaInt2(int dim1, int dim2) {
		this.dim1 = dim1;
		this.dim2 = dim2;
		
		pointerArray = new CUdeviceptr[dim1];
		for(int i = 0; i<pointerArray.length; i++) {
			pointerArray[i] = CudaUtils.allocateIntArray(dim2);
		}
		thePointer = CudaUtils.pointerArray(pointerArray);
	}
	/**Host to device*/
	public void push(int[][] data) {
		if(data.length!=dim1 || data[0].length!=dim2) throw new IllegalArgumentException("Dimension Missmatch");
		for(int x = 0; x<dim1; x++)
			JCudaDriver.cuMemcpyHtoD(pointerArray[x], Pointer.to(data[x]), Integer.BYTES * dim2);
	}
	/**Device to host*/
	public void pull(int[][] data){
		for(int i = 0; i<pointerArray.length; i++) {
			JCudaDriver.cuMemcpyDtoH(Pointer.to(data[i]), pointerArray[i], dim2*Integer.BYTES);
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
			try{JCudaDriver.cuMemFree(pointerArray[i]);}catch(CudaException ce) {};
		}
		try{JCudaDriver.cuMemFree(thePointer);}catch(CudaException ce) {};
	}
}
