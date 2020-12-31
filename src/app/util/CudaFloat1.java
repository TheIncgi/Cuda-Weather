package app.util;

import app.CudaUtils;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class CudaFloat1 implements AutoCloseable{
	private CUdeviceptr thePointer;
	private final int dim1;
	
	public CudaFloat1(int dim1) {
		this.dim1 = dim1;
		
		thePointer = CudaUtils.allocateFloatArray(dim1);
	}
	public void push(float[] data) {
		if(data.length!=dim1) throw new IllegalArgumentException("Dimension Missmatch");
		JCudaDriver.cuMemcpyHtoD(thePointer, Pointer.to(data), Float.BYTES * dim1);
	}
	
	/**Device to host*/
	public void pull(float[] data){
		JCudaDriver.cuMemcpyDtoH(Pointer.to(data), thePointer, dim1*Float.BYTES);
	}
	
	public CUdeviceptr getThePointer() {
		return thePointer;
	}
	
	public Pointer getArgPointer() {
		return Pointer.to(thePointer);
	}
	
	@Override
	public void close() {
		try{JCudaDriver.cuMemFree(thePointer);}catch(CudaException ce) {};
	}
}
