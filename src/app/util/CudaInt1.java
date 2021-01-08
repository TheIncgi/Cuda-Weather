package app.util;

import app.CudaUtils;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class CudaInt1 implements AutoCloseable{
	private CUdeviceptr thePointer;
	private final int dim1;
	
	public CudaInt1(int dim1) {
		this.dim1 = dim1;
		
		thePointer = CudaUtils.allocateFloatArray(dim1);
	}
	
	
	public void push(int[] data) {
		if(data.length!=dim1) throw new IllegalArgumentException("Dimension Missmatch");
		JCudaDriver.cuMemcpyHtoD(thePointer, Pointer.to(data), Integer.BYTES * dim1);
	}
	
	/**Device to host*/
	public void pull(int[] data){
		JCudaDriver.cuMemcpyDtoH(Pointer.to(data), thePointer, dim1*Integer.BYTES);
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
