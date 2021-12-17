package app.util;

import app.CudaUtils;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class CudaByte1 implements AutoCloseable{
	private CUdeviceptr thePointer;
	private final int dim1;
	
	public CudaByte1(int dim1) {
		this.dim1 = dim1;
		
		thePointer = CudaUtils.allocateByteArray(dim1);
	}
	
	
	public void push(byte[] data) {
		if(data.length!=dim1) throw new IllegalArgumentException("Dimension Missmatch");
		JCudaDriver.cuMemcpyHtoD(thePointer, Pointer.to(data), Byte.BYTES * dim1);
	}
	
	/**Device to host*/
	public void pull(byte[] data){
		JCudaDriver.cuMemcpyDtoH(Pointer.to(data), thePointer, dim1*Byte.BYTES);
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
