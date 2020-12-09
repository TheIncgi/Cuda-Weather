package app;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import app.util.Pair;
import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class CudaUtils {
	private static CUdevice device;
	private static CUcontext context;
	private static boolean init = false;
	
	
	public static void init() {
		if(init) {
			JCudaDriver.cuCtxSetCurrent(context);
			return;
		}
		init = true;
		JCudaDriver.setExceptionsEnabled(true);
		//JCudaDriver.setLogLevel(LogLevel.LOG_DEBUGTRACE);
		cuInit(0);
        device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
	}
	
	public static CUdevice getDevice() {
		return device;
	}
	public static CUcontext getContext() {
		return context;
	}
	
	public static CUmodule loadModule(String file) {
		CUmodule module = new CUmodule();
		JCudaDriver.cuModuleLoad(module, file);
		return module;
	}
	
	public static CUfunction getFunction(CUmodule module, String fName) {
		CUfunction func = new CUfunction();
		JCudaDriver.cuModuleGetFunction(func, module, fName);
		return func;
	}
	
	public static CUdeviceptr loadToGPU(float[] data) {
		 CUdeviceptr ptr = new CUdeviceptr();
	     cuMemAlloc(ptr, data.length * Sizeof.FLOAT);
	     cuMemcpyHtoD(ptr, Pointer.to(data),
	            data.length * Sizeof.FLOAT);
	     return ptr;
	}
	public static void loadToGPU(CUdeviceptr ptr, float[] data) {
		if(ptr==null) throw new NullPointerException("Missing pointer");
		if(data==null) throw new NullPointerException("Missing data");
		
		cuMemcpyHtoD(ptr, Pointer.to(data),
				data.length * Sizeof.FLOAT);
	}
	public static void loadToGPU(CUdeviceptr ptr, int[] data) {
		if(ptr==null) throw new NullPointerException("Missing pointer");
		if(data==null) throw new NullPointerException("Missing data");
		
		cuMemcpyHtoD(ptr, Pointer.to(data),
				data.length * Sizeof.INT);
	}
	
	public static CUdeviceptr loadToGPU(int[] data) {
		 CUdeviceptr ptr = new CUdeviceptr();
	     cuMemAlloc(ptr, data.length * Sizeof.INT);
	     cuMemcpyHtoD(ptr, Pointer.to(data),
	            data.length * Sizeof.FLOAT);
	     return ptr;
	}
	
	public static CUdeviceptr allocateFloatArray(int len) {
		CUdeviceptr out = new CUdeviceptr();
        cuMemAlloc(out, len * Sizeof.FLOAT);
        return out;
	}
	public static CUdeviceptr allocateIntArray(int len) {
		CUdeviceptr out = new CUdeviceptr();
        cuMemAlloc(out, len * Sizeof.INT);
        return out;
	}
	
	public static void getOutput(CUdeviceptr ptr, float[] out) {
        cuMemcpyDtoH(Pointer.to(out), ptr,
            out.length * Sizeof.FLOAT);
	}
	public static void getOutput(CUdeviceptr ptr, int[] out) {
        cuMemcpyDtoH(Pointer.to(out), ptr,
            out.length * Sizeof.FLOAT);
	}
	
	public static CUdeviceptr pointerArray(CUdeviceptr... pointers) {
		CUdeviceptr out = new CUdeviceptr();
		cuMemAlloc(out, Sizeof.POINTER * pointers.length);
		cuMemcpyHtoD(out, Pointer.to(pointers), Sizeof.POINTER * pointers.length);
		return out;
	}
}
