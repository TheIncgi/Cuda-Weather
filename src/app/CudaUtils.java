package app;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.util.HashMap;

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
	
	
	public static synchronized void init() {
		long x = System.currentTimeMillis();
		System.out.println("Cuda init on thread "+Thread.currentThread().getName());
		if(init) {
			JCudaDriver.cuCtxSetCurrent(context);
			System.out.println("Cuda Re-Init took" + (System.currentTimeMillis()-x)+" millis");
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
        System.out.println("Cuda Init took" + (System.currentTimeMillis()-x)+" millis");
        
	}
	
	public static synchronized void destroyContext() {
		JCudaDriver.cuCtxDestroy(context);
		init = false;
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
	
	private static HashMap<String, FileTime> lastMod = new HashMap<>();
	public static boolean modifiedSinceRead(String module) {
		try {
			File file = new File(module);
			BasicFileAttributes x= Files.readAttributes(file.toPath(), BasicFileAttributes.class);
			FileTime t = x.lastModifiedTime();
			boolean out = false;
			if(lastMod.containsKey(module)) {
				FileTime b = lastMod.get(module);
				out = b.compareTo(t) != 0;
			}
			lastMod.put(module, t);
			return out;
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
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
