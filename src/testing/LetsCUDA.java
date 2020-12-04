package testing;


import jcuda.*;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.*;
import static jcuda.driver.JCudaDriver.*;

//nvcc -ptx JCudaVectorAddKernel.cu -o JCudaVectorAddKernel.ptx

public class LetsCUDA {
	public static void main(String[] args) {
//		Pointer pointer = new Pointer();
//		JCuda.cudaMalloc(pointer, 4);
//		System.out.println("Test: "+ pointer);
//		JCuda.cudaFree(pointer);
		
		// Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
		
		cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
		
		CUmodule module = new CUmodule();
		JCudaDriver.cuModuleLoad(module, "JCudaVectorAddKernal.ptx");
		CUfunction func = new CUfunction();
		JCudaDriver.cuModuleGetFunction(func, module, "add");
		
		int numElements = 1024;

        // Allocate and fill the host input data
        float hostInputA[] = new float[numElements];
        float hostInputB[] = new float[numElements];
        for(int i = 0; i < numElements; i++)
        {
            hostInputA[i] = (float)i;
            hostInputB[i] = (float)i;
        }
        
        
        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA),
            numElements * Sizeof.FLOAT);
        
        CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB),
            numElements * Sizeof.FLOAT);

        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, numElements * Sizeof.FLOAT);
		
		Pointer kernalParams = Pointer.to(
				Pointer.to(new int[]{numElements}),
				Pointer.to(deviceInputA),
				Pointer.to(deviceInputB),
				Pointer.to(deviceOutput)
		);
		
		// Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)numElements / blockSizeX);
		
		JCudaDriver.cuLaunchKernel(func, 
		    gridSizeX,  1, 1,      // Grid dimension 
		    blockSizeX, 1, 1,      // Block dimension
		    0, null,               // Shared memory size and stream 
		    kernalParams, null // Kernel- and extra parameters
		); 
		
		
		cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[numElements];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
            numElements * Sizeof.FLOAT);

        // Verify the result
        boolean passed = true;
        for(int i = 0; i < numElements; i++)
        {
            float expected = i+i;
            if (Math.abs(hostOutput[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+hostOutput[i]+
                    " but expected "+expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        // Clean up.
        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);

	}
}
