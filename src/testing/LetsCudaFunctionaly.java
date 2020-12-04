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
import static app.CudaUtils.*;

//nvcc -ptx JCudaVectorAddKernel.cu -o JCudaVectorAddKernel.ptx

public class LetsCudaFunctionaly {
	
	private static CUdevice device;
	private static CUcontext context;
	
	public static void main(String[] args) {
//		Pointer pointer = new Pointer();
//		JCuda.cudaMalloc(pointer, 4);
//		System.out.println("Test: "+ pointer);
//		JCuda.cudaFree(pointer);
		
		// Enable exceptions and omit all subsequent error checks
        init();
		
		CUmodule module = loadModule("JCudaVectorAddKernal.ptx");
		CUfunction func = getFunction(module, "add");
		
		int numElements = 1024;

        // Allocate and fill the host input data
        float hostInputA[] = new float[numElements];
        float hostInputB[] = new float[numElements];
        for(int i = 0; i < numElements; i++)
            hostInputA[i] = hostInputB[i] = i;
        
        
        // Allocate to gpu
        CUdeviceptr deviceInputA = loadToGPU(hostInputA);        
        CUdeviceptr deviceInputB = loadToGPU(hostInputB);

        // Allocate device output memory
        CUdeviceptr deviceOutput = allocateFloatOut(numElements);
        
		
		Pointer kernalParams = Pointer.to(
				Pointer.to(new int[]{numElements}),
				Pointer.to(deviceInputA),
				Pointer.to(deviceInputB),
				Pointer.to(deviceOutput)
		);
		
		// Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)numElements / blockSizeX);
		
        System.out.printf("Block size x: %d\nGrid size x: %d\n", blockSizeX, gridSizeX);
        
		JCudaDriver.cuLaunchKernel(func,       
	//CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
		    gridSizeX,  1, 1,      // Grid dimension 
		    blockSizeX, 1, 1,      // Block dimension
		    0, null,               // Shared memory size and stream 
		    kernalParams, null // Kernel- and extra parameters
		); 
		
		
		cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[numElements];
        getOutput(deviceOutput, hostOutput);

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
