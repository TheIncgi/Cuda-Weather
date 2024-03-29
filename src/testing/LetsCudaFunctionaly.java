package testing;


import static app.CudaUtils.allocateFloatArray;
import static app.CudaUtils.getFunction;
import static app.CudaUtils.getOutput;
import static app.CudaUtils.loadModule;
import static app.CudaUtils.loadToGPU;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuMemFree;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

//nvcc -ptx JCudaVectorAddKernel.cu -o JCudaVectorAddKernel.ptx

public class LetsCudaFunctionaly {
	
//	private static CUdevice device;
//	private static CUcontext context;
	
	public static void main(String[] args) {
//		Pointer pointer = new Pointer();
//		JCuda.cudaMalloc(pointer, 4);
//		System.out.println("Test: "+ pointer);
//		JCuda.cudaFree(pointer);
		
		// Enable exceptions and omit all subsequent error checks
        
		
		CUmodule module = loadModule("cuda/ptx/JCudaVectorAddKernal.ptx");
		CUfunction func = getFunction(module, "add");
		
		int numElements = 1<<27;

        // Allocate and fill the host input data
        float hostInputA[] = new float[numElements];
        float hostInputB[] = new float[numElements];
        for(int i = 0; i < numElements; i++)
            hostInputA[i] = hostInputB[i] = i;
        
        
        // Allocate to gpu
        CUdeviceptr deviceInputA = loadToGPU(hostInputA);        
        CUdeviceptr deviceInputB = loadToGPU(hostInputB);

        // Allocate device output memory
        CUdeviceptr deviceOutput = allocateFloatArray(numElements);
        
		
		Pointer kernalParams = Pointer.to(
				Pointer.to(new int[]{numElements}),
				Pointer.to(deviceInputA),
				Pointer.to(deviceInputB),
				Pointer.to(deviceOutput)
		);
		
		double blocksNeeded = numElements/256d;
		
//		int dimLimit = 65535;
		int dim = (int) Math.ceil(Math.pow(blocksNeeded, 1/3d));
		
		// Call the kernel function.
        int blockSizeX = 256;
//        int gridSizeX = (int)Math.ceil((double)numElements / blockSizeX);
		
        System.out.printf("Block size x: %d\nGrid size x: %d\n", blockSizeX, dim);
        
		JCudaDriver.cuLaunchKernel(func,       
	//CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
		    dim,  dim, dim,      // Grid dimension 
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
