package testing;

import static app.CudaUtils.getFunction;
import static app.CudaUtils.loadModule;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

import app.CudaUtils;
import app.util.CudaFloat1;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

class CudaTest {

	@Test
	void test() {
		System.out.println("Test Cuda 1");
		CudaUtils.init();
		
		CUmodule module = loadModule("JCudaVectorAddKernal.ptx");
		CUfunction func = getFunction(module, "add");
		
		CudaFloat1 x = new CudaFloat1(4);
		CudaFloat1 y = new CudaFloat1(4);
		CudaFloat1 z = new CudaFloat1(4);
		x.push(new float[] {1,2,3,4});
		y.push(new float[] {1,2,3,4});
		Pointer kernel = Pointer.to(
				Pointer.to(new int[] {4}),
				Pointer.to(x.getThePointer()),
				Pointer.to(y.getThePointer()),
				Pointer.to(z.getThePointer())
				);
		
		JCudaDriver.cuLaunchKernel(func,
				1,1,1,
				4,1,1,
				0, null,
				kernel, null);
		float[] data = new float[4];
		z.pull(data);
		System.out.println(Arrays.toString(data));
		for(int i = 0; i<4; i++) {
			assertEquals((i+1)*2, data[i]);
		}
		System.out.println("Passed");
	}

}
