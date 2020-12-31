package testing;

import static org.junit.jupiter.api.Assertions.fail;

import org.junit.jupiter.api.Test;

class CubedRoot {

	@Test
	void test() {
		for(int i = 9; i<=2_000_000; i++) {
			double cr = Math.pow(i, 1/3d);
			long floor = (long) Math.floor(cr); //not good enough
			long ceil = (long) Math.ceil(cr);
			if(ceil * ceil * ceil < i)
				fail("Insufficent volume, "+ String.format("i: %d, floor: %d, ceil: %d", i, floor, ceil));
		}
	}
	
	

}
