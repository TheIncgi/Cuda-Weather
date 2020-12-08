package testing;

import java.util.ArrayList;

import app.util.Triplet;

public class Test {
	public static void main(String[] args) {
		long x = System.currentTimeMillis();
		int[] z = {7};
		for (int i = 0; i < 3_000_000; i++) {
			z[i%1] = z[i%1]+1;
		}
		long y = System.currentTimeMillis();
		System.out.println(y-x);
	}
	
	public static Triplet<Integer, Integer, Integer> calcBlockDim(long blocksNeeded){
		Triplet<Integer, Integer, Integer> out = new Triplet<Integer, Integer, Integer>();
		
		if(blocksNeeded >= 65536) {
			out.a = 65535;
			blocksNeeded /= 65536;
		}else {
			out.a = (int) blocksNeeded;
			blocksNeeded = 0;
		}
		if(blocksNeeded >= 65536) {
			out.b = 65535;
			blocksNeeded /= 65536;
		}else {
			out.b = (int) blocksNeeded;
			blocksNeeded = 0;
		}
		if(blocksNeeded >= 65536) 
			out.c = 65535;
		else
			out.c = (int) blocksNeeded;
		
		return out;
	}
	
	public static ArrayList<Integer> primes(int maxValue){
		ArrayList<Integer> out = new ArrayList<>();
		out.add(2);
		
		outter:
		for(int i = 3; i <= maxValue; i+=2) {
			int j = (int) Math.sqrt(i);
			for(int h = 0; out.get(h) <= j; h++) {
				if(i % out.get(h) == 0) continue outter;
			}
			out.add(i);
		}
		return out;
	}
}
