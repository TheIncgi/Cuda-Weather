package app.util;

/**Single int, still an array, but size 1, here for easy use */
public class CudaInt0 extends CudaInt1{
	public CudaInt0() {
		super(1);
	}
	public void push(int data) {
		push(new int[] {data});
	}
	public int pull() {
		int[] tmp = new int[1];
		super.pull(tmp);
		return tmp[0];
	}
}
