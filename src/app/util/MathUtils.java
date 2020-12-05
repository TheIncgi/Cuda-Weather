package app.util;

public class MathUtils {
	public static float map(float x, float inMin, float inMax, float outMin, float outMax) {
		return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
	}
	public static float clamp(float x, float min, float max) {
		return Math.max(min, Math.min(max, x));
	}
	public static int clamp(int x, int min, int max) {
		return Math.max(min, Math.min(max, x));
	}
	public static float areaTrapazoid(float base1, float base2, float height) {
		return (base1*base2)/2 * height;
	}
	public static float dist(float x1, float y1, float x2, float y2) {
		float dx = x2-x1;
		float dy = y2-y1;
		return (float) Math.sqrt(dx*dx + dy*dy);
	}
	public static float dist(float x1, float y1, float z1, float x2, float y2, float z2) {
		float dx = x2-x1;
		float dy = y2-y1;
		float dz = z2-z1;
		return (float) Math.sqrt(dx*dx + dy*dy + dz*dz);
	}
	public static boolean inRange(float x, float low, float hi) {
		return low <= x && x < hi;
	}
}
