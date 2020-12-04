package app;

public class MathUtils {
	public static float map(float x, float inMin, float inMax, float outMin, float outMax) {
		return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
	}
	public static float clamp(float x, float min, float max) {
		return Math.max(min, Math.min(max, x));
	}
}
