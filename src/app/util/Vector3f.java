package app.util;

import static java.lang.Math.cos;
import static java.lang.Math.sin;
public class Vector3f {
	public float x, y, z;
	
	
	public Vector3f() {
	}

	public Vector3f(float x, float y, float z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}
	
	public void rotateAboutZ(float yaw) {
		double nx = x*cos(yaw) - y*sin(yaw);
		double ny = x*sin(yaw) + y*cos(yaw);
		this.x = (float) nx;
		this.y = (float) ny;
	}
	public void rotateAboutY(float pitch) {
		double nx = x*cos(pitch) + z*sin(pitch);
		double nz = -x*sin(pitch) + z*cos(pitch);
		this.x = (float) nx;
		this.z = (float) nz;
	}
	
	public static float distance(Vector3f a, Vector3f b) {
		float dx = a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
		return (float) Math.sqrt(dx*dx + dy*dy + dz*dz);
	}
	
	public float distance(Vector3f b) {
		return distance(this, b);
	}
	
}
