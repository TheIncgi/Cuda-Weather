#pragma once
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

#include "vectors.cu"

//based on this https://en.wikipedia.org/wiki/Perlin_noise
//modified to create wrapping while still allowing offsets for seed
__device__ vec2 randomGradient(int ix, int iy) {
    // Random float. No precomputed gradients mean this works for any number of grid coordinates
    float random = 2920.f * sin(ix * 21942.f + iy * 171324.f + 8912.f) * cos(ix * 23157.f * iy * 217832.f + 9758.f);
    vec2 out = {cos(random), sin(random) };
    return out;
}

// Computes the dot product of the distance and gradient vectors.
__device__ float dotGridGradient(int ix, int iy, float x, float y) {
    // Get gradient from integer coordinates
    vec2 gradient = randomGradient(ix, iy);

    // Compute the distance vector
    float dx = x - ix;
    float dy = y - iy;

    // Compute the dot-product
    return (dx*gradient.x + dy*gradient.y);
}

// Compute Perlin noise at coordinates x, y
//world size added for texture wrapping
__device__ float perlin(double x, double y, bool wrapTop, bool wrapX1, bool wrapY, double scale) {
//    if(wrapY) y -= scale-1;
	// Determine grid cell coordinates
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    // ya, x       y1(+1), x
    // yb, x+1     y2(+1), x+1



    //if(x0%2==1 && y0 %2 == 1) return 1;
    //if(wrapY) return 1 ;
    // Determine interpolation weights
    // Could also use higher order polynomial/s-curve here
    double sx = x - x0;
    double sy = y - y0;

    double back = scale;//scale;
//    wrapY = false;
    y1       = wrapY? y1 - back : y1;
    double ym = wrapY? y  - back : y;




    // Interpolate between grid point gradients
    float n0, n1, ix0, ix1, value;

    n0 = dotGridGradient(x0, y0, x, y);
    n1 = dotGridGradient(x1, y0, x, y);
    ix0 = smoothstep(n0, n1, sx);

    n0 = dotGridGradient(x0, y1, x, ym);
    n1 = dotGridGradient(x1, y1, x, ym);
    ix1 = smoothstep(n0, n1, sx);

    value = smoothstep(ix0, ix1, sy);
    return value;
}

__device__ float perlin(float x, float y, double scale, float offsetX, float offsetY, int* worldSize){
	double dx = (x + offsetX*scale)/scale;//(x /*+  f1/(scale*2)*/   +offsetX)/scale;
	double dy = (y + offsetY*scale)/scale;//(y /*+  f2/(scale*2)*/   +offsetY)/scale;
	bool wrapTop = ((int)(x / scale)) == 0;
	bool wrapBottom = ((int)(x / scale)) == ((int)(worldSize[0] / scale))-1;
	bool wrapWidth = ((int)(y / scale)) == ((int)(worldSize[1] / scale))-1;
	return perlin(dx, dy, wrapTop, wrapBottom, wrapWidth, 2*worldSize[0]/scale);
}
