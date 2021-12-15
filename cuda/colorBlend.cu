#pragma once
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include "cuda.h"
#include "vectors.cu"
#include "math.h"
#include "constants.cu"

__device__ vec4 mixColors(vec4 bottomLayer, vec4 topLayer) {
	vec4 t;
	float bAmount = topLayer.w;
	float aAmount = 1-bAmount;
	//https://youtu.be/LKnqECcg6Gw?t=185
	t.x = sqrt( ((bottomLayer.x * bottomLayer.x) * aAmount) + ((topLayer.x * topLayer.x) * bAmount));
	t.y = sqrt( ((bottomLayer.y * bottomLayer.y) * aAmount) + ((topLayer.y * topLayer.y) * bAmount));
	t.z = sqrt( ((bottomLayer.z * bottomLayer.z) * aAmount) + ((topLayer.z * topLayer.z) * bAmount));
	t.w = 1 - ((1-bottomLayer.w) * (1-topLayer.w));
	//t.w = sqrt( ((a.w * a.w) * aAmount) + ((b.w * b.w) * bAmount));
	return t;
}
