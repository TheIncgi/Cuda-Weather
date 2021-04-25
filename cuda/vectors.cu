#pragma once
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

struct vec2{
	float x, y;
};
struct vec3{
	float x, y, z;
};
struct vec4{
	/**R/H*/
	float x;
	/**G/S*/
	float y;
	/**B/V*/
	float z;
	/**A*/
	float w;
};
struct dim2{
	int x,y;
};
__device__ bool operator==(const dim2& lhs, const dim2& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}
__device__ float dot(vec3 a, vec3 b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
}


//
// |~~~~~~~~~~~~~~~~~~~~~
// |    Color utils
// |~~~~~~~~~~~~~~~~~~~~~
__device__ vec4 hsvToArgb(vec4 in) {
	//h s v
	//r g b
	//x y z
	float c = in.z * in.y;
	float x = c * (1 - abs(fmod(in.x / 60.0 , 2.0) -1));
	float m = in.z - c;
	vec4 out;
	out.x = 0;
	out.y = 0;
	out.z = 0;
	out.w = in.w;

	if(in.x < 60) {
		out.x = c;
		out.y = x;
	}else if(in.x < 120){
		out.x = x;
		out.y = c;
	}else if(in.x <180){
		out.y = c;
		out.z = x;
	}else if(in.x < 240){
		out.y = x;
		out.z = c;
	}else if(in.x < 300){
		out.x = x;
		out.z = c;
	}else{
		out.x = c;
		out.z = x;
	}
	out.x = out.x+m;
	out.y = out.y+m;
	out.z = out.z+m;
	return out;
}
__device__ vec4 argbToHsv(vec4 in){
	vec4 out;
	float cmax = max(max(in.x, in.y), in.z);
	float cmin = min(min(in.x, in.y), in.z);
	float delta = cmax - cmin;
	float h = 0;
	//rgb
	//xyz
	if( delta==0 ){
		h = 0;
	}else if( cmax == in.x ){
		h = fmod(60.0 * ((in.y - in.z) / delta) , 6.0);
	}else if( cmax == in.y ){
		h = 60 * ((in.z - in.x) / delta) + 2;
	}else if( cmax == in.z ){
		h = 60 * ((in.x - in.y) / delta) + 4;
	}

	out.x = h;
	if(cmax==0)
		out.y = 0;
	else
		out.y = delta/cmax;
	out.z = cmax;
	out.w = in.w;
	return out;
}
__device__ int argbToHex(vec4 in){
	return
		(((int)(in.w * 255)) << 24) |
		(((int)(in.x * 255)) << 16) |
		(((int)(in.y * 255)) <<  8) |
		(((int)(in.z * 255))      ) ;
}
__device__ vec4 hexToArgb(int hex){
	vec4 out;
	out.w = (( hex >> 24 ) & 0xFF) / (float)255;
	out.x = (( hex >> 16 ) & 0xFF) / (float)255;
	out.y = (( hex >>  8 ) & 0xFF) / (float)255;
	out.z = (( hex       ) & 0xFF) / (float)255;
	return out;
}
__device__ int mixColors_hex(int a, int b, float bAmount) {
	vec4 i = hexToArgb(a);
	vec4 j = hexToArgb(b);
	vec4 t;
	float aAmount = 1-bAmount;
	//https://youtu.be/LKnqECcg6Gw?t=185
	t.x = sqrt( ((i.x * i.x) * aAmount) + ((j.x * j.x) * bAmount));
	t.y = sqrt( ((i.y * i.y) * aAmount) + ((j.y * j.y) * bAmount));
	t.z = sqrt( ((i.z * i.z) * aAmount) + ((j.z * j.z) * bAmount));
	t.w = sqrt( ((i.w * i.w) * aAmount) + ((j.w * j.w) * bAmount));
	return argbToHex(t);
}
__device__ vec4 mixColors_argb(vec4 a, vec4 b, float bAmount) {
	vec4 t;
	float aAmount = 1-bAmount;
	//https://youtu.be/LKnqECcg6Gw?t=185
	t.x = sqrt( ((a.x * a.x) * aAmount) + ((b.x * b.x) * bAmount));
	t.y = sqrt( ((a.y * a.y) * aAmount) + ((b.y * b.y) * bAmount));
	t.z = sqrt( ((a.z * a.z) * aAmount) + ((b.z * b.z) * bAmount));
	t.w = sqrt( ((a.w * a.w) * aAmount) + ((b.w * b.w) * bAmount));
	return t;
}

/**
 * alpha increases on combine
 * b.alpha effects mix amount
 * rgb *=
 * */
__device__ vec4 multipyColor(vec4 a, vec4 b) {
	vec4 out;
	float g = a.w + b.w;
	out.w = 1 - ((1-a.w) * (1-b.w));
	out.x = a.x * b.x;
	out.y = a.y * b.y;
	out.z = a.z * b.z;

	out = mixColors_argb(a, out, b.w);

	return out;
}
