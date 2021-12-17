#pragma once
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include<stdint.h>

__device__ bool fontDataSet = false;

struct FontData{
  uint8_t* bytes;
  int width = 0;
  int height = 0;
  int dataStart = (127-32+2)*4+8;
};


/* A utility function to reverse a string  */

__device__ void strReverse(char str[], int length)
{
    int start = 0;
    int end = length -1;
    while (start < end)
    {	
    	char s = str[start];
    	str[start] = str[end];
    	str[end] = s;
        start++;
        end--;
    }
}

__device__ char* intToStr(int num, char* str, int base)
{
    int i = 0;
    bool isNegative = false;
 
    /* Handle 0 explicitly, otherwise empty string is printed for 0 */
    if (num == 0)
    {
        str[i++] = '0';
        str[i] = '\0';
        return str;
    }
 
    // In standard itoa(), negative numbers are handled only with
    // base 10. Otherwise numbers are considered unsigned.
    if (num < 0 && base == 10)
    {
        isNegative = true;
        num = -num;
    }
 
    // Process individual digits
    while (num != 0)
    {
        int rem = num % base;
        str[i++] = (rem > 9)? (rem-10) + 'a' : rem + '0';
        num = num/base;
    }
 
    // If number is negative, append '-'
    if (isNegative)
        str[i++] = '-';
 
    str[i] = '\0'; // Append string terminator
 
    // Reverse the string
    strReverse(str, i);
 
    return str;
}

__device__ void loadFont( uint8_t* data, FontData &fontData ) {
	fontData.bytes = data;
	fontData.width  = data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3];
	fontData.height = data[4] << 24 | data[5] << 16 | data[6] << 8 | data[7];
}

__device__ int getFontCharStart(FontData &fontData, char c) {
	int charByte = 8 + (c-32) * 4;
	return fontData.bytes[charByte  ] << 24 | 
	       fontData.bytes[charByte+1] << 16 | 
	       fontData.bytes[charByte+2] << 8 | 
	       fontData.bytes[charByte+3];
}

__device__ int getFontCharEnd(FontData &fontData, char c) {
	int charByte = 8 + (c-32) * 4;
	return fontData.bytes[charByte+4] << 24 | 
	       fontData.bytes[charByte+5] << 16 | 
	       fontData.bytes[charByte+6] << 8 | 
	       fontData.bytes[charByte+7];
}

__device__ int getFontCharWidth(FontData &fontData, char c) {
	return getFontCharEnd(fontData, c) - getFontCharStart(fontData, c);
}

//return true if it's part of the char
//x & y are pixel coordinates of the char, not global
__device__ bool getFontCharPixel(FontData &fontData, char c,int x, int y){
	if( y<=0 || fontData.height<=y) return false;
	int xStart = getFontCharStart(fontData, c);
	int xEnd   = getFontCharEnd(fontData, c);
	if( x < 0 || x-xStart >= xEnd) return false;
	
	return !fontData.bytes[ xStart + x + y*fontData.width + fontData.dataStart ]; //not sure why this ended up inverted..
}

__device__ int getFontStringPixel(FontData &fontData, const char* str, int dx, int dy) {
	int w = 0;
	int i = 0;
	if(dy >= fontData.height) return -1;
	while(str[i]!=0) { //null term
		char c = str[i];
		int cw = getFontCharWidth(fontData, c);
		if( w <= dx && dx <= (w+cw) ){
			if(getFontCharPixel( fontData, c, dx-w, dy ))
				return i;
		}
		w += cw;
		i++;
	}
	return -1;
}