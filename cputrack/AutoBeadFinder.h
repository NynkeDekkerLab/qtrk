#pragma once

#include "utils.h"


struct AutoFind_BeadPos {
	AutoFind_BeadPos(int x=0,int y=0) : x(x),y(y) {}
	int x,y;
};

struct AutoFindConfig
{
	int roi;
	float img_distance; // relative to roi
	float similarity; // min value of pixels in 2D convolution

	float MinPixelDistance() { return 0.5f * img_distance * roi; }
};

// Performs automatic bead finding on a 2D float image
std::vector<AutoFind_BeadPos> AutoBeadFinder(ImageData* img, float* sample, AutoFindConfig* cfg);
std::vector<AutoFind_BeadPos> AutoBeadFinder(uint8_t* img, int w,int h, int smpCornerX, int smpCornerY, AutoFindConfig* cfg);
