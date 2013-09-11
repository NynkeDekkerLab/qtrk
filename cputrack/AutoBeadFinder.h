#pragma once

#include "utils.h"


struct AutoFind_BeadPos {
	int x,y;
};

struct AutoFindConfig
{
	int roi;
	float img_distance;
	float similarity; // min value of pixels in 2D convolution
};

// Performs automatic bead finding on a 2D float image
std::vector<AutoFind_BeadPos> AutoBeadFinder(ImageData* img, float* sample, AutoFindConfig* cfg);
std::vector<AutoFind_BeadPos> AutoBeadFinder(uint8_t* img, int w,int h, int smpCornerX, int smpCornerY, AutoFindConfig* cfg);
