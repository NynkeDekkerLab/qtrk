#pragma once

#include "utils.h"

namespace BeadFinder {

	struct Position {
		Position (int x=0,int y=0) : x(x),y(y) {}
		int x,y;
	};

	struct Config
	{
		int roi;
		float img_distance; // relative to roi
		float similarity; // min value of pixels in 2D convolution

		float MinPixelDistance() { return 0.5f * img_distance * roi; }
	};

	// Performs automatic bead finding on a 2D float image
	std::vector<Position> Find(ImageData* img, float* sample, Config* cfg);
	std::vector<Position> Find(uint8_t* img, int pitch, int w,int h, int smpCornerX, int smpCornerY, Config* cfg);

};

