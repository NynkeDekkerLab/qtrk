// BenchmarkLUT.h
// Generates a noise-free LUT intended to benchmark Z resolution with a theoretically optimal LUT

#pragma once
#include "utils.h"

class BenchmarkLUT
{
public:
	std::vector<float> normprof;
	float max_a, max_b, max_c; // maxline coefficients

	int lut_w,lut_h;

	BenchmarkLUT() { lut_w=lut_h=0; }
	BenchmarkLUT (ImageData* lut);
	BenchmarkLUT (const char *file);
	void Load(ImageData* lut);
	void Load(const char *file);

	void GenerateLUT(ImageData* lut);
	void GenerateSample(ImageData* image, vector3f pos, float minRadius, float maxRadius);
protected:

};
