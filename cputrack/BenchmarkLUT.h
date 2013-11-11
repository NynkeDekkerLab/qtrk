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

	BenchmarkLUT (ImageData* lut);
	BenchmarkLUT (const char *file);
	void Load(ImageData* lut);

	void GenerateLUT(ImageData* lut, float M);
	void GenerateSample(ImageData* image, vector3f pos, float rstep, float M);
protected:

};
