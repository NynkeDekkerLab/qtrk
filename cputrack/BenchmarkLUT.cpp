#include "std_incl.h"
#include "BenchmarkLUT.h"
#include "LsqQuadraticFit.h"

BenchmarkLUT::BenchmarkLUT (const char*file)
{
	Load(file);
}

BenchmarkLUT::BenchmarkLUT (ImageData* lut)
{
	Load(lut);
}

void BenchmarkLUT::Load(const char *file)
{
	ImageData d = ReadJPEGFile(file);
	Load(&d);
	d.free();
}

void BenchmarkLUT::Load(ImageData* lut)
{
	std::vector<float> max_x,max_y;

	lut_w = lut->w;
	lut_h = lut->h;

	for (int i=lut->h/5;i<lut->h*3/4;i++) {
		int maxpos = 0;
		float maxval = lut->at(0, i);
		for (int x=1;x<lut->w;x++)
			if (maxval < lut->at(x,i)) {
				maxval = lut->at(x,i);
				maxpos=x;
			}
		max_x.push_back(maxpos);
		max_y.push_back(i);
	}

	LsqSqQuadFit<float> fit(max_x.size(), &max_y[0],&max_x[0]);

	max_a = fit.a;
	max_b = fit.b;
	max_c = fit.c; //normalized

	std::vector<int> numsmp(lut->w);
	normprof.resize(lut->w);
	for (int r=0;r<lut->w;r++) {
		float sum=0.0f;
		int nsmp=0;
		for (int y=0;y<lut->h;y++) {
			float x=r*fit.compute(y)/fit.c;
			bool outside=false;
			float v = lut->interpolate(x,y,&outside);
			if (!outside) {
				sum+=v;
				nsmp++;
			}
		}
		numsmp[r]=nsmp;
		normprof[r]=sum;
	}
	for (int i=0;i<lut->w;i++) {
		if (numsmp[i] == 0 && i>0) normprof[i]=normprof[i-1];
		else normprof[i]=normprof[i]/numsmp[i];
	}
}

void BenchmarkLUT::GenerateLUT(ImageData* lut)
{
	float M = lut->w / (float)lut_w;

	for (int y=0;y<lut->h;y++) {
		for (int x=0;x<lut->w;x++) {
			float maxline=M*(max_a*y*y+max_b*y+max_c)/max_c;
			lut->at(x,y) = Interpolate1D(normprof, x/maxline);
		}
	}
}

void BenchmarkLUT::GenerateSample(ImageData* image, vector3f pos, float minRadius, float maxRadius)
{
	float radialDensity=lut_w / (maxRadius-minRadius);

	if(pos.z<0.0f) pos.z=0.0f;
	if(pos.z>lut_h-1) pos.z=lut_h-1;

	for (int y=0;y<image->h;y++)
		for (int x=0;x<image->w;x++)
		{
			float dx=x-pos.x;
			float dy=y-pos.y;
			float r = (sqrt(dx*dx+dy*dy)-minRadius)*radialDensity; // r in original radial bin pixels
			float maxline=(max_a*pos.z*pos.z+max_b*pos.z+max_c)/max_c;
			float profpos = r / maxline;
			image->at(x,y) = Interpolate1D(normprof, profpos);
		}
}
