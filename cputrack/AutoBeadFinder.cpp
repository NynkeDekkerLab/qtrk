#include "std_incl.h"
#include "utils.h"
#include "AutoBeadFinder.h"
#include "fft2D.h"

int NextPowerOf2(int x)
{
	int y=1;
	while (y<x)
		y*=2;
	return y;
}

std::vector<AutoFind_BeadPos> AutoBeadFinder(ImageData* img, float* sample, AutoFindConfig* cfg)
{
	// Compute nearest power of two dimension
	int w=NextPowerOf2(img->w);
	int h=NextPowerOf2(img->h);

	// Convert to std::complex and copy the image in there (subtract mean first)
	float mean = img->mean();
	std::complex<float>* cimg = new std::complex<float>[w*h];
	for (int y=0;y<img->h;y++) {
		for (int x=0;x<img->w;x++) {
			cimg[y*w+x] = img->data[y*img->w+x]-mean;
		}
	}

	// Inplace 2D fft
	fft2d::FFT2D(cimg, w, h, 1);

	// Create an image to put the sample in with size [s,s] as well
	std::complex<float>* smpimg = new std::complex<float>[w*h];
	std::fill(smpimg, smpimg+w*h, 0);
	float smpmean = ImageData(sample,cfg->roi,cfg->roi).mean();
	for (int y=0;y<cfg->roi;y++) 
		for (int x=0;x<cfg->roi;x++) 
			smpimg[ ((y-cfg->roi/2)%h)*w + (x-cfg->roi/2)%w ] = sample[cfg->roi*y+x]-smpmean;

	fft2d::FFT2D(smpimg, w,h, 1);
	for (int i=0;i<w*h;i++)
		cimg[i]*=smpimg[i];
	fft2d::FFT2D(cimg, w,h,-1);

	float maxVal = 0.0f;
	for (int i=0;i<w*h;i++) {
		maxVal=std::max(maxVal, cimg[i].real());
	}
	delete[] smpimg;
	delete[] cimg;

	std::vector<AutoFind_BeadPos> beads;
	return beads;
}

std::vector<AutoFind_BeadPos> AutoBeadFinder(uint8_t* img, int pitch, int w,int h, int smpCornerX, int smpCornerY, AutoFindConfig* cfg)
{
	ImageData fimg = ImageData::alloc(w,h);

	for (int y=0;y<h;y++)
		for (int x=0;x<w;x++) 
			fimg.at(x,y) = img[y*pitch+x];

	float *fsmp = new float[cfg->roi*cfg->roi];
	for (int y=0;y<cfg->roi;y++) {
		for (int x=0;x<cfg->roi;x++) {
			fsmp [y*cfg->roi+x] = fimg [ (y+smpCornerY) * h + x + smpCornerX ];
		}
	}

	std::vector<AutoFind_BeadPos> beads = AutoBeadFinder(&fimg, fsmp, cfg);

	delete[] fsmp;
	fimg.free();
	return beads;
}

