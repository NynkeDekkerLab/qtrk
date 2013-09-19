#include "std_incl.h"
#include "utils.h"
#include "kissfft.h"
#include <list>
#include "threads.h"
#include <functional>
#include "BeadFinder.h"

using namespace BeadFinder;

int NextPowerOf2(int x)
{
	int y=1;
	while (y<x)
		y*=2;
	return y;
}

float abs(std::complex<float> x) { return sqrtf(x.imag()*x.imag()+x.real()*x.real()); }

void ComplexToJPEGFile(const char* name, std::complex<float>* d, int w,int h, bool logMode=false)
{
	float minV=abs(d[0]),maxV=minV;
	for(int i=0;i<w*h;i++) {
		float v = abs(d[i]);
		minV=std::min(minV, v);
		maxV=std::max(maxV, v);
	}
	if (maxV==minV) {
		dbgprintf("ComplexToJPEGFile (%s): Min=max\n",name);
		maxV=minV+1;
	}

	uint8_t *img = new uint8_t[w*h];
	for (int i=0;i<w*h;i++) {
		float v = abs(d[i]);
		img[i] = (uint8_t) (0.5f + (logMode?10:1) * 255 * (v-minV)/(maxV-minV) );
	}
	WriteJPEGFile(img, w,h,name,99);
	delete[] img;
}

// Multithreaded 2D FFT
void FFT2D(std::complex<float> *d , int w,int h, bool inverse)
{
	kissfft<float> xfft(w, inverse);
	kissfft<float> yfft(h, inverse);

	auto fx = [&] (int y) {
		std::complex<float>* tmp = ALLOCA_ARRAY(std::complex<float>, std::max(w,h));
		xfft.transform(&d[w*y], tmp);
		for(int x=0;x<w;x++)
			d[w*y+x]=tmp[x];
	};
	ThreadPool<int, std::function< void (int index) > > xfftPool(fx);
	for (int y=0;y<h;y++)
		xfftPool.AddWork(y);
	xfftPool.WaitUntilDone();

	auto fy = [&] (int x) {
		std::complex<float>* src = ALLOCA_ARRAY(std::complex<float>, h);
		std::complex<float>* tmp = ALLOCA_ARRAY(std::complex<float>, std::max(w,h));
		for (int y=0;y<h;y++)
			src[y] = d[y*w+x];
		yfft.transform(src, tmp);
		for(int y=0;y<h;y++)
			d[w*y+x]=tmp[y];
	};
	ThreadPool<int, std::function<void (int x) > > yfftPool(fy);
	for (int x=0;x<w;x++)
		yfftPool.AddWork(x);
	yfftPool.WaitUntilDone();
}

// Search in the neighbourhood of the given position, and mark visited spots (so the next run cannot mark it)
Position SearchArea(std::complex<float>* cimg, int w, int h,int px,int py, int dist)
{
	Position best(px,py);
	float bestValue = cimg[py*w+px].real();

	int minY = std::max(0,py-dist);
	int minX = std::max(0,px-dist);
	int maxX = std::min(w-1,px+dist);
	int maxY = std::min(h-1,py+dist);

	int dist2 =dist*dist;
	for (int y=minY;y<=maxY;y++)
		for (int x=minX;x<=maxX;x++) {
			int dx=x-px,dy=y-py;
			if (dx*dx+dy*dy<=dist2)
			{
				float v = cimg[y*w+x].real();
				if(v > bestValue) {
					best = Position(x,y);
					bestValue = v;
				}
				cimg[y*w+x] = std::complex<float>();
			}
		}
	return best;
}


void RecenterAndFilter(ImageData* img, std::vector<Position> & pts, Config *cfg )
{
	int roi = cfg->roi;
	std::list<Position> newlist;

	for (int i=0;i<pts.size();i++) {

		// Compute COM
//		for (int 
		auto bp = pts[i];

		if (bp.x < 0 || bp.y < 0 || bp.x > img->w-1-roi || bp.y > img->h-1-roi)
			continue;

		float sum = 0.0f;
		for (int y=0;y<roi;y++) 
			for (int x=0;x<roi;x++) {
				float v = img->at(x+bp.x,y+bp.y);
				sum += v;
			}

		float sumX=0,sumY=0;
		float mean = sum/(roi*roi);
		sum=0;
		for (int y=0;y<roi;y++) {
			for (int x=0;x<roi;x++) {
				float v = img->at(x+bp.x,y+bp.y)-mean;
				v=v*v;
				sum += v;
				sumX += v*x;
				sumY += v*y;
			}
		}

		if (sum==0.0f)
			continue;

		bp.x += sumX/sum-roi/2;
		bp.y += sumY/sum-roi/2;

		// discard them if they are now getting outside of the image boundary
		if (bp.x < 0 || bp.y < 0 || bp.x > img->w-1-roi || bp.y > img->h-1-roi)
			continue;

		float m = cfg->MinPixelDistance();
		bool accepted=true;
		for (auto c=newlist.begin();c!=newlist.end();++c) {
			int dx=c->x-bp.x, dy=c->y-bp.y;
			if (dx*dx+dy*dy<m*m) {
				// bp is very close to *c, so they're probably both shitty beads
				newlist.erase(c);
				accepted=false;
				break;
			}
		}
	
		if (accepted)
			newlist.push_back(bp);
	}

	pts.clear();
	pts.insert(pts.begin(), newlist.begin(),newlist.end());
}

std::vector<Position> BeadFinder::Find(ImageData* img, float* sample, Config* cfg)
{
	typedef std::complex<float> pixelc_t;

	// Compute nearest power of two dimension
	int w=NextPowerOf2(img->w);
	int h=NextPowerOf2(img->h);

	// Convert to std::complex and copy the image in there (subtract mean first)
	float mean = img->mean();
	pixelc_t* cimg = new pixelc_t[w*h];
	std::fill(cimg, cimg+w*h, pixelc_t());
	for (int y=0;y<img->h;y++) {
		for (int x=0;x<img->w;x++) {
			cimg[y*w+x] = img->data[y*img->w+x]-mean;
		}
	}

	// Inplace 2D fft
	FFT2D(cimg,w,h,false);

	// Create an image to put the sample in with size [w,h] as well
	pixelc_t* smpimg = new pixelc_t[w*h];
	std::fill(smpimg, smpimg+w*h, pixelc_t());
	float smpmean = ImageData(sample,cfg->roi,cfg->roi).mean();
	for (int y=0;y<cfg->roi;y++) 
		for (int x=0;x<cfg->roi;x++) 
			smpimg[ ((y-cfg->roi)&(h-1)) * w + ( (x-cfg->roi)&(w-1))] = sample[cfg->roi*y+x]-smpmean;

//	ComplexToJPEGFile("smpimg.jpg", smpimg, w,h);
	FFT2D(smpimg,w,h,false);

	for (int i=0;i<w*h;i++)
		cimg[i]*=smpimg[i];
	FFT2D(cimg,w,h,true);

	float maxVal = 0.0f;
	for (int i=0;i<w*h;i++)
		maxVal=std::max(maxVal, cimg[i].real());

	std::vector<Position> pts;
	for (int y=0;y<img->h;y++) {
		for (int x=0;x<img->w;x++) {
			if (cimg[y*w+x].real()>maxVal*cfg->similarity )
				pts.push_back ( SearchArea(cimg, w, h, x,y, cfg->MinPixelDistance()) );
		}
	}
	//ComplexToJPEGFile("result.jpg", cimg, w,h);
	delete[] smpimg;
	delete[] cimg;

	RecenterAndFilter(img, pts, cfg);

	return pts;
}

std::vector<Position> BeadFinder::Find(uint8_t* img, int pitch, int w,int h, int smpCornerX, int smpCornerY, BeadFinder::Config* cfg)
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

	std::vector<Position> beads = Find(&fimg, fsmp, cfg);

	delete[] fsmp;
	fimg.free();
	return beads;
}

