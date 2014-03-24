#pragma once

#include "QueuedTracker.h"
#include "utils.h"
#include "scalar_types.h"
#include "kissfft.h"


typedef uchar pixel_t;


class XCor1DBuffer {
public:
	XCor1DBuffer(int xcorw) : fft_forward(xcorw, false), fft_backward(xcorw, true), xcorw(xcorw)
	{}

	kissfft<scalar_t> fft_forward, fft_backward;
	int xcorw;

	void XCorFFTHelper(complex_t* xc, complex_t* xcr, scalar_t* result);
};

class CPUTracker
{
public:
	int width, height, xcorw;
	int trackerID;

	float *srcImage, *debugImage;
	float mean, stdev; // Updated by ComputeBgCorrectedCOM()
#ifdef _DEBUG
	float maxImageValue;
#endif
	std::vector<vector2f> radialDirs; // full circle for ZLUT

	// The ZLUT system stores 'zlut_count' number of 2D zlut's, so every bead can be tracked with its own unique ZLUT.
	float* zluts; // size: zlut_planes*zlut_count*zlut_res,		indexing: zlut[index * (zlut_planes * zlut_res) + plane * zlut_res + r]
	bool zlut_memoryOwner; // is this instance the owner of the zluts memory, or is it external?
	int zlut_planes, zlut_res, zlut_count, zlut_angularSteps; 
	float zlut_minradius, zlut_maxradius;
	bool zlut_useCorrelation;
	std::vector<float> zlut_radialweights;
	kissfft<scalar_t> *qa_fft_forward, *qa_fft_backward;

	float* GetRadialZLUT(int index)  { return &zluts[zlut_res*zlut_planes*index]; }

	XCor1DBuffer* xcorBuffer;
	std::vector<vector2f> quadrantDirs; // single quadrant
	int qi_radialsteps;
	kissfft<scalar_t> *qi_fft_forward, *qi_fft_backward;

	class FFT2D {
	public:
		kissfft<float> xfft, yfft;
		std::complex<float> *cbuf;
		FFT2D(int w, int h) : xfft(w,false), yfft(h,false) {cbuf=new std::complex<float>[w*h]; }
		~FFT2D() { delete[] cbuf; }
		void Apply(float* d);
	};
	FFT2D *fft2d;

	float& GetPixel(int x, int y) { return srcImage[width*y+x]; }
	int GetWidth() { return width; }
	int GetHeight() { return height; }
	CPUTracker(int w, int h, int xcorwindow=128);
	~CPUTracker();
	bool KeepInsideBoundaries(vector2f *center, float radius);
	bool CheckBoundaries(vector2f center, float radius);
	vector2f ComputeXCorInterpolated(vector2f initial, int iterations, int profileWidth, bool& boundaryHit);
	vector2f ComputeQI(vector2f initial, int iterations, int radialSteps, int angularStepsPerQuadrant, float angStepIterationFactor, float minRadius, float maxRadius, bool& boundaryHit);

	struct Gauss2DResult {
		vector2f pos;
		float I0, bg;
	};

	Gauss2DResult Compute2DGaussianMLE(vector2f initial ,int iterations, float sigma);

	scalar_t QI_ComputeOffset(complex_t* qi_profile, int nr, int axisForDebug);
	scalar_t QuadrantAlign_ComputeOffset(complex_t* profile, complex_t* zlut_prof_fft, int nr, int axisForDebug);

	float ComputeAsymmetry(vector2f center, int radialSteps, int angularSteps, float minRadius, float maxRadius, float *dstAngProf=0);

	template<typename TPixel> void SetImage(TPixel* srcImage, uint srcpitch);
	void SetImage16Bit(ushort* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }
	void SetImage8Bit(uchar* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }
	void SetImageFloat(float* srcImage);
	void SaveImage(const char *filename);

	vector2f ComputeMeanAndCOM(float bgcorrection=0.0f);
	void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius, vector2f center, bool crp, bool* boundaryHit=0, bool normalize=true);
	void ComputeQuadrantProfile(scalar_t* dst, int radialSteps, int angularSteps, int quadrant, float minRadius, float maxRadius, vector2f center);

	float ComputeZ(vector2f center, int angularSteps, int zlutIndex, bool* boundaryHit=0, float* profile=0, float* cmpprof=0, bool normalizeProfile=true)
	{
		float* prof= ALLOCA_ARRAY(float, zlut_res);
		ComputeRadialProfile(prof,zlut_res,angularSteps, zlut_minradius, zlut_maxradius, center, false, boundaryHit, normalizeProfile);
		return LUTProfileCompare(prof, zlutIndex, cmpprof);
	}
	
	void FourierTransform2D();
	void FourierRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius);

	void Normalize(float *image=0);
	void SetRadialZLUT(float* data, int planes, int res, int num_zluts, float minradius, float maxradius, int angularSteps, bool copyMemory, bool useCorrelation, float* radialweights=0);
	float LUTProfileCompare(float* profile, int zlutIndex, float* cmpProf);
	float* GetDebugImage() { return debugImage; }

	void ApplyOffsetGain(float *offset, float *gain, float offsetFactor, float gainFactor);

	vector3d ZLUTAlignGradientStep(vector3d pos, int beadIndex,vector3d* diff, vector3d step, vector3d deriv_delta);
	vector3d ZLUTAlignNewtonRaphsonIndependentStep(vector3d pos, int beadIndex,vector3d* diff, vector3d deriv_delta);
	vector3d ZLUTAlignNewtonRaphson3DStep(vector3d pos, int beadIndex,vector3d* diff, vector3d deriv_delta);
	vector3d ZLUTAlignSecantMethod(vector3d pos, int beadIndex, int iterations, vector3d deriv_delta);

	double ZLUTAlign_ComputeScore(vector3d pos, int beadIndex);

	void AllocateQIFFTs(int nsteps);
	vector3f QuadrantAlign(vector3f initial, int beadIndex, int angularStepsPerQuadrant, bool& boundaryHit);
};


CPUTracker* CreateCPUTrackerInstance(int w,int h,int xcorw);

template<typename TPixel>
void CPUTracker::SetImage(TPixel* data, uint pitchInBytes)
{
	uchar* bp = (uchar*)data;

	for (int y=0;y<height;y++) {
		for (int x=0;x<width;x++) {
			srcImage[y*width+x] = ((TPixel*)bp)[x];
		}
		bp += pitchInBytes;
	}

	mean=0.0f;
}
