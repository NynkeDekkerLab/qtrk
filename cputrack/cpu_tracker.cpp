/*

CPU only tracker

*/


#include "std_incl.h"
#include <exception>
#include <cmath>

// 
//#define QI_DBG_EXPORT
#define ZLUT_CMPDATA

#pragma warning(disable: 4996) // Function call with parameters that may be unsafe

#include "QueuedTracker.h"
#include "cpu_tracker.h"
#include "LsqQuadraticFit.h"
#include "random_distr.h"
#include "CubicBSpline.h"

#define SFFT_BOTH 
#include "../cudatrack/simplefft.h"

const float XCorScale = 1.0f; // keep this at 1, because linear oversampling was a bad idea..

#include "DebugResultCompare.h"

static int round(scalar_t f) { return (int)(f+0.5f); }
template<typename T>
T conjugate(const T &v) { return T(v.real(),-v.imag()); }

const scalar_t QIWeights[QI_LSQFIT_NWEIGHTS] = QI_LSQFIT_WEIGHTS;
const float ZLUTWeights[ZLUT_LSQFIT_NWEIGHTS] = ZLUT_LSQFIT_WEIGHTS;
const double ZLUTWeights_d[ZLUT_LSQFIT_NWEIGHTS] = ZLUT_LSQFIT_WEIGHTS;

static int clamp(int v, int a,int b) { return std::max(a, std::min(b, v)); }


CPUTracker::CPUTracker(int w, int h, int xcorwindow)
{
	width = w;
	height = h;
	trackerID = 0;

	xcorBuffer = 0;
	
	mean=0.0f;
	srcImage = new float [w*h];
	debugImage = new float [w*h];
	std::fill(srcImage, srcImage+w*h, 0.0f);
	std::fill(debugImage, debugImage+w*h, 0.0f);

	zluts = 0;
	zlut_planes = zlut_res = zlut_count = 0;
	zlut_minradius = zlut_maxradius = 0.0f;
	xcorw = xcorwindow;
	qa_fft_forward = qa_fft_backward = 0;

	qi_radialsteps = 0;
	qi_fft_forward = qi_fft_backward = 0;

	fft2d=0;
}

CPUTracker::~CPUTracker()
{
	delete[] srcImage;
	delete[] debugImage;
	if (zluts && zlut_memoryOwner) 
		delete[] zluts;

	if (xcorBuffer)
		delete xcorBuffer;

	if (qi_fft_forward) { 
		delete qi_fft_forward;
		delete qi_fft_backward;
	}

	if (qa_fft_backward) {
		delete qa_fft_backward;
		delete qa_fft_forward;
	}

	if (fft2d)
		delete fft2d;
}

void CPUTracker::SetImageFloat(float *src)
{
	for (int k=0;k<width*height;k++)
		srcImage[k]=src[k];
	mean=0.0f;
}

void CPUTracker::ApplyOffsetGain(float* offset, float *gain, float offsetFactor, float gainFactor) 
{
	if (offset && !gain) {
		for (int i=0;i<width*height;i++)
			srcImage[i] = srcImage[i]+offset[i]*offsetFactor;
	}
	if (gain && !offset) {
		for (int i=0;i<width*height;i++)
			srcImage[i] = srcImage[i]*gain[i]*gainFactor;
	}
	if (gain && offset)  {
		for (int i=0;i<width*height;i++)
			srcImage[i] = (srcImage[i]+offset[i]*offsetFactor)*gain[i]*gainFactor;
	}
}


#ifdef _DEBUG

inline void markPixel(float* img, int x,int y, int w,int h, float maxv) {
	if (x>=0 && y>=0 && x < w && y<h)
		img[y*w+x]+=maxv*0.1f;
}

inline void _markPixels(float x,float y, float* img, int w, int h, float mv)
{
	int rx=(int)x, ry=(int)y;
	if (rx >=0 && ry >= 0 && rx+1<w && ry+1<h) {
		img[ry*w+rx] += mv;
		img[ry*w+rx+1] += mv;
		img[(ry+1)*w+rx] += mv;
		img[(ry+1)*w+rx+1] += mv;
	}
}
	#define MARKPIXEL(x,y) markPixel(debugImage, (x),(y),width,height,maxImageValue) 
	#define MARKPIXELI(x,y) _markPixels(x,y,debugImage, width, height, maxImageValue*0.1f);
#else
	#define MARKPIXEL(x,y)
	#define MARKPIXELI(x,y)
#endif



void XCor1DBuffer::XCorFFTHelper(complex_t* prof, complex_t *prof_rev, scalar_t* result)
{
	complex_t* fft_out = (complex_t*)ALLOCA(sizeof(complex_t)*xcorw);
	complex_t* fft_out_rev = (complex_t*)ALLOCA(sizeof(complex_t)*xcorw);

	fft_forward.transform(prof, fft_out);
	fft_forward.transform(prof_rev, fft_out_rev);

	// Multiply with conjugate of reverse
	for (int x=0;x<xcorw;x++) {
		fft_out[x] *= conjugate(fft_out_rev[x]);
	}

	fft_backward.transform(fft_out, fft_out_rev);

	for (int x=0;x<xcorw;x++)
		result[x] = fft_out_rev[ (x+xcorw/2) % xcorw ].real();
}

// Returns true if bounds are crossed
bool CPUTracker::KeepInsideBoundaries(vector2f* center, float radius)
{
	bool boundaryHit = center->x + radius >= width ||
		center->x - radius < 0 ||
		center->y + radius >= height ||
		center->y - radius < 0;

	if (center->x - radius < 0.0f)
		center->x = radius;

	if (center->y - radius < 0.0f)
		center->y = radius;

	if (center->x + radius >= width)
		center->x = width-radius-1;

	if (center->y + radius >= height)
		center->y = height-radius-1;
	return boundaryHit;
}

bool CPUTracker::CheckBoundaries(vector2f center, float radius)
{
	return center.x + radius >= width ||
		center.x - radius < 0 ||
		center.y + radius >= height ||
		center.y - radius < 0;
}

vector2f CPUTracker::ComputeXCorInterpolated(vector2f initial, int iterations, int profileWidth, bool& boundaryHit)
{
	// extract the image
	vector2f pos = initial;

	if (!xcorBuffer)
		xcorBuffer = new XCor1DBuffer(xcorw);

	if (xcorw < profileWidth)
		profileWidth = xcorw;

#ifdef _DEBUG
	std::copy(srcImage, srcImage+width*height, debugImage);
	maxImageValue = *std::max_element(srcImage,srcImage+width*height);
#endif

	if (xcorw > width || xcorw > height) {
		boundaryHit = true;
		return initial;
	}

	complex_t* prof = (complex_t*)ALLOCA(sizeof(complex_t)*xcorw);
	complex_t* prof_rev = (complex_t*)ALLOCA(sizeof(complex_t)*xcorw);
	scalar_t* prof_autocor = (scalar_t*)ALLOCA(sizeof(scalar_t)*xcorw);

	boundaryHit = false;
	for (int k=0;k<iterations;k++) {
		boundaryHit = CheckBoundaries(pos, XCorScale*xcorw/2);

		float xmin = pos.x - XCorScale * xcorw/2;
		float ymin = pos.y - XCorScale * xcorw/2;

		// generate X position xcor array (summing over y range)
		for (int x=0;x<xcorw;x++) {
			scalar_t s = 0.0f;
			int n=0;
			for (int y=0;y<profileWidth;y++) {
				scalar_t xp = x * XCorScale + xmin;
				scalar_t yp = pos.y + XCorScale * (y - profileWidth/2);
				bool outside;
				s += Interpolate(srcImage, width, height, xp, yp, &outside);
				n += outside?0:1;
				MARKPIXELI(xp, yp);
			}
			if (n >0) s/=n;
			else s=0;
			prof [x] = s;
			prof_rev [xcorw-x-1] = s;
		}

		xcorBuffer->XCorFFTHelper(prof, prof_rev, prof_autocor);
		scalar_t offsetX = ComputeMaxInterp<scalar_t,QI_LSQFIT_NWEIGHTS>::Compute(prof_autocor, xcorw, QIWeights) - (scalar_t)xcorw/2;

		// generate Y position xcor array (summing over x range)
		for (int y=0;y<xcorw;y++) {
			scalar_t s = 0.0f; 
			int n=0;
			for (int x=0;x<profileWidth;x++) {
				scalar_t xp = pos.x + XCorScale * (x - profileWidth/2);
				scalar_t yp = y * XCorScale + ymin;
				bool outside;
				s += Interpolate(srcImage,width,height, xp, yp, &outside);
				n += outside?0:1;
				MARKPIXELI(xp,yp);
			}
			if (n >0) s/=n;
			else s=0;
			prof[y] = s;
			prof_rev [xcorw-y-1] =s;
		}

		xcorBuffer->XCorFFTHelper(prof,prof_rev, prof_autocor);
		//WriteImageAsCSV("xcorautoconv.txt",&xcorBuffer->Y_result[0],xcorBuffer->Y_result.size(),1);
		scalar_t offsetY = ComputeMaxInterp<scalar_t, QI_LSQFIT_NWEIGHTS>::Compute(prof_autocor, xcorw, QIWeights) - (scalar_t)xcorw/2;

		pos.x += (offsetX - 1) * XCorScale * 0.5f;
		pos.y += (offsetY - 1) * XCorScale * 0.5f;
	}

	return pos;
}

void CPUTracker::AllocateQIFFTs(int nr)
{
	if(!qi_fft_forward || qi_radialsteps != nr) {
		if(qi_fft_forward) {
			delete qi_fft_forward;
			delete qi_fft_backward;
		}
		qi_radialsteps = nr;
		qi_fft_forward = new kissfft<scalar_t>(nr*2,false);
		qi_fft_backward = new kissfft<scalar_t>(nr*2,true);
	}
}

vector2f CPUTracker::ComputeQI(vector2f initial, int iterations, int radialSteps, int angularStepsPerQ, 
	float angStepIterationFactor, float minRadius, float maxRadius, bool& boundaryHit, float* radialWeights)
{
	int nr=radialSteps;
#ifdef _DEBUG
	std::copy(srcImage, srcImage+width*height, debugImage);
	maxImageValue = *std::max_element(srcImage,srcImage+width*height);
#endif

	if (angularStepsPerQ != quadrantDirs.size()) {
		quadrantDirs.resize(angularStepsPerQ);
		for (int j=0;j<angularStepsPerQ;j++) {
			float ang = 0.5*3.141593f*(j+0.5f)/(float)angularStepsPerQ;
			quadrantDirs[j] = vector2f( cosf(ang), sinf(ang) );
		}
	}

	AllocateQIFFTs(radialSteps);

	scalar_t* buf = (scalar_t*)ALLOCA(sizeof(scalar_t)*nr*4);
	scalar_t* q0=buf, *q1=buf+nr, *q2=buf+nr*2, *q3=buf+nr*3;
	complex_t* concat0 = (complex_t*)ALLOCA(sizeof(complex_t)*nr*2);
	complex_t* concat1 = concat0 + nr;

	vector2f center = initial;

	float pixelsPerProfLen = (maxRadius-minRadius)/radialSteps;
	boundaryHit = false;

	float angsteps = angularStepsPerQ / powf(angStepIterationFactor, iterations);
	
	for (int k=0;k<iterations;k++){
		// check bounds
		boundaryHit = CheckBoundaries(center, maxRadius);

		for (int q=0;q<4;q++) {
			ComputeQuadrantProfile(buf+q*nr, nr, angsteps, q, minRadius, maxRadius, center, radialWeights);
		}
#ifdef QI_DEBUG
		cmp_cpu_qi_prof.assign (buf,buf+4*nr);
#endif

		// Build Ix = [ qL(-r)  qR(r) ]
		// qL = q1 + q2   (concat0)
		// qR = q0 + q3   (concat1)
		for(int r=0;r<nr;r++) {
			concat0[nr-r-1] = q1[r]+q2[r];
			concat1[r] = q0[r]+q3[r];
		}

		scalar_t offsetX = QI_ComputeOffset(concat0, nr, 0);

		// Build Iy = [ qB(-r)  qT(r) ]
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			concat1[r] = q0[r]+q1[r];
			concat0[nr-r-1] = q2[r]+q3[r];
		}
		
		scalar_t offsetY = QI_ComputeOffset(concat0, nr, 1);

#ifdef QI_DBG_EXPORT
		std::copy(concat0, concat0+nr*2,tmp.begin()+nr*2);
		WriteComplexImageAsCSV("cpuprofxy.txt", &tmp[0], nr, 4);
		dbgprintf("[%d] OffsetX: %f, OffsetY: %f\n", k, offsetX, offsetY);
#endif

		center.x += offsetX * pixelsPerProfLen;
		center.y += offsetY * pixelsPerProfLen;

		angsteps *= angStepIterationFactor;
	}

	return center;
}


template<typename T>
double sum_diff(T* begin, T* end, T* other)
{
	double sd = 0.0;
	for (T* i = begin; i != end; i++, other++) {
		T d = *i - *other;
		sd += abs(d.real()) + abs(d.imag());
	}
	return sd;
}

// Profile is complex_t[nr*2]
scalar_t CPUTracker::QI_ComputeOffset(complex_t* profile, int nr, int axisForDebug)
{
	complex_t* reverse = ALLOCA_ARRAY(complex_t, nr*2);
	complex_t* fft_out = ALLOCA_ARRAY(complex_t, nr*2);
	complex_t* fft_out2 = ALLOCA_ARRAY(complex_t, nr*2);

	for(int x=0;x<nr*2;x++)
		reverse[x] = profile[nr*2-1-x];

	qi_fft_forward->transform(profile, fft_out);
	qi_fft_forward->transform(reverse, fft_out2); // fft_out2 contains fourier-domain version of reverse profile

	// multiply with conjugate
	for(int x=0;x<nr*2;x++)
		fft_out[x] *= conjugate(fft_out2[x]);

	qi_fft_backward->transform(fft_out, fft_out2);

#ifdef QI_DEBUG
	cmp_cpu_qi_fft_out.assign(fft_out2, fft_out2+nr*2);
#endif

	// fft_out2 now contains the autoconvolution
	// convert it to float
	scalar_t* autoconv = ALLOCA_ARRAY(scalar_t, nr*2);
	for(int x=0;x<nr*2;x++)  {
		autoconv[x] = fft_out2[(x+nr)%(nr*2)].real();
	}

	scalar_t maxPos = ComputeMaxInterp<scalar_t, QI_LSQFIT_NWEIGHTS>::Compute(autoconv, nr*2, QIWeights);
	return (maxPos - nr) * (3.14159265359f / 4);
}


// Profile is complex_t[nr*2]
scalar_t CPUTracker::QuadrantAlign_ComputeOffset(complex_t* profile, complex_t* zlut_prof_fft, int nr, int axisForDebug)
{
	complex_t* fft_out = ALLOCA_ARRAY(complex_t, nr*2);

//	WriteComplexImageAsCSV("qa_profile.txt", profile, nr*2, 1);

	qa_fft_forward->transform(profile, fft_out);

	// multiply with conjugate
	for(int x=0;x<nr*2;x++)
		fft_out[x] *= conjugate(zlut_prof_fft[x]);

	qa_fft_backward->transform(fft_out, profile);

#ifdef QI_DEBUG
	cmp_cpu_qi_fft_out.assign(fft_out2, fft_out2+nr*2);
#endif

	// profile now contains the autoconvolution
	// convert it to float
	scalar_t* autoconv = ALLOCA_ARRAY(scalar_t, nr*2);
	for(int x=0;x<nr*2;x++)  {
		autoconv[x] = profile[(x+nr)%(nr*2)].real();
	}

//	WriteImageAsCSV("qa_autoconv.txt", autoconv, nr*2, 1);

	scalar_t maxPos = ComputeMaxInterp<scalar_t, QI_LSQFIT_NWEIGHTS>::Compute(autoconv, nr*2, QIWeights);
	return (maxPos - nr) * (3.14159265359f / 4);
	
}


/*

- Compute quadrants
- Build interpolated profile from ZLUT
- Align X & Y against profile with FFT

Difference with QI: No mirroring of profile, instead of mirror we use ZLUT profile

*/
vector3f CPUTracker::QuadrantAlign(vector3f pos, int beadIndex, int angularStepsPerQuadrant, bool& boundaryHit)
{
	float* zlut = GetRadialZLUT(beadIndex);
	int res=zlut_res;
	complex_t* profile = ALLOCA_ARRAY(complex_t, res*2);
	complex_t* concat0 = ALLOCA_ARRAY(complex_t, res*2);
	complex_t* concat1 = concat0 + res;

	memset(concat0, 0, sizeof(complex_t)*res*2);

	int zp0 = clamp( (int)pos.z, 0, zlut_planes - 1);
	int zp1 = clamp( (int)pos.z + 1, 0, zlut_planes - 1);

	float* zlut0 = &zlut[ res * zp0 ];
	float* zlut1 = &zlut[ res * zp1 ];
	float frac = pos.z - (int)pos.z;
	for (int r=0;r<zlut_res;r++) {
	// Interpolate plane
		double zlutValue = Lerp(zlut0[r], zlut1[r], frac);
		concat0[res-r-1] = concat1[r] = zlutValue;
	}
//	WriteComplexImageAsCSV("qa_zlutprof.txt", concat0, res,2);
	qa_fft_forward->transform(concat0, profile);

	scalar_t* buf = ALLOCA_ARRAY(scalar_t, res*4);
	scalar_t* q0=buf, *q1=buf+res, *q2=buf+res*2, *q3=buf+res*3;

	boundaryHit = CheckBoundaries(vector2f(pos.x,pos.y), zlut_maxradius);
	for (int q=0;q<4;q++) {
		scalar_t *quadrantProfile = buf+q*res;
		ComputeQuadrantProfile(quadrantProfile, res, angularStepsPerQuadrant, q, zlut_minradius, zlut_maxradius, vector2f(pos.x,pos.y));
		NormalizeRadialProfile(quadrantProfile, res);
	}
	
	//WriteImageAsCSV("qa_qdr.txt" , buf, res,4);
	
	float pixelsPerProfLen = (zlut_maxradius-zlut_minradius)/zlut_res;
	boundaryHit = false;

	// Build Ix = [ qL(-r)  qR(r) ]
	// qL = q1 + q2   (concat0)
	// qR = q0 + q3   (concat1)
	for(int r=0;r<res;r++) {
		concat0[res-r-1] = q1[r]+q2[r];
		concat1[r] = q0[r]+q3[r];
	}

	scalar_t offsetX = QuadrantAlign_ComputeOffset(concat0, profile, res, 0);

	// Build Iy = [ qB(-r)  qT(r) ]
	// qT = q0 + q1
	// qB = q2 + q3
	for(int r=0;r<res;r++) {
		concat1[r] = q0[r]+q1[r];
		concat0[res-r-1] = q2[r]+q3[r];
	}
		
	scalar_t offsetY = QuadrantAlign_ComputeOffset(concat0, profile, res, 1);

	return vector3f(pos.x + offsetX * pixelsPerProfLen, pos.y + offsetY * pixelsPerProfLen, pos.z);
}

CPUTracker::Gauss2DResult CPUTracker::Compute2DGaussianMLE(vector2f initial, int iterations, float sigma)
{
	vector2f pos = initial;
	float I0 = mean*0.5f*width*height;
	float bg = mean*0.5f;

	const float _1oSq2Sigma = 1.0f / (sqrtf(2) * sigma);
	const float _1oSq2PiSigma = 1.0f / (sqrtf(2*3.14159265359f) * sigma);
	const float _1oSq2PiSigma3 = 1.0f / (sqrtf(2*3.14159265359f) * sigma*sigma*sigma);

	for (int i=0;i<iterations;i++)
	{
		double dL_dx = 0.0; 
		double dL_dy = 0.0; 
		double dL_dI0 = 0.0;
		double dL_dIbg = 0.0;
		double dL2_dx = 0.0;
		double dL2_dy = 0.0;
		double dL2_dI0 = 0.0;
		double dL2_dIbg = 0.0;

		double mu_sum = 0.0;
				
		for (int y=0;y<height;y++)
		{
			for (int x=0;x<width;x++)
			{
		        float Xexp0 = (x-pos.x + .5f) * _1oSq2Sigma;
				float Yexp0 = (y-pos.y + .5f) * _1oSq2Sigma;
        
				float Xexp1 = (x-pos.x - .5f) * _1oSq2Sigma;
				float Yexp1 = (y-pos.y - .5f) * _1oSq2Sigma;
				
				float DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				float DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
				float mu = bg + I0 * DeltaX * DeltaY;
				
				float dmu_dx = I0*_1oSq2PiSigma * ( expf(-Xexp1*Xexp1) - expf(-Xexp0*Xexp0)) * DeltaY;

				float dmu_dy = I0*_1oSq2PiSigma * ( expf(-Yexp1*Yexp1) - expf(-Yexp0*Yexp0)) * DeltaX;
				float dmu_dI0 = DeltaX*DeltaY;
				float dmu_dIbg = 1;
        
				float smp = GetPixel(x,y);
				float f = smp / mu - 1;
				dL_dx += dmu_dx * f;
				dL_dy += dmu_dy * f;
				dL_dI0 += dmu_dI0 * f;
				dL_dIbg += dmu_dIbg * f;

				float d2mu_dx = I0*_1oSq2PiSigma3 * ( (x - pos.x - .5f) * expf (-Xexp1*Xexp1) - (x - pos.x + .5f) * expf(-Xexp0*Xexp0) ) * DeltaY;
				float d2mu_dy = I0*_1oSq2PiSigma3 * ( (y - pos.y - .5f) * expf (-Yexp1*Yexp1) - (y - pos.y + .5f) * expf(-Yexp0*Yexp0) ) * DeltaX;
				dL2_dx += d2mu_dx * f - dmu_dx*dmu_dx * smp / (mu*mu);
				dL2_dy += d2mu_dy * f - dmu_dy*dmu_dy * smp / (mu*mu);
				dL2_dI0 += -dmu_dI0*dmu_dI0 * smp / (mu*mu);
				dL2_dIbg += -smp / (mu*mu);

				mu_sum += mu;
			}
		}

		double mean_mu = mu_sum / (width*height);

		pos.x -= dL_dx / dL2_dx;
		pos.y -= dL_dy / dL2_dy;
		I0 -= dL_dI0 / dL2_dI0;
		bg -= dL_dIbg / dL2_dIbg;
	}

	Gauss2DResult r;
	r.pos = pos;
	r.I0 = I0;
	r.bg = bg;

	return r;
}


float CPUTracker::ComputeAsymmetry(vector2f center, int radialSteps, int angularSteps, 
						float minRadius, float maxRadius, float *dstAngProf)
{
	vector2f* radialDirs = (vector2f*)ALLOCA(sizeof(vector2f)*angularSteps);
	for (int j=0;j<angularSteps;j++) {
		float ang = 2*3.141593f*j/(float)angularSteps;
		radialDirs[j] = vector2f (cosf(ang), sinf(ang));
	}

	// profile along single radial direction
	float* rline = (float*)ALLOCA(sizeof(float)*radialSteps);

	if (!dstAngProf) 
		dstAngProf = (float*)ALLOCA(sizeof(float)*radialSteps);

	double totalrmssum2 = 0.0f;
	float rstep = (maxRadius-minRadius) / radialSteps;

	for (int a=0;a<angularSteps;a++) {
		// fill rline
		// angularProfile[a] = rline COM 
		
		float r = minRadius;
		for (int i=0;i<radialSteps;i++) {
			float x = center.x + radialDirs[a].x * r;
			float y = center.y + radialDirs[a].y * r;
			rline[i] = Interpolate(srcImage,width,height, x,y);
			r += rstep;
		}

		// Compute rline COM
		dstAngProf[a] = ComputeBgCorrectedCOM1D(rline, radialSteps, 1.0f);
	}
	
	float stdev = ComputeStdDev(dstAngProf, angularSteps);
	return stdev;
}


void CPUTracker::ComputeQuadrantProfile(scalar_t* dst, int radialSteps, int angularSteps,
				int quadrant, float minRadius, float maxRadius, vector2f center ,float* radialWeights)
{
	const int qmat[] = {
		1, 1,
		-1, 1,
		-1, -1,
		1, -1 };
	int mx = qmat[2*quadrant+0];
	int my = qmat[2*quadrant+1];

	if (angularSteps < MIN_RADPROFILE_SMP_COUNT)
		angularSteps = MIN_RADPROFILE_SMP_COUNT;

	for (int i=0;i<radialSteps;i++)
		dst[i]=0.0f;

	scalar_t total = 0.0f;
	scalar_t rstep = (maxRadius - minRadius) / radialSteps;
	for (int i=0;i<radialSteps; i++) {
		scalar_t sum = 0.0f;
		scalar_t r = minRadius + rstep * i;

		int nPixels = 0;
		scalar_t angstepf = (scalar_t) quadrantDirs.size() / angularSteps;
		for (int a=0;a<angularSteps;a++) {
			int i = (int)angstepf * a;
			scalar_t x = center.x + mx*quadrantDirs[i].x * r;
			scalar_t y = center.y + my*quadrantDirs[i].y * r;

			bool outside;
			scalar_t v = Interpolate(srcImage,width,height, x,y, &outside);
			if (!outside) {
				sum += v;
				nPixels++;
				MARKPIXELI(x,y);
			}
		}

		dst[i] = nPixels>=MIN_RADPROFILE_SMP_COUNT ? sum/nPixels : mean;
		if (radialWeights) dst[i] *= radialWeights[i];
		total += dst[i];
	}
	#ifdef QI_DBG_EXPORT
	WriteImageAsCSV(SPrintf("qprof%d.txt", quadrant).c_str(), dst, 1, radialSteps);
	#endif
}



vector2f CPUTracker::ComputeMeanAndCOM(float bgcorrection)
{
	double sum=0, sum2=0;
	double momentX=0;
	double momentY=0;
	// Order is important 
	//COM: x:53.959133, y:53.958984
	//COM: x:53.958984, y:53.959133 

	for (int y=0;y<height;y++) {
		for (int x=0;x<width;x++)  {
			float v = GetPixel(x,y);
			sum += v;
			sum2 += v*v;
		}
	}

	float invN = 1.0f/(width*height);
	mean = sum * invN;
	stdev = sqrtf(sum2 * invN - mean * mean);
	sum = 0.0f;

	float *ymom = ALLOCA_ARRAY(float, width);
	float *xmom = ALLOCA_ARRAY(float, height);

	for(int x=0;x<width;x++)
		ymom[x]=0;
	for(int y=0;y<height;y++)
		xmom[y]=0;

	for(int x=0;x<width;x++) {
		for (int y=0;y<height;y++) {
			float v = GetPixel(x,y);
			v = std::max(0.0f, fabs(v-mean)-bgcorrection*stdev);
			sum += v;
//			xmom[y] += x*v;
	//		ymom[x] += y*v;
			momentX += x*(double)v;
			momentY += y*(double)v;
		}
	}
	
	//for (int 

	vector2f com;
	com.x = (float)( momentX / sum );
	com.y = (float)( momentY / sum );
	return com;
}


void CPUTracker::Normalize(float* d)
{
	if (!d) d=srcImage;
	normalize(d, width, height);
}



void CPUTracker::ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius, vector2f center, bool crp, bool* pBoundaryHit, bool normalize)
{
	bool boundaryHit = CheckBoundaries(center, maxradius);
	if (pBoundaryHit) *pBoundaryHit = boundaryHit;

	ImageData imgData (srcImage, width,height);
	if (crp) {
		ComputeCRP(dst, radialSteps, angularSteps, minradius, maxradius, center, &imgData, mean);
	}
	else {
		::ComputeRadialProfile(dst, radialSteps, angularSteps, minradius, maxradius, center, &imgData, mean, normalize);
	}
}

void CPUTracker::SetRadialZLUT(float* data, int planes, int res, int numLUTs, float minradius, float maxradius, bool copyMemory, bool useCorrelation)
{
	if (zluts && zlut_memoryOwner)
		delete[] zluts;
	
	if (copyMemory) {
		zluts = new float[planes*res*numLUTs];
		std::copy(data, data+(planes*res*numLUTs), zluts);
	} else
		zluts = data;
	zlut_memoryOwner = copyMemory;
	zlut_planes = planes;
	zlut_res = res;
	zlut_count = numLUTs;
	zlut_minradius = minradius;
	zlut_maxradius = maxradius;
	zlut_useCorrelation = useCorrelation;

	if (qa_fft_backward) {
		delete qa_fft_backward;
		delete qa_fft_forward;
	}

	qa_fft_forward = new kissfft<scalar_t>(res*2,false);
	qa_fft_backward = new kissfft<scalar_t>(res*2,true);
}

void CPUTracker::SetRadialWeights(float* radweights)
{
	if (radweights) {
		zlut_radialweights.resize(zlut_res);
		std::copy(radweights, radweights+zlut_res, zlut_radialweights.begin());
	} else
		zlut_radialweights.clear();
}


float CPUTracker::LUTProfileCompare (float* rprof, int zlutIndex, float* cmpProf, LUTProfileMaxComputeMode maxPosComputeMode, float* fitcurve, int *maxPos)
{
	if (!zluts)
		return 0.0f;
	
	double* rprof_diff = ALLOCA_ARRAY(double, zlut_planes);
	//WriteImageAsCSV("zlutradprof-cpu.txt", rprof, zlut_res, 1);

	// Now compare the radial profile to the profiles stored in Z
	float* zlut_sel = GetRadialZLUT(zlutIndex);

#if 0
	float* zlut_norm = ALLOCA_ARRAY(float, zlut_res);
	float* prof_norm = ALLOCA_ARRAY(float, zlut_res);
	for (int r=0;r<zlut_res;r++)
		prof_norm[r] = rprof[r] * zlut_radialweights[r];
	NormalizeRadialProfile(prof_norm, zlut_res);

	for (int k=0;k<zlut_planes;k++) {
		double diffsum = 0.0f;

		if (zlut_radialweights.empty()) {
			for (int r=0;r<zlut_res;r++)  zlut_norm[r]=zlut_sel[k*zlut_res+r];
		} else {
			for (int r=0;r<zlut_res;r++) {
				zlut_norm[r] = zlut_sel[k*zlut_res+r] * zlut_radialweights[r];
			}
		}
		NormalizeRadialProfile(zlut_norm, zlut_res);

		for (int r = 0; r<zlut_res;r++) {
			double d = prof_norm[r]-zlut_norm[r];
			d = -d*d;
			diffsum += d;
		}
		rprof_diff[k] = diffsum;
	}
#else
	for (int k=0;k<zlut_planes;k++) {
		double diffsum = 0.0f;
		for (int r = 0; r<zlut_res;r++) {
			double d = rprof[r]-zlut_sel[k*zlut_res+r];
			if (!zlut_radialweights.empty())
				d*=zlut_radialweights[r];
			d = -d*d;
			diffsum += d;
		}
		rprof_diff[k] = diffsum;
	}
#endif

	if (cmpProf) {
		//cmpProf->resize(zlut_planes);
		std::copy(rprof_diff, rprof_diff+zlut_planes, cmpProf);
	}

	if (maxPosComputeMode == LUTProfMaxQuadraticFit) {
		if (maxPos) {
			*maxPos = std::max_element(rprof_diff, rprof_diff+zlut_planes) - rprof_diff;
		}

		if (fitcurve) {
			LsqSqQuadFit<double> fit;
			float z= ComputeMaxInterp<double, ZLUT_LSQFIT_NWEIGHTS>::Compute(rprof_diff, zlut_planes, ZLUTWeights_d, &fit);
			int iMax = std::max_element(rprof_diff, rprof_diff+zlut_planes) - rprof_diff;
			for (int i=0;i<zlut_planes;i++)
				fitcurve[i] = fit.compute(i-iMax);
			return z;
		} else {
			return ComputeMaxInterp<double, ZLUT_LSQFIT_NWEIGHTS>::Compute(rprof_diff, zlut_planes, ZLUTWeights_d);
		}
	}
	else if (maxPosComputeMode == LUTProfMaxSimpleInterp) {
		assert(0);
		return 0;
	} else 
		return ComputeSplineFitMaxPos(rprof_diff, zlut_planes);
}


float CPUTracker::LUTProfileCompareAdjustedWeights(float* rprof, int zlutIndex, float z_estim)
{
	if (!zluts)
		return 0.0f;
	
	double* rprof_diff = ALLOCA_ARRAY(double, zlut_planes);

	// Now compare the radial profile to the profiles stored in Z
	float* zlut_sel = GetRadialZLUT(zlutIndex);

	int zi = (int)z_estim;
	if (zi > zlut_planes-2) zi = zlut_planes-2;
	float* z0 = &zlut_sel[zi*zlut_res];
	float* z1 = &zlut_sel[(zi+1)*zlut_res];
	double* zlut_weights = ALLOCA_ARRAY(double, zlut_res);
	for (int i=0;i<zlut_res;i++)
		zlut_weights[i] = fabs((double)z1[i]-(double)z0[i]) * ( zlut_minradius + (zlut_maxradius - zlut_minradius) * i/(float)zlut_res );

	for (int k=0;k<zlut_planes;k++) {
		double diffsum = 0.0f;
		for (int r = 0; r<zlut_res;r++) {
			double d = (double)rprof[r]-(double)zlut_sel[k*zlut_res+r];
			d = -d*d;
			d *= zlut_weights[r];
			diffsum += d;
		}
		rprof_diff[k] = diffsum;
	}
	//return ComputeSplineFitMaxPos(rprof_diff,zlut_planes);
	return ComputeMaxInterp<double, ZLUT_LSQFIT_NWEIGHTS>::Compute(rprof_diff, zlut_planes, ZLUTWeights_d);
}


void CPUTracker::SaveImage(const char *filename)
{
	FloatToJPEGFile(filename, srcImage, width, height);
}


void CPUTracker::FourierTransform2D()
{
	//kissfft<float> yfft(h, inverse);
	if (!fft2d){
		fft2d = new FFT2D(width, height);
	}

	fft2d->Apply(srcImage);
}

void CPUTracker::FFT2D::Apply(float* d)
{
	int w = xfft.nfft();
	int h = yfft.nfft();

	std::complex<float>* tmpsrc = ALLOCA_ARRAY(std::complex<float>, h);
	std::complex<float>* tmpdst = ALLOCA_ARRAY(std::complex<float>, h);
	
	for(int y=0;y<h;y++) {
		for (int x=0;x<w;x++)
			tmpsrc[x]=d[y*w+x];
		xfft.transform(tmpsrc, &cbuf[y*w]);
	}
	for (int x=0;x<w;x++) {
		for (int y=0;y<h;y++)
			tmpsrc[y]=cbuf[y*w+x];
		yfft.transform(tmpsrc, tmpdst);
		for (int y=0;y<h;y++)
			cbuf[y*w+x]=tmpdst[y];
	}
	// copy and shift
	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++) {
			int dx=(x+w/2)%w;
			int dy=(y+h/2)%h;
			auto v=cbuf[x+y*w];
			d[dx+dy*w] = v.real()*v.real() + v.imag()*v.imag();
		}
	}

//	delete[] tmpsrc;
//	delete[] tmpdst;
}

void CPUTracker::FourierRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius)
{
	FourierTransform2D();
	ImageData img(srcImage,width,height);
	::ComputeRadialProfile(dst, radialSteps, angularSteps, 1, width*0.3f, vector2f(width/2,height/2), &img, 0, false);
	for (int i=0;i<radialSteps;i++) {
		dst[i]=logf(dst[i]);
	}
	//NormalizeRadialProfile(dst, radialSteps);
}

