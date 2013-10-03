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

#define SFFT_BOTH 
#include "../cudatrack/simplefft.h"

const float XCorScale = 1.0f; // keep this at 1, because linear oversampling was obviously a bad idea..

#include "DebugResultCompare.h"

static int round(xcor_t f) { return (int)(f+0.5f); }
template<typename T>
T conjugate(const T &v) { return T(v.real(),-v.imag()); }

const float QIWeights[QI_LSQFIT_NWEIGHTS] = QI_LSQFIT_WEIGHTS;
const float ZLUTWeights[ZLUT_LSQFIT_NWEIGHTS] = ZLUT_LSQFIT_WEIGHTS;

CPUTracker::CPUTracker(int w, int h, int xcorwindow)
{
	width = w;
	height = h;

	xcorBuffer = 0;
	
	mean=0.0f;
	srcImage = new float [w*h];
	debugImage = new float [w*h];
	std::fill(srcImage, srcImage+w*h, 0.0f);
	std::fill(debugImage, debugImage+w*h, 0.0f);

	zluts = 0;
	zlut_planes = zlut_res = zlut_count = zlut_angularSteps = 0;
	zlut_minradius = zlut_maxradius = 0.0f;
	xcorw = xcorwindow;

	qi_radialsteps = 0;
	qi_fft_forward = qi_fft_backward = 0;
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
}

void CPUTracker::SetImageFloat(float *src)
{
	for (int k=0;k<width*height;k++)
		srcImage[k]=src[k];
	mean=0.0f;
}

void CPUTracker::ApplyOffsetGain(float* offset, float *gain, float offsetFactor, float gainFactor) 
{
	for (int i=0;i<width*height;i++)
		srcImage[i] = (srcImage[i]+offset[i]*offsetFactor)*gain[i]*gainFactor;
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



void XCor1DBuffer::XCorFFTHelper(complexc* prof, complexc *prof_rev, xcor_t* result)
{
	complexc* fft_out = (complexc*)ALLOCA(sizeof(complexc)*xcorw);
	complexc* fft_out_rev = (complexc*)ALLOCA(sizeof(complexc)*xcorw);

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

	complexc* prof = (complexc*)ALLOCA(sizeof(complexc)*xcorw);
	complexc* prof_rev = (complexc*)ALLOCA(sizeof(complexc)*xcorw);
	float* prof_autocor = (xcor_t*)ALLOCA(sizeof(float)*xcorw);

	boundaryHit = false;
	for (int k=0;k<iterations;k++) {
		boundaryHit = CheckBoundaries(pos, XCorScale*xcorw/2);

		float xmin = pos.x - XCorScale * xcorw/2;
		float ymin = pos.y - XCorScale * xcorw/2;

		// generate X position xcor array (summing over y range)
		for (int x=0;x<xcorw;x++) {
			xcor_t s = 0.0f;
			int n=0;
			for (int y=0;y<profileWidth;y++) {
				float xp = x * XCorScale + xmin;
				float yp = pos.y + XCorScale * (y - profileWidth/2);
				bool outside;
				s += Interpolate(srcImage, width, height, xp, yp, &outside);
				n += outside?0:1;
				MARKPIXELI(xp, yp);
			}
			s/=n;
			prof [x] = s;
			prof_rev [xcorw-x-1] = s;
		}

		xcorBuffer->XCorFFTHelper(prof, prof_rev, prof_autocor);
		xcor_t offsetX = ComputeMaxInterp<float,QI_LSQFIT_NWEIGHTS>::Compute(prof_autocor, xcorw, QIWeights) - (xcor_t)xcorw/2;

		// generate Y position xcor array (summing over x range)
		for (int y=0;y<xcorw;y++) {
			xcor_t s = 0.0f; 
			int n=0;
			for (int x=0;x<profileWidth;x++) {
				float xp = pos.x + XCorScale * (x - profileWidth/2);
				float yp = y * XCorScale + ymin;
				bool outside;
				s += Interpolate(srcImage,width,height, xp, yp, &outside);
				n += outside?0:1;
				MARKPIXELI(xp,yp);
			}
			s/=n;
			prof[y] = s;
			prof_rev [xcorw-y-1] =s;
		}

		xcorBuffer->XCorFFTHelper(prof,prof_rev, prof_autocor);
		//WriteImageAsCSV("xcorautoconv.txt",&xcorBuffer->Y_result[0],xcorBuffer->Y_result.size(),1);
		xcor_t offsetY = ComputeMaxInterp<xcor_t, QI_LSQFIT_NWEIGHTS>::Compute(prof_autocor, xcorw, QIWeights) - (xcor_t)xcorw/2;

		pos.x += (offsetX - 1) * XCorScale * 0.5f;
		pos.y += (offsetY - 1) * XCorScale * 0.5f;
	}

	return pos;
}

vector2f CPUTracker::ComputeQI(vector2f initial, int iterations, int radialSteps, int angularStepsPerQ, 
	float angStepIterationFactor, float minRadius, float maxRadius, bool& boundaryHit)
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
	if(!qi_fft_forward || qi_radialsteps != nr) {
		if(qi_fft_forward) {
			delete qi_fft_forward;
			delete qi_fft_backward;
		}
		qi_radialsteps = nr;
		qi_fft_forward = new kissfft<qi_t>(nr*2,false);
		qi_fft_backward = new kissfft<qi_t>(nr*2,true);
	}

	qi_t* buf = (qi_t*)ALLOCA(sizeof(qi_t)*nr*4);
	qi_t* q0=buf, *q1=buf+nr, *q2=buf+nr*2, *q3=buf+nr*3;
	qic_t* concat0 = (qic_t*)ALLOCA(sizeof(qic_t)*nr*2);
	qic_t* concat1 = concat0 + nr;

	vector2f center = initial;

	float pixelsPerProfLen = (maxRadius-minRadius)/radialSteps;
	boundaryHit = false;

	if (width < maxRadius || height < maxRadius) {
		boundaryHit = true;
		return initial;
	}

	float angsteps = angularStepsPerQ / powf(angStepIterationFactor, iterations);
	
	for (int k=0;k<iterations;k++){
		// check bounds
		boundaryHit = CheckBoundaries(center, maxRadius);

		for (int q=0;q<4;q++) {
			ComputeQuadrantProfile(buf+q*nr, nr, angsteps, q, minRadius, maxRadius, center);
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

		float offsetX = QI_ComputeOffset(concat0, nr, 0);

		// Build Iy = [ qB(-r)  qT(r) ]
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			concat1[r] = q0[r]+q1[r];
			concat0[nr-r-1] = q2[r]+q3[r];
		}
		
		float offsetY = QI_ComputeOffset(concat0, nr, 1);

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

// Profile is complexc[nr*2]
CPUTracker::qi_t CPUTracker::QI_ComputeOffset(CPUTracker::qic_t* profile, int nr, int axisForDebug)
{
	qic_t* reverse = ALLOCA_ARRAY(qic_t, nr*2);
	qic_t* fft_out = ALLOCA_ARRAY(qic_t, nr*2);
	qic_t* fft_out2 = ALLOCA_ARRAY(qic_t, nr*2);

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
	qi_t* autoconv = ALLOCA_ARRAY(qi_t, nr*2);
	for(int x=0;x<nr*2;x++)  {
		autoconv[x] = fft_out2[(x+nr)%(nr*2)].real();
	}

	float maxPos = ComputeMaxInterp<qi_t, QI_LSQFIT_NWEIGHTS>::Compute(autoconv, nr*2, QIWeights);
	return (maxPos - nr) / (3.14159265359f * 0.5f);
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
			}
		}

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


void CPUTracker::ComputeQuadrantProfile(CPUTracker::qi_t* dst, int radialSteps, int angularSteps,
				int quadrant, float minRadius, float maxRadius, vector2f center)
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

	float total = 0.0f;
	float rstep = (maxRadius - minRadius) / radialSteps;
	for (int i=0;i<radialSteps; i++) {
		float sum = 0.0f;
		float r = minRadius + rstep * i;

		int nPixels = 0;
		float angstepf = (float) quadrantDirs.size() / angularSteps;
		for (int a=0;a<angularSteps;a++) {
			int i = (int)angstepf * a;
			float x = center.x + mx*quadrantDirs[i].x * r;
			float y = center.y + my*quadrantDirs[i].y * r;

			bool outside;
			float v = Interpolate(srcImage,width,height, x,y, &outside);
			if (!outside) {
				sum += v;
				nPixels++;
				MARKPIXELI(x,y);
			}
		}

		dst[i] = nPixels>=MIN_RADPROFILE_SMP_COUNT ? sum/nPixels : mean;
		total += dst[i];
	}
	#ifdef QI_DBG_EXPORT
	WriteImageAsCSV(SPrintf("qprof%d.txt", quadrant).c_str(), dst, 1, radialSteps);
	#endif
}



vector2f CPUTracker::ComputeMeanAndCOM(float bgcorrection)
{
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;

	for (int y=0;y<height;y++)
		for (int x=0;x<width;x++) {
			float v = GetPixel(x,y);
			sum += v;
			sum2 += v*v;
		}

	float invN = 1.0f/(width*height);
	mean = sum * invN;
	stdev = sqrtf(sum2 * invN - mean * mean);
	sum = 0.0f;

	for (int y=0;y<height;y++)
		for(int x=0;x<width;x++)
		{
			float v = GetPixel(x,y);
			v = std::max(0.0f, fabs(v-mean)-bgcorrection*stdev);
			sum += v;
			momentX += x*v;
			momentY += y*v;
		}
	vector2f com;
	com.x = momentX / (float)sum;
	com.y = momentY / (float)sum;
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

void CPUTracker::SetRadialZLUT(float* data, int planes, int res, int numLUTs, float minradius, float maxradius, int angularSteps, bool copyMemory, bool useCorrelation, float* radweights)
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
	zlut_angularSteps = angularSteps;
	zlut_useCorrelation = useCorrelation;

	if (radweights) {
		zlut_radialweights.resize(zlut_res);
		std::copy(radweights, radweights+zlut_res, zlut_radialweights.begin());
	} else
		zlut_radialweights.clear();
}



float CPUTracker::ComputeZ(vector2f center, int angularSteps, int zlutIndex, bool crp, bool* boundaryHit, float* profile, float* cmpProf, bool normalize)
{
	if (!zluts)
		return 0.0f;
	
	float* rprof = ALLOCA_ARRAY(float, zlut_res);
	float* rprof_diff = ALLOCA_ARRAY(float, zlut_planes);

	ComputeRadialProfile(rprof, zlut_res, angularSteps, zlut_minradius, zlut_maxradius, center, crp, boundaryHit, normalize);
	if (profile) std::copy(rprof, rprof+zlut_res, profile);

	//WriteImageAsCSV("zlutradprof-cpu.txt", rprof, zlut_res, 1);

	// Now compare the radial profile to the profiles stored in Z

	float* zlut_sel = GetRadialZLUT(zlutIndex);

	for (int k=0;k<zlut_planes;k++) {
		float diffsum = 0.0f;
		for (int r = 0; r<zlut_res;r++) {
			float d;
			if (zlut_useCorrelation) 
				d = rprof[r]*zlut_sel[k*zlut_res+r];
			else {
				float diff = rprof[r]-zlut_sel[k*zlut_res+r];
				d = -diff*diff;
			}
			if(!zlut_radialweights.empty())
				d *= zlut_radialweights[r];
			diffsum += d;
		}
		rprof_diff[k] = diffsum;
	}

	if (cmpProf) {
		//cmpProf->resize(zlut_planes);
		std::copy(rprof_diff, rprof_diff+zlut_planes, cmpProf);
	}
	
#ifdef _DEBUG
	static int started=0;
	std::string file = GetLocalModulePath() + "/zlutcmp-cpu.txt";
	if( zlutIndex == 0) {
		if (started++==0) remove(file.c_str());
		WriteArrayAsCSVRow(file.c_str(), rprof_diff, zlut_planes, true);
	}
#endif

	float z = ComputeMaxInterp<float, ZLUT_LSQFIT_NWEIGHTS>::Compute(rprof_diff, zlut_planes, ZLUTWeights);
	return z;
}

vector2f CPUTracker::Compute2DXCor()
{
	return vector2f();
}