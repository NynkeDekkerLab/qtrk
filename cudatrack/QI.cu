#include "std_incl.h"
#include "utils.h"
#include "QueuedCUDATracker.h"
#include "ImageSampler.h"

#ifdef QI_DEBUG
inline void DbgCopyResult(device_vec<float2>& src, std::vector< std::complex<float> >& dst) {
	cudaDeviceSynchronize();
	std::vector<float2> x(src.size);
	src.copyToHost(x,false,0);
	dst.resize(src.size);
	for(int i=0;i<x.size();i++)
		dst[i]=std::complex<float>(x[i].x,x[i].y);
}
inline void DbgCopyResult(device_vec<float>& src, std::vector< float >& dst) {
	cudaDeviceSynchronize();
	src.copyToHost(dst,false,0);
}
#else
inline void DbgCopyResult(device_vec<float2> src, std::vector< std::complex<float> >& dst) {} 
inline void DbgCopyResult(device_vec<float> src, std::vector< float>& dst) {}
#endif


template<typename TImageSampler>
__device__ void ComputeQuadrantProfile(cudaImageListf& images, int idx, float* dst, const QIParams& params, int quadrant, float2 center, float mean, int angularSteps)
{
	const int qmat[] = {
		1, 1,
		-1, 1,
		-1, -1,
		1, -1 };
	int mx = qmat[2*quadrant+0];
	int my = qmat[2*quadrant+1];

	for (int i=0;i<params.radialSteps;i++)
		dst[i]=0.0f;
	
	float asf = (float)params.trigtablesize / angularSteps;
	float total = 0.0f;
	float rstep = (params.maxRadius - params.minRadius) / params.radialSteps;
	for (int i=0;i<params.radialSteps; i++) {
		float sum = 0.0f;
		float r = params.minRadius + rstep * i;
		int count=0;

		for (int a=0;a<angularSteps;a++) {
			int j = (int)(asf * a);
			float x = center.x + mx*params.cos_sin_table[j].x * r;
			float y = center.y + my*params.cos_sin_table[j].y * r;
			bool outside=false;
			float v = TImageSampler::Interpolated(images, x,y, idx, outside);
			if (!outside) {
				sum += v;
				count ++;
			}
		}

		dst[i] = count >= MIN_RADPROFILE_SMP_COUNT ? sum/count : mean;
		total += dst[i];
	}
}

template<typename TImageSampler>
__global__ void QI_ComputeProfile(BaseKernelParams kp, float3* positions, float* quadrants, float2* profiles, float2* reverseProfiles, const QIParams *params, int angularSteps)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < kp.njobs) {
		int fftlen = params->radialSteps*2;
		float* img_qdr = &quadrants[ idx * params->radialSteps * 4 ];
		for (int q=0;q<4;q++) {
			ComputeQuadrantProfile<TImageSampler> (kp.images, idx, &img_qdr[q*params->radialSteps], *params, q, 
				make_float2(positions[idx].x, positions[idx].y), kp.imgmeans[idx], angularSteps);
		}

		int nr = params->radialSteps;
		qicomplex_t* imgprof = (qicomplex_t*) &profiles[idx * fftlen*2];
		qicomplex_t* x0 = imgprof;
		qicomplex_t* x1 = imgprof + nr*1;
		qicomplex_t* y0 = imgprof + nr*2;
		qicomplex_t* y1 = imgprof + nr*3;

		qicomplex_t* revprof = (qicomplex_t*)&reverseProfiles[idx*fftlen*2];
		qicomplex_t* xrev = revprof;
		qicomplex_t* yrev = revprof + nr*2;

		float* q0 = &img_qdr[0];
		float* q1 = &img_qdr[nr];
		float* q2 = &img_qdr[nr*2];
		float* q3 = &img_qdr[nr*3];

		// Build Ix = qL(-r) || qR(r)
		// qL = q1 + q2   (concat0)
		// qR = q0 + q3   (concat1)
		for(int r=0;r<nr;r++) {
			x0[nr-r-1] = make_float2(q1[r]+q2[r], 0);
			x1[r] = make_float2(q0[r]+q3[r],0);
		}

		// Build Iy = [ qB(-r)  qT(r) ]
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			y1[r] = make_float2(q0[r]+q1[r],0);
			y0[nr-r-1] = make_float2(q2[r]+q3[r],0);
		}

		for(int r=0;r<nr*2;r++)
			xrev[r] = x0[nr*2-r-1];
		for(int r=0;r<nr*2;r++)
			yrev[r] = y0[nr*2-r-1];
	}
}


__global__ void QI_MultiplyWithConjugate(int n, cufftComplex* a, cufftComplex* b)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		cufftComplex A = a[idx];
		cufftComplex B = b[idx];
	
		a[idx] = make_float2(A.x*B.x + A.y*B.y, A.y*B.x -A.x*B.y); // multiplying with conjugate
	}
}

__device__ float QI_ComputeAxisOffset(cufftComplex* autoconv, int fftlen, float* shiftbuf)
{
	typedef float compute_t;
	int nr = fftlen/2;
	for(int x=0;x<fftlen;x++)  {
		shiftbuf[x] = autoconv[(x+nr)%(nr*2)].x;
	}

	const float QIWeights[QI_LSQFIT_NWEIGHTS] = QI_LSQFIT_WEIGHTS;

	compute_t maxPos = ComputeMaxInterp<compute_t>::Compute(shiftbuf, fftlen, QIWeights);
	compute_t offset = (maxPos - nr) / (3.14159265359f * 0.5f);
	return offset;
}

__global__ void QI_OffsetPositions(int njobs, float3* current, float3* dst, cufftComplex* autoconv, int fftLength, float2* offsets, float pixelsPerProfLen, float* shiftbuf)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < njobs) {
		float* shifted = &shiftbuf[ idx * fftLength ];		

		// X
		cufftComplex* autoconvX = &autoconv[idx * fftLength * 2];
		float xoffset = QI_ComputeAxisOffset(autoconvX, fftLength, shifted);

		cufftComplex* autoconvY = autoconvX + fftLength;
		float yoffset = QI_ComputeAxisOffset(autoconvY, fftLength, shifted);

		dst[idx].x = current[idx].x + xoffset * pixelsPerProfLen;
		dst[idx].y = current[idx].y + yoffset * pixelsPerProfLen;

		if (offsets) 
			offsets[idx] = make_float2( xoffset, yoffset);
	}
}



/*
		q0: xprof[r], yprof[r]
		q1: xprof[len-r-1], yprof[r]
		q2: xprof[len-r-1], yprof[len-r-1]
		q3: xprof[r], yprof[len-r-1]

	kernel gets called with dim3(images.count, radialsteps, 4) elements
*/
template<typename TImageSampler>
__global__ void QI_ComputeQuadrants(BaseKernelParams kp, float3* positions, float* dst_quadrants, const QIParams params, int angularSteps)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int rIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int quadrant = threadIdx.z;

	if (jobIdx < kp.njobs && rIdx < params.radialSteps && quadrant < 4) {

		const int qmat[] = {
			1, 1,
			-1, 1,
			-1, -1,
			1, -1 };

		int mx = qmat[2*quadrant+0];
		int my = qmat[2*quadrant+1];
		float* qdr = &dst_quadrants[ (jobIdx * 4 + quadrant) * params.radialSteps ];

		float rstep = (params.maxRadius - params.minRadius) / params.radialSteps;
		float sum = 0.0f;
		float r = params.minRadius + rstep * rIdx;
		float3 pos = positions[jobIdx];
//		float mean = imgmeans[jobIdx];

		int count=0;
		for (int a=0;a<angularSteps;a++) {
			float x = pos.x + mx*params.cos_sin_table[a].x * r;
			float y = pos.y + my*params.cos_sin_table[a].y * r;
			bool outside=false;
			sum += TImageSampler::Interpolated(kp.images, x,y,jobIdx, outside);
			if (!outside) count++;
		}
		qdr[rIdx] = count>MIN_RADPROFILE_SMP_COUNT ? sum/count : kp.imgmeans[jobIdx];
	}
}



__global__ void QI_QuadrantsToProfiles(BaseKernelParams kp, float* quadrants, float2* profiles, float2* reverseProfiles, const QIParams params)
{
//ComputeQuadrantProfile(cudaImageListf& images, int idx, float* dst, const QIParams& params, int quadrant, float2 center)
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < kp.njobs) {
		int fftlen = params.radialSteps*2;
		float* img_qdr = &quadrants[ idx * params.radialSteps * 4 ];
	//	for (int q=0;q<4;q++)
			//ComputeQuadrantProfile<TImageSampler> (images, idx, &img_qdr[q*params.radialSteps], params, q, img_means[idx], make_float2(positions[idx].x, positions[idx].y));

		int nr = params.radialSteps;
		qicomplex_t* imgprof = (qicomplex_t*) &profiles[idx * fftlen*2];
		qicomplex_t* x0 = imgprof;
		qicomplex_t* x1 = imgprof + nr*1;
		qicomplex_t* y0 = imgprof + nr*2;
		qicomplex_t* y1 = imgprof + nr*3;

		qicomplex_t* revprof = (qicomplex_t*)&reverseProfiles[idx*fftlen*2];
		qicomplex_t* xrev = revprof;
		qicomplex_t* yrev = revprof + nr*2;

		float* q0 = &img_qdr[0];
		float* q1 = &img_qdr[nr];
		float* q2 = &img_qdr[nr*2];
		float* q3 = &img_qdr[nr*3];

		// Build Ix = qL(-r) || qR(r)
		// qL = q1 + q2   (concat0)
		// qR = q0 + q3   (concat1)
		for(int r=0;r<nr;r++) {
			x0[nr-r-1] = make_float2(q1[r]+q2[r],0);
			x1[r] = make_float2(q0[r]+q3[r],0);
		}
		// Build Iy = [ qB(-r)  qT(r) ]
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			y1[r] = make_float2(q0[r]+q1[r],0);
			y0[nr-r-1] = make_float2(q2[r]+q3[r],0);
		}

		for(int r=0;r<nr*2;r++)
			xrev[r] = x0[nr*2-r-1];
		for(int r=0;r<nr*2;r++)
			yrev[r] = y0[nr*2-r-1];
	}
}


void QI::Execute (BaseKernelParams& p, const QTrkComputedConfig& cfg, QI::StreamInstance* s, QI::DeviceInstance* d, device_vec<float3>* initial, device_vec<float3> *output, bool useTextureCache) 
{
	float angsteps = cfg.qi_angstepspq / powf(cfg.qi_angstep_factor, cfg.qi_iterations);

	for (int a=0;a<cfg.qi_iterations;a++) {
		if (useTextureCache) 
			Iterate< ImageSampler_Tex > (p, a==0 ? initial : output, output, s, d, std::max(MIN_RADPROFILE_SMP_COUNT, (int)angsteps) );
		else
			Iterate< ImageSampler_MemCopy > (p, a==0 ? initial : output, output, s, d, std::max(MIN_RADPROFILE_SMP_COUNT, (int)angsteps) );
		angsteps *= cfg.qi_angstep_factor;
	}
}

template<typename TImageSampler>
void QI::Iterate(BaseKernelParams& p, device_vec<float3>* initial, device_vec<float3>* newpos, StreamInstance *s, DeviceInstance* d, int angularSteps)
{
	if (0) {
	/*	dim3 qdrThreads(16, 8);
		dim3 qdrDim( (p.njobs + qdrThreads.x - 1) / qdrThreads.x, (params.radialSteps + qdrThreads.y - 1) / qdrThreads.y, 4 );
		QI_ComputeQuadrants<TImageSampler> <<< qdrDim , qdrThreads, 0, s->stream >>> 
			(p, initial->data, s->d_quadrants.data, d->d_qiparams);

		QI_QuadrantsToProfiles <<< blocks(p.njobs), threads(), 0, s->stream >>> 
			(p, s->d_quadrants.data, s->d_QIprofiles.data, s->d_QIprofiles_reverse.data, d->d_qiparams);*/
	}
	else {
		QI_ComputeProfile <TImageSampler> <<< blocks(p.njobs), threads(), 0, s->stream >>> (p, initial->data, 
			s->d_quadrants.data, s->d_QIprofiles.data, s->d_QIprofiles_reverse.data, d->d_qiparams, angularSteps);

#ifdef QI_DEBUG
		DbgCopyResult(s->d_quadrants, cmp_gpu_qi_prof);
#endif
	}

	cufftComplex* prof = (cufftComplex*)s->d_QIprofiles.data;
	cufftComplex* revprof = (cufftComplex*)s->d_QIprofiles_reverse.data;

	cufftExecC2C(s->fftPlan, prof, prof, CUFFT_FORWARD);
	cufftExecC2C(s->fftPlan, revprof, revprof, CUFFT_FORWARD);

	int nval = qi_FFT_length * 2 * batchSize, nthread=256;
	QI_MultiplyWithConjugate<<< dim3( (nval + nthread - 1)/nthread ), dim3(nthread), 0, s->stream >>>(nval, prof, revprof);
	cufftExecC2C(s->fftPlan, prof, prof, CUFFT_INVERSE);

#ifdef QI_DEBUG
	DbgCopyResult(s->d_QIprofiles, cmp_gpu_qi_fft_out);
#endif

	float2* d_offsets=0;
	float pixelsPerProfLen = (params.maxRadius-params.minRadius)/params.radialSteps;
	dim3 nBlocks=blocks(p.njobs), nThreads=threads();
	QI_OffsetPositions<<<nBlocks, nThreads, 0, s->stream>>>
		(p.njobs, initial->data, newpos->data, prof, qi_FFT_length, d_offsets, pixelsPerProfLen, s->d_shiftbuffer.data); 
}

void QI::InitDevice(DeviceInstance*d, QTrkComputedConfig& cc)
{
	std::vector<float2> qi_radialgrid(cc.qi_angstepspq);
	for (int i=0;i<cc.qi_angstepspq;i++)  {
		float ang = 0.5f*3.141593f*(i+0.5f)/(float)cc.qi_angstepspq;
		qi_radialgrid[i]=make_float2(cos(ang), sin(ang));
	}
	d->qi_trigtable = qi_radialgrid;

	QIParams dp = params;
	dp.cos_sin_table = d->qi_trigtable.data;

	cudaMalloc(&d->d_qiparams, sizeof(QIParams));
	cudaMemcpy(d->d_qiparams, &dp, sizeof(QIParams), cudaMemcpyHostToDevice);
}

void QI::InitStream(StreamInstance* s, QTrkComputedConfig& cc, cudaStream_t stream, int batchSize)
{
	int fftlen = cc.qi_radialsteps*2;
	s->stream = stream;
	s->d_quadrants.init(fftlen*batchSize*2);
	s->d_QIprofiles.init(batchSize*2*fftlen); // (2 axis) * (2 radialsteps) = 8 * nr = 2 * fftlen
	s->d_QIprofiles_reverse.init(batchSize*2*fftlen);
	s->d_shiftbuffer.init(fftlen * batchSize);
		
	// 2* batchSize, since X & Y both need an FFT transform
	//cufftResult_t r = cufftPlanMany(&s->fftPlan, 1, &fftlen, 0, 1, fftlen, 0, 1, fftlen, CUFFT_C2C, batchSize*4);
	cufftResult_t r = cufftPlan1d(&s->fftPlan, fftlen, CUFFT_C2C, batchSize*2);

	if(r != CUFFT_SUCCESS) {
		throw std::runtime_error( SPrintf("CUFFT plan creation failed. FFT len: %d. Batchsize: %d\n", fftlen, batchSize*4));
	}
	cufftSetCompatibilityMode(s->fftPlan, CUFFT_COMPATIBILITY_NATIVE);
	cufftSetStream(s->fftPlan, stream);

	this->qi_FFT_length = fftlen;
}

void QI::Init(QTrkComputedConfig& cfg, int batchSize)
{
	QIParams& qi = params;
	qi.trigtablesize = cfg.qi_angstepspq;
	qi.iterations = cfg.qi_iterations;
	qi.maxRadius = cfg.qi_maxradius;
	qi.minRadius = cfg.qi_minradius;
	qi.radialSteps = cfg.qi_radialsteps;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	numThreads = prop.warpSize;

	this->batchSize = batchSize;
}


