#pragma once



template<typename T>
static __device__ T interpolate(T a, T b, float x) { return a + (b-a)*x; }

template<typename TImageSampler>
__device__ float2 BgCorrectedCOM(int idx, cudaImageListf images, float correctionFactor, float* pMean)
{
	int imgsize = images.w*images.h;
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;

	for (int y=0;y<images.h;y++)
		for (int x=0;x<images.w;x++) {
			float v = TImageSampler::Index(images, x, y, idx);
			sum += v;
			sum2 += v*v;
		}

	float invN = 1.0f/imgsize;
	float mean = sum * invN;
	float stdev = sqrtf(sum2 * invN - mean * mean);
	sum = 0.0f;

	for (int y=0;y<images.h;y++)
		for(int x=0;x<images.w;x++)
		{
			float v = TImageSampler::Index(images, x,y,idx);
			v = fabsf(v-mean)-correctionFactor*stdev;
			if(v<0.0f) v=0.0f;
			sum += v;
			momentX += x*v;
			momentY += y*v;
		}

	if (pMean)
		*pMean = mean;

	float2 com;
	com.x = momentX / (float)sum;
	com.y = momentY / (float)sum;
	return com;
}

template<typename TImageSampler>
__global__ void BgCorrectedCOM(int count, cudaImageListf images,float3* d_com, float bgCorrectionFactor, float* d_imgmeans) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < count) {
		float mean;
		float2 com = BgCorrectedCOM<TImageSampler> (idx, images, bgCorrectionFactor, &mean);
		d_com[idx] = make_float3(com.x,com.y,0.0f);
		d_imgmeans[idx] = mean;
	}
}

__global__ void ZLUT_ProfilesToZLUT(int njobs, cudaImageListf images, ZLUTParams params, float3* positions, LocalizationParams* locParams, float* profiles)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < njobs) {
		auto m = locParams[idx];
		float* dst = params.GetRadialZLUT(m.zlutIndex, m.zlutPlane );

		for (int i=0;i<params.radialSteps();i++)
			dst [i] += profiles [ params.radialSteps()*idx + i ];
	}
}

// Compute a single ZLUT radial profile element (looping through all the pixels at constant radial distance)
template<typename TImageSampler>
__global__ void ZLUT_RadialProfileKernel(int njobs, cudaImageListf images, ZLUTParams params, float3* positions, float* profiles, float* means)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int radialIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if (jobIdx >= njobs || radialIdx >= params.radialSteps()) 
		return;

	float* dstprof = &profiles[params.radialSteps() * jobIdx];
	float r = params.minRadius + (params.maxRadius-params.minRadius)*radialIdx/params.radialSteps();
	float sum = 0.0f;
	int count = 0;
	
	for (int i=0;i<params.angularSteps;i++) {
		float x = positions[jobIdx].x + params.trigtable[i].x * r;
		float y = positions[jobIdx].y + params.trigtable[i].y * r;

		bool outside=false;
		sum += TImageSampler::Interpolated(images, x,y, jobIdx, outside);
		if (!outside) count++;
	}
	dstprof [radialIdx] = count>MIN_RADPROFILE_SMP_COUNT ? sum/count : means[jobIdx];
}


__global__ void ZLUT_ComputeZ (int njobs, ZLUTParams params, float3* positions, float* compareScoreBuf)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (jobIdx < njobs) {
		float* cmp = &compareScoreBuf [params.planes * jobIdx];

		const float ZLUTFittingWeights[ZLUT_LSQFIT_NWEIGHTS] = ZLUT_LSQFIT_WEIGHTS;
		float maxPos = ComputeMaxInterp<float, ZLUT_LSQFIT_NWEIGHTS>::Compute(cmp, params.planes, ZLUTFittingWeights);
		positions[jobIdx].z = maxPos;
	}
}

__global__ void ZLUT_ComputeProfileMatchScores(int njobs, ZLUTParams params, float *profiles, float* compareScoreBuf, LocalizationParams *locParams)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int zPlaneIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if (jobIdx >= njobs || zPlaneIdx >= params.planes)
		return;

	float* prof = &profiles [jobIdx * params.radialSteps()];
	auto mapping = locParams[jobIdx];
	float diffsum = 0.0f;
	for (int r=0;r<params.radialSteps();r++) {
		float d = prof[r] - params.img.pixel(r, zPlaneIdx, mapping.zlutIndex);
		if (params.zcmpwindow)
			d *= params.zcmpwindow[r];
		diffsum += d*d;
	}

	compareScoreBuf[ params.planes * jobIdx + zPlaneIdx ] = -diffsum;
}

__global__ void ZLUT_NormalizeProfiles(int njobs, ZLUTParams params, float* profiles)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (jobIdx < njobs) {
		float* prof = &profiles[params.radialSteps()*jobIdx];

		// First, subtract mean
		float mean = 0.0f;
		for (int i=0;i<params.radialSteps();i++) {
			mean += prof[i];
		}
		mean /= params.radialSteps();

		float rmsSum2 = 0.0f;
		for (int i=0;i<params.radialSteps();i++){
			prof[i] -= mean;
			rmsSum2 += prof[i]*prof[i];
		}

		// And make RMS power equal 1
		float invTotalRms = 1.0f / sqrt(rmsSum2/params.radialSteps());
		for (int i=0;i<params.radialSteps();i++)
			prof[i] *= invTotalRms;
	}
}


__global__ void ApplyOffsetGain (BaseKernelParams kp, cudaImageListf calib_gain, cudaImageListf calib_offset, float gainFactor, float offsetFactor)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int jobIdx = threadIdx.z + blockIdx.z * blockDim.z;

	if (x < kp.images.w && y < kp.images.h && jobIdx < kp.njobs) {
		int bead = kp.locParams[jobIdx].zlutIndex;

		float value = kp.images.pixel(x,y,jobIdx);
		float offset = calib_offset.pixel(x,y,bead);
		float gain = calib_gain.pixel(x,y,bead);
		kp.images.pixel(x,y,jobIdx) = (value + offset*offsetFactor) * gain*gainFactor;
	}
}


// Simple gaussian 2D MLE implementation. A better solution would be to distribute CUDA threads over each pixel, but this is a very straightforward implementation
template<typename TImageSampler>
__global__ void G2MLE_Compute(BaseKernelParams kp, float sigma, int iterations, float3* initial, float3 *positions, float* I_bg, float* I_0)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (jobIdx >= kp.njobs)
		return;

	float2 pos = make_float2(initial[jobIdx].x, initial[jobIdx].y);
	float mean = kp.imgmeans[jobIdx];
	float I0 = mean*0.5f*kp.images.w*kp.images.h;
	float bg = mean*0.5f;

	const float _1oSq2Sigma = 1.0f / (sqrtf(2) * sigma);
	const float _1oSq2PiSigma = (1.0f / (sqrtf(2*3.14159265359f))) / sigma;
	const float _1oSq2PiSigma3 = (1.0f / (sqrtf(2*3.14159265359f))) / (sigma*sigma*sigma);

	for (int i=0;i<iterations;i++)
	{
		float dL_dx = 0.0; 
		float dL_dy = 0.0; 
		float dL_dI0 = 0.0;
		float dL_dIbg = 0.0;
		float dL2_dx = 0.0;
		float dL2_dy = 0.0;
		float dL2_dI0 = 0.0;
		float dL2_dIbg = 0.0;
				
		for (int y=0;y<kp.images.h;y++)
		{
			for (int x=0;x<kp.images.w;x++)
			{
		        float Xexp0 = (x-pos.x + .5f) * _1oSq2Sigma;
				float Yexp0 = (y-pos.y + .5f) * _1oSq2Sigma;
        
				float Xexp1 = (x-pos.x - .5f) * _1oSq2Sigma;
				float Yexp1 = (y-pos.y - .5f) * _1oSq2Sigma;
				
				float DeltaX = 0.5f * erff(Xexp0) - 0.5f * erff(Xexp1);
				float DeltaY = 0.5f * erff(Yexp0) - 0.5f * erff(Yexp1);
				float mu = bg + I0 * DeltaX * DeltaY;
				
				float dmu_dx = I0*_1oSq2PiSigma * ( expf(-Xexp1*Xexp1) - expf(-Xexp0*Xexp0)) * DeltaY;

				float dmu_dy = I0*_1oSq2PiSigma * ( expf(-Yexp1*Yexp1) - expf(-Yexp0*Yexp0)) * DeltaX;
				float dmu_dI0 = DeltaX*DeltaY;
				float dmu_dIbg = 1;
        
				float smp = TImageSampler::Index(kp.images, x,y, jobIdx);
				float f = smp / mu - 1;
				dL_dx += dmu_dx * f;
				dL_dy += dmu_dy * f;
				dL_dI0 += dmu_dI0 * f;
				dL_dIbg += dmu_dIbg * f;

				float d2mu_dx = I0*_1oSq2PiSigma3 * ( (x - pos.x - .5f) * expf (-Xexp1*Xexp1) - (x - pos.x + .5) * expf(-Xexp0*Xexp0) ) * DeltaY;
				float d2mu_dy = I0*_1oSq2PiSigma3 * ( (y - pos.y - .5f) * expf (-Yexp1*Yexp1) - (y - pos.y + .5) * expf(-Yexp0*Yexp0) ) * DeltaX;
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
	

	positions[jobIdx].x = pos.x;
	positions[jobIdx].y = pos.y;
	if (I_bg) I_bg[jobIdx] = bg;
	if (I_0) I_0[jobIdx] = I0;
}

surface<void, cudaSurfaceType2DLayered> image_lut_surface;

template<typename TImageSampler>
__global__ void ImageLUT_Build(BaseKernelParams kp, ImageLUTConfig ilc, float3* positions, cudaImage4D<float>::KernelInst lut)
{
	// add sampled image data to 
	int idx = threadIdx.x;
	if (idx < kp.njobs) {

		float invMean = 1.0f / kp.imgmeans[idx];

		float startx = positions[idx].x - ilc.w/2*ilc.xscale;
		float starty = positions[idx].y - ilc.h/2*ilc.yscale;
		int2 imgpos = lut.getImagePos(kp.locParams[idx].zlutPlane);

		for (int y=0;y<kp.images.h;y++)
			for (int x=0;x<kp.images.w;x++) {
				float px = startx + x*ilc.xscale;
				float py = starty + y*ilc.yscale;

				bool outside=false;
				float v = TImageSampler::Interpolated(kp.images, px, py, idx, outside);

				//float org;
				//surf2DLayeredread (&org, image_lut_surface, (int)( sizeof(float)*(x+dstx)), y, kp.locParams[idx].zlutIndex, cudaBoundaryModeTrap);
				int z = kp.locParams[idx].zlutIndex;
				float org = lut.readSurfacePixel(image_lut_surface, x + imgpos.x, y + imgpos.y, z);
				lut.writeSurfacePixel(image_lut_surface, x + imgpos.x, y + imgpos.y, z, org+v*invMean);
			}
	}
}

