#pragma once

#include "random_distr.h"
#include <direct.h> // _mkdir()

#include "FisherMatrix.h"

/*
void ResampleFourierLUT(QueuedTracker* qtrk, ImageData* orglut, ImageData* flut, int zplanes, const char *jpgfile)
{
	if (buildLUTFlags & BUILDLUT_FOURIER) {
		tracker.SetRadialZLUT(0, 1, zplanes);
		for (int i=0;i<zplanes;i++)
		{
			GenerateImageFromLUT(&img, lut, 0, cfg.width/2, vector3f(cfg.width/2, cfg.height/2, i/(float)zplanes * lut->h));
			img.normalize();
			tracker.BuildLUT(img.data, sizeof(float)*img.w, QTrkFloat, buildLUTFlags & ~BUILDLUT_FOURIER, i);
		}
		tracker.FinalizeLUT();
		tracker.GetRadialZLUT(newlut->data);

		if (jpgfile) {
			float* zlut_result=new float[zplanes*cfg.zlut_radialsteps*1];
			tracker.GetRadialZLUT(zlut_result);
			FloatToJPEGFile(SPrintf("gen-%s", jpgfile).c_str(), zlut_result, cfg.zlut_radialsteps, zplanes);
			delete[] zlut_result;
		}
	}
}*/


// Generate a LUT by creating new image samples and using the tracker in BuildZLUT mode
// This will ensure equal settings for radial profiles etc
template<typename T>
void ResampleLUT(T* qtrk, ImageData* lut, int zplanes, ImageData* newlut, const char *jpgfile=0, uint buildLUTFlags=0)
{
	QTrkComputedConfig& cfg = qtrk->cfg;
	ImageData img = ImageData::alloc(cfg.width,cfg.height);

	qtrk->SetRadialZLUT(0, 1, zplanes);
	qtrk->BeginLUT(buildLUTFlags);
	for (int i=0;i<zplanes;i++)
	{
		vector2f pos(cfg.width/2,cfg.height/2);
		GenerateImageFromLUT(&img, lut, qtrk->cfg.zlut_minradius, qtrk->cfg.zlut_maxradius, vector3f(pos.x,pos.y, i/(float)zplanes * lut->h), true);
		img.normalize();
		if (i == zplanes/2 && jpgfile)
			WriteJPEGFile(SPrintf("smp-%s",jpgfile).c_str(), img);
		qtrk->BuildLUT(img.data, sizeof(float)*img.w, QTrkFloat, i, &pos);
	}
	qtrk->FinalizeLUT();

	*newlut = ImageData::alloc(cfg.zlut_radialsteps, zplanes);
	qtrk->GetRadialZLUT(newlut->data);
	newlut->normalize();

	img.free();

	if (jpgfile) {
		float* zlut_result=new float[zplanes*cfg.zlut_radialsteps*1];
		qtrk->GetRadialZLUT(zlut_result);
		FloatToJPEGFile(jpgfile, zlut_result, cfg.zlut_radialsteps, zplanes);
		delete[] zlut_result;
	}
}


static std::vector<vector3f> FetchResults(QueuedTracker* trk)
{
	int N = trk->GetResultCount();
	std::vector<vector3f> results (N);
	for (int i=0;i<N;i++) {
		LocalizationResult j;
		trk->FetchResults(&j,1);
		results[j.job.frame] = j.pos;
	}
	return results;
}


static void RandomFill (float* d,int size, float mean, float std_deviation)
{
	for (int i=0;i<size;i++)
		d [i] = std::max(0.0f,  mean + rand_normal<float>() * std_deviation);
}

/*
This test runs a localization series with varying amounts of CMOS per-pixel gain variation:

- Simulate drift over some pixels
- Generate images with standard Poisson noise
- Repeat this with several amounts of CMOS per-pixel noise settings
*/
template<typename TrackerType>
void TestCMOSNoiseInfluence(const char *lutfile)
{
	int nNoiseLevels = 5;
	float gainStDevScale = 0.3f;
#ifdef _DEBUG
	int nDriftSteps = 100;
#else
	int nDriftSteps = 2000;
#endif
	float driftDistance = 4; // pixels

	ImageData lut = ReadJPEGFile(lutfile);
	ImageData img = ImageData::alloc(120,120);

	QTrkSettings cfg;
	cfg.width = img.w; cfg.height = img.h;
	TrackerType trk(cfg);

	int nBeads=1;
	float zlutMinR=4,zlutMaxR = cfg.width/2-5;
	ResampleLUT(&trk, &lut,zlutMinR, zlutMaxR,100, "resample-zlut.jpg");
	
	float *offset = new float [nBeads * cfg.width * cfg.height];
	float *gain = new float [nBeads * cfg.width * cfg.height];

	bool useCalib = false;
	bool saveImgs = true;

	float *info = new float [nNoiseLevels*2];

	trk.SetLocalizationMode((LocMode_t)(LT_QI|LT_LocalizeZ|LT_NormalizeProfile));

	for (int nl=0;nl<nNoiseLevels;nl++) {

		float offset_stdev = info[nl*2+0] = nl*0.01f;
		float gain_stdev = info[nl*2+1] = nl*gainStDevScale;

		srand(0);
		RandomFill(offset, nBeads * cfg.width * cfg.height, 0, offset_stdev);
		RandomFill(gain, nBeads * cfg.width * cfg.height, 1, gain_stdev);

		if (useCalib) trk.SetPixelCalibrationImages(offset, gain);

		std::string dirname= SPrintf("noiselev%d", nl);
		if (saveImgs) _mkdir(dirname.c_str());

		// drift
		float dx = driftDistance / (nDriftSteps-1);
		for (int d=0;d<nDriftSteps;d++) {
			GenerateImageFromLUT(&img, &lut, zlutMinR, zlutMaxR, vector3f(img.w/2+dx*d,img.h/2, 20));
			img.normalize();

			if (!useCalib) {
				for (int i=0;i<cfg.width*cfg.height;i++)
					img[i]*=gain[i];
				ApplyPoissonNoise(img, 255);
				for (int i=0;i<cfg.width*cfg.height;i++)
					img[i]+=offset[i];
			} else 
				ApplyPoissonNoise(img, 255);

			if (saveImgs && d<40) {
				FloatToJPEGFile(SPrintf("%s\\nl%d-smp%d.jpg",dirname.c_str(),nl,d).c_str(),img.data, img.w,img.h);
			}

			if (d==0) {
				FloatToJPEGFile(SPrintf("nl%d.jpg",nl).c_str(),img.data, img.w,img.h);
			}

			if ( d%(std::max(1,nDriftSteps/10)) == 0 )
				dbgprintf("Generating images for test %d...\n", d);

			LocalizationJob job(d, d,0,0);
			trk.ScheduleLocalization((uchar*)img.data, sizeof(float)*img.w, QTrkFloat, &job);
		}
		trk.Flush();

		while (!trk.IsIdle());
		int nResults = trk.GetResultCount(); 
		dbgprintf("noiselevel: %d done. Writing %d results\n", nl, nResults);
		auto results = FetchResults(&trk);
		WriteTrace(SPrintf("%s\\trace.txt", dirname.c_str()), &results[0], results.size());

	}
	WriteImageAsCSV("offset_gain_stdev.txt", info, 2, nNoiseLevels);

	img.free();
	lut.free();

	delete[] gain;
	delete[] offset;
	delete[] info;
}

static void EnableGainCorrection(QueuedTracker* qtrk)
{
	int nb, _pl, _r;
	qtrk->GetRadialZLUTSize(nb, _pl, _r);

	int w=qtrk->cfg.width, h=qtrk->cfg.height;
	float *offset = new float [w*h*nb];
	float *gain = new float [w*h*nb];
	RandomFill(offset, w*h*nb, 0, 0.1f);
	RandomFill(gain, w*h*nb, 1, 0.2f);
	qtrk->SetPixelCalibrationImages(offset, gain);
	delete[] offset; delete[] gain;
}

template<typename TrackerType>
std::vector<vector3f> Gauss2DTest(
#ifdef _DEBUG
	int NumImages=10, int JobsPerImg=10
#else
	int NumImages=10, int JobsPerImg=1000
#endif
	,	bool useGC = false )
{
	QTrkSettings cfg;
	cfg.width = cfg.height = 20;
	cfg.gauss2D_iterations = 8;
	LocMode_t lt = LT_Gaussian2D;
	TrackerType qtrk(cfg);
	std::vector<float*> images;

	srand(0);

	qtrk.SetRadialZLUT(0, 1, 1); // need to indicate 1 bead, as the pixel calibration images are per-bead
	if (useGC) EnableGainCorrection(&qtrk);
	
	// Schedule images to localize on

	dbgprintf("Gauss2D: Generating %d images...\n", NumImages);
	std::vector<float> truepos(NumImages*2);
//	srand(time(0));

	for (int n=0;n<NumImages;n++) {
		double t1 = GetPreciseTime();
		float xp = cfg.width/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = cfg.height/2+(rand_uniform<float>() - 0.5) * 5;
		truepos[n*2+0] = xp;
		truepos[n*2+1] = yp;

		float *image = new float[cfg.width*cfg.height];
		images.push_back(image);

		ImageData img(image,cfg.width,cfg.height);
		GenerateGaussianSpotImage(&img, vector2f(xp,yp), cfg.gauss2D_sigma, 1000, 4);
		ApplyPoissonNoise(img, 1.0f);

		FloatToJPEGFile(SPrintf("gauss2d-%d.jpg", n).c_str(), image,cfg.width,cfg.height);
	}

	// Measure speed
	dbgprintf("Localizing on %d images...\n", NumImages*JobsPerImg);
	double tstart = GetPreciseTime();

	qtrk.SetLocalizationMode(lt);
	for (int n=0;n<NumImages;n++) {
		for (int k=0;k<JobsPerImg;k++) {
			LocalizationJob job(n,0,0,0);
			qtrk.ScheduleLocalization((uchar*)images[n], cfg.width*sizeof(float), QTrkFloat, &job);
		}
	}

	qtrk.Flush();

	int total = NumImages*JobsPerImg;
	int rc = qtrk.GetResultCount(), displayrc=0;
	do {
		rc = qtrk.GetResultCount();
		while (displayrc<rc) {
			if( displayrc%JobsPerImg==0) dbgprintf("Done: %d / %d\n", displayrc, total);
			displayrc++;
		}
		Sleep(10);
	} while (rc != total);

	double tend = GetPreciseTime();

	// Wait for last jobs
	rc = NumImages*JobsPerImg;
	double errX=0.0, errY=0.0;

	std::vector<vector3f> results (NumImages);
	while(rc>0) {
		LocalizationResult result;

		if (qtrk.FetchResults(&result, 1)) {
			int iid = result.job.frame;
			float x = fabs(truepos[iid*2+0]-result.pos.x);
			float y = fabs(truepos[iid*2+1]-result.pos.y);

			results [result.job.frame] = result.pos;

			errX += x; errY += y;
			rc--;
		}
	}
	dbgprintf("Localization Speed: %d (img/s), using %d threads\n", (int)( total/(tend-tstart) ), qtrk.cfg.numThreads);
	dbgprintf("ErrX: %f, ErrY: %f\n", errX/total, errY/total);
	DeleteAllElems(images);
	return results;
}


static double WaitForFinish(QueuedTracker* qtrk, int N)
{
	double t0 = GetPreciseTime();
	qtrk->Flush();
	int displayrc=0,rc=0;
	while ( (rc = qtrk->GetResultCount ()) != N || displayrc<rc) {
		while (displayrc<rc) {
			if(displayrc%std::max(1,N/10)==0) dbgprintf("Done: %d / %d\n", displayrc, N);
			displayrc++;
		}
		Threads::Sleep(50);
	}
	double t1 = GetPreciseTime();
	return t1-t0;
}


static void MeanStDevError(const std::vector<vector3f>&  truepos, const std::vector<vector3f>&  v, vector3f &meanErr, vector3f & stdev) 
{
	meanErr=vector3f();
	for (size_t i=0;i<v.size();i++) meanErr+=v[i]-truepos[i]; 
	meanErr*=1.0f/v.size();

	vector3f r;
	for (uint i=0;i<v.size();i++) {
		vector3f d = (v[i]-truepos[i])-meanErr;
		r+= d*d;
	}
	stdev = sqrt(r/v.size());
}

struct RunTrackerResults{
	std::vector<vector3f> output;
	std::vector<vector3f> truepos;

	vector3f meanErr, stdev;

	void computeStats() {
		MeanStDevError(truepos,output,meanErr,stdev);
	}
};

template<typename TrkType>
RunTrackerResults RunTracker(const char *lutfile, QTrkSettings *cfg, bool useGC, const char* name, LocMode_t locMode, int N=
#ifdef _DEBUG
	1
#else
	2000
#endif
	, float noiseFactor=28, float zpos=10, ImageData* pRescaledLUT=0, vector3f samplePosRange=vector3f(1,1,1) )
{
	std::vector<vector3f> results, truepos;

	ImageData lut = ReadJPEGFile(lutfile);
	ImageData img = ImageData::alloc(cfg->width,cfg->height);

	ImageData* rescaledLUT;
	ImageData rescaledBuffer;

	TrkType trk(*cfg);
	if (pRescaledLUT)  {
		rescaledLUT = pRescaledLUT;
		if (rescaledLUT->data == 0) {
			*rescaledLUT = ImageData::alloc(cfg->width, cfg->height);
			ResampleLUT(&trk, &lut, lut.h, rescaledLUT, SPrintf("%s-zlut.jpg",name).c_str(), ((locMode&LT_FourierLUT) ? BUILDLUT_FOURIER : 0));
		}
	} else {
		rescaledBuffer = ImageData::alloc(cfg->width, cfg->height);
		rescaledLUT = &rescaledBuffer;
		ResampleLUT(&trk, &lut, lut.h, rescaledLUT, SPrintf("%s-zlut.jpg",name).c_str(), ((locMode&LT_FourierLUT) ? BUILDLUT_FOURIER : 0));
	}

	if (useGC) EnableGainCorrection(&trk);

	//trk.SetConfigValue("use_texturecache" , "0");

	srand(0);
	trk.SetLocalizationMode(locMode);
	for (int i=0;i<N;i++)
	{
		vector3f pos(
			cfg->width/2 + samplePosRange.x*(rand_uniform<float>()-0.5f),
			cfg->height/2 + samplePosRange.y*(rand_uniform<float>()-0.5f), 
			zpos + samplePosRange.z*(rand_uniform<float>()-0.5f)
		);
		GenerateImageFromLUT(&img, rescaledLUT, trk.cfg.zlut_minradius, trk.cfg.zlut_maxradius, vector3f( pos.x,pos.y, pos.z));
		if (noiseFactor>0)	ApplyPoissonNoise(img, noiseFactor * 255, 255);
		truepos.push_back(pos);

		LocalizationJob job(i, 0, 0, 0);
		trk.ScheduleImageData(&img, &job);
	}

	dbgprintf("Scheduled %d images\n" ,  N);

	WaitForFinish(&trk, N);

	results.resize(trk.GetResultCount());
	for (uint i=0;i<results.size();i++) {
		LocalizationResult r;
		trk.FetchResults(&r,1);
		results[r.job.frame]=r.pos;
	}

	double tend = GetPreciseTime();
	img.free();
	lut.free();

	RunTrackerResults r;
	r.output=results;
	r.truepos=truepos;

	if (!pRescaledLUT)
		rescaledBuffer.free();

	return r;
}


static void ResizeLUT(ImageData& lut, ImageData& resized, QTrkSettings* cfg)
{
	QTrkComputedConfig cc(*cfg);

	int nsmp = (int) ceil( (float)lut.w/cc.zlut_radialsteps );
	resized = ImageData::alloc(cc.zlut_radialsteps, lut.h);
	float *c = ALLOCA_ARRAY(float, cc.zlut_radialsteps);
	for (int y=0;y<lut.h;y++) {
		for (int i=0;i<cc.zlut_radialsteps;i++)c[i]=0;
		for(int x=0;x<lut.w;x++) {
			float pos = x/(float)lut.w * cc.zlut_radialsteps;
			float frac= (int)pos - pos;
			int p=(int)pos;
			float v=lut.at(x,y);
			c[p] += frac;
			resized.at(p,y) += frac*v;
			if (p<cc.zlut_radialsteps-1) {
				c[p+1] += 1-frac;
				resized.at(p+1,y) += (1-frac)*v;
			}
		}
		for (int x=0;x<cc.zlut_radialsteps;x++)
			resized.at(x,y) /= c[x];
	}
}


template<typename T>
std::vector<T> logspace(T a, T b, int N)
{
	std::vector<T> r (N);

	T a_ = log(a);
	T b_ = log(b);

	for (int i=0;i<N;i++)
		r[i] = exp( (b_ - a_) * (i/(float)(N-1)) + a_);
	return r;
}

template<typename T>
std::vector<T> linspace(T a, T b, int N)
{
	std::vector<T> r (N);
	for (int i=0;i<N;i++)
		r[i] = (b - a) * (i/(float)(N-1)) + a;
	return r;
}


struct SpeedAccResult{
	vector3f acc,bias,crlb;
	int speed;
	void Compute(const std::vector<vector3f>& results, std::function< vector3f(int x) > truepos) {

		vector3f s;
		for(int i=0;i<results.size();i++) {
			s+=results[i]-truepos(i);
		}
		s*=1.0f/results.size();
		bias=s;

		acc=vector3f();
		for (uint i=0;i<results.size();i++) {
			vector3f d = results[i]-truepos(i); 
			vector3f errMinusMeanErr = d - s;
			acc += errMinusMeanErr*errMinusMeanErr;
		}
		acc = sqrt(acc/results.size());

	}
};




static SpeedAccResult SpeedAccTest(ImageData& lut, QTrkSettings *cfg, int N, vector3f centerpos, vector3f range, const char *name, int MaxPixelValue, int extraFlags=0, int buildLUTFlags=0)
{
	typedef QueuedTracker TrkType;
	int NImg=N;//std::max(1,N/20);
	std::vector<vector3f> results, truepos (NImg);

	std::vector<ImageData> imgs(NImg);
	const float R=5;
	
	QueuedTracker* trk = new QueuedCPUTracker(*cfg);// CreateQueuedTracker(*cfg);

	ImageData resizedLUT;
	ResampleLUT(trk, &lut, lut.h, &resizedLUT, 0, buildLUTFlags);

	if (buildLUTFlags&BUILDLUT_BIASCORRECT)
		trk->ComputeZBiasCorrection(lut.h*10, 0, 4, true);

	std::vector<Matrix3X3> fishers(NImg);

//	for (int i=0;i<NImg;i++) {
	parallel_for(NImg, [&](int i) {
		imgs[i]=ImageData::alloc(cfg->width,cfg->height);
		vector3f pos = centerpos + range*vector3f(rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f)*1;
		GenerateImageFromLUT(&imgs[i], &resizedLUT, trk->cfg.zlut_minradius, trk->cfg.zlut_maxradius, vector3f( pos.x,pos.y, pos.z));

		SampleFisherMatrix fm(MaxPixelValue);
		fishers[i] = fm.Compute(pos, vector3f(1,1,1)*0.001f, resizedLUT, cfg->width,cfg->height, trk->cfg.zlut_minradius,trk->cfg.zlut_maxradius);

		imgs[i].normalize();
		if (MaxPixelValue> 0) ApplyPoissonNoise(imgs[i], MaxPixelValue);
		if(i==0) WriteJPEGFile(name, imgs[i]);

		truepos[i]=pos;
	});

	Matrix3X3 fisher;
	for (int i=0;i<NImg;i++) fisher+=fishers[i];

	int flags= LT_LocalizeZ|LT_NormalizeProfile|extraFlags;
	if (cfg->qi_iterations>0) flags|=LT_QI;

	trk->SetLocalizationMode((LocMode_t)flags);
	double tstart=GetPreciseTime();

	int img=0;
	for (int i=0;i<N;i++)
	{
		LocalizationJob job(i, 0, 0, 0);
		trk->ScheduleLocalization((uchar*)imgs[i%NImg].data, sizeof(float)*cfg->width, QTrkFloat, &job);
	}


	WaitForFinish(trk, N);
	double tend = GetPreciseTime();

	results.resize(trk->GetResultCount());
	for (uint i=0;i<results.size();i++) {
		LocalizationResult r;
		trk->FetchResults(&r,1);
		results[r.job.frame]=r.pos;
	}

	for (int i=0;i<NImg;i++)
		imgs[i].free();

	SpeedAccResult r;
	r.Compute(results, [&](int index) { return truepos[index]; });

	r.speed = N/(tend-tstart);
	fisher *= 1.0f/NImg;
	r.crlb = sqrt(fisher.Inverse().diag());
	resizedLUT.free();
	delete trk;
	return r;
}

