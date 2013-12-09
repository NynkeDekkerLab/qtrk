#pragma once

#include "random_distr.h"
#include <direct.h> // _mkdir()


// Generate a LUT by creating new image samples and using the tracker in BuildZLUT mode
// This will ensure equal settings for radial profiles etc
static void ResampleLUT(QueuedTracker* qtrk, ImageData* lut, float M, int zplanes=100, const char *jpgfile=0, ImageData* newlut=0)
{
	QTrkComputedConfig& cfg = qtrk->cfg;
	ImageData img = ImageData::alloc(cfg.width,cfg.height);

	qtrk->SetRadialZLUT(0, 1, zplanes);
	for (int i=0;i<zplanes;i++)
	{
		GenerateImageFromLUT(&img, lut, 0, cfg.width/2, vector2f(cfg.width/2, cfg.height/2), i/(float)zplanes * lut->h, M);
		img.normalize();
		if (i == 0)
			WriteJPEGFile(SPrintf("smp-%s",jpgfile).c_str(), img);

		qtrk->BuildLUT(img.data, sizeof(float)*img.w, QTrkFloat, 0, i);
	}
	qtrk->FinalizeLUT();
	img.free();

	if (newlut) {
		newlut->alloc(cfg.zlut_radialsteps, zplanes);
		qtrk->GetRadialZLUT(newlut->data);
		newlut->normalize();
	}

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
			GenerateImageFromLUT(&img, &lut, zlutMinR, zlutMaxR, vector2f(img.w/2+dx*d,img.h/2), 20, 1.0f);
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
	cfg.gauss2D_iterations = 4;
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




template<typename TrkType>
std::vector<vector3f> RunTracker(const char *lutfile, QTrkSettings *cfg, bool useGC, const char* name, LocMode_t locMode, int N=
#ifdef _DEBUG
	1
#else
	2000
#endif
	)
{
	std::vector<vector3f> results, truepos;

	ImageData lut = ReadJPEGFile(lutfile);
	ImageData img = ImageData::alloc(cfg->width,cfg->height);

	TrkType trk(*cfg);
	ResampleLUT(&trk, &lut, 1, 100, SPrintf("%s-zlut.jpg",name).c_str());

	if (useGC) EnableGainCorrection(&trk);

	//trk.SetConfigValue("use_texturecache" , "0");

	float R=5;
	srand(0);
	trk.SetLocalizationMode(locMode);
	for (int i=0;i<N;i++)
	{
		vector3f pos(cfg->width/2 + R*(rand_uniform<float>()-0.5f),cfg->height/2 + R*(rand_uniform<float>()-0.5f), lut.h/4+rand_uniform<float>()*lut.h/2);
		GenerateImageFromLUT(&img, &lut, 2.0f, cfg->width/2-3,vector2f( pos.x,pos.y), pos.z, 1.0f);
		truepos.push_back(pos);

		LocalizationJob job(i, 0, 0, 0);
		trk.ScheduleLocalization((uchar*)img.data, sizeof(float)*cfg->width, QTrkFloat, &job);
	}

	WaitForFinish(&trk, N);

	results.resize(trk.GetResultCount());
	for (int i=0;i<results.size();i++) {
		LocalizationResult r;
		trk.FetchResults(&r,1);
		results[r.job.frame]=r.pos;
	}

	double tend = GetPreciseTime();
	img.free();
	lut.free();

	auto Mean = [&] (const std::vector<vector3f>& v) -> vector3f {
		vector3f s;
		for (int i=0;i<v.size();i++) s+=v[i]-truepos[i]; s*=1.0f/v.size();
		return s;
	};
	auto StDev = [&] (std::vector<vector3f>& v) -> vector3f {
		vector3f r;
		vector3f mean = Mean(v);
		for (int i=0;i<v.size();i++) {
			vector3f d = v[i]-mean; r+= d*d;
		//	dbgprintf("dx:%f,dy:%f\n", d.x,d.y);
		}
		return sqrt(r/v.size());
	};

	dbgprintf("Mean (%s): %f.  St. Dev: %f\n", name, Mean(results).x, StDev(results).x);

	return results;
}

template<typename Tracker>
void TestBuildRadialZLUT(const char *rlutfile)
{
	QTrkSettings cfg;
	cfg.width = 100;
	cfg.height = 100;
	
	ImageData rlut=ReadJPEGFile(rlutfile);
	int nbeads=3;
	ImageData imgbuf=ImageData::alloc(cfg.width,cfg.height * nbeads);

	Tracker trk(cfg);

	int nplanes = 50;
	trk.SetRadialZLUT(0, nbeads, nplanes);
	for (int z=0;z<nplanes;z++) {
		dbgprintf("z=%d\n", z);
		for (int n=0;n<1;n++) {
			for (int b=0;b<nbeads;b++) {
				ImageData tmpimg = ImageData( &imgbuf.data[imgbuf.w * cfg.height *b], cfg.width,cfg.height); 
				vector2f pos = vector2f::random( vector2f( cfg.width/2,cfg.height/2 ), 2.0f );
				GenerateImageFromLUT(&tmpimg, &rlut, 0.0f, rlut.w-1, pos, z/(float)nplanes * rlut.h, 0.5f + 0.2f*b);
			}
			//ApplyPoissonNoise (img, 255);
			WriteJPEGFile("rlut-test-img.jpg", imgbuf);
			trk.BuildLUT(imgbuf.data, imgbuf.pitch(), QTrkFloat, false, z);
		}
	}
	trk.FinalizeLUT();

	ImageData result = ImageData::alloc(trk.cfg.zlut_radialsteps, nplanes * nbeads);
	trk.GetRadialZLUT(result.data);

	WriteJPEGFile("rlut-test.jpg", result);

	result.free();

	imgbuf.free();
	rlut.free();
}

