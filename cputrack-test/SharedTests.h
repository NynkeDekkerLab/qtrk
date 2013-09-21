#pragma once


#include <direct.h> // _mkdir()


// Generate a LUT by creating new image samples and using the tracker in BuildZLUT mode
// This will ensure equal settings for radial profiles etc
static void ResampleLUT(QueuedTracker* qtrk, ImageData* lut, float zlutMinRadius, float zlutMaxRadius, int zplanes=100, const char *jpgfile=0)
{
	QTrkComputedConfig& cfg = qtrk->cfg;
	ImageData img = ImageData::alloc(cfg.width,cfg.height);

	qtrk->SetZLUT(0, 1, zplanes);

	for (int i=0;i<zplanes;i++)
	{
		GenerateImageFromLUT(&img, lut, zlutMinRadius, zlutMaxRadius, vector2f(cfg.width/2, cfg.height/2), i/(float)zplanes * lut->h, 1.0f);
		img.normalize();

		LocalizationJob job((LocalizeType)(LT_QI|LT_BuildZLUT|LT_NormalizeProfile), i, 0, i,0);
		qtrk->ScheduleLocalization((uchar*)img.data, sizeof(float)*img.w, QTrkFloat, &job);
	}
	img.free();

	while(!qtrk->IsIdle());
	qtrk->ClearResults();

	if (jpgfile) {
		float* zlut_result=new float[zplanes*cfg.zlut_radialsteps*1];
		qtrk->GetZLUT(zlut_result);
		FloatToJPEGFile(jpgfile, zlut_result, cfg.zlut_radialsteps, zplanes);
		delete[] zlut_result;
	}
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
	float gainStDevScale = 0.05f;
#ifdef _DEBUG
	int nDriftSteps = 100;
#else
	int nDriftSteps = 5000;
#endif
	float driftDistance = 4; // pixels

	ImageData lut = ReadJPEGFile(lutfile);
	ImageData img = ImageData::alloc(120,120);

	float z = 30.0f;
	QTrkSettings cfg;
	cfg.width = img.w; cfg.height = img.h;
	TrackerType trk(cfg);

	int nBeads=1;
	float zlutMinR=4,zlutMaxR = cfg.width/2-5;
	ResampleLUT(&trk, &lut,100, zlutMinR,zlutMaxR, "resample-zlut.jpg");
//	trk.SetZLUT(lut.data, nBeads, lut.h);
	
	float *offset = new float [nBeads * cfg.width * cfg.height];
	float *gain = new float [nBeads * cfg.width * cfg.height];

	bool useCalib = false;
	bool saveImgs = true;

	float *info = new float [nNoiseLevels*2];

	for (int nl=0;nl<nNoiseLevels;nl++) {

		float offset_stdev = info[nl*2+0] = nl*0.01f;
		float gain_stdev = info[nl*2+1] = nl*gainStDevScale;

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


			LocalizationJob job((LocalizeType)(LT_QI|LT_LocalizeZ|LT_NormalizeProfile),d, d,0,0);
			trk.ScheduleLocalization((uchar*)img.data, sizeof(float)*img.w, QTrkFloat, &job);
		}

		while (!trk.IsIdle());
		int nResults = trk.GetResultCount();
		auto results = new LocalizationResult[nResults];
		trk.FetchResults(results, nResults);
		WriteTrace(SPrintf("%s\\trace.txt", dirname.c_str()), results, nResults);
		delete[] results;

		dbgprintf("noiselevel: %d\n", nl);
	}
	WriteImageAsCSV("offset_gain_stdev.txt", info, 2, nNoiseLevels);

	img.free();
	lut.free();

	delete[] gain;
	delete[] offset;
	delete[] info;
}

void EnableGainCorrection(QueuedTracker* qtrk)
{
	int nb, _pl, _r;
	qtrk->GetZLUTSize(nb, _pl, _r);

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
	LocalizeType lt = LT_Gaussian2D;
	TrackerType qtrk(cfg);
	std::vector<float*> images;

	srand(0);

	qtrk.SetZLUT(0, 1, 1); // need to indicate 1 bead, as the pixel calibration images are per-bead
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

	for (int n=0;n<NumImages;n++) {
		for (int k=0;k<JobsPerImg;k++) {
			LocalizationJob job(lt,n,0,0,0);
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


