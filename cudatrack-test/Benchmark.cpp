
#include "QueuedTracker.h"
#include "QueuedCPUTracker.h"
#include "../cputrack-test/SharedTests.h"
#include <functional>
#include "BenchmarkLUT.h"

const float img_mean = 75, img_sigma = 1.62f; // as measured
// electrons have a Poisson distribution (variance=mean), so we measure this value by looking at the mean/variance
const float ElectronsPerBit = img_mean / (img_sigma*img_sigma); 


struct SpeedAccResult{
	vector3f acc,bias;
	int speed;
};


// Generate a LUT by creating new image samples and using the tracker in BuildZLUT mode
// This will ensure equal settings for radial profiles etc
static void ResampleBMLUT(QueuedTracker* qtrk, BenchmarkLUT* lut, float M, int zplanes=100, const char *jpgfile=0, ImageData* newlut=0)
{
	QTrkComputedConfig& cfg = qtrk->cfg;
	ImageData img = ImageData::alloc(cfg.width,cfg.height);

	std::vector<float> stetsonWindow = ComputeStetsonWindow(cfg.zlut_radialsteps);
	qtrk->SetRadialZLUT(0, 1, zplanes, &stetsonWindow[0]);
	qtrk->SetLocalizationMode( (LocMode_t)(LT_QI|LT_BuildRadialZLUT|LT_NormalizeProfile) );
	for (int i=0;i<zplanes;i++)
	{
		lut->GenerateSample(&img, vector3f(cfg.width/2, cfg.height/2, i/(float)zplanes * lut->lut_h), cfg.width*cfg.zlut_roi_coverage/2);
		//GenerateImageFromLUT(&img, lut, 0, lut->w, , M);
		img.normalize();

		LocalizationJob job(i, 0, i,0);
		qtrk->ScheduleLocalization((uchar*)img.data, sizeof(float)*img.w, QTrkFloat, &job);
	}
	img.free();

	qtrk->Flush();
	while(!qtrk->IsIdle());
	qtrk->ClearResults();

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

SpeedAccResult SpeedAccTest(ImageData& lut, QTrkSettings *cfg, int N, vector3f centerpos, vector3f range, const char *name)
{
	typedef QueuedTracker TrkType;
	std::vector<vector3f> results, truepos;

	int NImg=std::max(1,N/20);
	std::vector<ImageData> imgs(NImg);
	const float R=5;
	
	QueuedTracker* trk = CreateQueuedTracker(*cfg);
	
	BenchmarkLUT bml(&lut);
	ImageData resizedLUT = ImageData::alloc(trk->cfg.zlut_radialsteps, lut.h);
	bml.GenerateLUT(&resizedLUT, (float)trk->cfg.zlut_radialsteps/lut.w);
	WriteJPEGFile( SPrintf("resizedLUT-%s.jpg", name).c_str(), resizedLUT);

	ResampleBMLUT(trk, &bml, 1.0f, lut.h);

	//std::vector<float> stetsonWindow = ComputeStetsonWindow(trk.cfg.zlut_radialsteps);
	//trk.SetRadialZLUT(resizedLUT.data, 1, lut.h, &stetsonWindow[0]);

	float M = cfg->width / (float) (2.0f * lut.w);

	for (int i=0;i<NImg;i++) {
		imgs[i]=ImageData::alloc(cfg->width,cfg->height);
		vector3f pos = centerpos + range*vector3f(rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f)*2;

		bml.GenerateSample(&imgs[i], pos, trk->cfg.width*trk->cfg.zlut_roi_coverage/2);
		//GenerateImageFromLUT(&imgs[i], &resizedLUT, 0.0f, lut.w, vector2f( pos.x,pos.y), pos.z, M);
		imgs[i].normalize();
		ApplyPoissonNoise(imgs[i], 255 * ElectronsPerBit, 255);
		if(i==0) WriteJPEGFile(name, imgs[i]);

		truepos.push_back(pos);
	}

	trk->SetLocalizationMode((LocMode_t)(LT_QI|LT_LocalizeZ| LT_NormalizeProfile));
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
	for (int i=0;i<results.size();i++) {
		LocalizationResult r;
		trk->FetchResults(&r,1);
		results[r.job.frame]=r.pos;
	}

	for (int i=0;i<NImg;i++)
		imgs[i].free();

	vector3f s;
	for(int i=0;i<N;i++) {
		s+=results[i]-truepos[i%NImg]; 
	}
	s*=1.0f/N;

	vector3f acc;
	for (int i=0;i<results.size();i++) {
		vector3f d = results[i]-truepos[i%NImg]; 
		vector3f errMinusMeanErr = d - s;
		acc += errMinusMeanErr*errMinusMeanErr;
	}
	acc = sqrt(acc/N);

	SpeedAccResult r;
	r.bias = s;
	r.acc = acc;
	r.speed = N/(tend-tstart);
	dbgprintf("Speed=%d img/s, Mean.X: %f.  St. Dev.X: %f;  Mean.Z: %f.  St. Dev.Z: %f\n",r.speed, r.bias.x, r.acc.x, r.bias.z, r.acc.z);
	resizedLUT.free();
	delete trk;
	return r;
}

void BenchmarkROISizes(int n)
{
	std::vector<SpeedAccResult> results;
	std::vector<int> rois;

	const char *lutfile = "refbeadlut.jpg";
	ImageData lut = ReadJPEGFile(lutfile);

	for (int roi=50;roi<=160;roi+=10) {
	//for (int roi=90;roi<100;roi+=10) {
		QTrkSettings cfg;
		cfg.qi_angstep_factor = 2;
		cfg.qi_iterations = 4;
		cfg.qi_angular_coverage = 0.7f;
		cfg.qi_roi_coverage = 1;
		cfg.qi_radial_coverage = 2.5f;
		cfg.zlut_angular_coverage = 0.7f;
		cfg.zlut_roi_coverage = 1;
		cfg.zlut_radial_coverage = 2.5f;

		rois.push_back(roi);

		cfg.width = roi;
		cfg.height = roi;

		vector3f pos(cfg.width/2, cfg.height/2, lut.h/3);
		results.push_back(SpeedAccTest (lut, &cfg, n, pos, vector3f(0,0,0), SPrintf("roi%dtestimg.jpg", cfg.width).c_str()));
	}
	lut.free();

	for (int i=0;i<results.size();i++) {
		auto r = results[i];
		float row[] = { rois[i], r.acc.x, r.acc.y, r.acc.z, r.bias.x, r.bias.y, r.bias.z, r.speed };
		WriteArrayAsCSVRow("roi-sizes.txt", row, sizeof(row)/sizeof(float),i>0);
	}
}

void BenchmarkZAccuracy(int n)
{
	std::vector<SpeedAccResult> results;
	std::vector<int> zplanes;
	
	const char *lutfile = "refbeadlut.jpg";
	ImageData lut = ReadJPEGFile(lutfile);

	for (int z=0;z<lut.h;z+=5) {
		QTrkSettings cfg;
		cfg.qi_angstep_factor = 2;
		cfg.qi_iterations = 4;
		cfg.qi_angular_coverage = 0.7f;
		cfg.qi_roi_coverage = 1;
		cfg.qi_radial_coverage = 2.5f;
		cfg.zlut_angular_coverage = 0.7f;
		cfg.zlut_roi_coverage = 1;
		cfg.zlut_radial_coverage = 2.5f;

		cfg.width = 100;
		cfg.height = 100;

		vector3f pos(cfg.width/2, cfg.height/2, z);
		results.push_back(SpeedAccTest (lut, &cfg, n, pos, vector3f(0,0,0), SPrintf("zrange-z%d.jpg",z).c_str()));
		zplanes.push_back(z);
	}
	lut.free();

	for (int i=0;i<results.size();i++) {
		auto r = results[i];
		float row[] = { zplanes[i], r.acc.x, r.acc.y, r.acc.z, r.bias.x, r.bias.y, r.bias.z, r.speed };
		WriteArrayAsCSVRow("lutpos.txt", row, sizeof(row)/sizeof(float),i>0);
	}
}

void BenchmarkParams()
{
	/*
	- Accuracy vs ROIsize
	- Speed vs ROIsize
	*/
#ifdef _DEBUG
	int n = 50;
#else
	int n = 6000;
#endif

	BenchmarkROISizes(n);
	BenchmarkZAccuracy(n);
}
