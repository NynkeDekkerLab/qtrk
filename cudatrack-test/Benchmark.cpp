
#include "QueuedTracker.h"
#include "QueuedCPUTracker.h"
#include "../cputrack-test/SharedTests.h"
#include <functional>
#include "BenchmarkLUT.h"
#include "FisherMatrix.h"

const float img_mean = 75, img_sigma = 1.62f; // as measured
// electrons have a Poisson distribution (variance=mean), so we measure this value by looking at the mean/variance
const float ElectronsPerBit = img_mean / (img_sigma*img_sigma); 


struct SpeedAccResult{
	vector3f acc,bias,crlb;
	int speed;
};




SpeedAccResult SpeedAccTest(ImageData& lut, QTrkSettings *cfg, int N, vector3f centerpos, vector3f range, const char *name, int MaxPixelValue)
{
	typedef QueuedTracker TrkType;
	std::vector<vector3f> results, truepos;

	int NImg=std::max(1,N/20);
	std::vector<ImageData> imgs(NImg);
	const float R=5;
	
	QueuedTracker* trk = new QueuedCPUTracker(*cfg);// CreateQueuedTracker(*cfg);

	ImageData resizedLUT = ImageData::alloc(trk->cfg.zlut_radialsteps, lut.h);
	ResampleLUT(trk, &lut, lut.h, &resizedLUT, SPrintf("lut_resized_%s.jpg", name).c_str());

	Matrix3X3 fisher;

	for (int i=0;i<NImg;i++) {
		imgs[i]=ImageData::alloc(cfg->width,cfg->height);
		vector3f pos = centerpos + range*vector3f(rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f)*1;
		GenerateImageFromLUT(&imgs[i], &resizedLUT, trk->cfg.zlut_minradius, trk->cfg.zlut_maxradius, vector3f( pos.x,pos.y, pos.z));

		SampleFisherMatrix fm(MaxPixelValue);
		fisher += fm.Compute(pos, vector3f(1,1,1)*0.001f, resizedLUT, cfg->width,cfg->height, trk->cfg.zlut_minradius,trk->cfg.zlut_maxradius);

		imgs[i].normalize();
		if (MaxPixelValue> 0) ApplyPoissonNoise(imgs[i], MaxPixelValue);
		if(i==0) WriteJPEGFile(name, imgs[i]);

		truepos.push_back(pos);
	}

	int flags= LT_LocalizeZ|LT_NormalizeProfile;//|LT_LocalizeZWeighted;
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
	fisher *= 1.0f/NImg;
	r.crlb = sqrt(fisher.Inverse().diag());
	r.speed = N/(tend-tstart);
	dbgprintf("ROI:%d, Speed=%d img/s, Mean.X: %f.  St. Dev.X: %f;  Mean.Z: %f.  St. Dev.Z: %f\n", cfg->width, r.speed, r.bias.x, r.acc.x, r.bias.z, r.acc.z);
	resizedLUT.free();
	delete trk;
	return r;
}

void BenchmarkROISizes(const char *name, int n, int MaxPixelValue, int qi_iterations)
{
	std::vector<SpeedAccResult> results;
	std::vector<int> rois;

	const char *lutfile = "refbeadlut.jpg";
	ImageData lut = ReadJPEGFile(lutfile);

	for (int roi=30;roi<=180;roi+=10) {
	//for (int roi=90;roi<100;roi+=10) {
		QTrkSettings cfg;
		cfg.qi_angstep_factor = 1.3f;
		cfg.qi_iterations = qi_iterations;
		cfg.qi_angular_coverage = 0.7f;
		cfg.qi_roi_coverage = 1;
		cfg.qi_radial_coverage = 2.5f;
		cfg.zlut_angular_coverage = 0.7f;
		cfg.zlut_roi_coverage = 1;
		cfg.zlut_radial_coverage = 2.5f;
		cfg.zlut_minradius = 0;
		cfg.qi_minradius = 0;
		rois.push_back(roi);

		cfg.width = roi;
		cfg.height = roi;

		vector3f pos(cfg.width/2, cfg.height/2, lut.h/4);
		results.push_back(SpeedAccTest(lut, &cfg, n, pos, vector3f(2,2,2), SPrintf("roi%dtestimg.jpg", cfg.width).c_str(), MaxPixelValue));
	}
	lut.free();

	for (int i=0;i<results.size();i++) {
		auto r = results[i];
		float row[] = { rois[i], r.acc.x, r.acc.z, r.bias.x, r.bias.z,  r.crlb.x, r.crlb.z, r.speed };
		WriteArrayAsCSVRow(name, row, sizeof(row)/sizeof(float),i>0);
	}
}


template<typename T>
void BenchmarkConfigParamRange(int n, T QTrkSettings::* param, QTrkSettings* config, std::vector<T> param_values, const char *name, int MaxPixelValue)
{
	std::vector<SpeedAccResult> results;

	const char *lutfile = "refbeadlut.jpg";
	ImageData lut = ReadJPEGFile(lutfile);

	for(int i =0; i<param_values.size();i++) {
		QTrkSettings cfg = *config;
		cfg.*param = param_values[i];

		vector3f pos(cfg.width/2, cfg.height/2, lut.h/3);
		std::string pvname = SPrintf("%s-%d.jpg", name, i);
		results.push_back(SpeedAccTest (lut, &cfg, n, pos, vector3f(0,0,0), pvname.c_str(), MaxPixelValue ));
	}
	lut.free();

	for (int i=0;i<results.size();i++) {
		auto r = results[i];
		float row[] = { param_values[i], r.acc.x, r.acc.y, r.acc.z, r.bias.x, r.bias.y, r.bias.z, r.speed };
		WriteArrayAsCSVRow(SPrintf("%s-results.txt", name).c_str(), row, sizeof(row)/sizeof(float),i>0);
	}
}

void BenchmarkZAccuracy(const char *name, int n, int MaxPixelValue)
{
	std::vector<SpeedAccResult> results;
	std::vector<int> zplanes;
	
	const char *lutfile = "refbeadlut.jpg";
	ImageData lut = ReadJPEGFile(lutfile);

	for (int z=5;z<lut.h;z+=5) {
		QTrkSettings cfg;
		cfg.qi_angstep_factor = 2;
		cfg.qi_iterations = 3;
		cfg.qi_angular_coverage = 0.7f;
		cfg.qi_roi_coverage = 1;
		cfg.qi_radial_coverage = 1.5f;
		cfg.zlut_angular_coverage = 0.7f;
		cfg.zlut_roi_coverage = 1;
		cfg.zlut_radial_coverage = 2.5f;

		cfg.width = 100;
		cfg.height = 100;

		vector3f pos(cfg.width/2, cfg.height/2, z);
		results.push_back(SpeedAccTest (lut, &cfg, n, pos, vector3f(2,2,1), SPrintf("%s-zrange-z%d.jpg",name,z).c_str(), MaxPixelValue));
		zplanes.push_back(z);
	}
	lut.free();

	for (int i=0;i<results.size();i++) {
		auto r = results[i];
		float row[] = { zplanes[i], r.acc.x, r.acc.y, r.acc.z, r.bias.x, r.bias.y, r.bias.z, r.speed };
		WriteArrayAsCSVRow(name, row, sizeof(row)/sizeof(float),i>0);
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


void BenchmarkParams()
{
	/*
	- Accuracy vs ROIsize
	- Speed vs ROIsize
	*/
#ifdef _DEBUG
	int n = 50;
#else
	int n = 10000;
#endif

	QTrkSettings basecfg;
	basecfg.width = 80;
	basecfg.height = 80;
	basecfg.qi_iterations = 4;
	basecfg.qi_roi_coverage = 1;
	basecfg.qi_radial_coverage = 1.5f;
	basecfg.qi_angular_coverage = 0.7f;
	basecfg.zlut_roi_coverage = 1;
	basecfg.zlut_radial_coverage = 1.5f;
	basecfg.com_bgcorrection = 0;
	basecfg.qi_angstep_factor = 2;
	basecfg.zlut_angular_coverage = 0.7f;


	int mpv = 0;//255  *28;

	for (int i=0;i<4;i++)
		BenchmarkROISizes(SPrintf("roi_qi%d.txt",i).c_str(), n, mpv, i);

/*	BenchmarkConfigParamRange (n, &QTrkSettings::qi_radial_coverage, &basecfg, linspace(0.2f, 4.0f, 20), "qi_rad_cov_noise", mpv );
	BenchmarkConfigParamRange (n, &QTrkSettings::zlut_radial_coverage, &basecfg, linspace(0.2f, 4.0f, 20), "zlut_rad_cov_noise", mpv);
	BenchmarkConfigParamRange (n, &QTrkSettings::qi_iterations, &basecfg, linspace(1, 6, 6), "qi_iterations_noise", mpv);
	BenchmarkZAccuracy("zpos-noise.txt", n, mpv);

	BenchmarkROISizes("roi-sizes.txt", n, 0);
	BenchmarkConfigParamRange (n, &QTrkSettings::qi_radial_coverage, &basecfg, linspace(0.2f, 4.0f, 20), "qi_rad_cov", 0);
	BenchmarkConfigParamRange (n, &QTrkSettings::zlut_radial_coverage, &basecfg, linspace(0.2f, 4.0f, 20), "zlut_rad_cov", 0);
	BenchmarkConfigParamRange (n, &QTrkSettings::qi_iterations, &basecfg, linspace(1, 6, 6), "qi_iterations", 0);
	BenchmarkZAccuracy("zpos.txt", n, 0);*/
}
