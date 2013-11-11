
#include "QueuedTracker.h"
#include "QueuedCPUTracker.h"
#include "../cputrack-test/SharedTests.h"
#include <functional>

const float img_mean = 75, img_sigma = 1.62f; // as measured
// electrons have a Poisson distribution (variance=mean), so we measure this value by looking at the mean/variance
const float ElectronsPerBit = img_mean / (img_sigma*img_sigma); 



struct SpeedAccResult{
	vector3f acc,bias;
	int speed;
};

SpeedAccResult SpeedAccTest(ImageData& lut, QTrkSettings *cfg, int N, vector3f centerpos, vector3f range, const char *name)
{
	typedef QueuedCPUTracker TrkType;
	std::vector<vector3f> results, truepos;

	int NImg=std::max(1,N/50);
	std::vector<ImageData> imgs(NImg);
	const float R=5;
	
	TrkType trk(*cfg);
	float M=0.5f* cfg->width /(float) lut.w;
	ResampleLUT(&trk, &lut, M, lut.h, "resample-lut.jpg");

	for (int i=0;i<NImg;i++) {
		imgs[i]=ImageData::alloc(cfg->width,cfg->height);
		vector3f pos = centerpos + range*vector3f(rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f)*2;

		truepos.push_back(pos);
	}

	auto f = [&] (int i) {
		vector3f pos = truepos[i];

		GenerateImageFromLUT(&imgs[i], &lut, 0.0f, lut.w, vector2f( pos.x,pos.y), pos.z, M);
		imgs[i].normalize();
		ApplyPoissonNoise(imgs[i], 255 * ElectronsPerBit);
		if(i==0) WriteJPEGFile(name, imgs[i]);
	};

	ThreadPool<int, std::function<void (int index)> > pool(f);

	for (int i=0;i<truepos.size();i++) {
		pool.AddWork(i);
	}
	pool.WaitUntilDone();

	trk.SetLocalizationMode((LocMode_t)(LT_QI|LT_LocalizeZ| LT_NormalizeProfile));
	double tstart=GetPreciseTime();

	int img=0;
	for (int i=0;i<N;i++)
	{
		LocalizationJob job(i, 0, 0, 0);
		trk.ScheduleLocalization((uchar*)imgs[i%NImg].data, sizeof(float)*cfg->width, QTrkFloat, &job);
	}

	WaitForFinish(&trk, N);
	double tend = GetPreciseTime();

	results.resize(trk.GetResultCount());
	for (int i=0;i<results.size();i++) {
		LocalizationResult r;
		trk.FetchResults(&r,1);
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
		acc += d*d;
	}
	acc = sqrt(acc/N);

	SpeedAccResult r;
	r.bias = s;
	r.acc = acc;
	r.speed = N/(tend-tstart);
	dbgprintf("Speed=%d img/s, Mean.X: %f.  St. Dev.X: %f;  Mean.Z: %f.  St. Dev.Z: %f\n",r.speed, r.bias.x, r.acc.x, r.bias.z, r.acc.z);
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

		vector3f pos(cfg.width/2, cfg.height/2, lut.h/2);
		results.push_back(SpeedAccTest (lut, &cfg, n, pos, vector3f(5,5,lut.h/4), SPrintf("roi%dtestimg.jpg", cfg.width).c_str()));
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
		results.push_back(SpeedAccTest (lut, &cfg, n, pos, vector3f(5,5,3), SPrintf("zrange-z%d.jpg",z).c_str()));
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
