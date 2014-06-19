
#include "QueuedTracker.h"
#include "QueuedCPUTracker.h"
#include "SharedTests.h"
#include <functional>
#include "BenchmarkLUT.h"
#include "FisherMatrix.h"

const float img_mean = 75, img_sigma = 1.62f; // as measured
// electrons have a Poisson distribution (variance=mean), so we measure this value by looking at the mean/variance
const float ElectronsPerBit = img_mean / (img_sigma*img_sigma); 


void BenchmarkROISizes(const char *name, int n, int MaxPixelValue, int qi_iterations, int extraFlags, float range_in_nm, float pixel_size, float lutstep, int buildLUTFlags)
{
	std::vector<SpeedAccResult> results;
	std::vector<int> rois;

	const char *lutfile = "zrange\\exp_qi.radialzlut#169";
	ImageData lut = ReadLUTFile(lutfile);

	for (int roi=20;roi<=180;roi+=10) {
	//for (int roi=90;roi<100;roi+=10) {
		QTrkSettings cfg;
		cfg.qi_angstep_factor = 1.3f;
		cfg.qi_iterations = qi_iterations;
		cfg.qi_angular_coverage = 0.7f;
		cfg.qi_roi_coverage = 1;
		cfg.qi_radial_coverage = 2.5f;
		cfg.qi_minradius=2;
		cfg.zlut_minradius=2;
		cfg.zlut_angular_coverage = 0.7f;
		cfg.zlut_roi_coverage = 1;
		cfg.zlut_radial_coverage = 2.5f;
		cfg.com_bgcorrection = 0;
		cfg.xc1_profileLength = roi*0.8f;
		cfg.xc1_profileWidth = roi*0.2f;
		cfg.xc1_iterations = 1;
		rois.push_back(roi);

		cfg.width = roi;
		cfg.height = roi;

		vector3f pos(cfg.width/2, cfg.height/2, lut.h/2);
		vector3f range(range_in_nm / pixel_size, range_in_nm / pixel_size, range_in_nm / lutstep);
		results.push_back(SpeedAccTest(lut, &cfg, n, pos, range, SPrintf("roi%dtestimg.jpg", cfg.width).c_str(), MaxPixelValue, extraFlags, buildLUTFlags));
		auto lr = results.back();
		dbgprintf("ROI:%d, #QI:%d, Speed=%d img/s, Mean.X: %f.  St. Dev.X: %f;  Mean.Z: %f.  St. Dev.Z: %f\n", roi, qi_iterations, lr.speed, lr.bias.x, lr.acc.x, lr.bias.z, lr.acc.z);
	}
	lut.free();

	for (uint i=0;i<results.size();i++) {
		auto r = results[i];
		float row[] = { rois[i], r.acc.x, r.acc.z, r.bias.x, r.bias.z,  r.crlb.x, r.crlb.z, r.speed, n };
		WriteArrayAsCSVRow(name, row, sizeof(row)/sizeof(float),i>0);
	}
}


template<typename T>
void BenchmarkConfigParamRange(int n, T QTrkSettings::* param, QTrkSettings* config, std::vector<T> param_values, const char *name, int MaxPixelValue, vector3f range)
{
	std::vector<SpeedAccResult> results;

	const char *lutfile = "lut000.jpg";
	ImageData lut = ReadJPEGFile(lutfile);

	for(uint i =0; i<param_values.size();i++) {
		QTrkSettings cfg = *config;
		cfg.*param = param_values[i];

		vector3f pos(cfg.width/2, cfg.height/2, lut.h/3);
		std::string pvname = SPrintf("%s-%d.jpg", name, i);
		results.push_back(SpeedAccTest (lut, &cfg, n, pos, range, pvname.c_str(), MaxPixelValue ));
	}
	lut.free();

	for (int i=0;i<results.size();i++) {
		auto r = results[i];
		float row[] = { param_values[i], r.acc.x, r.acc.y, r.acc.z, r.bias.x, r.bias.y, r.bias.z, r.speed, n };
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

	for (uint i=0;i<results.size();i++) {
		auto r = results[i];
		float row[] = { zplanes[i], r.acc.x, r.acc.y, r.acc.z, r.bias.x, r.bias.y, r.bias.z, r.speed };
		WriteArrayAsCSVRow(name, row, sizeof(row)/sizeof(float),i>0);
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
	int n = 300;
#endif

	int mpv = 10000;
	float pixel_size = 120, lutstep = 50;

	for (int zlutbias=0;zlutbias<2;zlutbias++) {
		float range_in_nm=0;
		for (int bias=0;bias<2;bias++) {
			for (int i=0;i<5;i++)
				BenchmarkROISizes(SPrintf("roi_qi%d_bias%d_zlutbias%d.txt",i,bias, zlutbias).c_str(), n, mpv, i, 0, range_in_nm, pixel_size, lutstep, zlutbias ? BUILDLUT_BIASCORRECT : 0);
	//		for (int i=0;i<5;i++)
		//		BenchmarkROISizes(SPrintf("roi_qi%d_bias%d_wz.txt",i,bias).c_str(), n, mpv, i, LT_LocalizeZWeighted, range_in_nm, pixel_size, lutstep);
			BenchmarkROISizes( SPrintf("roi_xcor_bias%d_zlutbias%d.txt", bias, zlutbias).c_str(), n, mpv, 0, LT_XCor1D, range_in_nm, pixel_size, lutstep, zlutbias ? BUILDLUT_BIASCORRECT : 0);
	//		BenchmarkROISizes( SPrintf("roi_xcor_bias%d_wz.txt",bias).c_str(), n, mpv, 0, LT_XCor1D | LT_LocalizeZWeighted, range_in_nm, pixel_size, lutstep);
			range_in_nm = 200;
		}
	}

	QTrkSettings basecfg;
	basecfg.width = 80;
	basecfg.height = 80;
	basecfg.qi_iterations = 4;
	basecfg.qi_roi_coverage = 1;
	basecfg.qi_minradius=1;
	basecfg.zlut_minradius=1;
	basecfg.qi_radial_coverage = 2.5f;
	basecfg.qi_angular_coverage = 0.7f;
	basecfg.zlut_roi_coverage = 1;
	basecfg.zlut_radial_coverage = 1.5f;
	basecfg.com_bgcorrection = 0;
	basecfg.qi_angstep_factor = 1.1f;
	basecfg.zlut_angular_coverage = 0.7f;

	//BenchmarkConfigParamRange (20000, &QTrkSettings::qi_iterations, &basecfg, linspace(1, 6, 6), "qi_iterations_noise", mpv);
/*
	BenchmarkConfigParamRange (n, &QTrkSettings::qi_radial_coverage, &basecfg, linspace(0.2f, 4.0f, 20), "qi_rad_cov_noise", mpv );
	BenchmarkConfigParamRange (n, &QTrkSettings::zlut_radial_coverage, &basecfg, linspace(0.2f, 4.0f, 20), "zlut_rad_cov_noise", mpv);
	BenchmarkConfigParamRange (n, &QTrkSettings::qi_iterations, &basecfg, linspace(1, 6, 6), "qi_iterations_noise", mpv);
	BenchmarkZAccuracy("zpos-noise.txt", n, mpv);

	BenchmarkROISizes("roi-sizes.txt", n, 0);
	BenchmarkConfigParamRange (n, &QTrkSettings::qi_radial_coverage, &basecfg, linspace(0.2f, 4.0f, 20), "qi_rad_cov", 0);
	BenchmarkConfigParamRange (n, &QTrkSettings::zlut_radial_coverage, &basecfg, linspace(0.2f, 4.0f, 20), "zlut_rad_cov", 0);
	BenchmarkConfigParamRange (n, &QTrkSettings::qi_iterations, &basecfg, linspace(1, 6, 6), "qi_iterations", 0);
	BenchmarkZAccuracy("zpos.txt", n, 0);*/
}
