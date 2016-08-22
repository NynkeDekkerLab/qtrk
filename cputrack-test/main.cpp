#include "../cputrack/std_incl.h"
#include "../cputrack/cpu_tracker.h"
#include "../cputrack/random_distr.h"
#include "../cputrack/QueuedCPUTracker.h"
#include "../cputrack/FisherMatrix.h"
#include "../cputrack/BeadFinder.h"
#include "../cputrack/LsqQuadraticFit.h"
#include "../utils/ExtractBeadImages.h"
#include "../cputrack/BenchmarkLUT.h"
#include "../cputrack/CubicBSpline.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include "testutils.h"
#include "SharedTests.h"
#include "ResultManager.h"

const float ANGSTEPF = 1.5f;

const bool InDebugMode = 
#ifdef _DEBUG
	true
#else
	false
#endif
	;

void RescaleLUT(CPUTracker* trk, ImageData* lut)
{

}

void SpeedTest()
{
#ifdef _DEBUG
	int N = 20;
#else
	int N = 1000;
#endif
	int qi_iterations = 7;
	int xcor_iterations = 7;
	CPUTracker* tracker = new CPUTracker(150,150, 128);

	int radialSteps = 64, zplanes = 120;
	float zmin = 2, zmax = 8;
	float zradius = tracker->xcorw/2;

	float* zlut = new float[radialSteps*zplanes];
	for (int x=0;x<zplanes;x++)  {
		vector2f center (tracker->GetWidth()/2, tracker->GetHeight()/2);
		float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
		GenerateTestImage(ImageData(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight()), center.x, center.y, s, 0.0f);
		tracker->mean = 0.0f;
		tracker->ComputeRadialProfile(&zlut[x*radialSteps], radialSteps, 64, 1, zradius, center, false);	
	}
	tracker->SetRadialZLUT(zlut, zplanes, radialSteps, 1,1, zradius, true, true);
	delete[] zlut;

	// Speed test
	vector2f comdist, xcordist, qidist;
	float zdist=0.0f;
	double zerrsum=0.0f;
	double tcom = 0.0, tgen=0.0, tz = 0.0, tqi=0.0, txcor=0.0;
	for (int k=0;k<N;k++)
	{
		double t0 = GetPreciseTime();
		float xp = tracker->GetWidth()/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = tracker->GetHeight()/2+(rand_uniform<float>() - 0.5) * 5;
		float z = zmin + 0.1f + (zmax-zmin-0.2f) * rand_uniform<float>();

		GenerateTestImage(ImageData(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight()), xp, yp, z, 0);

		double t1 = GetPreciseTime();
		vector2f com = tracker->ComputeMeanAndCOM();
		vector2f initial(com.x, com.y);
		double t2 = GetPreciseTime();
		bool boundaryHit = false;
		vector2f xcor = tracker->ComputeXCorInterpolated(initial, xcor_iterations, 16, boundaryHit);
		if (boundaryHit)
			dbgprintf("xcor boundaryhit!!\n");

		comdist.x += fabsf(com.x - xp);
		comdist.y += fabsf(com.y - yp);

		xcordist.x +=fabsf(xcor.x - xp);
		xcordist.y +=fabsf(xcor.y - yp);
		double t3 = GetPreciseTime();
		boundaryHit = false;
		vector2f qi = tracker->ComputeQI(initial, qi_iterations, 64, 16,ANGSTEPF, 5,50, boundaryHit);
		qidist.x += fabsf(qi.x - xp);
		qidist.y += fabsf(qi.y - yp);
		double t4 = GetPreciseTime();
		if (boundaryHit)
			dbgprintf("qi boundaryhit!!\n");

		boundaryHit = false;
		float est_z = zmin + (zmax-zmin)*tracker->ComputeZ(qi, 64, 0, &boundaryHit, 0) / (zplanes-1);
		zdist += fabsf(est_z-z);
		zerrsum += est_z-z;

		if (boundaryHit)
			dbgprintf("computeZ boundaryhit!!\n");
		double t5 = GetPreciseTime();
	//	dbgout(SPrintf("xpos:%f, COM err: %f, XCor err: %f\n", xp, com.x-xp, xcor.x-xp));
		if (k>0) { // skip first initialization round
			tgen+=t1-t0;
			tcom+=t2-t1;
			txcor+=t3-t2;
			tqi+=t4-t3;
			tz+=t5-t4;
		}
	}

	int Nns = N-1;
	dbgprintf("Image gen. (img/s): %f\nCenter-of-Mass speed (img/s): %f\n", Nns/tgen, Nns/tcom);
	dbgprintf("XCor estimation (img*it/s): %f\n", (Nns*xcor_iterations)/txcor);
	dbgprintf("COM+XCor(%d) (img/s): %f\n", xcor_iterations, Nns/(tcom+txcor));
	dbgprintf("Z estimation (img/s): %f\n", Nns/tz);
	dbgprintf("QI speed: %f (img*it/s)\n", (Nns*qi_iterations)/tqi);
	dbgprintf("Average dist: COM x: %f, y: %f\n", comdist.x/N, comdist.y/N);
	dbgprintf("Average dist: Cross-correlation x: %f, y: %f\n", xcordist.x/N, xcordist.y/N);
	dbgprintf("Average dist: QI x: %f, y: %f\n", qidist.x/N, qidist.y/N);
	dbgprintf("Average dist: Z: %f. Mean error:%f\n", zdist/N, zerrsum/N); 
	
	delete tracker;
}

void OnePixelTest()
{
	CPUTracker* tracker = new CPUTracker(32,32, 16);

	tracker->GetPixel(15,15) = 1;
	dbgout(SPrintf("Pixel at 15,15\n"));
	vector2f com = tracker->ComputeMeanAndCOM();
	dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));
	
	vector2f initial(15,15);
	bool boundaryHit = false;
	vector2f xcor = tracker->ComputeXCorInterpolated(initial,2, 16, boundaryHit);
	dbgout(SPrintf("XCor: %f,%f\n", xcor.x, xcor.y));

	assert(xcor.x == 15.0f && xcor.y == 15.0f);
	delete tracker;
}
 
void SmallImageTest()
{
	CPUTracker *tracker = new CPUTracker(50,50, 16);

	GenerateTestImage(ImageData(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight()), tracker->width/2,tracker->height/2, 9, 0.0f);
	FloatToJPEGFile("smallimg.jpg", tracker->srcImage, tracker->width, tracker->height);

	vector2f com = tracker->ComputeMeanAndCOM(0);
	dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));
	
	vector2f initial(25,25);
	bool boundaryHit = false;
	vector2f xcor = tracker->ComputeXCorInterpolated(initial, 2, 16, boundaryHit);
	dbgout(SPrintf("XCor: %f,%f\n", xcor.x, xcor.y));
	//assert(fabsf(xcor.x-15.0f) < 1e-6 && fabsf(xcor.y-15.0f) < 1e-6);

	int I=4;
	vector2f pos = initial;
	for (int i=0;i<I;i++) {
		bool bhit;
		vector2f np = tracker->ComputeQI(pos, 1, 32, 4, 1, 1, 16, bhit);
		dbgprintf("qi[%d]. New=%.4f, %.4f;\tOld=%.4f, %.4f\n", i, np.x, np.y, pos.x, pos.y);
	}


	FloatToJPEGFile("debugimg.jpg", tracker->GetDebugImage(), tracker->width, tracker->height);
	delete tracker;
}

void OutputProfileImg()
{
	CPUTracker *tracker = new CPUTracker(128,128, 16);
	bool boundaryHit;

	for (int i=0;i<10;i++) {
		float xp = tracker->GetWidth()/2+(rand_uniform<float>() - 0.5) * 20;
		float yp = tracker->GetHeight()/2+(rand_uniform<float>() - 0.5) * 20;
		
		GenerateTestImage(ImageData(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight()), xp, yp, 1, 0.0f);

		vector2f com = tracker->ComputeMeanAndCOM();
		dbgout(SPrintf("COM: %f,%f\n", com.x-xp, com.y-yp));
	
		vector2f initial = com;
		boundaryHit=false;
		vector2f xcor = tracker->ComputeXCorInterpolated(initial, 3, 16, boundaryHit);
		dbgprintf("XCor: %f,%f. Err: %d\n", xcor.x-xp, xcor.y-yp, boundaryHit);

		boundaryHit=false;
		vector2f qi = tracker->ComputeQI(initial, 3, 64, 32, ANGSTEPF, 1, 10, boundaryHit);
		dbgprintf("QI: %f,%f. Err: %d\n", qi.x-xp, qi.y-yp, boundaryHit);
	}

	delete tracker;
}
 
void TestBoundCheck()
{
	CPUTracker *tracker = new CPUTracker(32,32, 16);
	bool boundaryHit;

	for (int i=0;i<10;i++) {
		float xp = tracker->GetWidth()/2+(rand_uniform<float>() - 0.5) * 20;
		float yp = tracker->GetHeight()/2+(rand_uniform<float>() - 0.5) * 20;
		
		GenerateTestImage(ImageData(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight()), xp, yp, 1, 0.0f);

		vector2f com = tracker->ComputeMeanAndCOM();
		dbgout(SPrintf("COM: %f,%f\n", com.x-xp, com.y-yp));
	
		vector2f initial = com;
		boundaryHit=false;
		vector2f xcor = tracker->ComputeXCorInterpolated(initial, 3, 16, boundaryHit);
		dbgprintf("XCor: %f,%f. Err: %d\n", xcor.x-xp, xcor.y-yp, boundaryHit);

		boundaryHit=false;
		vector2f qi = tracker->ComputeQI(initial, 3, 64, 32, ANGSTEPF, 1, 10, boundaryHit);
		dbgprintf("QI: %f,%f. Err: %d\n", qi.x-xp, qi.y-yp, boundaryHit);
	}

	delete tracker;
}

void PixelationErrorTest()
{
	CPUTracker *tracker = new CPUTracker(128,128, 64);

	float X = tracker->GetWidth()/2;
	float Y = tracker->GetHeight()/2;
	int N = 20;
	for (int x=0;x<N;x++)  {
		float xpos = X + 2.0f * x / (float)N;
		GenerateTestImage(ImageData(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight()), xpos, X, 1, 0.0f);

		vector2f com = tracker->ComputeMeanAndCOM();
		//dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));

		vector2f initial(X,Y);
		bool boundaryHit = false;
		vector2f xcorInterp = tracker->ComputeXCorInterpolated(initial, 3, 32, boundaryHit);
		vector2f qipos = tracker->ComputeQI(initial, 3, tracker->GetWidth(), 128, 1, 2.0f, tracker->GetWidth()/2-10, boundaryHit);
		dbgprintf("xpos:%f, COM err: %f, XCorInterp err: %f. QI err: %f\n", xpos, com.x-xpos, xcorInterp.x-xpos, qipos.x-xpos);

	}
	delete tracker;
}

void BuildConvergenceMap(int iterations)
{
	int W=80, H=80;
	char* data=new char[W*H];
	FILE* f=fopen("singlebead.bin", "rb");
	fread(data,1,80*80,f);
	fclose(f);

	float testrange=20;
	int steps=100;
	float step=testrange/steps;
	vector2f errXCor, errQI;
	CPUTracker trk(W,H,40);

	// Get a good starting estimate
	trk.SetImage8Bit((uchar*)data,W);
	vector2f com = trk.ComputeMeanAndCOM();
	bool boundaryHit;
	vector2f cmp = trk.ComputeQI(com,8,80,64,ANGSTEPF,2,25,boundaryHit);

	float *xcorErrMap = new float[steps*steps];
	float *qiErrMap = new float[steps*steps];

	for (int y=0;y<steps;y++){
		for (int x=0;x<steps;x++)
		{
			vector2f initial (cmp.x+step*(x-steps/2), cmp.y+step*(y-steps/2) );
			vector2f xcor = trk.ComputeXCorInterpolated(initial, iterations, 64, boundaryHit);
			vector2f qi = trk.ComputeQI(initial, iterations, 80, 64,ANGSTEPF,2,30,boundaryHit);

			errXCor.x += fabs(xcor.x-cmp.x);
			errXCor.y += fabs(xcor.y-cmp.y);
			xcorErrMap[y*steps+x] = distance(xcor,cmp);

			errQI.x += fabs(qi.x-cmp.x);
			errQI.y += fabs(qi.y-cmp.y);
			qiErrMap[y*steps+x] = distance(qi,cmp);
		}
		dbgprintf("y=%d\n", y);
	}
	

	WriteImageAsCSV(SPrintf("xcor-err-i%d.csv", iterations).c_str(), xcorErrMap, steps,steps);
	WriteImageAsCSV(SPrintf("qi-err-i%d.csv", iterations).c_str(), qiErrMap, steps,steps);

	delete[] qiErrMap;
	delete[] xcorErrMap;
	delete[] data;
}

void CRP_TestGeneratedData()
{
	int w=64,h=64;
	float* img = new float[w*h];
	const char* lutname = "LUTexample25X.jpg";
	std::vector<uchar> lutdata = ReadToByteBuffer(lutname);

	int lutw,luth;
	uchar* lut;
	ReadJPEGFile(&lutdata[0], lutdata.size(), &lut, &lutw, &luth);
	delete[] img;
}

void CorrectedRadialProfileTest()
{
	// read image
	const char* imgname = "SingleBead.jpg";
	ImageData img = ReadJPEGFile(imgname);

	// localize
	CPUTracker trk(img.w,img.h);
	trk.SetImageFloat(img.data);
	vector2f com = trk.ComputeMeanAndCOM();
	bool boundaryHit;
	vector2f qi = trk.ComputeQI(com, 4, 64, 64,ANGSTEPF, 1, 30, boundaryHit);
	dbgprintf("%s: COM: %f, %f. QI: %f, %f\n", imgname, com.x, com.y, qi.x, qi.y);

	std::vector<float> angularProfile(128);
	float asym = trk.ComputeAsymmetry(qi, 64, angularProfile.size(), 1, 30, &angularProfile[0]);
//	ComputeAngularProfile(&angularProfile[0], 64, angularProfile.size(), 1, 30, qi, &img, trk.mean);
	WriteImageAsCSV("angprof.csv", &angularProfile[0], angularProfile.size(), 1);
	dbgprintf("Asymmetry value: %f\n", asym);
	std::vector<float> crp(128);
	float* crpmap = new float[angularProfile.size()*crp.size()];
	ComputeCRP(&crp[0], crp.size(), angularProfile.size(), 1, 30, qi, &img, 0.0f); 
	WriteImageAsCSV("crpmap.csv", crpmap, crp.size(), angularProfile.size());
	delete[] crpmap;
	delete[] img.data;

	//CRP_TestGeneratedData();

//	GenerateImageFromLUT(ImageData(img,w,h), ImageData
}

void WriteRadialProf(const char *file, ImageData& d)
{
	CPUTracker trk(d.w,d.h);
	trk.SetImageFloat(d.data);
	vector2f com = trk.ComputeMeanAndCOM();
	bool bhit;
	vector2f qipos = trk.ComputeQI(com, 4, 64, 64,ANGSTEPF, 5, 50, bhit);

	const int radialsteps=64;
	float radprof[radialsteps];
	trk.ComputeRadialProfile(radprof, radialsteps, 128, 5, 40, qipos, false);

	WriteImageAsCSV(file, radprof, radialsteps, 1);
}

std::vector<float> ComputeRadialWeights(int rsteps, float minRadius, float maxRadius)
{
	std::vector<float> wnd(rsteps);
	for(int x=0;x<rsteps;x++)
		wnd[x]=Lerp(minRadius, maxRadius, x/(float)rsteps) / (0.5f * (minRadius+maxRadius));
	return wnd;
}

void TestBias()
{
	ImageData lut = ReadLUTFile("lut000.jpg");

	ImageData img = ImageData::alloc(80,80);
	CPUTracker trk(img.w,img.h);

	float o = 2.03f;
	vector3f pos (img.w/2+o, img.h/2+o, lut.h/2);
	NormalizeZLUT(lut.data, 1, lut.h, lut.w);
	lut.normalize();

	srand(time(0));
	for (int i=0;i<10;i++) {
		GenerateImageFromLUT(&img, &lut, 0, img.w/2, pos, true);
		ApplyPoissonNoise(img, 11050);
		float stdev = StdDeviation(img.data, img.data + img.w);
		dbgprintf("Noise level std: %f\n",stdev);
	}

	WriteJPEGFile("TestBias-smp.jpg", img);
	trk.SetImageFloat(img.data);

	vector2f com = trk.ComputeMeanAndCOM();
	dbgprintf("COM: x:%f, y:%f\n", com.x,com.y);
	bool bhit;
	auto rw = ComputeRadialBinWindow(img.w);
	vector2f qi = trk.ComputeQI(com, 3, img.w, 3*img.w/4, 1, 0, img.w/2, bhit, &rw[0]);
	dbgprintf("QI: x: %f, y:%f\n", qi.x,qi.y);

	trk.SetRadialZLUT(lut.data, lut.h, lut.w, 1, 0, img.w/2, true, false);
	float z = trk.ComputeZ(qi,3*img.w, 0, 0, 0, 0, false);
	float znorm = trk.ComputeZ(qi,3*img.w, 0, 0, 0, 0, true);

	dbgprintf("Z: %f, ZNorm: %f, true:%f\n", z, znorm, pos.z);

	img.free();
	lut.free();
}

void TestZRangeBias(const char *name, const char *lutfile, bool normProf)
{
	ImageData lut = ReadLUTFile(lutfile);

	QTrkComputedConfig settings;
	settings.width=settings.height=100;
	settings.Update();
	ImageData img=ImageData::alloc(settings.width,settings.height);

	CPUTracker trk(settings.width,settings.height);
	NormalizeZLUT(lut.data, 1, lut.h, lut.w);
	trk.SetRadialZLUT(lut.data, lut.h, lut.w, 1, 1, settings.qi_maxradius, true, false);
	float* prof= ALLOCA_ARRAY(float,lut.w);

	int N=1000;
	std::vector< vector2f> results (N);
	for (int i=0;i<N;i++) {
		vector3f pos (settings.width/2, settings.height/2, i/(float)N*lut.h);
		GenerateImageFromLUT(&img, &lut, 1, settings.qi_maxradius, pos);
		if(InDebugMode&&i==0){
			WriteJPEGFile(SPrintf("%s-smp.jpg",name).c_str(),img);
		}

		trk.SetImageFloat(img.data);
		//trk.ComputeRadialProfile(prof,lut.w,settings.zlut_angularsteps, settings.zlut_minradius, settings.zlut_maxradius, pos.xy(), false);
		//float z=trk.LUTProfileCompare(prof, 0, 0, normProf ? CPUTracker::LUTProfMaxSplineFit : CPUTracker::LUTProfMaxQuadraticFit); result: splines no go!
		float z = trk.ComputeZ(pos.xy(), settings.zlut_angularsteps, 0, 0, 0, 0,true);
		results[i].x = pos.z;
		results[i].y = z-pos.z;
	}

	WriteImageAsCSV(SPrintf("%s-results.txt", name).c_str(),(float*) &results[0], 2, N);

	lut.free();
	img.free();
}

enum RWeightMode { RWNone, RWUniform, RWRadial, RWDerivative, RWStetson };

void TestZRange(const char *name, const char *lutfile, int extraFlags, int clean_lut, RWeightMode weightMode=RWNone, bool biasMap=false, bool biasCorrect=false)
{
	ImageData lut = ReadLUTFile(lutfile);
	vector3f delta(0.001f,0.001f, 0.001f);

	if(PathSeperator(lutfile).extension != "jpg"){
		WriteJPEGFile(SPrintf("%s-lut.jpg",name).c_str(), lut);
	}

	if (clean_lut) {
		BenchmarkLUT::CleanupLUT(lut);
		WriteJPEGFile( std::string(lutfile).substr(0, strlen(lutfile)-4).append("_bmlut.jpg").c_str(), lut );
	}

	QTrkComputedConfig settings;
	settings.qi_iterations = 2;
	settings.zlut_minradius = 1;
	settings.qi_minradius = 1;
	settings.width = settings.height = 100;
	settings.Update();
	
	float maxVal=10000;
	std::vector<float> stdv;
	dbgprintf("High-res LUT range...\n");
	SampleFisherMatrix fm( maxVal);

	QueuedCPUTracker trk(settings);
	ImageData rescaledLUT;
	ResampleLUT(&trk, &lut, lut.h, &rescaledLUT);

	if (biasCorrect) {
		CImageData result;
		trk.ComputeZBiasCorrection(lut.h*10, &result, 4, true);

		WriteImageAsCSV(SPrintf("%s-biasc.txt", name).c_str(), result.data, result.w, result.h);
	}

	int f = 0;
	if (weightMode == RWDerivative)
		f |= LT_LocalizeZWeighted;
	else if(weightMode == RWRadial) {
		std::vector<float> w(settings.zlut_radialsteps);
		for (int i=0;i<settings.zlut_radialsteps;i++)
			w[i]= settings.zlut_minradius + i/(float)settings.zlut_radialsteps*settings.zlut_maxradius;
		trk.SetRadialWeights(&w[0]);
	}
	else if (weightMode == RWStetson)
		trk.SetRadialWeights( ComputeRadialBinWindow(settings.zlut_radialsteps) );

	trk.SetLocalizationMode(LT_QI|LT_LocalizeZ|LT_NormalizeProfile|extraFlags|f);

	uint nstep= InDebugMode ? 20 : 1000;
	uint smpPerStep = InDebugMode ? 2 : 200;
	if (biasMap) {
		smpPerStep=1;
		nstep=InDebugMode? 200 : 2000;
	}

	std::vector<vector3f> truepos, positions,crlb;
	std::vector<float> stdevz;
	for (uint i=0;i<nstep;i++)
	{
		float z = 1 + i / (float)nstep * (rescaledLUT.h-2);
		vector3f pos = vector3f(settings.width/2, settings.height/2, z);
		truepos.push_back(pos);
		Matrix3X3 invFisherLUT = fm.Compute(pos, delta, rescaledLUT, settings.width, settings.height, settings.zlut_minradius, settings.zlut_maxradius).Inverse();
		crlb.push_back(sqrt(invFisherLUT.diag()));

		ImageData img=ImageData::alloc(settings.width,settings.height);

		for (uint j=0;j<smpPerStep; j++) {
			vector3f rndvec(rand_uniform<float>(), rand_uniform<float>(), rand_uniform<float>());
			if (biasMap) rndvec=vector3f();
			vector3f rndpos = pos + vector3f(1,1,0.1) * (rndvec-0.5f); // 0.1 plane is still a lot larger than the 0.02 typical accuracy
			GenerateImageFromLUT(&img, &rescaledLUT, settings.zlut_minradius, settings.zlut_maxradius, rndpos, true);
			img.normalize();
			if (!biasMap) ApplyPoissonNoise(img, maxVal);
			LocalizationJob job(positions.size(), 0, 0, 0);
			trk.ScheduleImageData(&img, &job);
			positions.push_back(rndpos);
			if(j==0 && InDebugMode) {
				WriteJPEGFile(SPrintf("%s-sampleimg.jpg",name).c_str(), img);
			}
		}
		dbgprintf("[%d] z=%f Min std deviation: X=%f, Y=%f, Z=%f.\n", i, z, crlb[i].x,crlb[i].y,crlb[i].z);
		img.free();
	}
	WaitForFinish(&trk, positions.size());
	std::vector<vector3f> trkmean(nstep), trkstd(nstep);
	std::vector<vector3f> resultpos(nstep*smpPerStep);
	for (uint i=0;i<positions.size();i++) {
		LocalizationResult lr;
		trk.FetchResults(&lr, 1);
		resultpos[lr.job.frame]=lr.pos;
	} 
	for (uint i=0;i<nstep;i++) {
		for (uint j=0;j<smpPerStep;j ++) {
			vector3f err=resultpos[i*smpPerStep+j]-positions[i*smpPerStep+j];
			trkmean[i]+=err;
		}
		trkmean[i]/=smpPerStep;
		vector3f variance;
		for (uint j=0;j<smpPerStep;j ++) {
			vector3f r = resultpos[i*smpPerStep+j];
			vector3f t = positions[i*smpPerStep+j];;
			vector3f err=r-t;
			err -= trkmean[i];
			variance += err*err;

			if (InDebugMode) {
				dbgprintf("Result: x=%f,y=%f,z=%f. True: x=%f,y=%f,z=%f\n", r.x,r.y,r.z,t.x,t.y,t.z);
			}
		}
		if (biasMap) trkstd[i]=vector3f();
		else trkstd[i] = sqrt(variance / (smpPerStep-1));
	}

	vector3f mean_std;
	std::vector<float> output;
	for(uint i=0;i<nstep;i++) {
		dbgprintf("trkstd[%d]:%f crlb=%f bias=%f true=%f\n", i, trkstd[i].z, crlb[i].z, trkmean[i].z, truepos[i].z);
		output.push_back(truepos[i].z);
		output.push_back(trkmean[i].x);
		output.push_back(trkstd[i].x);
		output.push_back(trkmean[i].z);
		output.push_back(trkstd[i].z);
		output.push_back(crlb[i].x);
		output.push_back(crlb[i].z);
		
		mean_std += trkstd[i];
	}
	dbgprintf("mean z err: %f\n", (mean_std/nstep).z);
	WriteImageAsCSV( SPrintf("%s_%d_flags%d_cl%d.txt",name, weightMode, extraFlags,clean_lut).c_str(), &output[0], 7, output.size()/7);
	lut.free();
	rescaledLUT.free();
}

void AutoBeadFindTest()
{
	auto img = ReadJPEGFile("00008153.jpg");
	auto smp = ReadJPEGFile("00008153-s.jpg");
	BeadFinder::Config cfg;
	cfg.img_distance = 0.5f;
	cfg.roi = 80;
	cfg.similarity = 0.5;

	auto results=BeadFinder::Find(&img, smp.data, &cfg);

	for (uint i=0;i<results.size();i++) {
		dbgprintf("beadpos: x=%d, y=%d\n", results[i].x, results[i].y);
		img.at(results[i].x+cfg.roi/2, results[i].y+cfg.roi/2) = 1.0f;
	}
	dbgprintf("%d beads total\n", results.size());

	FloatToJPEGFile("autobeadfind.jpg", img.data, img.w, img.h);

	img.free();
	smp.free();
}

void TestFourierLUT()
{
	QTrkSettings cfg;
	cfg.width = cfg.height = 60;
	cfg.zlut_minradius=3;
	cfg.zlut_roi_coverage=1;
	
//	auto locMode = (LocMode_t)(LT_ZLUTAlign | LT_NormalizeProfile | LT_LocalizeZ);
//	auto resultsCOM = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "com-zlutalign", locMode, 100 );

	const float NF=28;
	float zpos=10;
	
	auto locMode = (LocMode_t)(LT_QI | LT_FourierLUT | LT_NormalizeProfile | LT_LocalizeZ);
	auto resultsZA = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "qi-fourierlut",	locMode, 200, NF,zpos);

	auto locModeQI = (LocMode_t)(LT_QI | LT_NormalizeProfile | LT_LocalizeZ);
	auto resultsQI = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "qi",	locModeQI, 200, NF, zpos);

	resultsZA.computeStats(); 
	resultsQI.computeStats();

	dbgprintf("FourierLUT: X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsZA.meanErr.x, resultsZA.stdev.x, resultsZA.meanErr.z, resultsZA.stdev.z);
	dbgprintf("Only QI:   X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsQI.meanErr.x, resultsQI.stdev.x, resultsQI.meanErr.z, resultsQI.stdev.z);
}

void TestFourierLUTOnDataset()
{
	const char*basepath= "D:/jcnossen1/datasets/RefBeads 2013-09-02/2013-09-02/";

	process_bead_dir(SPrintf("%s/%s", basepath, "tmp_001").c_str(), 80, [&] (ImageData *img, int bead, int frame) {
		


	}, 1000);
}

void TestZLUTAlign()
{
	QTrkSettings cfg;
	cfg.width = cfg.height = 60;
	
//	auto locMode = (LocMode_t)(LT_ZLUTAlign | LT_NormalizeProfile | LT_LocalizeZ);
//	auto resultsCOM = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "com-zlutalign", locMode, 100 );

	const float NF=28;

	auto locModeQI = (LocMode_t)(LT_QI | LT_NormalizeProfile | LT_LocalizeZ);
	auto resultsQI = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "qi", locModeQI, 200, NF);

	auto locMode = (LocMode_t)(LT_QI | LT_ZLUTAlign | LT_NormalizeProfile | LT_LocalizeZ);
	auto resultsZA = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "qi-zlutalign", locMode, 200, NF );

	resultsZA.computeStats(); 
	resultsQI.computeStats();

	dbgprintf("ZLUTAlign: X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsZA.meanErr.x, resultsZA.stdev.x, resultsZA.meanErr.z, resultsZA.stdev.z);
	dbgprintf("Only QI:   X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsQI.meanErr.x, resultsQI.stdev.x, resultsQI.meanErr.z, resultsQI.stdev.z);
}

void TestQuadrantAlign()
{
	QTrkSettings cfg;
	cfg.width = cfg.height = 60;
	cfg.numThreads=1;
	
//	auto locMode = (LocMode_t)(LT_ZLUTAlign | LT_NormalizeProfile | LT_LocalizeZ);
//	auto resultsCOM = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "com-zlutalign", locMode, 100 );

	const float NF=10;

#ifdef _DEBUG
	int N=1;
	cfg.numThreads=1;
#else
	int N=2000;
#endif

	auto locModeQI = (LocMode_t)(LT_QI | LT_NormalizeProfile | LT_LocalizeZ);
	auto resultsQI = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "qa", locModeQI, N, NF);

	auto locMode = (LocMode_t)(LT_QI | LT_ZLUTAlign | LT_NormalizeProfile | LT_LocalizeZ);
	auto resultsZA = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "qa-qalign", locMode, N, NF );
	
	resultsZA.computeStats(); 
	resultsQI.computeStats();

	dbgprintf("QuadrantAlign: X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsZA.meanErr.x, resultsZA.stdev.x, resultsZA.meanErr.z, resultsZA.stdev.z);
	dbgprintf("Only QI:   X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsQI.meanErr.x, resultsQI.stdev.x, resultsQI.meanErr.z, resultsQI.stdev.z);
}

void SimpleTest()
{
	QTrkSettings cfg;
	cfg.qi_minradius=0;
	cfg.zlut_minradius = 0;
	cfg.width = cfg.height = 30;
	auto locModeQI = (LocMode_t)(LT_QI | LT_NormalizeProfile | LT_LocalizeZ);
	auto results = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "qi", locModeQI, 1000, 10000/255 );

	results.computeStats();
	dbgprintf("X= %f. stdev: %f\tZ=%f,  stdev: %f\n", 
		results.meanErr.x, results.stdev.x, results.meanErr.z, results.stdev.z);
}

static void TestBSplineMax(float maxpos)
{
	const int N=100;
	float v[N];

	for (int i=0;i<N;i++)
		v[i] = -sqrt(1+(i-maxpos)*(i-maxpos));
	float max = ComputeSplineFitMaxPos(v, N);
	dbgprintf("Max: %f, true: %f\n", max, maxpos); 
}

void GenerateZLUTFittingCurve(const char *lutfile)
{
	QTrkSettings settings;
	settings.width = settings.height = 80;

	QueuedCPUTracker qt(settings);
	ImageData lut = ReadJPEGFile(lutfile);
	ImageData nlut;
	ResampleLUT(&qt, &lut, lut.h, &nlut);

	CPUTracker trk(settings.width,settings.height);

	ImageData smp = ImageData::alloc(settings.width,settings.height);

	trk.SetRadialZLUT(nlut.data, nlut.h, nlut.w, 1, qt.cfg.zlut_minradius, qt.cfg.zlut_maxradius, false, false);

	int N=8;
	for (int z=0;z<6;z++) {
		vector3f pos(settings.width/2,settings.height/2, nlut.h * (1+z) / (float)N + 0.123f);
		GenerateImageFromLUT(&smp, &nlut, qt.cfg.zlut_minradius, qt.cfg.zlut_maxradius, pos);
		ApplyPoissonNoise(smp, 10000);
		WriteJPEGFile( SPrintf("zlutfitcurve-smpimg-z%d.jpg", z).c_str(), smp);
		trk.SetImageFloat(smp.data);
		std::vector<float> profile(qt.cfg.zlut_radialsteps), cmpProf(nlut.h), fitted(nlut.h);
		trk.ComputeRadialProfile(&profile[0], qt.cfg.zlut_radialsteps, qt.cfg.zlut_angularsteps, qt.cfg.zlut_minradius, qt.cfg.zlut_maxradius, pos.xy(), false);
		trk.LUTProfileCompare(&profile[0], 0, &cmpProf[0], CPUTracker::LUTProfMaxQuadraticFit, &fitted[0]);

		WriteArrayAsCSVRow("zlutfitcurve-profile.txt", &profile[0], profile.size(), z>0);
		WriteArrayAsCSVRow("zlutfitcurve-cmpprof.txt", &cmpProf[0], cmpProf.size(), z>0);
		WriteArrayAsCSVRow("zlutfitcurve-fitted.txt", &fitted[0], fitted.size(), z>0);
	}

	smp.free();
	nlut.free();
	lut.free();
}

void BenchmarkParams();

static SpeedAccResult AccBiasTest(ImageData& lut, QueuedTracker *trk, int N, vector3f centerpos, vector3f range, const char *name, int MaxPixelValue, int extraFlags=0)
{
	typedef QueuedTracker TrkType;
	std::vector<vector3f> results, truepos;

	int NImg=N;//std::max(1,N/20);
	std::vector<ImageData> imgs(NImg);
	const float R=5;
	
	int flags= LT_LocalizeZ|LT_NormalizeProfile|extraFlags;
	if (trk->cfg.qi_iterations>0) flags|=LT_QI;

	trk->SetLocalizationMode((LocMode_t)flags);
	Matrix3X3 fisher;
	for (int i=0;i<NImg;i++) {
		imgs[i]=ImageData::alloc(trk->cfg.width,trk->cfg.height);
		vector3f pos = centerpos + range*vector3f(rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f)*1;
		GenerateImageFromLUT(&imgs[i], &lut, trk->cfg.zlut_minradius, trk->cfg.zlut_maxradius, vector3f( pos.x,pos.y, pos.z));

		SampleFisherMatrix fm(MaxPixelValue);
		fisher += fm.Compute(pos, vector3f(1,1,1)*0.001f, lut, trk->cfg.width,trk->cfg.height, trk->cfg.zlut_minradius,trk->cfg.zlut_maxradius);

		imgs[i].normalize();
		if (MaxPixelValue> 0) ApplyPoissonNoise(imgs[i], MaxPixelValue);
		//if(i==0) WriteJPEGFile(name, imgs[i]);

		LocalizationJob job(i, 0, 0, 0);
		trk->ScheduleLocalization((uchar*)imgs[i%NImg].data, sizeof(float)*trk->cfg.width, QTrkFloat, &job);
		truepos.push_back(pos);
	}
	WaitForFinish(trk, N);

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

	fisher *= 1.0f/NImg;
	r.crlb = sqrt(fisher.Inverse().diag());
	return r;
}

void ScatterBiasArea(int roi, float scan_width, int steps, int samples, int qi_it, float angstep)
{
	std::vector<float> u=linspace(roi/2-scan_width/2,roi/2+scan_width/2, steps);
	
	QTrkComputedConfig cfg;
	cfg.width=cfg.height=roi;
	cfg.qi_angstep_factor = angstep;
	cfg.qi_iterations = qi_it;
	cfg.qi_angular_coverage = 0.7f;
	cfg.qi_roi_coverage = 1;
	cfg.qi_radial_coverage = 1.5f;
	cfg.qi_minradius=0;
	cfg.zlut_minradius=0;
	cfg.zlut_angular_coverage = 0.7f;
	cfg.zlut_roi_coverage = 1;
	cfg.zlut_radial_coverage = 1.5f;
	cfg.zlut_minradius = 0;
	cfg.qi_minradius = 0;
	cfg.com_bgcorrection = 0;
	cfg.xc1_profileLength = roi*0.8f;
	cfg.xc1_profileWidth = roi*0.2f;
	cfg.xc1_iterations = 1;
	cfg.Update();

	ImageData lut,orglut = ReadLUTFile("10x.radialzlut#4");
	vector3f ct(roi/2,roi/2,lut.h/2 + 0.123f);
	float dx = scan_width/steps;

	QueuedCPUTracker trk(cfg);
	ResampleLUT(&trk, &orglut, orglut.h, &lut);
	int maxval = 10000;

	ImageData tmp=ImageData::alloc(roi,roi);
	GenerateImageFromLUT(&tmp, &lut, 0, cfg.zlut_maxradius, vector3f(roi/2,roi/2,lut.h/2));
	ApplyPoissonNoise(tmp, maxval);

	std::string fn = SPrintf( "sb_area_roi%d_scan%d_steps%d_qit%d_N%d", roi, (int)scan_width, steps, qi_it, samples);
	WriteJPEGFile( (fn + ".jpg").c_str(), tmp);
	tmp.free();

	fn += ".txt";
	for (int y=0;y<steps;y++)  {
		for (int x=0;x<steps;x++)
		{
			vector3f cpos( (x+0.5f-steps/2) * dx, (y+0.5f-steps/2) * dx, 0 );

			cfg.qi_iterations = qi_it;
			auto r= AccBiasTest(orglut, &trk, samples, cpos+ct, vector3f(), 0, maxval, qi_it < 0 ? LT_XCor1D : 0);
			
			float row[] = { r.acc.x, r.acc.y, r.acc.z, r.bias.x, r.bias.y, r.bias.z,  r.crlb.x, r.crlb.z, samples };
			WriteArrayAsCSVRow(fn.c_str(), row, 9, x+y>0);

			dbgprintf("X=%d,Y=%d\n", x,y);
		}
	}
	orglut.free();
	lut.free();
}

void RunCOMAndQI(ImageData img, outputter* output){
	char buf[256];
	CPUTracker trk(img.w,img.h);
	trk.SetImageFloat(img.data);
	double t = GetPreciseTime();
	vector2f com = trk.ComputeMeanAndCOM();
	t = GetPreciseTime() - t;
	//float asym = trk.ComputeAsymmetry(com,64,64,5,50,dstAngProf);
	sprintf(buf,"%f %f %f",com.x,com.y,t);
	output->outputString(buf);

	vector2f initial(com.x, com.y);
	
	bool boundaryHit = false;
	for(int qi_iterations = 1; qi_iterations < 10; qi_iterations++){
		t = GetPreciseTime();
		vector2f qi = trk.ComputeQI(initial, qi_iterations, 64, 16,ANGSTEPF, 5,50, boundaryHit);
		t = GetPreciseTime() - t;
		//float asym = trk.ComputeAsymmetry(qi,64,64,5,50,dstAngProf);
		sprintf(buf,"%f %f %f",qi.x,qi.y,t);
		output->outputString(buf);
		boundaryHit = false;
	}
}

float SkewParam(ImageData img){
	int size = img.w * 2 + img.h * 2 - 4;
	float* outeredge = new float[size];
	GetOuterEdges(outeredge,size,img);
	float* max = std::max_element(img.data,img.data+(img.w*img.h));
	float* min = std::min_element(img.data,img.data+(img.w*img.h));
	float* outer_max = std::max_element(outeredge,outeredge+size);
	float* outer_min = std::min_element(outeredge,outeredge+size);

	float out = (*outer_max-*outer_min)/(*max-*min);
	delete[] outeredge;
	return out;
}

void TestROIDisplacement(std::vector<BeadPos> beads, ImageData oriImg, outputter* output, int ROISize, int maxdisplacement = 0)
{
	for(uint ii = 0; ii < beads.size(); ii++){

		output->newFile(SPrintf("%d,%d-ROI-%d",beads.at(ii).x,beads.at(ii).y,ROISize));
		
		for(int x_i = -maxdisplacement; x_i <= maxdisplacement; x_i++){
			for(int y_i = -maxdisplacement; y_i <= maxdisplacement; y_i++){
				int x = beads.at(ii).x + x_i - ROISize/2;
				int y = beads.at(ii).y + y_i - ROISize/2;
				ImageData img = CropImage(oriImg,x,y,ROISize,ROISize);
				output->outputImage(img,SPrintf("%d,%d-Crop",beads.at(ii).x + x_i, beads.at(ii).y + y_i));
				output->outputString(SPrintf("%d,%d - ROI (%d,%d) -> (%d,%d)",x_i,y_i,x,y,x+ROISize,y+ROISize));
				RunCOMAndQI(img,output);
				img.free();	
			}
		}
	}
}

void TestInterference(std::vector<BeadPos> beads, ImageData oriImg, outputter* output, int ROISize, vector2f displacement = vector2f(60,0))
{
	ImageData added = AddImages(oriImg,oriImg,displacement);
	//output->outputImage(added,"Added");

	for(uint ii = 0; ii < beads.size(); ii++){
		output->newFile(SPrintf("%d,%d-Inter",beads.at(ii).x,beads.at(ii).y));

		int x = beads.at(ii).x - ROISize/2;
		int y = beads.at(ii).y - ROISize/2;

		ImageData img = CropImage(added,x,y,ROISize,ROISize);
		output->outputImage(img,SPrintf("%d,%d-Inter",beads.at(ii).x,beads.at(ii).y));

		RunCOMAndQI(img,output);
		img.free();
	}

	added.free();
}

void TestSkew(std::vector<BeadPos> beads, ImageData oriImg, outputter* output, int ROISize)
{
	for(uint ii = 0; ii < beads.size(); ii++){
		output->newFile(SPrintf("%d,%d-Skew",beads.at(ii).x,beads.at(ii).y));

		int x = beads.at(ii).x - ROISize/2;
		int y = beads.at(ii).y - ROISize/2;

		ImageData img = CropImage(oriImg,x,y,ROISize,ROISize);
		ImageData skewImg = SkewImage(img,20);
		output->outputImage(skewImg,SPrintf("%d,%d-Skew",beads.at(ii).x,beads.at(ii).y));
		float imgSkew	  = SkewParam(img);
		float skewImgSkew = SkewParam(skewImg);
		output->outputString(SPrintf("%f %f",imgSkew,skewImgSkew));
		RunCOMAndQI(skewImg,output);
		img.free();
		skewImg.free();
	}
}

void TestBackground(std::vector<BeadPos> beads, ImageData oriImg, outputter* output, int ROISize)
{
	std::string out;
	
	out = "Beads-Background";
	output->newFile(out);
	
	for(uint ii = 0; ii < beads.size(); ii++){
		int x = beads.at(ii).x - ROISize/2;
		int y = beads.at(ii).y - ROISize/2;
		ImageData img = CropImage(oriImg,x,y,ROISize,ROISize);
		output->outputImage(img,SPrintf("%d,%d-Crop",beads.at(ii).x, beads.at(ii).y));
		output->outputString(SPrintf("%d,%d-Crop",beads.at(ii).x, beads.at(ii).y));
		output->outputString(SPrintf("median: %f",BackgroundMedian(img)));
		output->outputString(SPrintf("sigma: %f",BackgroundStdDev(img)));
		output->outputString(SPrintf("rms: %f",BackgroundRMS(img)));
		output->outputString("");
		img.free();	
	}
}

void BuildZLUT(std::string folder, outputter* output)
{
	int ROISize = 100;
	std::vector<BeadPos> beads = read_beadlist(SPrintf("%sbeadlist.txt",folder.c_str()));


	int numImgInStack = 1218;
	int numPositions = 1001; // 10nm/frame
	float range = 10.0f; // total range 25.0 um -> 35.0 um
	float umPerImg = range/numImgInStack;
	
	QTrkComputedConfig cfg;
	cfg.width=cfg.height = ROISize;
	cfg.qi_angstep_factor = 1;
	cfg.qi_iterations = 6;
	cfg.qi_angular_coverage = 0.7f;
	cfg.qi_roi_coverage = 1;
	cfg.qi_radial_coverage = 1.5f;
	cfg.qi_minradius=0;
	cfg.zlut_minradius=0;
	cfg.zlut_angular_coverage = 0.7f;
	cfg.zlut_roi_coverage = 1;
	cfg.zlut_radial_coverage = 1.5f;
	cfg.zlut_minradius = 0;
	cfg.qi_minradius = 0;
	cfg.com_bgcorrection = 0;
	cfg.xc1_profileLength = ROISize*0.8f;
	cfg.xc1_profileWidth = ROISize*0.2f;
	cfg.xc1_iterations = 1;
	cfg.Update();
	cfg.WriteToFile();

	int zplanes = 50;

	QueuedCPUTracker* qtrk = new QueuedCPUTracker(cfg);
	qtrk->SetLocalizationMode(LT_NormalizeProfile | LT_QI);
	qtrk->SetRadialZLUT(0, beads.size(), zplanes);
	qtrk->BeginLUT(0);

	int pxPerBead = ROISize*ROISize;
	int memSizePerBead = pxPerBead*sizeof(float);
	int startFrame = 400;
	for(int plane = 0; plane < zplanes; plane++){
		output->outputString(SPrintf("Frame %d/%d",plane+1,zplanes),true);
		int frameNum = startFrame+(int)(numImgInStack-startFrame)*((float)plane/zplanes);
		std::string file = SPrintf("%s\img%05d.jpg",folder.c_str(),frameNum);
		
		ImageData frame = ReadJPEGFile(file.c_str());

		float* data = new float[beads.size()*pxPerBead];

		for(uint ii = 0; ii < beads.size(); ii++){	
			vector2f pos;
			pos.x = beads.at(ii).x - ROISize/2;
			pos.y = beads.at(ii).y - ROISize/2;
			ImageData crop = CropImage(frame,pos.x,pos.y,ROISize,ROISize);
			//output->outputImage(crop,SPrintf("%d-%05d",ii,plane));
			memcpy(data+ii*pxPerBead,crop.data,memSizePerBead);
			crop.free();
		}
		
		/*
		// To verify seperate frame bead stack generation
		output->newFile(SPrintf("data-plane-%d",plane));
		output->outputArray(data,beads.size()*pxPerBead);

		ImageData allBeads = ImageData(data,ROISize,ROISize*beads.size());
		output->outputImage(allBeads,SPrintf("allBeads-%05d",frameNum));//*/
		
		qtrk->BuildLUT(data, sizeof(float)*ROISize, QTrkFloat, plane);
		
		frame.free();
		delete[] data;
	}

	qtrk->FinalizeLUT();
	
	for(int ii = 0; ii < beads.size(); ii++){
		ImageData lut = ImageData::alloc(cfg.zlut_radialsteps,zplanes);
		memcpy(lut.data,qtrk->GetZLUTByIndex(ii),cfg.zlut_radialsteps*zplanes*sizeof(float));
		//output->outputImage(lut,SPrintf("lut%03d,%d",beads.at(ii).x,beads.at(ii).y));
		output->outputImage(lut,SPrintf("lut%03d",ii));
		lut.free();
	}

	qtrk->Flush();
	delete qtrk;
}

void RunZTrace(std::string imagePath, std::string zlutPath, outputter* output)
{
	int ROISize = 100;
	std::vector<BeadPos> beads = read_beadlist(SPrintf("%sbeadlist.txt",imagePath.c_str()));
	//std::vector<BeadPos> beads = read_labview_beadlist(SPrintf("%sbeadlist.txt",imagePath.c_str()));
	if(beads.size() == 0){
		output->outputString("Empty beadlist!",true);
		return;
	}
	
	QTrkComputedConfig cfg;
	/*
	output->outputString("Enter used ZLUT settings.",true);
	output->outputString(SPrintf("ROI size (currently %d)",ROISize),true);
	std::cin >> ROISize;
	output->outputString("ZLUT radial sample density (default 1)",true);
	std::cin >> cfg.zlut_radial_coverage;
	output->outputString("ZLUT minradius (default 2)",true);
	std::cin >> cfg.zlut_minradius;
	output->outputString("ZLUT ROI coverage (default 1)",true);
	std::cin >> cfg.zlut_roi_coverage;
	output->outputString("ZLUT angular coverage (default 0.7)",true);
	std::cin >> cfg.zlut_angular_coverage;
	*/
	/*
	cfg.width = cfg.height = ROISize;
	cfg.qi_angstep_factor = 1;
	cfg.qi_iterations = 6;
	cfg.qi_angular_coverage = 0.7f;
	cfg.qi_roi_coverage = 1;
	cfg.qi_radial_coverage = 1.5f;
	cfg.qi_minradius = 0;
	cfg.zlut_minradius = 2;
	cfg.zlut_radial_coverage = 2;
	cfg.zlut_angular_coverage = 0.7f;
	cfg.zlut_roi_coverage = 1;
	cfg.com_bgcorrection = 0;
	cfg.xc1_profileLength = ROISize*0.8f;
	cfg.xc1_profileWidth = ROISize*0.2f;
	cfg.xc1_iterations = 1;
	cfg.testRun = true;
	*/
	cfg.width=cfg.height = ROISize;
	cfg.qi_angstep_factor = 1;
	cfg.qi_iterations = 6;
	cfg.qi_angular_coverage = 0.7f;
	cfg.qi_roi_coverage = 1;
	cfg.qi_radial_coverage = 1.5f;
	cfg.qi_minradius=0;
	cfg.zlut_minradius=0;
	cfg.zlut_angular_coverage = 0.7f;
	cfg.zlut_roi_coverage = 1;
	cfg.zlut_radial_coverage = 1.5f;
	cfg.zlut_minradius = 0;
	cfg.qi_minradius = 0;
	cfg.com_bgcorrection = 0;
	cfg.xc1_profileLength = ROISize*0.8f;
	cfg.xc1_profileWidth = ROISize*0.2f;
	cfg.xc1_iterations = 1;
	cfg.testRun = true;
	cfg.Update();
	cfg.WriteToFile();

	std::string file = SPrintf("%s\lut%03d.jpg",zlutPath.c_str(),0);
	ImageData lut = ReadJPEGFile(file.c_str());

	if(cfg.zlut_radialsteps != lut.w){
		output->outputString("ZLUT settings do not match LUT image sizes!",true);
		lut.free();
		return;
	}

	int zplanes = lut.h;
	lut.free();
	int res = cfg.zlut_radialsteps;
		
	float* zluts = new float[zplanes*res*beads.size()];
	ROIPosition* positions = new ROIPosition[beads.size()];
	for(int ii = 0; ii < beads.size(); ii++){
		std::string file = SPrintf("%s\lut%03d.jpg",zlutPath.c_str(),ii);
		ImageData lut = ReadJPEGFile(file.c_str());
		memcpy(zluts+ii*res*zplanes,lut.data,res*zplanes*sizeof(float));
		
		lut.free();

		positions[ii].x = beads.at(ii).x - ROISize/2;
		positions[ii].y = beads.at(ii).y - ROISize/2;
	}

	QueuedCPUTracker* qtrk = new QueuedCPUTracker(cfg);
	qtrk->SetRadialZLUT(zluts,beads.size(),zplanes);
	qtrk->FinalizeLUT();
	
	/*for(int ii = 0; ii < beads.size(); ii++){
		ImageData lut = ImageData::alloc(cfg.zlut_radialsteps,zplanes);
		memcpy(lut.data,qtrk->GetZLUTByIndex(ii),cfg.zlut_radialsteps*zplanes*sizeof(float));
		output->outputImage(lut,SPrintf("lut-%d,%d",beads.at(ii).x,beads.at(ii).y));
		lut.free();
	}*/
	//qtrk->ZLUTSelfTest();

	qtrk->SetLocalizationMode(LT_QI | LT_LocalizeZ | LT_NormalizeProfile);

	ResultManagerConfig RMcfg;
	RMcfg.numBeads = beads.size();
	RMcfg.numFrameInfoColumns = 0;
	RMcfg.scaling = vector3f(1.0f,1.0f,1.0f);
	RMcfg.offset  = vector3f(0.0f,0.0f,0.0f);
	RMcfg.writeInterval = 25;
	RMcfg.maxFramesInMemory = 100;
	RMcfg.binaryOutput = false;

	std::vector<std::string> colnames;
	for(int ii = 0;ii<RMcfg.numFrameInfoColumns;ii++){
		colnames.push_back(SPrintf("%d",ii));
	}

	ResultManager* RM = new ResultManager(
		SPrintf("%s\\RMOutput.txt",output->folder.c_str()).c_str(),
		SPrintf("%s\\RMFrameInfo.txt",output->folder.c_str()).c_str(),
		&RMcfg, colnames);
	
	RM->SetTracker(qtrk);

	int numFramesToTrack = NumJpgInDir(imagePath + "*");
	for(int ii = 0; ii < numFramesToTrack; ii++){
		std::string file = SPrintf("%s\img%05d.jpg",imagePath.c_str(),ii);
		ImageData img = ReadJPEGFile(file.c_str());

		LocalizationJob job(ii, 0, 0, 0);
		int queued = qtrk->ScheduleFrame(img.data,sizeof(float)*img.w,img.w,img.h,positions,beads.size(),QTrkFloat,&job);
		//output->outputString(SPrintf("Queueing frame %d. Current Queue size: %d. Added: %d.",ii,qtrk->GetQueueLength(),queued),true);
		ResultManager::FrameCounters cnt = RM->GetFrameCounters();
		output->outputString(SPrintf("Queueing frame %d. Current Queue size: %d.\n\tcaptured: %d\n\tprocessed: %d\n\tlocalizations: %d\n\tLOST: %d\n",ii,qtrk->GetQueueLength(),cnt.capturedFrames,cnt.processedFrames,cnt.localizationsDone,cnt.lostFrames),true);

		img.free();
	}

	while(qtrk->GetQueueLength() != 0);
	RM->Flush();
	delete RM;
	
	/*  NO RESULT MANAGER

	int numFramesToTrack = NumJpgInDir(imagePath + "*");
	for(int ii = 0; ii < numFramesToTrack; ii++){
		std::string file = SPrintf("%s\img%05d.jpg",imagePath.c_str(),ii);
		ImageData img = ReadJPEGFile(file.c_str());

		LocalizationJob job(ii, 0, 0, 0);
		int queued = qtrk->ScheduleFrame(img.data,sizeof(float)*img.w,img.w,img.h,positions,beads.size(),QTrkFloat,&job);
		output->outputString(SPrintf("Queueing frame %d. Current Queue size: %d. Added: %d.",ii,qtrk->GetQueueLength(),queued),true);
		img.free();
		
		while(qtrk->GetResultCount() != 0) {
			LocalizationResult lr;
			qtrk->FetchResults(&lr,1);
			output->outputString(SPrintf("frame %d, bead %d: %f %f %f",lr.job.frame,lr.job.zlutIndex,lr.pos.x,lr.pos.y,lr.pos.z));
			//delete &(lr.job);
		}
	}	

	while(qtrk->GetQueueLength() != 0) {
		if(qtrk->GetResultCount() != 0) {
			LocalizationResult lr;
			qtrk->FetchResults(&lr,1);
			output->outputString(SPrintf("frame %d, bead %d: %f %f %f",lr.job.frame,lr.job.zlutIndex,lr.pos.x,lr.pos.y,lr.pos.z));
		}
	}
	*/
	/*ImageData allLuts = ImageData::alloc(res,zplanes*beads.size());
	memcpy(allLuts.data,zluts,res*zplanes*beads.size()*sizeof(float));
	output->outputImage(allLuts,"allLuts");
	allLuts.free();//*/

	qtrk->ClearResults();
	delete qtrk;
	delete positions;
	delete[] zluts;	
}

enum Tests{
	ROIDis,
	Inter,
	Skew,
	Backg
};

void RunTest(Tests test, const char* image, outputter* output, int ROISize)
{
	ImageData source = ReadJPEGFile(image);
	std::vector<BeadPos> beads = read_beadlist("D:\\TestImages\\beadlist.txt");
	output->newFile("TestInfo","a");

	if(test == ROIDis)
		output->outputString("ROI Displacement test");
	else if(test == Inter)
		output->outputString("Interference test");
	else if(test == Skew)
		output->outputString("Skew test");
	else if(test == Backg)
		output->outputString("Background test");
	output->outputString(SPrintf("Image %s\nBeadlist D:\\TestImages\\beadlist.txt\nNumBeads %d\nROISize %d",image,beads.size(),ROISize));
	
	double t0 = GetPreciseTime();

	switch(test){
	case ROIDis:
		TestROIDisplacement(beads,source,output,ROISize);
		break;
	case Inter:
		TestInterference(beads,source,output,ROISize);
		break;
	case Skew:
		TestSkew(beads,source,output,ROISize);
		break;
	case Backg:
		TestBackground(beads,source,output,ROISize);
		break;
	};

	double t1 = GetPreciseTime();
	output->newFile("TestInfo","a");
	output->outputString(SPrintf("Duration %f\n",t1-t0));
	source.free();
}

void ManTest()
{
	int ROISize = 101;
	float displacement = 0;
	float skewFact = 0.0f;
	float bgCorr = 2;
	int imgSel = 0;

	char inChar;
	do
	{
		char selChar;
		do{
			printf_s("Current settings: image %s\nROI %d, skewFact %f, displacement %f, bgCorr %f\n\n", (imgSel == 0)? "generated":"CroppedBead.jpg", ROISize, skewFact, displacement, bgCorr);
			std::cout << "What setting to change? (? for list, 0 to run)\n";
			std::cin >> selChar;
			switch(selChar){
			case '?':
				printf("i: imgSel (0: Generated, 1: CroppedBead.jpg)\nr: ROISize\nb: COM Background correction\n");
				printf("\nOnly for generated image:\n\ts: skewFact\n\td: displacement\n");
				break;
			case 'i':
				std::cin >> imgSel;
				break;
			case 'r':
				std::cin >> ROISize;
				break;
			case 's':
				std::cin >> skewFact;
				break;
			case 'd':
				std::cin >> displacement;
				break;
			case 'b':
				std::cin >> bgCorr;
				break;
			default:
				break;
			}
			std::cout << std::endl;
		} while(selChar != '0');
		ImageData img;
		if(imgSel == 1){
			ImageData imgRaw = ReadJPEGFile("D:\\TestImages\\CroppedBead.jpg");
			ROISize = imgRaw.w;
			img = SkewImage(imgRaw,skewFact);
			img.normalize();
			imgRaw.free();
		} else {
			ImageData imgRaw = ImageData::alloc(ROISize,ROISize);
			GenerateTestImage(imgRaw,(float)ROISize/2+displacement,(float)ROISize/2+displacement,5,0);
			img = SkewImage(imgRaw,skewFact);
			img.normalize();
			imgRaw.free();
		}
		FloatToJPEGFile("D:\\TestImages\\test.jpg",img.data,ROISize,ROISize);

		QTrkComputedConfig cfg;
		cfg.width = cfg.height = ROISize;
		cfg.qi_angstep_factor = 1;
		cfg.qi_iterations = 10;
		cfg.qi_angular_coverage = 0.7f;
		cfg.qi_roi_coverage = 1.0f;
		cfg.qi_radial_coverage = 1.5f;
		cfg.qi_minradius = 0;
		cfg.zlut_minradius = 2;
		cfg.zlut_radial_coverage = 2;
		cfg.zlut_angular_coverage = 0.7f;
		cfg.zlut_roi_coverage = 1;
		cfg.com_bgcorrection = bgCorr;
		cfg.xc1_profileLength = ROISize*0.8f;
		cfg.xc1_profileWidth = ROISize*0.2f;
		cfg.xc1_iterations = 4;	
		cfg.testRun = true;
	
		cfg.Update();

		QueuedCPUTracker* qtrk = new QueuedCPUTracker(cfg);
		qtrk->SetLocalizationMode(LT_QI | LT_LocalizeZ | LT_NormalizeProfile);
		LocalizationJob job(0, 0, 0, 0);
		ROIPosition pos;
		pos.x = 0;
		pos.y = 0;
		int queued = qtrk->ScheduleFrame(img.data,sizeof(float)*img.w,img.w,img.h,&pos,1,QTrkFloat,&job);
		printf("q: %d\n",queued);
		while(qtrk->GetQueueLength() != 0);
		while(qtrk->GetResultCount() != 0) {
			LocalizationResult lr;
			qtrk->FetchResults(&lr,1);
			
			printf("%f %f %f %f\n",lr.firstGuess.x,lr.firstGuess.y,lr.pos.x,lr.pos.y);
			
			float* prof = ALLOCA_ARRAY(float,cfg.zlut_radialsteps);
			bool boundaryHit;
			ImageData imgData (img.data,ROISize,ROISize);
			ComputeRadialProfile(prof,cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, lr.pos.xy(), &imgData, lr.imageMean, false);
			WriteArrayAsCSVRow("D:\\TestImages\\RadialProfile.txt",prof,cfg.zlut_radialsteps,false);
			ComputeRadialProfile(prof,cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, lr.pos.xy(), &imgData, lr.imageMean, true);
			WriteArrayAsCSVRow("D:\\TestImages\\RadialProfile.txt",prof,cfg.zlut_radialsteps,true);
		}

		img.free();
		qtrk->Flush();
		delete qtrk;
		std::cout << "\nContinue?\n";
		std::cin >> inChar;
	} while(inChar != '0');
}

void PrintMenu(outputter* output)
{
	output->outputString("0. Quit",true);
	output->outputString("1. ROI Displacement",true);
	output->outputString("2. Interference",true);
	output->outputString("3. Skew",true);
	output->outputString("4. Background",true);
	output->outputString("5. Build ZLUTs",true);
	output->outputString("6. Run Trace",true);
	output->outputString("m. Manual",true);
	output->outputString("r. Change ROI size",true);
	output->outputString("?. Menu",true);
}

void SelectTests(const char* image, int OutputMode)
{
	outputter* output = new outputter(OutputMode);
	int testNum = 1;
	int ROISize = 120;
	
	PrintMenu(output);
	char inChar;
	do
	{
		output->outputString("Select test or ? for menu:",true);
		std::string imagesPath, LUTPath;
		std::cin >> inChar;
		switch(inChar)
		{
			case '0':
				break;
			case '1':
				RunTest(ROIDis,image,output,ROISize);
				output->outputString("Test done!",true);
				break;
			case '2':
				RunTest(Inter,image,output,ROISize);
				output->outputString("Test done!",true);
				break;
			case '3':
				RunTest(Skew,image,output,ROISize);
				output->outputString("Test done!",true);
				break;
			case '4':
				RunTest(Backg,image,output,ROISize);
				output->outputString("Test done!",true);
				break;
			case '5':
				//BuildZLUT("L:\\BN\\ND\\Shared\\Jordi\\TestMovie150507_2\\images\\jpg\\Zstack\\",output);
				BuildZLUT("D:\\TestImages\\TestMovie150507_2\\images\\jpg\\Zstack\\",output);
				output->outputString("ZLUTs Built",true);
				break;
			case '6':
				/*while(!(DirExists(imagesPath) && NumJpgInDir(imagesPath) > 0)){
					output->outputString("Enter image folder (with beadlist):",true);
					std::cin >> imagesPath;
					output->outputString(SPrintf("%d jpgs in folder",NumJpgInDir(imagesPath)),true);
				}
				while(!(DirExists(LUTPath) && NumJpgInDir(LUTPath) > 0)){
					output->outputString("Enter LUT folder:",true);
					std::cin >> LUTPath;
					output->outputString(SPrintf("%d jpgs in folder",NumJpgInDir(LUTPath)),true);
				}
				RunZTrace(imagesPath,LUTPath,output);*/
				//RunZTrace("L:\\BN\\ND\\Shared\\Jordi\\20150804_TestMovie\\images\\jordi_test\\jpg\\","L:\\BN\\ND\\Shared\\Jordi\\20150804_TestMovie\\images\\lut\\",output);
				//RunZTrace("D:\\TestImages\\20150804_TestMovie\\images\\jordi_test\\jpg\\","D:\\TestImages\\20150804_TestMovie\\images\\lut\\",output);
				RunZTrace("C:\\TestImages\\TestMovie150507_2\\images\\jpg\\Zstack\\","C:\\TestImages\\TestMovie150507_2\\ZLUTS_50planes\\",output);
				break;
			case 'm':
				ManTest();
				break;
			case 'R':
			case 'r':
				output->outputString(SPrintf("Enter new ROI size (currently %d)",ROISize),true);
				std::cin >> ROISize;
				break;
			default:
				output->outputString("Wrong input",true);
			case '?':
				PrintMenu(output);
				break;
		}
	} while(inChar != '0');
	
	delete output;
}

int main()
{
#ifdef _DEBUG
//	Matrix3X3::test();
#endif
	SelectTests("D:\\TestImages\\img00095.jpg", Files+Images);
//	ManTest();

//	SimpleTest();

//	GenerateZLUTFittingCurve("lut000.jpg");

	/*TestBias();
	TestZRangeBias("ref169-norm", "zrange\\exp_qi.radialzlut#169", true);
	TestZRangeBias("ref169-raw", "zrange\\exp_qi.radialzlut#169", false);

//	SmallROITest("lut000.jpg");

	//TestZRange("lut227-ref","lut227.jpg", 0, 0, RWStetson);
	TestZRange("zrange\\lut169ref-biasmap-c","zrange\\exp_qi.radialzlut#169", 0, 0, RWStetson, true, true);
	TestZRange("zrange\\lut169ref-biasmap","zrange\\exp_qi.radialzlut#169", 0, 0, RWStetson, true, false);
	TestZRange("zrange\\lut169ref","zrange\\exp_qi.radialzlut#169", 0, 0, RWStetson, false, false);
	TestZRange("zrange\\lut169ref-c","zrange\\exp_qi.radialzlut#169", 0, 0, RWStetson, false, true);
	TestZRange("zrange\\lut013tether","zrange\\exp_qi.radialzlut#13", 0, 0, RWStetson, false, false);
	TestZRange("zrange\\lut013tether-c","zrange\\exp_qi.radialzlut#13", 0, 0, RWStetson, false, true);*/
/*
	TestZRange("zrange\\longlut1-c","zrange\\long.radialzlut#1", 0,0, RWStetson, false, true);
	TestZRange("zrange\\longlut1","zrange\\long.radialzlut#1", 0,0, RWStetson, false, false);
	TestZRange("zrange\\longlut3-c","zrange\\long.radialzlut#3", 0,0, RWStetson, false, true);
	TestZRange("zrange\\longlut3","zrange\\long.radialzlut#3", 0,0, RWStetson, false, false);
*/
	//TestZRange("cleanlut1", "lut000.jpg", LT_LocalizeZWeighted, 0);
	//TestZRange("cleanlut1", "lut000.jpg", LT_LocalizeZWeighted, 1);
	//TestZRange("cleanlut10", "lut10.jpg", LT_LocalizeZWeighted, 1);
	//TestZRange("cleanlut10", "lut10.jpg", LT_LocalizeZWeighted, 1);
	
//	BenchmarkParams();
//	int N=50;
	/*ScatterBiasArea(80, 4, 100, N, 3, 1);
	ScatterBiasArea(80, 4, 100, N, 4, 1);
	ScatterBiasArea(80, 4, 100, N, 1, 1);
	ScatterBiasArea(80, 4, 100, N, 2, 1);
	ScatterBiasArea(80, 4, 100, N, 0, 1);
	ScatterBiasArea(80, 4, 100, N, -1, 1);
	*/
/*
	ImageData img=ReadLUTFile("lut000.jpg");
	img.mean();

	TestZRange("rbin1x", "1x.radialzlut#4", 0, 0, RWUniform);
	TestZRange("rbin1x", "1x.radialzlut#4", 0, 0, RWRadial);
	TestZRange("rbin1x", "1x.radialzlut#4", 0, 0, RWDerivative);

	TestZRange("rbin10x", "10x.radialzlut#4", 0, 0, RWUniform);
	TestZRange("rbin10x", "10x.radialzlut#4", 0, 0, RWRadial);
	TestZRange("rbin10x", "10x.radialzlut#4", 0, 0, RWDerivative);
	*/
//	QTrkTest();
//	TestCMOSNoiseInfluence<QueuedCPUTracker>("lut000.jpg");

	//AutoBeadFindTest();
//	Gauss2DTest<QueuedCPUTracker>();
	
	//SpeedTest();
	//PixelationErrorTest();
	//ZTrackingTest();
	//Test2DTracking();
	//TestBoundCheck();
	//QTrkTest();
	//for (int i=1;i<8;i++)
	//	BuildConvergenceMap(i);

	//TestFourierLUT();
//	TestQuadrantAlign();
	//TestZLUTAlign();
	//TestImageLUT();
//	TestBuildRadialZLUT<QueuedCPUTracker>( "lut000.jpg" );
	//TestImageLUT();

	//CorrectedRadialProfileTest();
	
	//system("pause");
	return 0;
}
