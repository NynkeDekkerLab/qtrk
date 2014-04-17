#include "../cputrack/std_incl.h"
#include "../cputrack/cpu_tracker.h"
#include "../cputrack/random_distr.h"
#include "../cputrack/QueuedCPUTracker.h"
#include "../cputrack/FisherMatrix.h"
#include "../cputrack/BeadFinder.h"
#include "../cputrack/LsqQuadraticFit.h"
#include "../utils/ExtractBeadImages.h"
#include "../cputrack/BenchmarkLUT.h"
#include <time.h>



#include "SharedTests.h"

template<typename T> T sq(T x) { return x*x; }
template<typename T> T distance(T x, T y) { return sqrt(x*x+y*y); }

float distance(vector2f a,vector2f b) { return distance(a.x-b.x,a.y-b.y); }

const float ANGSTEPF = 1.5f;

const bool InDebugMode = 
#ifdef _DEBUG
	true
#else
	false
#endif
	;

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
	tracker->SetRadialZLUT(zlut, zplanes, radialSteps, 1,1, zradius, 64, true, true);
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

void TestFisherFast(const char *lutfile)
{
	ImageData lut = ReadJPEGFile(lutfile);

	float z = 60.0f;
	vector3f delta(0.001f,0.001f, 0.001f);

	float zlutMin=0;
	float zlutMax=40;
	QTrkComputedConfig settings;
	settings.zlut_minradius = zlutMin;
	settings.zlut_maxradius = zlutMax;
	settings.width = settings.height = 80;
	settings.Update();
	
	float maxVal=10000;

	vector3f smppos(settings.width/2,settings.height/2, z);
	std::vector<float> stdv;
	dbgprintf("High-res LUT range...\n");
	SampleFisherMatrix fm( maxVal);

	QueuedCPUTracker trk(settings);
	ImageData rescaledLUT;
	ResampleLUT(&trk, &lut, 150, &rescaledLUT);
	trk.SetLocalizationMode(LT_QI|LT_LocalizeZ);

	int nstep= InDebugMode ? 50 : 1000;
	std::vector<vector3f> positions;
	std::vector<float> stdevz, imgsmp;
	for (int i=0;i<nstep;i++)
	{
		float z = 1 + i / (float)nstep * (rescaledLUT.h-2);
		vector3f pos = vector3f( smppos.x, smppos.y, z);
		positions.push_back(pos);
		Matrix3X3 invFisherLUT = fm.Compute(pos, delta, rescaledLUT, settings.width, settings.height, zlutMin, zlutMax).Inverse();
		vector3f stdev = sqrt(invFisherLUT.diag());
		stdevz.push_back(stdev.z);

		ImageData img=ImageData::alloc(settings.width,settings.height);
		GenerateImageFromLUT(&img, &rescaledLUT, zlutMin, zlutMax, pos);
		LocalizationJob job(i, 0, 0, 0);
		trk.ScheduleImageData(&img, &job);
		imgsmp.push_back(img.at(img.w/4, img.h/2)); // 
		img.free();
//		dbgprintf("[%d] z=%f Min std deviation: X=%f nm, Y=%f nm, Z=%f nm.\n", i, z, stdev.x,stdev.y,stdev.z);
	}
	WaitForFinish(&trk, nstep);
	std::vector<float> output;
	std::vector<float> errz(nstep);
	std::vector<float> zresult(nstep);
	for (int i=0;i<nstep;i++) {
		LocalizationResult lr;
		trk.FetchResults(&lr, 1);
		zresult[lr.job.frame]=lr.pos.z;
		errz[lr.job.frame] = zresult[lr.job.frame] - positions[lr.job.frame].z;
	} 

	for(int i=0;i<nstep;i++) {
		dbgprintf("errz[%d]:%f.  z=%f, true=%f\n", i,  errz[i], zresult[i], positions[i].z);
		output.push_back(positions[i].z);
		output.push_back(errz[i]);
		output.push_back(stdevz[i]);
		output.push_back(imgsmp[i]);
	}
	WriteImageAsCSV("lutrange-zerr.txt", &output[0], 4, output.size()/4);
	lut.free();
	rescaledLUT.free();
}

void TestFisher(const char *lutfile)
{
	ImageData lut = ReadJPEGFile(lutfile);

	float z = 60.0f;
	float zlutMin=0;
	float zlutMax=40;
	vector3f delta(0.001f,0.001f, 0.001f);

	QTrkComputedConfig settings;
	settings.zlut_minradius = zlutMin; 
	settings.zlut_maxradius = zlutMax;
	settings.width = settings.height = 80;
	settings.Update();
	float maxVal=10000;

	vector3f smppos(settings.width/2,settings.height/2, z);
	std::vector<float> stdv;
	dbgprintf("High-res LUT range...\n");
	SampleFisherMatrix fm( maxVal);

	ImageData rescaledLUT;

	int nstep=2000;
	for (int i=0;i<nstep;i++) {
		float z = 1 + i / (float)nstep * (rescaledLUT.h-2);
		vector3f pos = vector3f( smppos.x, smppos.y, z);
		Matrix3X3 invFisherLUT = fm.Compute(pos, delta, rescaledLUT, settings.width, settings.height, zlutMin, zlutMax).Inverse();

		ImageData img=ImageData::alloc(settings.width,settings.height);
		GenerateImageFromLUT(&img, &rescaledLUT, zlutMin, zlutMax, pos);

		vector3f stdev = sqrt(invFisherLUT.diag());

		stdv.push_back(stdev.x);
		stdv.push_back(stdev.z);
		stdv.push_back(img.at(img.w/4, img.h/2)); // 

		img.free();

		dbgprintf("[%d] z=%f Min std deviation: X=%f nm, Y=%f nm, Z=%f nm.\n", i, z, stdev.x,stdev.y,stdev.z);

		RunTrackerResults trkresults = RunTracker<QueuedCPUTracker> (lutfile, &settings, false, "fishercmp", LT_QI | LT_LocalizeZ, InDebugMode ? 10 : 2000, maxVal / 255, pos.z, &rescaledLUT);
		trkresults.computeStats();
		stdv.push_back(trkresults.stdev.x);
		stdv.push_back(trkresults.stdev.z);
		dbgprintf("[%d] tracker: X=%f nm, Y=%f nm, Z=%f nm.\n", i, trkresults.stdev.x,trkresults.stdev.y,trkresults.stdev.z);

	}
	WriteImageAsCSV("fisher_lx_lz_smpL_trkx_trkz.txt", &stdv[0], 5, stdv.size()/5);
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

	for (int i=0;i<results.size();i++) {
		dbgprintf("beadpos: x=%d, y=%d\n", results[i].x, results[i].y);
		img.at(results[i].x+cfg.roi/2, results[i].y+cfg.roi/2) = 1.0f;
	}
	dbgprintf("%d beads total\n", results.size());

	FloatToJPEGFile("autobeadfind.jpg", img.data, img.w, img.h);

	img.free();
	smp.free();
}


void TestImageLUT()
{
	QTrkSettings cfg;
	cfg.width=cfg.height=100;

	QueuedCPUTracker trk(cfg);

	// [ count, planes, height, width ] 
	int nplanes=10;
	int dims[] = { 1, nplanes, cfg.height*0.7f,cfg.width*0.7f };
	trk.SetImageZLUT(0, 0, dims);

	ImageData img=ImageData::alloc(cfg.width,cfg.height);
	ImageData lut = ReadJPEGFile("refbeadlut.jpg");

	for (int i=0;i<nplanes;i++) {
		GenerateImageFromLUT(&img, &lut, trk.cfg.zlut_minradius, trk.cfg.zlut_maxradius, vector3f(img.w/2,img.h/2, i));
		trk.BuildLUT(img.data, img.pitch(), QTrkFloat, true, i);
	}
	trk.FinalizeLUT();

	float *ilut = new float [dims[0]*dims[1]*dims[2]*dims[3]];
	trk.GetImageZLUT(ilut);
	ImageData ilutImg (ilut, dims[3], dims[1]*dims[2]);
	WriteJPEGFile("ilut.jpg", ilutImg);
	delete[] ilut;

	int nsmp = 10;
	for (int i=0;i<nsmp;i++) {
		vector3f pos(img.w/2+rand_uniform<float>()-0.5f ,img.h/2-rand_uniform<float>()-0.5f,nplanes/2.0f);
		GenerateImageFromLUT(&img, &lut, trk.cfg.zlut_minradius, trk.cfg.zlut_maxradius, vector3f(pos.x,pos.y,pos.z));
		ApplyPoissonNoise(img, 28 * 255, 255);
		CPUTracker ct(cfg.width,cfg.height);
		bool bhit;
		ct.SetImageFloat(img.data);
		vector2f qipos = ct.ComputeQI(ct.ComputeMeanAndCOM(), 2, trk.cfg.qi_radialsteps, trk.cfg.qi_angstepspq, trk.cfg.qi_angstep_factor, trk.cfg.qi_minradius, trk.cfg.qi_maxradius, bhit);

		dbgprintf("QIPos: %f,%f;\t", qipos.x,qipos.y);
		dbgprintf("QI Error: %f,%f\n", qipos.x-pos.x,qipos.y-pos.y);
	}

	lut.free();
	img.free();
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
	auto resultsQI = RunTracker<QueuedCPUTracker> ("lut000.jpg", &cfg, false, "qi",				locModeQI, 200, NF, zpos);

	resultsZA.computeStats(); 
	resultsQI.computeStats();

	dbgprintf("FourierLUT: X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsZA.mean.x, resultsZA.stdev.x, resultsZA.mean.z, resultsZA.stdev.z);
	dbgprintf("Only QI:   X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsQI.mean.x, resultsQI.stdev.x, resultsQI.mean.z, resultsQI.stdev.z);
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

	dbgprintf("ZLUTAlign: X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsZA.mean.x, resultsZA.stdev.x, resultsZA.mean.z, resultsZA.stdev.z);
	dbgprintf("Only QI:   X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsQI.mean.x, resultsQI.stdev.x, resultsQI.mean.z, resultsQI.stdev.z);
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

	dbgprintf("QuadrantAlign: X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsZA.mean.x, resultsZA.stdev.x, resultsZA.mean.z, resultsZA.stdev.z);
	dbgprintf("Only QI:   X= %f. stdev: %f\tZ=%f,  stdev: %f\n", resultsQI.mean.x, resultsQI.stdev.x, resultsQI.mean.z, resultsQI.stdev.z);
}


int main()
{
	/*
	const int N=5;
	float x[N],y[N],w[N]={0.1f, 0.5f, 1.0f, 0.4f, 0.1f };
	for (int i=0;i<N;i++) { x[i]=i-2; y[i]=x[i]*x[i]-2*x[i]+1.0f; }
	LsqSqQuadFit<float> fit(N, x, y, w);
	dbgprintf("f(1)(fit) = %f.  f(1)=%f\n", fit.compute(1.0f), sq(1)-2*1+1);
	LsqSqQuadFit<float> fit2(N, x, y, w);
	dbgprintf("f(1) = %f\n", fit2.compute(1.0f));
	*/

#ifdef _DEBUG
	Matrix3X3::test();
#endif

	TestFisher("lut000.jpg");

	//QTrkTest();
//	TestCMOSNoiseInfluence<QueuedCPUTracker>("lut000.jpg");

	//AutoBeadFindTest();
//	Gauss2DTest<QueuedCPUTracker>();
	
	//SpeedTest();
	//SmallImageTest();
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
	return 0;
}
