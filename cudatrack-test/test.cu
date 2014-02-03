
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "std_incl.h"
#include "utils.h"

#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>
#include <valarray>

#include "random_distr.h"

#include <stdint.h>
#include "gpu_utils.h"
#include "QueuedCUDATracker.h"
#include "QueuedCPUTracker.h"

#include "../cputrack-test/SharedTests.h"
#include "BenchmarkLUT.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

void BenchmarkParams();

std::string getPath(const char *file)
{
	std::string s = file;
	int pos = s.length()-1;
	while (pos>0 && s[pos]!='\\' && s[pos]!= '/' )
		pos--;
	
	return s.substr(0, pos);
}


inline __device__ float2 mul_conjugate(float2 a, float2 b)
{
	float2 r;
	r.x = a.x*b.x + a.y*b.y;
	r.y = a.y*b.x - a.x*b.y;
	return r;
}


void ShowCUDAError() {
	cudaError_t err = cudaGetLastError();
	dbgprintf("Cuda error: %s\n", cudaGetErrorString(err));
}

__shared__ float cudaSharedMem[];

__device__ float compute(int idx, float* buf, int s)
{
	// some random calcs to make the kernel unempty
	float k=0.0f;
	for (int x=0;x<s;x++ ){
		k+=cosf(x*0.1f*idx);
		buf[x]=k;
	}
	for (int x=0;x<s/2;x++){
		buf[x]=buf[x]*buf[x];
	}
	float sum=0.0f;
	for (int x=s-1;x>=1;x--) {
		sum += buf[x-1]/(fabsf(buf[x])+0.1f);
	}
	return sum;
}


__global__ void testWithGlobal(int n, int s, float* result, float* buf) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		result [idx] = compute(idx, &buf [idx * s],s);
	}
}

__global__ void testWithShared(int n, int s, float* result) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		result [idx] = compute(idx, &cudaSharedMem[threadIdx.x * s],s);
	}
}

void TestSharedMem()
{
	int n=100, s=200;
	dim3 nthreads(32), nblocks( (n+nthreads.x-1)/nthreads.x);
	device_vec<float> buf(n*s);
	device_vec<float> result_s(n), result_g(n);

	double t0 = GetPreciseTime();
	testWithGlobal<<<nblocks,nthreads>>>(n,s,result_g.data,buf.data);
	cudaDeviceSynchronize();
	double t1 = GetPreciseTime();
	testWithShared <<<nblocks,nthreads,s*sizeof(float)*nthreads.x>>>(n,s,result_s.data);
	cudaDeviceSynchronize();
	double t2 = GetPreciseTime();

	std::vector<float> rs = result_s, rg = result_g;
	for (int x=0;x<n;x++) {
		dbgprintf("result_s[%d]=%f.   result_g[%d]=%f\n", x,rs[x], x,rg[x]);
	}

	dbgprintf("Speed of shared comp: %f, speed of global comp: %f\n", n/(t2-t1), n/(t1-t0));
}

void QTrkCompareTest()
{
	QTrkSettings cfg;
	cfg.width = cfg.height = 40;
	cfg.qi_iterations = 1;
	cfg.xc1_iterations = 2;
	cfg.xc1_profileLength = 64;
	cfg.numThreads = -1;
	cfg.com_bgcorrection = 0.0f;
	bool haveZLUT = false;
#ifdef _DEBUG
	cfg.numThreads = 2;
	cfg.qi_iterations=1;
	int total= 10;
	int batchSize = 2;
	haveZLUT=false;
#else
	cfg.numThreads = 4;
	int total = 10000;
	int batchSize = 512;
#endif

	QueuedCUDATracker qtrk(cfg, batchSize);
	QueuedCPUTracker qtrkcpu(cfg);
	ImageData img = ImageData::alloc(cfg.width,cfg.height);
	bool cpucmp = true;

	qtrk.EnableTextureCache(true);

	srand(1);

	// Generate ZLUT
	int zplanes=100;
	float zmin=0.5,zmax=3;
	qtrk.SetRadialZLUT(0, 1, zplanes);
	if (cpucmp) qtrkcpu.SetRadialZLUT(0, 1, zplanes);
	if (haveZLUT) {
		qtrk.SetLocalizationMode(LT_BuildRadialZLUT|LT_OnlyCOM);
		qtrkcpu.SetLocalizationMode(LT_BuildRadialZLUT|LT_OnlyCOM);
		for (int x=0;x<zplanes;x++)  {
			vector2f center ( cfg.width/2, cfg.height/2 );
			float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
			GenerateTestImage(img, center.x, center.y, s, 0.0f);
			WriteJPEGFile("qtrkzlutimg.jpg", img);

			LocalizationJob jobInfo;
			jobInfo.frame = jobInfo.zlutPlane = x;
			jobInfo.zlutIndex = 0;
			qtrk.ScheduleLocalization((uchar*)img.data, cfg.width*sizeof(float),QTrkFloat, &jobInfo);
			if (cpucmp) qtrkcpu.ScheduleLocalization((uchar*)img.data, cfg.width*sizeof(float),QTrkFloat, &jobInfo);
		}
		qtrk.Flush();
		if (cpucmp) qtrkcpu.Flush();
		// wait to finish ZLUT
		while(true) {
			int rc = qtrk.GetResultCount();
			if (rc == zplanes) break;
			Sleep(100);
			dbgprintf(".");
		}
		if (cpucmp) {
			while(qtrkcpu.GetResultCount() != zplanes);
		}
	}
	float* zlut = new float[qtrk.cfg.zlut_radialsteps*zplanes];
	qtrk.GetRadialZLUT(zlut);
	if (cpucmp) { 
		float* zlutcpu = new float[qtrkcpu.cfg.zlut_radialsteps*zplanes];
		qtrkcpu.GetRadialZLUT(zlutcpu);
		
		WriteImageAsCSV("zlut-cpu.txt", zlutcpu, qtrkcpu.cfg.zlut_radialsteps, zplanes);
		WriteImageAsCSV("zlut-gpu.txt", zlut, qtrkcpu.cfg.zlut_radialsteps, zplanes);
		delete[] zlutcpu;
	}
	qtrk.ClearResults();
	if (cpucmp) qtrkcpu.ClearResults();
	FloatToJPEGFile ("qtrkzlutcuda.jpg", zlut, qtrk.cfg.zlut_radialsteps, zplanes);
	delete[] zlut;
	
	// Schedule images to localize on
	dbgprintf("Benchmarking...\n", total);
	GenerateTestImage(img, cfg.width/2, cfg.height/2, (zmin+zmax)/2, 0);
	double tstart = GetPreciseTime();
	int rc = 0, displayrc=0;
	LocMode_t flags = (LocMode_t)(LT_NormalizeProfile |LT_QI| (haveZLUT ? LT_LocalizeZ : 0) );
	qtrk.SetLocalizationMode(flags);
	qtrkcpu.SetLocalizationMode(flags);
	for (int n=0;n<total;n++) {
		LocalizationJob jobInfo;
		jobInfo.frame = n;
		jobInfo.zlutIndex = 0;
		qtrk.ScheduleLocalization((uchar*)img.data, cfg.width*sizeof(float), QTrkFloat,&jobInfo);
		if (cpucmp) qtrkcpu.ScheduleLocalization((uchar*)img.data, cfg.width*sizeof(float), QTrkFloat, &jobInfo);
		if (n % 10 == 0) {
			rc = qtrk.GetResultCount();
			while (displayrc<rc) {
				if( displayrc%(total/10)==0) dbgprintf("Done: %d / %d\n", displayrc, total);
				displayrc++;
			}
		}
	}
	if (cpucmp) qtrkcpu.Flush();
	WaitForFinish(&qtrk, total);
	
	// Measure speed
	double tend = GetPreciseTime();

	if (cpucmp) {
		dbgprintf("waiting for cpu results..\n");
		while (total != qtrkcpu.GetResultCount())
			Sleep(10);
	}
	

	img.free();

	const int NumResults = 20;
	LocalizationResult results[NumResults], resultscpu[NumResults];
	int rcount = std::min(NumResults,total);
	for (int i=0;i<rcount;i++) {
		qtrk.FetchResults(&results[i], 1);
		if (cpucmp) qtrkcpu.FetchResults(&resultscpu[i], 1);
	}

	// if you wonder about this syntax, google C++ lambda functions
	std::sort(results, results+rcount, [](LocalizationResult a, LocalizationResult b) -> bool { return a.job.frame > b.job.frame; });
	if(cpucmp) std::sort(resultscpu, resultscpu+rcount, [](LocalizationResult a, LocalizationResult b) -> bool { return a.job.frame > b.job.frame; });
	for (int i=0;i<rcount;i++) {
		LocalizationResult& r = results[i];
		dbgprintf("gpu [%d] x: %f, y: %f. z: %+g, COM: %f, %f\n", i,r.pos.x, r.pos.y, r.pos.z, r.firstGuess.x, r.firstGuess.y);

		if (cpucmp) {
			r = resultscpu[i];
			dbgprintf("cpu [%d] x: %f, y: %f. z: %+g, COM: %f, %f\n", i,r.pos.x, r.pos.y, r.pos.z, r.firstGuess.x, r.firstGuess.y);
		}
	}

	dbgprintf("Localization Speed: %d (img/s)\n", (int)( total/(tend-tstart) ));
}

void listDevices()
{
	cudaDeviceProp prop;
	int dc;
	cudaGetDeviceCount(&dc);
	for (int k=0;k<dc;k++) {
		cudaGetDeviceProperties(&prop, k);
		dbgprintf("Device[%d] = %s\n", k, prop.name);
		dbgprintf("\tMax texture width: %d\n" ,prop.maxTexture2D[0]);
	}

}

__global__ void SimpleKernel(int N, float* a){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		for (int x=0;x<1000;x++)
			a[idx] = asin(a[idx]+x);
	}
}


void TestAsync()
{
	int N =100000;
	int nt = 32;

	pinned_array<float> a(N); 
//	cudaMallocHost(&a, sizeof(float)*N, 0);

	device_vec<float> A(N);

	cudaStream_t s0;
	cudaEvent_t done;

	cudaStreamCreate(&s0);
	cudaEventCreate(&done,0);

	for (int x=0;x<N;x++)
		a[x] = cos(x*0.01f);

	for (int x=0;x<1;x++) {
		{ MeasureTime mt("a->A"); A.copyToDevice(a.data(), N, true); }
		{ MeasureTime mt("func(A)"); 
		SimpleKernel<<<dim3( (N+nt-1)/nt ), dim3(nt)>>>(N, A.data);
		}
		{ MeasureTime mt("A->a"); A.copyToHost(a.data(), true); }
	}
	cudaEventRecord(done);

	{
	MeasureTime("sync..."); while (cudaEventQuery(done) != cudaSuccess); 
	}
	
	cudaStreamDestroy(s0);
	cudaEventDestroy(done);
}

__global__ void emptyKernel()
{}

float SpeedTest(const QTrkSettings& cfg, QueuedTracker* qtrk, int count, bool haveZLUT, LocMode_t locType, float* scheduleTime, bool gaincorrection=false)
{
	ImageData img=ImageData::alloc(cfg.width,cfg.height);
	srand(1);

	// Generate ZLUT
	int zplanes=100;
	float zmin=0.5,zmax=3;
	qtrk->SetRadialZLUT(0, 1, zplanes);
	if (gaincorrection) EnableGainCorrection(qtrk);
	if (haveZLUT) {
		qtrk->SetLocalizationMode(LT_BuildRadialZLUT|LT_OnlyCOM|LT_NormalizeProfile);

		for (int x=0;x<zplanes;x++)  {
			vector2f center( cfg.width/2, cfg.height/2 );
			float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
			GenerateTestImage(img, center.x, center.y, s, 0.0f);
			LocalizationJob job;
			job.frame = 0;
			job.zlutPlane = job.frame = x;
			qtrk->ScheduleLocalization((uchar*)img.data, cfg.width*sizeof(float),QTrkFloat, &job);
		}
		qtrk->Flush();
		// wait to finish ZLUT
		while(true) {
			int rc = qtrk->GetResultCount();
			if (rc == zplanes) break;
			Sleep(100);
			dbgprintf(".");
		}
	}
	qtrk->ClearResults();
	
	// Schedule images to localize on
	dbgprintf("Benchmarking...\n", count);
	GenerateTestImage(img, cfg.width/2, cfg.height/2, (zmin+zmax)/2, 0);
	double tstart = GetPreciseTime();
	int rc = 0, displayrc=0;
	double maxScheduleTime = 0.0f;
	double sumScheduleTime2 = 0.0f;
	double sumScheduleTime = 0.0f;
	qtrk->SetLocalizationMode(locType| (haveZLUT ? LT_LocalizeZ : 0));
	for (int n=0;n<count;n++) {
		double t0 = GetPreciseTime();
		///qtrk->ScheduleLocalization((uchar*)image, cfg.width*sizeof(float), QTrkFloat, flags, n, 0, 0, 0, 0);
		ROIPosition roipos[]={ {0,0} };
		LocalizationJob job(n, 0, 0,0);
		qtrk->ScheduleFrame((uchar*)img.data, cfg.width*sizeof(float),cfg.width,cfg.height, roipos, 1, QTrkFloat, &job);
		double dt = GetPreciseTime() - t0;
		maxScheduleTime = std::max(maxScheduleTime, dt);
		sumScheduleTime += dt;
		sumScheduleTime2 += dt*dt;

		if (n % 10 == 0) {
			rc = qtrk->GetResultCount();
			while (displayrc<rc) {
				if( displayrc%(count/10)==0) dbgprintf("Done: %d / %d\n", displayrc, count);
				displayrc++;
			}
		}
	}
	WaitForFinish(qtrk, count);
	
	// Measure speed
	double tend = GetPreciseTime();
	img.free();

	float mean = sumScheduleTime / count;
	float stdev = sqrt(sumScheduleTime2 / count - mean * mean);
	dbgprintf("Scheduletime: Avg=%f, Max=%f, Stdev=%f\n", mean*1000, maxScheduleTime*1000, stdev*1000);
	*scheduleTime = mean;

	return count/(tend-tstart);
}

int NearestPowerOfTwo(int v)
{
	int r=1;
	while (r < v) 
		r *= 2;
	if ( fabsf(r-v) < fabsf(r/2-v) )
		return r;
	return r/2;
}

int SmallestPowerOfTwo(int minval)
{
	int r=1;
	while (r < minval)
		r *= 2;
	return r;
}



struct SpeedInfo {
	float speed_cpu, speed_gpu;
	float sched_cpu, sched_gpu;
};

SpeedInfo SpeedCompareTest(int w,bool gc=false)
{
	int cudaBatchSize = 1024;
	int count = 20000;

#ifdef _DEBUG
	count = 100;
	cudaBatchSize = 32;
#endif
	bool haveZLUT = false;
	LocMode_t locType = (LocMode_t)( LT_QI|LT_NormalizeProfile );

	QTrkComputedConfig cfg;
	cfg.width = cfg.height = w;
	cfg.qi_iterations = 4;
	//std::vector<int> devices(1); devices[0]=1;
	//SetCUDADevices(devices);
	cfg.cuda_device = QTrkCUDA_UseAll;
	cfg.numThreads = -1;
	cfg.com_bgcorrection = 0.0f;
	cfg.Update();
	dbgprintf("Width: %d, QI radius: %f, radialsteps: %d\n", w, cfg.qi_maxradius, cfg.qi_radialsteps);

	SpeedInfo info;
	QueuedCPUTracker *cputrk = new QueuedCPUTracker(cfg);
	info.speed_cpu = SpeedTest(cfg, cputrk, count, haveZLUT, locType, &info.sched_cpu, gc);
	delete cputrk;

	QueuedCUDATracker *cudatrk = new QueuedCUDATracker(cfg, cudaBatchSize);
	info.speed_gpu = SpeedTest(cfg, cudatrk, count, haveZLUT, locType, &info.sched_gpu, gc);
	//info.speed_gpu = SpeedTest(cfg, cudatrk, count, haveZLUT, locType, &info.sched_gpu);
	std::string report = cudatrk->GetProfileReport();
	delete cudatrk;

	dbgprintf("CPU tracking speed: %d img/s\n", (int)info.speed_cpu);
	dbgprintf("GPU tracking speed: %d img/s\n", (int)info.speed_gpu);

	return info;
}

void ProfileSpeedVsROI()
{
	int N=24;
	float* values = new float[N*2];

	for (int i=0;i<N;i++) {
		int roi = 40+i*5;
		SpeedInfo info = SpeedCompareTest(roi);
		values[i*2+0] = info.speed_cpu;
		values[i*2+1] = info.speed_gpu;
	}

	const char *labels[] = { "CPU", "CUDA" };
	WriteImageAsCSV("speeds.txt", values, 2, N, labels);
	delete[] values;
}

std::vector<vector3f> LocalizeGeneratedImages(const QTrkSettings& cfg, QueuedTracker* qtrk, bool haveZLUT, LocMode_t locType, std::vector<vector3f> positions)
{
	ImageData img = ImageData::alloc(cfg.width,cfg.height);
	srand(1);

	// Generate ZLUT
	int zplanes=100;
	int count = positions.size();
	float zmin=0.5,zmax=3;
	qtrk->SetRadialZLUT(0, 1, zplanes);
	if (haveZLUT) {
		qtrk->SetLocalizationMode(LT_BuildRadialZLUT|LT_QI|LT_NormalizeProfile);
		for (int x=0;x<zplanes;x++)  {
			vector2f center( cfg.width/2, cfg.height/2 );
			float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
			GenerateTestImage(img, center.x, center.y, s, 0.0f);
			qtrk->ScheduleLocalization((uchar*)img.data, cfg.width*sizeof(float),QTrkFloat, x, 0,0, 0, x);
		}
		qtrk->Flush();
		// wait to finish ZLUT
		while (qtrk->GetResultCount() != zplanes) {
			Sleep(100);
			dbgprintf(".");
		}
	}
	qtrk->ClearResults();
	qtrk->SetLocalizationMode(locType| (haveZLUT ? LT_LocalizeZ : 0) );
	for (int n=0;n<count;n++) {
		vector3f pos = positions[n];
		float s = zmin + (zmax-zmin) * pos.z/zplanes;
		GenerateTestImage(img, cfg.width/2 + pos.x, cfg.height/2 + pos.y, s, 0);
		//if (n<5) FloatToJPEGFile(SPrintf("tracker-%d.jpg", n).c_str(), image, cfg.width,cfg.height);
		qtrk->ScheduleLocalization((uchar*)img.data, cfg.width*sizeof(float), QTrkFloat, n, 0, 0, 0, 0);
	}
	qtrk->Flush();
	while (qtrk->GetResultCount() != count) Sleep(10);

	std::vector<LocalizationResult> results (count);
	qtrk->FetchResults( &results[0], count );
	std::sort (results.begin(), results.end(), 
		[](LocalizationResult& a, LocalizationResult& b) { return a.job.frame < b.job.frame; } );

	std::vector<vector3f> resultPos(count);
	for (int i=0;i<count;i++) {
		resultPos[i] = results[i].pos;
	}

	img.free();
	return resultPos;
}


void CompareAccuracy (const char *lutfile)
{
	QTrkSettings cfg;
	cfg.width=150;
	cfg.height=150;
	cfg.numThreads=1;

	auto cpu = RunTracker<QueuedCPUTracker> (lutfile, &cfg, false, "cpu", LT_QI);
	auto gpu = RunTracker<QueuedCUDATracker>(lutfile, &cfg, false, "gpu", LT_QI);
//	auto cpugc = RunTracker<QueuedCPUTracker>(lutfile, &cfg, true, "cpugc");
//	auto gpugc = RunTracker<QueuedCUDATracker>(lutfile, &cfg, true, "gpugc");

	for (int i=0;i<std::min((int)cpu.output.size(),20);i++) {
		dbgprintf("CPU-GPU: %f, %f\n", cpu.output[i].x-gpu.output[i].x,cpu.output[i].y-gpu.output[i].y);
	}

/*	dbgprintf("CPU\tGPU\tCPU(gc)\tGPU(gc)\n");
	dbgprintf("St Dev. : CPU: %.2f\tGPU: %.2f\tCPU(gc)%.2f\tGPU(gc)%.2f\n", StDev(cpu).x, StDev(gpu).x, StDev(cpugc).x, StDev(gpugc).x); 
	dbgprintf("Mean err: CPU: %.2f\tGPU: %.2f\tCPU(gc)%.2f\tGPU(gc)%.2f\n", Mean(cpu).x, Mean(gpu).x, Mean(cpugc).x, Mean(gpugc).x); 
*/
}

texture<float, cudaTextureType2D, cudaReadModeElementType> test_tex(0, cudaFilterModePoint); // Un-normalized
texture<float, cudaTextureType2D, cudaReadModeElementType> test_tex_lin(0, cudaFilterModeLinear); // Un-normalized


__global__ void TestSampling(int n , cudaImageListf img, float *rtex, float *rtex2, float *rmem, float2* pts)
{
	int idx = threadIdx.x+blockDim.x * blockIdx.x;

	if (idx < n) {
		float x = pts[idx].x;
		float y = pts[idx].y;
		int ii = 1;
		rtex[idx] = tex2D(test_tex_lin, x+0.5f, y+0.5f+img.h*ii);
		bool outside;
		rtex2[idx] = img.interpolateFromTexture(test_tex, x, y, ii, outside);
		rmem[idx] = img.interpolate(x,y,ii, outside);
	}
}

void TestTextureFetch()
{
	int w=8,h=4;
	cudaImageListf img = cudaImageListf::alloc(w,h,2);
	float* himg = new float[w*h*2];

	int N=10;
	std::vector<vector2f> pts(N);
	for(int i=0;i<N;i++) {
		pts[i]=vector2f( rand_uniform<float>() * (w-1), rand_uniform<float>() * (h-1) );
	}
	device_vec<vector2f> dpts;
	dpts.copyToDevice(pts, false);

	srand(1);
	for (int i=0;i<w*h*2;i++)
		himg[i]=i;
	img.copyToDevice(himg,false);

	img.bind(test_tex);
	img.bind(test_tex_lin);
	device_vec<float> rtex(N),rmem(N),rtex2(N);
	int nt=32;
	TestSampling<<< dim3( (N+nt-1)/nt ), dim3(nt) >>> (N, img, rtex.data,rtex2.data,rmem.data, (float2*)dpts.data);
	img.unbind(test_tex_lin);
	img.unbind(test_tex);

	auto hmem = rmem.toVector();
	auto htex = rtex.toVector();
	auto htex2 = rtex2.toVector();
	for (int x=0;x<N;x++) {
		dbgprintf("[%.2f, %.2f]: %f (tex), %f(tex2),  %f (mem).  tex-mem: %f,  tex2-mem: %f\n",
		pts[x].x, pts[x].y, htex[x], htex2[x],	hmem[x],	htex[x]-hmem[x],htex2[x]-hmem[x]);
	}
}




void BasicQTrkTest()
{
	QTrkComputedConfig cc;
	cc.width = cc.height = 100;
	cc.Update();
	QueuedCUDATracker qtrk(cc);

	float zmin=1,zmax=5;
	ImageData img = ImageData::alloc(cc.width,cc.height);
	GenerateTestImage(img, cc.width/2, cc.height/2, (zmin+zmax)/2, 0);
	
	int N=4000;
#ifdef _DEBUG
	N=400;
#endif
	qtrk.SetLocalizationMode((LocMode_t)(LT_QI|LT_NormalizeProfile));
	for (int i=0;i<N;i++)
	{
		LocalizationJob job ( i, 0, 0, 0);
		qtrk.ScheduleLocalization((uchar*)img.data, sizeof(float)*cc.width, QTrkFloat, &job);
	}

	double t = WaitForFinish(&qtrk, N);

	dbgprintf("Speed: %d imgs/s (Only QI)", (int)(N / t));
	img.free();
}

void TestGauss2D(bool calib)
{
	int N=20, R=1000;
#ifdef _DEBUG
	R=1;
#endif
	std::vector<vector3f> rcpu = Gauss2DTest<QueuedCPUTracker>(N, R, calib);
	std::vector<vector3f> rgpu = Gauss2DTest<QueuedCUDATracker>(N, R, calib);

	for (int i=0;i<std::min(20,N);i++) {
		dbgprintf("[%d] CPU: X:%.5f, Y:%.5f\t;\tGPU: X:%.5f, Y:%.5f. \tDiff: X:%.5f, Y:%.5f\n", 
			i, rcpu[i].x, rcpu[i].y, rgpu[i].x, rgpu[i].y, rcpu[i].x-rgpu[i].x, rcpu[i].y-rgpu[i].y);
	}
}

void TestRadialLUTGradientMethod()
{

}


std::vector< float > cmp_cpu_qi_prof;
std::vector< float > cmp_gpu_qi_prof;

std::vector< std::complex<float> > cmp_cpu_qi_fft_out;
std::vector< std::complex<float> > cmp_gpu_qi_fft_out;


void QICompare(const char *lutfile )
{
	QTrkSettings cfg;
	cfg.qi_iterations=1;
	cfg.width = 150;
	cfg.height = 150;
	cfg.numThreads=1;	
	QueuedCUDATracker gpu(cfg, 1);
	QueuedCPUTracker cpu(cfg);

	ImageData lut=ReadJPEGFile(lutfile);
	ImageData img=ImageData::alloc(cfg.width,cfg.height);

	srand(0);
	const int N=1;
	gpu.SetLocalizationMode( LT_QI);
	cpu.SetLocalizationMode(LT_QI);
	for (int i=0;i<N;i++) {
		LocalizationJob job(i, 0, 0, 0);
		vector2f pos(cfg.width/2,cfg.height/2);
		pos.x += rand_uniform<float>();
		pos.y += rand_uniform<float>();
		GenerateImageFromLUT(&img, &lut, 1, cfg.width/2, pos, lut.h/2,1.0f);
		gpu.ScheduleLocalization( (uchar*)img.data, sizeof(float)*img.w, QTrkFloat, &job);
		cpu.ScheduleLocalization( (uchar*)img.data, sizeof(float)*img.w, QTrkFloat, &job);
	}
	gpu.Flush();
	cpu.Flush();
	while(cpu.GetResultCount() != N || gpu.GetResultCount() != N );
	
	ImageData dbgImg = cpu.DebugImage(0);
	FloatToJPEGFile("qidbgimg.jpg", dbgImg.data, dbgImg.w, dbgImg.h);

	auto rcpu = FetchResults(&cpu), rgpu = FetchResults(&gpu);
	for (int i=0;i<N;i++) {
		vector3f d=rcpu[i]-rgpu[i];
		dbgprintf("[%d]: CPU: x=%f, y=%f. GPU: x=%f, y=%f.\tGPU-CPU: x:%f, y:%f\n", i, rcpu[i].x, rcpu[i].y, rgpu[i].x, rgpu[i].y, d.x,d.y);
	}

	// Profiles
	for(int i=0;i<cmp_cpu_qi_prof.size();i++) {
		dbgprintf("QIPROF[%d]. CPU=%f, GPU=%f, Diff: %f\n", i, cmp_cpu_qi_prof[i], cmp_gpu_qi_prof[i], cmp_gpu_qi_prof[i]-cmp_cpu_qi_prof[i]);
	}
	// FFT out
	for(int i=0;i<cmp_cpu_qi_fft_out.size();i++) {
		dbgprintf("fft-out[%d]. CPU=%f, GPU=%f, Diff: %f\n", i, cmp_cpu_qi_fft_out[i].real(), cmp_gpu_qi_fft_out[i].real(), cmp_gpu_qi_fft_out[i].real()-cmp_cpu_qi_fft_out[i].real());
	}

	img.free();
	lut.free();
}


surface<void, 2> test_surf;

__global__ void writeSurf(int w, int h, float val)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x< w && y < h) {
//		float v;
//		surf2Dread(&v, test_surf, x*4,y);

		surf2Dwrite(val, test_surf, x*4, y);
	}
}


__global__ void cosSurf(int w, int h)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x< w && y < h) {
		float v;
		surf2Dread(&v, test_surf, x*4,y);

		surf2Dwrite(cos(v), test_surf, x*4, y);
	}
}


__global__ void readSurf(int w, int h, float* dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x< w && y < h) {
		float v;
		surf2Dread(&v, test_surf, x*4,y);

		dst[y*w+x] = v;
	}
}


void TestSurfaceReadWrite()
{
	cudaArray *a;
	int W=128, H=64;

	auto cd = cudaCreateChannelDesc<float>();

	float* rnd = new float[W*H];
	for (int i=0;i<W*H;i++) rnd[i]=rand_uniform<float>();

	// Copy random test data to cuda array
	CheckCUDAError(cudaMallocArray(&a, &cd,sizeof(float)* W,H, cudaArraySurfaceLoadStore));
	CheckCUDAError(cudaMemcpy2DToArray(a, 0, 0, rnd, W*sizeof(float), sizeof(float)*W, H, cudaMemcpyHostToDevice) );
	
	CheckCUDAError(cudaBindSurfaceToArray(test_surf, a) );
	dim3 nt (16,8);
	device_vec<float> d_dst(W*H);
	readSurf<<< dim3( (W+nt.x-1)/nt.x, (H+nt.y-1)/nt.y ), nt >>> (W,H, d_dst.data);

	std::vector<float> result = d_dst;
	for (int i=0;i<W*H;i++)
		assert(result[i]==rnd[i]);
	cudaFreeArray(a);

	// Test 2D layered
/*	cudaArray_t a3;
	auto ext=make_cudaExtent(20, 10, 5);
	CheckCUDAError(cudaMalloc3DArray(&a3, &cd, ext, cudaArraySurfaceLoadStore | cudaArrayLayered ));
	cudaMemcpy3DParms p ={0};
	p.dstArray=a3;
	p.extent = ext;
	p.kind = cudaMemcpyHostToDevice;
	CheckCUDAError(cudaMemcpy3D(&p));*/
}

surface<void, cudaSurfaceType2DLayered> img4d_test;
//surface<void, 3> img4d_test;


__global__ void set_img4d_value(Image4DCudaArray<float>::KernelInst p, int L, float val)
{
	for (int y=0;y<p.imgh;y++) {
		for (int x=0;x<p.imgw;x++)
		{
			//p.writeSurfacePixel(img4d_test, x,y, L, val);
			surf2DLayeredwrite(val, img4d_test, sizeof(float)*x, y, L, cudaBoundaryModeTrap);
			//surf3Dwrite(val, img4d_test, sizeof(float)*x, y, L, cudaBoundaryModeTrap);

			float v2;
			surf2DLayeredread(&v2, img4d_test, sizeof(float)*x, y, L, cudaBoundaryModeTrap);

		}
	}
}

void TestImage4D()
{
	int W=32,H=32,D=10,L=5;
	Image4DCudaArray<float> img(W,H,D,L);
	int N=W*H*D*L;
	float *src = new float[N], *test=new float[N];

	for (int i=0;i<N;i++)
		src[i] = rand_uniform<float>();

	img.clear();

	img.copyToDevice(src);
	img.copyToHost(test);
	img.copyToHost(test);

	for(int i=0;i<N;i++)
		assert(src[i]==test[i]);

	auto cd = cudaCreateChannelDesc<float>();
	//cudaBindSurfaceToArray(&img4d_test, img.array, &cd);
	img.bind(img4d_test);
	set_img4d_value<<< dim3(1,1,1), dim3(1,1,1) >>> (img.kernelInst(), 0, 1.0f);
	cudaDeviceSynchronize();

	img.copyToHost(test);
	for (int i=0;i<W*H;i++) {
		assert(test[i] == 1.0f);
	}

	delete[] src; delete[] test;
}

void TestImage4DMemory()
{
	int W=32,H=32, D=5, L=10;
	Image4DMemory<float> img(W,H,D,L);
	
	int N=W*H*D*L;
	float *src = new float[N], *test=new float[N];

	for (int i=0;i<N;i++)
		src[i] = rand_uniform<float>();

	img.clear();

	img.copyToDevice(src);
	img.copyToHost(test);
	img.copyToHost(test);

	for(int i=0;i<N;i++)
		assert(src[i]==test[i]);
}

void TestBenchmarkLUT()
{
	BenchmarkLUT bml("refbeadlut.jpg");

	ImageData img=ImageData::alloc(120,120);

	ImageData lut = ImageData::alloc(bml.lut_w, bml.lut_h);
	bml.GenerateLUT(&lut,1.0f);
	WriteJPEGFile("refbeadlut-lutsmp.jpg", lut);
	lut.free();
	
	bml.GenerateSample(&img, vector3f(img.w/2,img.h/2,bml.lut_h/2), img.w/2-5);
	WriteJPEGFile("refbeadlut-bmsmp.jpg", img);
	img.free();
}

int main(int argc, char *argv[])
{
	listDevices();

//	TestBenchmarkLUT();
//	testLinearArray();
//	TestTextureFetch();
//	TestGauss2D(true);
//	MultipleLUTTest();

//	TestSurfaceReadWrite();
//	TestImage4D();
//	TestImage4DMemory();
//	TestImageLUT("../cputrack-test/lut000.jpg");
	//TestRadialLUTGradientMethod();
	TestBuildRadialZLUT<QueuedCUDATracker> ("../cputrack-test/lut000.jpg");

//	BenchmarkParams();

//	BasicQTrkTest();
//	TestCMOSNoiseInfluence<QueuedCUDATracker>("../cputrack-test/lut000.jpg");

#ifdef QI_DEBUG
	QICompare("../cputrack-test/lut000.jpg");
#endif

//CompareAccuracy("../cputrack-test/lut000.jpg");
//QTrkCompareTest();
	//	ProfileSpeedVsROI();
	/*auto info = SpeedCompareTest(80, false);
	auto infogc = SpeedCompareTest(80, true);
	dbgprintf("[gainc=false] CPU: %f, GPU: %f\n", info.speed_cpu, info.speed_gpu); 
	dbgprintf("[gainc=true] CPU: %f, GPU: %f\n", infogc.speed_cpu, infogc.speed_gpu); 
	*/
	return 0;
}
