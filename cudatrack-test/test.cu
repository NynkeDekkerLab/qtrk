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

#include "FisherMatrix.h"

#include "testutils.h"
#include "ResultManager.h"

#include "ExtractBeadImages.h"

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
		for (int x=0;x<zplanes;x++)  {
			vector2f center ( cfg.width/2, cfg.height/2 );
			float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
			GenerateTestImage(img, center.x, center.y, s, 0.0f);
			WriteJPEGFile("qtrkzlutimg.jpg", img);

			qtrk.BuildLUT(img.data,img.pitch(),QTrkFloat, 0, (vector2f*)(0));
			if (cpucmp) 
				qtrkcpu.BuildLUT(img.data,img.pitch(),QTrkFloat, 0);
		}
		qtrk.FinalizeLUT();
		if (cpucmp) qtrkcpu.FinalizeLUT();
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
		for (int x=0;x<zplanes;x++)  {
			vector2f center( cfg.width/2, cfg.height/2 );
			float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
			GenerateTestImage(img, center.x, center.y, s, 0.0f);
			qtrk->BuildLUT(img.data,img.pitch(),QTrkFloat, 0);
		}
		qtrk->FinalizeLUT();
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

SpeedInfo SpeedCompareTest(int w, LocalizeModeEnum locMode, bool haveZLUT, int qi_iterations = 5)
{
	int cudaBatchSize = 1024;
	int count = 60000;

#ifdef _DEBUG
	count = 100;
	cudaBatchSize = 32;
#endif
	LocMode_t locType = (LocMode_t)( locMode|LT_NormalizeProfile );

	QTrkComputedConfig cfg;
	cfg.width = cfg.height = w;
	cfg.qi_iterations = qi_iterations;
	cfg.qi_radial_coverage = 1.5f;
	cfg.qi_angstep_factor = 1.5f;
	cfg.qi_angular_coverage = 0.7f;
	cfg.zlut_radial_coverage = 2.0f;
	//std::vector<int> devices(1); devices[0]=1;
	//SetCUDADevices(devices);
	cfg.cuda_device = QTrkCUDA_UseAll;
	cfg.numThreads = -1;
	cfg.com_bgcorrection = 0.0f;
	cfg.Update();
	dbgprintf("Width: %d, QI radius: %f, radialsteps: %d\n", w, cfg.qi_maxradius, cfg.qi_radialsteps);

	SpeedInfo info;
	QueuedCPUTracker *cputrk = new QueuedCPUTracker(cfg);
	info.speed_cpu = SpeedTest(cfg, cputrk, count, haveZLUT, locType, &info.sched_cpu, false);
	delete cputrk;

	QueuedCUDATracker *cudatrk = new QueuedCUDATracker(cfg, cudaBatchSize);
	info.speed_gpu = SpeedTest(cfg, cudatrk, count, haveZLUT, locType, &info.sched_gpu, false);
	//info.speed_gpu = SpeedTest(cfg, cudatrk, count, haveZLUT, locType, &info.sched_gpu);
	std::string report = cudatrk->GetProfileReport();
	delete cudatrk;

	dbgprintf("CPU tracking speed: %d img/s\n", (int)info.speed_cpu);
	dbgprintf("GPU tracking speed: %d img/s\n", (int)info.speed_gpu);

	return info;
}

void ProfileSpeedVsROI(LocalizeModeEnum locMode, const char *outputcsv, bool haveZLUT, int qi_iterations)
{
	std::vector<float> values;

	for (int roi=20;roi<=180;roi+=10) { // same as BenchmarkROIAccuracy()
		SpeedInfo info = SpeedCompareTest(roi, locMode, haveZLUT, qi_iterations);
		values.push_back( roi);
		values.push_back(info.speed_cpu);
		values.push_back( info.speed_gpu);
	}

	const char *labels[] = { "ROI", "CPU", "CUDA" };
	WriteImageAsCSV(outputcsv, &values[0], 3, values.size()/3, labels);
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
/*
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
*/
void BasicQTrkTest()
{
	QTrkComputedConfig cc;
	cc.width = cc.height = 100;
	cc.Update();
	QueuedCUDATracker qtrk(cc);

	float zmin=1,zmax=5;
	ImageData img = ImageData::alloc(cc.width,cc.height);

	float pos_x = cc.width/2 - 5;
	float pos_y = cc.height/2 + 3;
	GenerateTestImage(img, pos_x, pos_y, (zmin+zmax)/2, 0);
	
	int N = 100000;
#ifdef _DEBUG
	N = 10000;
#endif
	double t = GetPreciseTime();
	qtrk.SetLocalizationMode((LocMode_t)(LT_QI|LT_NormalizeProfile));
	for (int i=0;i<N;i++)
	{
		LocalizationJob job ( i, 0, 0, 0);
		qtrk.ScheduleLocalization((uchar*)img.data, sizeof(float)*cc.width, QTrkFloat, &job);
		if(i%std::max(1,(int)(N*0.1))==0) dbgprintf("Queued: %d / %d\n", i, N);
	}
	WaitForFinish(&qtrk, N);
	t = GetPreciseTime() - t;
	dbgprintf("Speed: %d imgs/s (Only QI, %d iterations)\n", (int)(N / t), cc.qi_iterations);
	int count = 0;

	while(qtrk.GetResultCount() != 0){
		LocalizationResult res;
		qtrk.FetchResults(&res,1);
		if( res.pos.x > pos_x + 0.01f || res.pos.x < pos_x - 0.01f || res.pos.y > pos_y + 0.01f || res.pos.y < pos_y - 0.01f ){
			if(count < 100)
				dbgprintf("Location frame %d: (%02f,%02f)\n",res.job.frame, res.pos.x, res.pos.y);
			count++;
		}
	}

	dbgprintf("Position was (%.3f, %.3f) \n", pos_x, pos_y);
	dbgprintf("Errors: %d/%d (%f%%)\n", count, N, (float)100 * count / N);
	img.free();
}

void BasicQTrkTest_RM()
{
	QTrkComputedConfig cc;
	//cc.qi_iterations = 10;
	cc.width = cc.height = 100;
	cc.Update();
	QueuedCUDATracker qtrk(cc);

	float zmin=1,zmax=5;
	ImageData img = ImageData::alloc(cc.width,cc.height);

	// Positions to set
	float pos_x = cc.width/2 - 5;
	float pos_y = cc.height/2 + 3;
	GenerateTestImage(img, pos_x, pos_y, (zmin+zmax)/2, 0);
	
	int N = 100000;
#ifdef _DEBUG
	N = 100000;
#endif
	qtrk.SetLocalizationMode((LocMode_t)(LT_QI|LT_NormalizeProfile));

	ResultManagerConfig RMcfg;
	RMcfg.numBeads = 1;
	RMcfg.numFrameInfoColumns = 0;
	RMcfg.scaling = vector3f(1.0f,1.0f,1.0f);
	RMcfg.offset  = vector3f(0.0f,0.0f,0.0f);
	RMcfg.writeInterval = 4000;
	RMcfg.maxFramesInMemory = 0;
	RMcfg.binaryOutput = false;

	std::vector<std::string> colnames;
	for(int ii = 0;ii<RMcfg.numFrameInfoColumns;ii++){
		colnames.push_back(SPrintf("%d",ii));
	}

	outputter output(Files+Images);

	ResultManager RM(
		SPrintf("%s\\RMOutput.txt",output.folder.c_str()).c_str(),
		SPrintf("%s\\RMFrameInfo.txt",output.folder.c_str()).c_str(),
		&RMcfg, colnames);
	
	RM.SetTracker(&qtrk);
	double t = GetPreciseTime();
	for (int i=0;i<N;i++)
	{
		LocalizationJob job ( i, 0, 0, 0);
		qtrk.ScheduleLocalization((uchar*)img.data, sizeof(float)*cc.width, QTrkFloat, &job);
		//if(i%std::max(1,N/1000)==0) dbgprintf("Queued: %d / %d\n", i, N);
	}
	printf("\nDone queueing!\n");
	// Tell the tracker to perform the localizations left in the queue regardless of batchSize
	qtrk.Flush();
	
	// Halt the test (=timer) until all localizations are done.
	while(RM.GetFrameCounters().localizationsDone < N);
	t = GetPreciseTime() - t;
	
	// Tell the resultmanager to print the final available results regardless of writeInterval
	RM.Flush();
	while(RM.GetFrameCounters().lastSaveFrame != N);

	dbgprintf("Speed: %d imgs/s (Only QI, %d iterations)\n", (int)(N / t), cc.qi_iterations);

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
	gpu.SetLocalizationMode(LT_QI);
	cpu.SetLocalizationMode(LT_QI);
	for (int i=0;i<N;i++) {
		LocalizationJob job(i, 0, 0, 0);
		vector3f pos(cfg.width/2,cfg.height/2, lut.h/2);
		pos.x += rand_uniform<float>();
		pos.y += rand_uniform<float>();
		GenerateImageFromLUT(&img, &lut, 1, cfg.width/2, pos);
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
	for(uint i=0;i<cmp_cpu_qi_prof.size();i++) {
		dbgprintf("QIPROF[%d]. CPU=%f, GPU=%f, Diff: %f\n", i, cmp_cpu_qi_prof[i], cmp_gpu_qi_prof[i], cmp_gpu_qi_prof[i]-cmp_cpu_qi_prof[i]);
	}
	// FFT out
	for(uint i=0;i<cmp_cpu_qi_fft_out.size();i++) {
		dbgprintf("fft-out[%d]. CPU=%f, GPU=%f, Diff: %f\n", i, cmp_cpu_qi_fft_out[i].real(), cmp_gpu_qi_fft_out[i].real(), cmp_gpu_qi_fft_out[i].real()-cmp_cpu_qi_fft_out[i].real());
	}

	img.free();
	lut.free();
}

void TestBenchmarkLUT()
{
	BenchmarkLUT bml("refbeadlut.jpg");

	ImageData img=ImageData::alloc(120,120);

	ImageData lut = ImageData::alloc(bml.lut_w, bml.lut_h);
	bml.GenerateLUT(&lut);
	WriteJPEGFile("refbeadlut-lutsmp.jpg", lut);
	lut.free();
	
	bml.GenerateSample(&img, vector3f(img.w/2,img.h/2,bml.lut_h/2), 0, img.w/2-5);
	WriteJPEGFile("refbeadlut-bmsmp.jpg", img);
	img.free();
}

template<typename T>
void check_arg(const std::vector<std::string>& args, const char *name, T *param)
{
	for (uint i=0;i<args.size();i++) {
		if (args[i] == name) {
			*param = (T)atof(args[i+1].c_str());
			return;
		}
	}
}

void check_strarg(const std::vector<std::string>& args, const char *name, std::string* param)
{
	for (uint i=0;i<args.size();i++) {
		if (args[i] == name) {
			*param = args[i+1];
			return;
		}
	}
}

int CmdLineRun(int argc, char*argv[])
{
	QTrkSettings cfg;
	std::vector<std::string> args(argc-1);
	for (int i=0;i<argc-1;i++)
		args[i]=argv[i+1];
	
	check_arg(args, "roi", &cfg.width);
	cfg.height=cfg.width;

	int count=100;
	check_arg(args, "count", &count);

	std::string outputfile, fixlutfile, inputposfile, bmlutfile, rescaledlutfile;
	std::string radialWeightsFile;
	check_strarg(args, "output", &outputfile);
	check_strarg(args, "fixlut", &fixlutfile);
	check_strarg(args, "bmlut", &bmlutfile);
	check_strarg(args, "inputpos", &inputposfile);
	check_strarg(args, "regenlut", &rescaledlutfile);
	check_strarg(args, "radweights", &radialWeightsFile);

	std::string crlboutput;
	check_strarg(args, "crlb", &crlboutput);

	std::vector< vector3f > inputPos;
	if (!inputposfile.empty()) {
		inputPos = ReadVector3CSV(inputposfile.c_str());
		count = inputPos.size();
	}

	check_arg(args, "zlut_minradius", &cfg.zlut_minradius);
	check_arg(args, "zlut_radial_coverage", &cfg.zlut_radial_coverage);
	check_arg(args, "zlut_angular_coverage", &cfg.zlut_angular_coverage);
	check_arg(args, "zlut_roi_coverage", &cfg.zlut_roi_coverage);

	check_arg(args, "qi_iterations", &cfg.qi_iterations);
	check_arg(args, "qi_minradius", &cfg.qi_minradius);
	check_arg(args, "qi_radial_coverage", &cfg.qi_radial_coverage);
	check_arg(args, "qi_angular_coverage", &cfg.qi_angular_coverage);
	check_arg(args, "qi_roi_coverage", &cfg.qi_roi_coverage);
	check_arg(args, "qi_angstep_factor", &cfg.qi_angstep_factor);
	check_arg(args, "downsample", &cfg.downsample);

	int zlutAlign=0;
	check_arg(args, "zlutalign", &zlutAlign);

	float pixelmax = 28 * 255;
	check_arg(args, "pixelmax", &pixelmax);

	std::string lutsmpfile;
	check_strarg(args, "lutsmpfile", &lutsmpfile);

	int cuda=1;
	check_arg(args, "cuda", &cuda);
	QueuedTracker* qtrk;

	if (cuda) qtrk = new QueuedCUDATracker(cfg);
	else qtrk = new QueuedCPUTracker(cfg);

	ImageData lut;
	BenchmarkLUT bmlut;

	if (!fixlutfile.empty()) 
	{
		lut = ReadJPEGFile(fixlutfile.c_str());

		if(!rescaledlutfile.empty()) {
			// rescaling allowed
			ImageData newlut;
			ResampleLUT(qtrk, &lut, lut.h, &newlut, rescaledlutfile.c_str()); 
			lut.free();
			lut=newlut;
		}
		else if (lut.w != qtrk->cfg.zlut_radialsteps) {
			lut.free();
			dbgprintf("Invalid LUT size (%d). Expecting %d radialsteps\n", lut.w, qtrk->cfg.zlut_radialsteps);
			delete qtrk;
			return -1;
		}

		qtrk->SetRadialZLUT(lut.data,1,lut.h);
	}
	else
	{
		if (bmlutfile.empty()) {
			delete qtrk;
			dbgprintf("No lut file\n");
			return -1;
		}

		bmlut.Load(bmlutfile.c_str());
		lut = ImageData::alloc(qtrk->cfg.zlut_radialsteps, bmlut.lut_h);
		bmlut.GenerateLUT(&lut);

		if (!rescaledlutfile.empty())
			WriteJPEGFile(rescaledlutfile.c_str(), lut);

		qtrk->SetRadialZLUT(lut.data,1,lut.h);
	}

	if (inputPos.empty()) {
		inputPos.resize(count);
		for (int i=0;i<count;i++){
			inputPos[i]=vector3f(cfg.width/2,cfg.height/2,lut.h/2);
		}
	}

	if (!radialWeightsFile.empty())
	{
		auto rwd = ReadCSV(radialWeightsFile.c_str());
		std::vector<float> rw(rwd.size());
		if (rw.size() == qtrk->cfg.zlut_radialsteps)
			qtrk->SetRadialWeights(&rw[0]);
		else  {
			dbgprintf("Invalid # radial weights");
			delete qtrk;
		}
	}

	std::vector<ImageData> imgs (inputPos.size());

	std::vector<vector3f> crlb(inputPos.size());

	for (uint i=0;i<inputPos.size();i++) {
		imgs[i]=ImageData::alloc(cfg.width, cfg.height);
		//vector3f pos = centerpos + range*vector3f(rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f, rand_uniform<float>()-0.5f)*2;

		auto p = inputPos[i];
		if (!bmlut.lut_w) {
			GenerateImageFromLUT(&imgs[i], &lut, qtrk->cfg.zlut_minradius, qtrk->cfg.zlut_maxradius, p, false);
			if (!crlboutput.empty()) {
				SampleFisherMatrix sfm(pixelmax);
				crlb[i]=sfm.Compute(p, vector3f(1,1,1)*0.001f, lut, qtrk->cfg.width,qtrk->cfg.height, qtrk->cfg.zlut_minradius, qtrk->cfg.zlut_maxradius).Inverse().diag();
			}
		} else
			bmlut.GenerateSample(&imgs[i], p, qtrk->cfg.zlut_minradius, qtrk->cfg.zlut_maxradius);
		imgs[i].normalize();
		if (pixelmax > 0) ApplyPoissonNoise(imgs[i], pixelmax, 255);
		if(i==0 && !lutsmpfile.empty()) WriteJPEGFile(lutsmpfile.c_str(), imgs[i]);
	}

	int locMode = LT_LocalizeZ | LT_NormalizeProfile | LT_LocalizeZWeighted;
	if (qtrk->cfg.qi_iterations > 0) 
		locMode |= LT_QI;
	if (zlutAlign)
		locMode |= LT_ZLUTAlign;

	qtrk->SetLocalizationMode((LocMode_t)locMode);
	double tstart=GetPreciseTime();

	for (uint i=0;i<inputPos.size();i++)
	{
		LocalizationJob job(i, 0, 0, 0);
		qtrk->ScheduleImageData(&imgs[i], &job);
	}

	WaitForFinish(qtrk, inputPos.size());
	double tend = GetPreciseTime();

	std::vector<vector3f> results(inputPos.size());
	for (uint i=0;i<inputPos.size();i++) {
		LocalizationResult r;
		qtrk->FetchResults(&r,1);
		results[r.job.frame]=r.pos;
	}
	vector3f meanErr, stdevErr;
	MeanStDevError(inputPos, results, meanErr, stdevErr);
	dbgprintf("Mean err X=%f,Z=%f. St deviation: X=%f,Z=%f\n", meanErr.x,meanErr.y,stdevErr.x,stdevErr.z);

	if (!crlboutput.empty())
		WriteTrace(crlboutput, &crlb[0], crlb.size());

	WriteTrace(outputfile, &results[0], inputPos.size());
	
	if (lut.data) lut.free();
	delete qtrk;

	return 0;
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

	QueuedCUDATracker* qtrk = new QueuedCUDATracker(cfg);
	//qtrk->SetLocalizationMode(LT_NormalizeProfile | LT_QI);
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
	float* luts = new float[beads.size()*(zplanes*cfg.zlut_radialsteps)];
	qtrk->GetRadialZLUT(luts);
	
	for(int ii = 0; ii < beads.size(); ii++){
		ImageData lut = ImageData::alloc(cfg.zlut_radialsteps, zplanes);
		memcpy(lut.data, &luts[ii*cfg.zlut_radialsteps*zplanes], cfg.zlut_radialsteps*zplanes*sizeof(float));
		//memcpy(lut.data,qtrk->GetZLUTByIndex(ii),cfg.zlut_radialsteps*zplanes*sizeof(float));
		//output->outputImage(lut,SPrintf("lut%03d,%d",beads.at(ii).x,beads.at(ii).y));
		output->outputImage(lut, SPrintf("lut%03d",ii));
		lut.free();
	}

	qtrk->Flush();
	delete qtrk;
}

int main(int argc, char *argv[])
{
	//listDevices();

	printf("%d, %d\n",sizeof(long),sizeof(int));

	if (argc > 1)
	{
		return CmdLineRun(argc, argv);
	}

	try {
	//	outputter output(Files+Images);
	//	BuildZLUT("C:\\TestImages\\TestMovie150507_2\\images\\jpg\\Zstack\\", &output);
		BasicQTrkTest();
	//	BasicQTrkTest_RM();


	//	TestBenchmarkLUT();
	//	testLinearArray();
	//	TestTextureFetch();
	//	TestGauss2D(true);
	//	MultipleLUTTest();

	//	TestSurfaceReadWrite();
	//	TestImage4D();
	//	TestImage4DMemory();
	//	TestImageLUT("../cputrack-test/lut000.jpg");
	//	TestRadialLUTGradientMethod();
		
	//	BenchmarkParams();
	//	TestTextureFetch();
	//	QICompare("../cputrack-test/lut000.jpg");
	//	TestCMOSNoiseInfluence<QueuedCUDATracker>("../cputrack-test/lut000.jpg");

	//	CompareAccuracy("../cputrack-test/lut000.jpg");
	//	QTrkCompareTest();
		/*
		ProfileSpeedVsROI(LT_OnlyCOM, "speeds-com.txt", false, 0);
		ProfileSpeedVsROI(LT_OnlyCOM, "speeds-com-z.txt", true, 0);
		ProfileSpeedVsROI(LT_XCor1D, "speeds-xcor.txt", true, 0);
		for (int qi_it=1;qi_it<=4;qi_it++) {
			ProfileSpeedVsROI(LT_QI, SPrintf("speeds-qi-%d-iterations.txt",qi_it).c_str(), true, qi_it);
		}*/

	/*	auto info = SpeedCompareTest(80, false);
		auto infogc = SpeedCompareTest(80, true);
		dbgprintf("[gainc=false] CPU: %f, GPU: %f\n", info.speed_cpu, info.speed_gpu); 
		dbgprintf("[gainc=true] CPU: %f, GPU: %f\n", infogc.speed_cpu, infogc.speed_gpu); 
	*/
	} catch (const std::exception& e) {
		dbgprintf("Exception: %s\n", e.what());
	}
	system("pause");
	return 0;
}