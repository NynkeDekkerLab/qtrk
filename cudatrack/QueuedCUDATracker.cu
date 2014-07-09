/*
CUDA implementations of a variety of tracking algorithms: COM, Quadrant Interpolation, 2D Gaussian with Max-Likelihood estimation.
Copyright 2012-2013, Jelmer Cnossen

It will automatically use all available CUDA devices if using the QTrkCUDA_UseAll value for QTrkSettings::cuda_device

Method:

-Load images into host-side image buffer
-Scheduling thread executes any batch that is filled

- Mutexes:
	* JobQueueMutex: controlling access to state and jobs. 
		Used by ScheduleLocalization, scheduler thread, and GetQueueLen
	* ResultMutex: controlling access to the results list, 
		locked by the scheduler whenever results come available, and by calling threads when they run GetResults/Count

-Running batch:
	- Async copy host-side buffer to device
	- Bind image
	- Run COM kernel
	- QI loop: {
		- Run QI kernel: Sample from texture into quadrant profiles
		- Run CUFFT. Each iteration per axis does 2x forward FFT, and 1x backward FFT.
		- Run QI kernel: Compute positions
	}
	- Compute ZLUT profiles
	- Depending on localize flags:
		- copy ZLUT profiles (for ComputeBuildZLUT flag)
		- generate compare profile kernel + compute Z kernel (for ComputeZ flag)
	- Unbind image
	- Async copy results to host

Issues:
- Due to FPU operations on texture coordinates, there are small numerical differences between localizations of the same image at a different position in the batch
*/
#include "std_incl.h"
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_types.h"
#include <cstdint>
#include "utils.h"

#include "cpu_tracker.h"

#include "QueuedCUDATracker.h"
#include "gpu_utils.h"
#include "ImageSampler.h"


#include "Kernels.h"
#include "DebugResultCompare.h"

#include "QI_impl.h"


// Do CPU-side profiling of kernel launches?
#define TRK_PROFILE

#ifdef TRK_PROFILE
	class ScopedCPUProfiler
	{
		double* time;
		double start;
	public:
		ScopedCPUProfiler(double *time) :  time(time) {
			start = GetPreciseTime();
		}
		~ScopedCPUProfiler() {
			double end = GetPreciseTime();
			*time += start-end;
		}
	};
#else
	class ScopedCPUProfiler {
	public:
		ScopedCPUProfiler(double* time) {}
	};
#endif

static std::vector<int> cudaDeviceList; 

void SetCUDADevices(int* dev, int numdev) {
	cudaDeviceList.assign(dev,dev+numdev);
}



QueuedTracker* CreateQueuedTracker(const QTrkComputedConfig& cc)
{
	return new QueuedCUDATracker(cc);
}


static int GetBestCUDADevice()
{
	int bestScore;
	int bestDev;
	int numDev;
	cudaGetDeviceCount(&numDev);
	for (int a=0;a<numDev;a++) {
		int score;
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, a);
		score = prop.multiProcessorCount * prop.clockRate;
		if (a==0 || bestScore < score) {
			bestScore = score;
			bestDev = a;
		}
	}
	return bestDev;
}

void QueuedCUDATracker::InitializeDeviceList()
{
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	// Select the most powerful one
	if (cfg.cuda_device == QTrkCUDA_UseBest) {
		cfg.cuda_device = GetBestCUDADevice();
		devices.push_back(new Device(cfg.cuda_device));
	} else if(cfg.cuda_device == QTrkCUDA_UseAll) {
		// Use all devices
		for (int i=0;i<numDevices;i++)
			devices.push_back(new Device(i));
	} else if (cfg.cuda_device == QTrkCUDA_UseList) {
		for (uint i=0;i<cudaDeviceList.size();i++)
			devices.push_back(new Device(cudaDeviceList[i]));
	} else {
		devices.push_back (new Device(cfg.cuda_device));
	}
	deviceReport = "Using devices: ";
	for (uint i=0;i<devices.size();i++) {
		cudaDeviceProp p; 
		cudaGetDeviceProperties(&p, devices[i]->index);
		deviceReport += SPrintf("%s%s", p.name, i<devices.size()-1?", ":"\n");
	}
}


QueuedCUDATracker::QueuedCUDATracker(const QTrkComputedConfig& cc, int batchSize) 
	: resultMutex("result"), jobQueueMutex("jobqueue")
{
	cfg = cc;

	InitializeDeviceList();

	// We take numThreads to be the number of CUDA streams
	if (cfg.numThreads < 1) {
		cfg.numThreads = devices.size()*4;
	}
	int numStreams = cfg.numThreads;

	cudaGetDeviceProperties(&deviceProp, devices[0]->index);

	if (deviceProp.kernelExecTimeoutEnabled) {
		throw std::runtime_error(SPrintf("CUDA Kernel execution timeout is enabled for %s. Disable WDDM Time-out Detection and Recovery (TDR) in the nVidia NSight Monitor before running this code", deviceProp.name));
	}

	numThreads = deviceProp.warpSize;
	
	if(batchSize<0) batchSize = 256;
	while (batchSize * cfg.height > deviceProp.maxTexture2D[1]) {
		batchSize/=2;
	}
	this->batchSize = batchSize;

	dbgprintf("CUDA Hardware: %s. \n# of CUDA processors:%d. Using %d streams\n", deviceProp.name, deviceProp.multiProcessorCount, numStreams);
	dbgprintf("Warp size: %d. Max threads: %d, Batch size: %d\n", deviceProp.warpSize, deviceProp.maxThreadsPerBlock, batchSize);
	
	qi.Init(cfg, batchSize);

	std::vector<float2> zlut_radialgrid(cfg.zlut_angularsteps);
	for (int i=0;i<cfg.zlut_angularsteps;i++) {
		float ang = 2*3.141593f*i/(float)cfg.zlut_angularsteps;
		zlut_radialgrid[i]=make_float2(cos(ang),sin(ang));
	}

	for (uint i=0;i<devices.size();i++) {
		Device* d = devices[i];
		cudaSetDevice(d->index);
		qi.InitDevice(&d->qi_instance, cfg);
		d->zlut_trigtable = zlut_radialgrid;
	}
	
	streams.reserve(numStreams);
	try {
		for (int i=0;i<numStreams;i++)
			streams.push_back( CreateStream( devices[i%devices.size()], i ) );
	}
	catch(...) {
		DeleteAllElems(streams);
		throw;
	}

	streams[0]->OutputMemoryUse();

	batchesDone = 0;
	useTextureCache = true;
	resultCount = 0;
	zlut_build_flags=0;

	quitScheduler = false;
	schedulingThread = Threads::Create(SchedulingThreadEntryPoint, this);

	gc_offsetFactor = gc_gainFactor = 1.0f;
	localizeMode = LT_OnlyCOM;

	ForceCUDAKernelsToLoad<<< dim3(),dim3() >>> ();
}

QueuedCUDATracker::~QueuedCUDATracker()
{
	quitScheduler = true;
	Threads::WaitAndClose(schedulingThread);

	DeleteAllElems(streams);
	DeleteAllElems(devices);
}

QueuedCUDATracker::Device::~Device()
{
	cudaSetDevice(index);
	radial_zlut.free();
	calib_gain.free();
	calib_offset.free();
}

void QueuedCUDATracker::SchedulingThreadEntryPoint(void *param)
{
	((QueuedCUDATracker*)param)->SchedulingThreadMain();
}

void QueuedCUDATracker::SchedulingThreadMain()
{
	std::vector<Stream*> activeStreams;

	while (!quitScheduler) {
		jobQueueMutex.lock();
		Stream* s = 0;
		for (int i=0;i<streams.size();i++) 
			if (streams[i]->state == Stream::StreamPendingExec) {
				s=streams[i];
				s->state = Stream::StreamExecuting;
			//	dbgprintf("Executing stream %p [%d]. %d jobs\n", s, i, s->JobCount());
				break;
			}
		jobQueueMutex.unlock();

		if (s) {
			s->imageBufMutex.lock();

			// Launch filled batches, or if flushing launch every batch with nonzero jobs
			if (useTextureCache)
				ExecuteBatch<ImageSampler_Tex> (s);
			else
				ExecuteBatch<ImageSampler_MemCopy> (s);
			s->imageBufMutex.unlock();
			activeStreams.push_back(s);
		}

		// Fetch results
		for (int a=0;a<activeStreams.size();a++) {
			Stream* s = activeStreams[a];
			if (s->IsExecutionDone()) {
		//		dbgprintf("Stream %p done.\n", s);
				CopyStreamResults(s);
				s->localizeFlags = 0; // reset this for the next batch
				jobQueueMutex.lock();
				s->jobs.clear();
				s->state = Stream::StreamIdle;
				jobQueueMutex.unlock();
				activeStreams.erase(std::find(activeStreams.begin(),activeStreams.end(),s));
				break;
			}
		}

		Threads::Sleep(1);
	}
}


QueuedCUDATracker::Stream::Stream(int streamIndex)
	: imageBufMutex(SPrintf("imagebuf%d", streamIndex).c_str())
{ 
	device = 0;
	hostImageBuf = 0; 
	images.data=0; 
	stream=0;
	state=StreamIdle;
	localizeFlags=0;
}

QueuedCUDATracker::Stream::~Stream() 
{
	cudaSetDevice(device->index);

	if(images.data) images.free();
	cudaEventDestroy(localizationDone);
	cudaEventDestroy(qiDone);
	cudaEventDestroy(comDone);
	cudaEventDestroy(imageCopyDone);
	cudaEventDestroy(zcomputeDone);
	cudaEventDestroy(imapDone);
	cudaEventDestroy(batchStart);

	if (stream)
		cudaStreamDestroy(stream); // stream can be zero if in debugStream mode.
}


bool QueuedCUDATracker::Stream::IsExecutionDone()
{
	cudaSetDevice(device->index);
	return cudaEventQuery(localizationDone) == cudaSuccess;
}


void QueuedCUDATracker::Stream::OutputMemoryUse()
{
	int deviceMem = d_com.memsize() + d_locParams.memsize() + qi_instance.memsize() + d_radialprofiles.memsize() +
		d_resultpos.memsize() + d_zlutcmpscores.memsize() + images.totalNumBytes();

	int hostMem = hostImageBuf.memsize() + com.memsize() + locParams.memsize() + results.memsize();

	dbgprintf("Stream memory use: %d kb pinned on host, %d kb device memory (%d for images). \n", hostMem / 1024, deviceMem/1024, images.totalNumBytes()/1024);
}


QueuedCUDATracker::Stream* QueuedCUDATracker::CreateStream(Device* device, int streamIndex)
{
	Stream* s = new Stream(streamIndex);

	try {
		s->device = device;
		cudaSetDevice(device->index);
		cudaStreamCreate(&s->stream);

		s->images = cudaImageListf::alloc(cfg.width, cfg.height, batchSize);
		s->images.allocateHostImageBuffer(s->hostImageBuf);

		s->jobs.reserve(batchSize);
		s->results.init(batchSize);
		s->com.init(batchSize);
		s->d_com.init(batchSize);
		s->d_resultpos.init(batchSize);
		s->results.init(batchSize);
		s->locParams.init(batchSize);
		s->imgMeans.init(batchSize);
		s->d_imgmeans.init(batchSize);
		s->d_locParams.init(batchSize);
		s->d_radialprofiles.init(cfg.zlut_radialsteps*batchSize);

		qi.InitStream(&s->qi_instance, cfg, s->stream, batchSize);
		qalign.InitStream(&s->qalign_instance, cfg, s->stream, batchSize);

		cudaEventCreate(&s->localizationDone);
		cudaEventCreate(&s->comDone);
		cudaEventCreate(&s->imageCopyDone);
		cudaEventCreate(&s->zcomputeDone);
		cudaEventCreate(&s->qiDone);
		cudaEventCreate(&s->imapDone);
		cudaEventCreate(&s->batchStart);
	} catch (...) {
		delete s;
		throw;
	}
	return s;
}


 // get a stream that is not currently executing, and still has room for images
QueuedCUDATracker::Stream* QueuedCUDATracker::GetReadyStream()
{
	while (true) {
		jobQueueMutex.lock();

		Stream *best = 0;
		for (int i=0;i<streams.size();i++) 
		{
			Stream*s = streams[i];

			if (s->state == Stream::StreamIdle) {
				if (!best || (s->JobCount() > best->JobCount()))
					best = s;
			}
		}

		jobQueueMutex.unlock();

		if (best) 
			return best;

		Threads::Sleep(1);
	}
}


bool QueuedCUDATracker::IsIdle()
{
	int ql = GetQueueLength(0);
	return ql == 0;
}

int QueuedCUDATracker::GetQueueLength(int *maxQueueLen)
{
	jobQueueMutex.lock();
	int qlen = 0;
	for (uint a=0;a<streams.size();a++){
		qlen += streams[a]->JobCount();
	}
	jobQueueMutex.unlock();

	if (maxQueueLen) {
		*maxQueueLen = streams.size()*batchSize;
	}

	return qlen;
}


void QueuedCUDATracker::SetLocalizationMode(int mode)
{
	Flush();
	while (!IsIdle());

	jobQueueMutex.lock();
	localizeMode = mode;
	jobQueueMutex.unlock();
}


void QueuedCUDATracker::ScheduleLocalization(void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob* jobInfo )
{
	Stream* s = GetReadyStream();

	jobQueueMutex.lock();
	int jobIndex = s->jobs.size();
	LocalizationJob job = *jobInfo;
	s->jobs.push_back(job);
	s->localizeFlags = localizeMode; // which kernels to run
	s->locParams[jobIndex].zlutIndex = jobInfo->zlutIndex;

	if (s->jobs.size() == batchSize)
		s->state = Stream::StreamPendingExec;
	jobQueueMutex.unlock();

	s->imageBufMutex.lock();
	// Copy the image to the batch image buffer (CPU side)
	float* hostbuf = &s->hostImageBuf[cfg.height*cfg.width*jobIndex];
	CopyImageToFloat( (uchar*)data, cfg.width, cfg.height, pitch, pdt, hostbuf);
	if (localizeMode & LT_ClearFirstFourPixels) {
		for(int i=0;i<4;i++) hostbuf[i]=0;
	}
	s->imageBufMutex.unlock();

	//dbgprintf("Job: %d\n", jobIndex);
}

// 
__global__ void AddProfilesToZLUT(float* d_src, int nbeads, int radialsteps, int plane, cudaImageListf zlut)
{
	int b = threadIdx.x + blockIdx.x * blockDim.x;
	int r = threadIdx.y + blockIdx.y * blockDim.y;

	if (b < nbeads && r < radialsteps) {
		zlut.pixel(r, plane, b) += d_src[ b * radialsteps + r ];
	}
}


void QueuedCUDATracker::BeginLUT(uint flags)
{
	zlut_build_flags = flags;
}


void QueuedCUDATracker::BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, int plane, vector2f* known_pos)
{
	// Copy to image 
	Device* d = streams[0]->device;
	cudaSetDevice(d->index);

	int nbeads = d->radial_zlut.count; // jobcount
	std::vector<vector2f> positions(nbeads);

	float *profiles = new float [nbeads * cfg.zlut_radialsteps];
	memset(profiles, 0, sizeof(float) * nbeads * cfg.zlut_radialsteps);

	// Should be fast enough for zlut building
	parallel_for(nbeads, [&] (int i) {
		CPUTracker trk (cfg.width,cfg.height);
		void *img_data = (uchar*)data + pitch * cfg.height * i;

		if (pdt == QTrkFloat)
			trk.SetImage((float*)img_data, pitch);
		else if (pdt == QTrkU8)
			trk.SetImage8Bit((uchar*)img_data, pitch);
		else
			trk.SetImage16Bit((ushort*)img_data,pitch);

		CPU_ApplyOffsetGain(&trk, i);

		if(known_pos)
			positions[i] = known_pos[i];
		else {
			vector2f com = trk.ComputeMeanAndCOM();
			bool bhit;
			positions[i] = trk.ComputeQI(com, cfg.qi_iterations, cfg.qi_radialsteps, cfg.qi_angstepspq, cfg.qi_angstep_factor, cfg.qi_minradius, cfg.qi_maxradius, bhit);
			trk.ComputeRadialProfile(&profiles[i * cfg.zlut_radialsteps], cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, positions[i], false, 0, true);
		}
	});

	// add to device 0 LUT
	device_vec<float> d_profiles (nbeads * cfg.zlut_radialsteps);
	d_profiles.copyToDevice(profiles, nbeads * cfg.zlut_radialsteps);
	delete[] profiles;

	dim3 numThreads(4, 64);
	dim3 numBlocks( (nbeads + numThreads.x - 1) / numThreads.x, (cfg.zlut_radialsteps + numThreads.y - 1) / numThreads.y);
	AddProfilesToZLUT<<< numBlocks, numThreads, 0, streams[0]->stream >>> (d_profiles.data, nbeads, cfg.zlut_radialsteps, plane, d->radial_zlut);
	cudaStreamSynchronize(streams[0]->stream);
}

void QueuedCUDATracker::FinalizeLUT()
{
	Device* srcd = devices[0];
	cudaSetDevice(srcd->index);

	cudaImageListf& src = srcd->radial_zlut;
	float *tmp = new float [src.w*src.h*src.count];
	srcd->radial_zlut.copyToHost(tmp);

	for (int i=0;i<src.h*src.count;i++){
		NormalizeRadialProfile(&tmp[src.w*i],src.w);
	}

	SetRadialZLUT(tmp, src.count, src.h);
	delete[] tmp;
}


void QueuedCUDATracker::Flush()
{
	jobQueueMutex.lock();
	for (int i=0;i<streams.size();i++) {
		if(streams[i]->JobCount()>0 && streams[i]->state != Stream::StreamExecuting)
			streams[i]->state = Stream::StreamPendingExec;
	}
	jobQueueMutex.unlock();
}


#ifdef QI_DBG_EXPORT
static unsigned long hash(unsigned char *str, int n)
{
    unsigned long hash = 5381;
    
    for (int i=0;i<n;i++) {
		int c = str[i];
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	}

    return hash;
}
#endif

template<typename T>
void checksum(T* data, int elemsize, int numelem, const char *name)
{
#ifdef QI_DBG_EXPORT
	uchar* cp = (uchar*)ALLOCA(elemsize*numelem*sizeof(T));
	cudaDeviceSynchronize();
	cudaMemcpy(cp, data, sizeof(T)*elemsize*numelem, cudaMemcpyDeviceToHost);

	dbgprintf("%s:\n", name);
	for (int i=0;i<numelem;i++) {
		uchar *elem = cp+elemsize*sizeof(T)*i;
		dbgprintf("[%d]: %d\n", i, hash(elem, elemsize));
	}
#endif
}


template<typename TImageSampler>
void QueuedCUDATracker::ExecuteBatch(Stream *s)
{
	if (s->JobCount()==0)
		return;
	//dbgprintf("Sending %d images to GPU stream %p...\n", s->jobCount, s->stream);

	Device *d = s->device;
	cudaSetDevice(d->index);

	BaseKernelParams kp;
	kp.imgmeans = s->d_imgmeans.data;
	kp.images = s->images;
	kp.njobs = s->JobCount();
	kp.locParams = s->d_locParams.data;

	cudaEventRecord(s->batchStart, s->stream);
//	dbgprintf("copying %d jobs to gpu\n", s->JobCount());
	s->d_locParams.copyToDevice(s->locParams.data(), s->JobCount(), true, s->stream);

	{ScopedCPUProfiler p(&cpu_time.imageCopy);
		s->images.copyToDevice(s->hostImageBuf.data(), true, s->stream); 
	}

	if (!d->calib_gain.isEmpty() || !d->calib_offset.isEmpty()) {
		dim3 numThreads(16, 16, 2);
		dim3 numBlocks((cfg.width + numThreads.x - 1 ) / numThreads.x,
				(cfg.height + numThreads.y - 1) / numThreads.y,
				(s->JobCount() + numThreads.z - 1) / numThreads.z);

		gc_mutex.lock();
		float of = gc_offsetFactor, gf = gc_gainFactor;
		gc_mutex.unlock();

		ApplyOffsetGain <<< numBlocks, numThreads, 0, s->stream >>>	
			(kp, s->device->calib_gain, s->device->calib_offset, gf, of);
	}

	cudaEventRecord(s->imageCopyDone, s->stream);

	TImageSampler::BindTexture(s->images);
	{ ScopedCPUProfiler p(&cpu_time.com);
		BgCorrectedCOM<TImageSampler> <<< blocks(s->JobCount()), threads(), 0, s->stream >>> 
			(s->JobCount(), s->images, s->d_com.data, cfg.com_bgcorrection, s->d_imgmeans.data);
		checksum(s->d_com.data, 1, s->JobCount(), "com");
	}
	cudaEventRecord(s->comDone, s->stream);

	device_vec<float3> *curpos = &s->d_com;
	if (s->localizeFlags & LT_QI) {
		ScopedCPUProfiler p(&cpu_time.qi);
		qi.Execute <TImageSampler> (kp, cfg, &s->qi_instance, &s->device->qi_instance, &s->d_com, &s->d_resultpos);
		curpos = &s->d_resultpos;
	}

	if (s->localizeFlags & LT_Gaussian2D) {
		G2MLE_Compute<TImageSampler> <<< blocks(s->JobCount()), threads(), 0, s->stream >>>
			(kp, cfg.gauss2D_sigma, cfg.gauss2D_iterations, s->d_com.data, s->d_resultpos.data, 0, 0);
		curpos = &s->d_resultpos;
	}

	cudaEventRecord(s->qiDone, s->stream);

	int numZIterations = (s->localizeFlags & LT_ZLUTAlign) ? 3 : 1;
	for (int i=0;i<numZIterations;i++) {
		{ScopedCPUProfiler p(&cpu_time.zcompute);
		ZLUTParams zp;
		zp.planes = d->radial_zlut.h;
		zp.angularSteps = cfg.zlut_angularsteps;
		zp.maxRadius = cfg.zlut_maxradius;
		zp.minRadius = cfg.zlut_minradius;
		zp.img = d->radial_zlut;
		zp.trigtable = d->zlut_trigtable.data;
		zp.zcmpwindow = d->zcompareWindow.data;

		// Compute radial profiles
		if (s->localizeFlags & LT_LocalizeZ) {
			dim3 numThreads(16, 16);
			dim3 numBlocks( (s->JobCount() + numThreads.x - 1) / numThreads.x, 
					(cfg.zlut_radialsteps + numThreads.y - 1) / numThreads.y);
			ZLUT_RadialProfileKernel<TImageSampler> <<< numBlocks , numThreads, 0, s->stream >>>
				(s->JobCount(), s->images, zp, curpos->data, s->d_radialprofiles.data, s->d_imgmeans.data);
			ZLUT_NormalizeProfiles<<< blocks(s->JobCount()), threads(), 0, s->stream >>> (s->JobCount(), zp, s->d_radialprofiles.data);
		}
		// Compute Z 
		if (s->localizeFlags & LT_LocalizeZ) {
			dim3 numThreads(8, 16);
			ZLUT_ComputeProfileMatchScores <<< dim3( (s->JobCount() + numThreads.x - 1) / numThreads.x, (zp.planes  + numThreads.y - 1) / numThreads.y), numThreads, 0, s->stream >>> 
				(s->JobCount(), zp, s->d_radialprofiles.data, s->d_zlutcmpscores.data, s->d_locParams.data);
			ZLUT_ComputeZ <<< blocks(s->JobCount()), threads(), 0, s->stream >>> (s->JobCount(), zp, curpos->data, s->d_zlutcmpscores.data);
		}}

		if (i>0) {
			ScopedCPUProfiler p(&cpu_time.zlutAlign);
			qalign.Execute<TImageSampler> (kp, cfg, &s->qalign_instance, &s->device->qalign_instance, &s->d_resultpos, &s->d_resultpos);
		}
	}

	TImageSampler::UnbindTexture(s->images);
	cudaEventRecord(s->zcomputeDone, s->stream);

	{ ScopedCPUProfiler p(&cpu_time.getResults);
		s->d_com.copyToHost(s->com.data(), true, s->stream);
		curpos->copyToHost(s->results.data(), true, s->stream);
		s->d_imgmeans.copyToHost(s->imgMeans.data(), true, s->stream);
	}

	// Make sure we can query the all done signal
	cudaEventRecord(s->localizationDone, s->stream);
}


void QueuedCUDATracker::CopyStreamResults(Stream *s)
{
	resultMutex.lock();
	for (int a=0;a<s->JobCount();a++) {
		LocalizationJob& j = s->jobs[a];
		LocalizationResult r;
		r.job = j;
		r.firstGuess =  vector2f( s->com[a].x, s->com[a].y );
		r.pos = vector3f( s->results[a].x , s->results[a].y, s->results[a].z);
		r.imageMean = s->imgMeans[a];
		r.pos.z = ZLUTBiasCorrection(s->results[a].z, devices[0]->radial_zlut.h, j.zlutIndex);
		results.push_back(r);
	}
	resultCount+=s->JobCount();
	resultMutex.unlock();

	// Update times
	float qi, com, imagecopy, zcomp, getResults;
	cudaEventElapsedTime(&imagecopy, s->batchStart, s->imageCopyDone);
	cudaEventElapsedTime(&com, s->imageCopyDone, s->comDone);
	cudaEventElapsedTime(&qi, s->comDone, s->qiDone);
	cudaEventElapsedTime(&zcomp, s->qiDone, s->zcomputeDone);
	cudaEventElapsedTime(&getResults, s->zcomputeDone, s->localizationDone);
	time.com += com;
	time.qi += qi;
	time.imageCopy += imagecopy;
	time.zcompute += zcomp;
	time.getResults += getResults;
	batchesDone ++;
}

int QueuedCUDATracker::FetchResults(LocalizationResult* dstResults, int maxResults)
{
	resultMutex.lock();
	int numResults = 0;
	while (numResults < maxResults && !results.empty()) {
		dstResults[numResults++] = results.front();
		results.pop_front();
		resultCount--;
	}
	resultMutex.unlock();
	return numResults;
}


void QueuedCUDATracker::SetPixelCalibrationImages(float* offset, float* gain)
{
	for (uint i=0;i<devices.size();i++) {
		devices[i]->SetPixelCalibrationImages(offset, gain, cfg.width, cfg.height);
	}

	// Copy to CPU side buffers for BuildLUT
	int nelem = devices[0]->radial_zlut.count * cfg.width * cfg.height;
	if (offset && gc_offset.size()!=nelem) { 
		gc_offset.resize(nelem);
		gc_offset.assign(offset,offset+nelem);
	}
	if (!offset) gc_offset.clear();

	if (gain && gc_gain.size()!=nelem) {
		gc_gain.reserve(nelem);
		gc_gain.assign(gain, gain+nelem);
	}
	if (!gain) gc_gain.clear();
}


void QueuedCUDATracker::CPU_ApplyOffsetGain(CPUTracker* trk, int beadIndex)
{
	if (!gc_offset.empty() || !gc_gain.empty()) {
		int index = cfg.width*cfg.height*beadIndex;

		gc_mutex.lock();
		float gf = gc_gainFactor, of = gc_offsetFactor;
		gc_mutex.unlock();

		trk->ApplyOffsetGain(gc_offset.empty() ? 0:  &gc_offset[index] , gc_gain.empty() ? 0: &gc_gain[index], of, gf);
//		if (j->job.frame%100==0)
	}
}

void QueuedCUDATracker::SetPixelCalibrationFactors(float offsetFactor, float gainFactor)
{
	gc_mutex.lock();
	gc_gainFactor = gainFactor;
	gc_offsetFactor = offsetFactor;
	gc_mutex.unlock();
}

void QueuedCUDATracker::Device::SetPixelCalibrationImages(float* offset, float* gain, int img_width, int img_height)
{
	cudaSetDevice(index);

	if (offset == 0)
		calib_offset.free();

	if (gain == 0)
		calib_gain.free();

	if (radial_zlut.count > 0) {

		if (gain)
			calib_gain = cudaImageListf::alloc(img_width,img_height,radial_zlut.count);

		if (offset)
			calib_offset = cudaImageListf::alloc(img_width,img_height,radial_zlut.count);

		for (int j=0;j<radial_zlut.count;j++) {
			if (gain) calib_gain.copyImageToDevice(j, &gain[img_width*img_height*j]);
			if (offset) calib_offset.copyImageToDevice(j, &offset[img_width*img_height*j]);
		}
	}
}

// data can be zero to allocate ZLUT data
void QueuedCUDATracker::SetRadialZLUT(float* data,  int numLUTs, int planes) 
{
	for (uint i=0;i<devices.size();i++) {
		devices[i]->SetRadialZLUT(data, cfg.zlut_radialsteps, planes, numLUTs);
	}

	for (uint i=0;i<streams.size();i++) {
		StreamUpdateZLUTSize(streams[i]);
	}
}

void QueuedCUDATracker::SetRadialWeights(float *zcmp)
{
	for (uint i=0;i<devices.size();i++) {
		devices[i]->SetRadialWeights(zcmp);
	}
}

void QueuedCUDATracker::StreamUpdateZLUTSize(Stream* s)
{		
	cudaSetDevice(s->device->index);
	s->d_zlutcmpscores.init(s->device->radial_zlut.h * batchSize);
	// radialsteps never changes
}

void QueuedCUDATracker::Device::SetRadialZLUT(float *data, int radialsteps, int planes, int numLUTs)
{
	cudaSetDevice(index);

	radial_zlut.free();
	radial_zlut = cudaImageListf::alloc(radialsteps, planes, numLUTs);
	if (data) {
		for (int i=0;i<numLUTs;i++)
			radial_zlut.copyImageToDevice(i, &data[planes*radialsteps*i]);
	}
	else radial_zlut.clear();
}


void QueuedCUDATracker::Device::SetRadialWeights(float* zcmp)
{
	cudaSetDevice(index);
	if (zcmp)
		zcompareWindow.copyToDevice(zcmp, radial_zlut.w, false);
	else 
		zcompareWindow.free();
}


void QueuedCUDATracker::GetRadialZLUT(float* data)
{
	cudaImageListf* zlut = &devices[0]->radial_zlut;

	if (zlut->data) {
		for (int i=0;i<zlut->count;i++) {
			float* img = &data[i*cfg.zlut_radialsteps*zlut->h];
			zlut->copyImageToHost(i, img);
		}
	}
}

void QueuedCUDATracker::GetRadialZLUTSize(int& count, int &planes, int& rsteps)
{
	count = devices[0]->radial_zlut.count;
	planes = devices[0]->radial_zlut.h;
	rsteps = cfg.zlut_radialsteps;
}

int QueuedCUDATracker::GetResultCount()
{
	resultMutex.lock();
	int r = resultCount;
	resultMutex.unlock();
	return r;
}

void QueuedCUDATracker::ClearResults()
{
	resultMutex.lock();
	results.clear();
	resultCount=0;
	resultMutex.unlock();
}


std::string QueuedCUDATracker::GetProfileReport()
{
	float f = 1.0f/batchesDone;

	return deviceReport + "Time profiling: [GPU], [CPU] \n" +
		SPrintf("%d batches done of size %d, on %d streams", batchesDone, batchSize, streams.size()) + "\n" +
		SPrintf("Image copying: %.2f,\t%.2f ms\n", time.imageCopy*f, cpu_time.imageCopy*f) +
		SPrintf("QI:            %.2f,\t%.2f ms\n", time.qi*f, cpu_time.qi*f) +
		SPrintf("COM:           %.2f,\t%.2f ms\n", time.com*f, cpu_time.com*f) +
		SPrintf("Z Computing:   %.2f,\t%.2f ms\n", time.zcompute*f, cpu_time.zcompute*f);
}


QueuedCUDATracker::ConfigValueMap QueuedCUDATracker::GetConfigValues()
{
	ConfigValueMap cvm;
	cvm["use_texturecache"] = useTextureCache ? "1" : "0";
	return cvm;
}

void QueuedCUDATracker::SetConfigValue(std::string name, std::string value)
{
	if (name == "use_texturecache")
		useTextureCache = atoi(value.c_str()) != 0;
}



