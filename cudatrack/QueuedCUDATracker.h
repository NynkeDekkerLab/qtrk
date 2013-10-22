// CUDA implementation of QueuedTracker interface

// Thread-Safety:

// We assume 2 threads concurrently accessing the tracker functions:
//	- Queueing thread: ScheduleLocalization, SetRadialZLUT, GetRadialZLUT, Flush, IsQueueFilled, IsIdle
//	- Fetching thread: FetchResults, GetResultCount, ClearResults

#pragma once
#include "QueuedTracker.h"
#include "threads.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <list>
#include <vector>
#include <map>
#include "gpu_utils.h"
#include "cudaImageList.h"


template<typename T>
struct cudaImageList;
typedef cudaImageList<float> cudaImageListf;

struct LocalizationParams {
	int zlutIndex, zlutPlane; // if <0 then, no zlut for this job
};


struct BaseKernelParams {
	int njobs;
	LocalizationParams* locParams;
	float* imgmeans;
	cudaImageListf images;
};

#include "QI.h"

struct ZLUTParams {
	CUBOTH float* GetRadialZLUT(int bead, int plane) { return img.pixelAddress(0, plane, bead); }
	float minRadius, maxRadius;
	float* zcmpwindow;
	int angularSteps;
	int planes;
	cudaImageListf img;
	CUBOTH int radialSteps() { return img.w; }
	float2* trigtable; // precomputed radial directions (cos,sin pairs)
};

struct ImageLUTConfig
{
	static ImageLUTConfig empty() {
		ImageLUTConfig v;
		v.nLUTs = v.planes=v.w=v.h=0;
		v.xscale=v.yscale=1.0f;
		return v;
	}
	int nLUTs;
	int planes;
	int w, h;
	float xscale, yscale;

	int lutWidth() { return w * planes; }
	int lutNumPixels() { return w * h * planes; }
};

class QueuedCUDATracker : public QueuedTracker {
public:
	QueuedCUDATracker(const QTrkComputedConfig& cc, int batchSize=-1);
	~QueuedCUDATracker();
	void EnableTextureCache(bool useTextureCache) { this->useTextureCache=useTextureCache; }
	
	void SetLocalizationMode(LocMode_t locType) override;
	void ScheduleLocalization(void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) override;
	void ClearResults() override;

	// data can be zero to allocate ZLUT data.
	void SetRadialZLUT(float* data,  int numLUTs, int planes, float* zcmp=0) override; 
	void GetRadialZLUT(float* data) override; // delete[] memory afterwards
	void GetRadialZLUTSize(int& count, int& planes, int &radialSteps) override;
	int FetchResults(LocalizationResult* results, int maxResults) override;

	void GetImageZLUTSize(int* dims);
	void GetImageZLUT(float* dst);
	void SetImageZLUT(float* dst, int* dims);

	std::string GetProfileReport() override;

	// Force the current waiting batch to be processed. 
	void Flush() override;
	
	int GetQueueLength(int* maxQueueLen) override;
	bool IsIdle() override;
	int GetResultCount() override;

	void SetPixelCalibrationImages(float* offset,float* gain) override;
	void SetPixelCalibrationFactors(float offsetFactor, float gainFactor) override;

	ConfigValueMap GetConfigValues() override;
	void SetConfigValue(std::string name, std::string value) override;

protected:

	struct Device {
		Device(int index) {
			this->index=index; 
			radial_zlut=calib_offset=calib_gain=cudaImageListf::emptyList(); 
		}
		~Device(); 
		void SetRadialZLUT(float *data, int radialsteps, int planes, int numLUTs, float* zcmp);
		void SetPixelCalibrationImages(float* offset, float* gain, int img_width, int img_height);
		void SetImageLUT(float* data, ImageLUTConfig* cfg);
				
		cudaImageListf radial_zlut;
		cudaImageListf calib_offset, calib_gain;
		device_vec<float> zcompareWindow;
		QI::DeviceInstance qi_instance;
		device_vec<float2> zlut_trigtable;
		int index;
	};

	struct Stream {
		Stream(int streamIndex);
		~Stream();
		bool IsExecutionDone();
		void OutputMemoryUse();
		int JobCount() { return jobs.size(); }
		
		pinned_array<float3> results;
		pinned_array<float3> com;
		pinned_array<LocalizationParams> locParams;
		device_vec<LocalizationParams> d_locParams;
		std::vector<LocalizationJob> jobs;
		
		cudaImageListf images; 
//		pinned_array<float, cudaHostAllocWriteCombined> hostImageBuf; // original image format pixel buffer
		pinned_array<float> hostImageBuf; // original image format pixel buffer
		Threads::Mutex imageBufMutex;

		// CUDA objects
		cudaStream_t stream; // Stream used

		// Events
		cudaEvent_t localizationDone; // all done.
		// Events for profiling
		cudaEvent_t imageCopyDone, comDone, qiDone, zcomputeDone, imapDone, batchStart;

		// Intermediate data
		device_vec<float3> d_resultpos;
		device_vec<float3> d_com; // z is zero

		QI::StreamInstance qi_instance;

		device_vec<float> d_imgmeans; // [ njobs ]
		device_vec<float> d_radialprofiles;// [ radialsteps * njobs ] for Z computation
		device_vec<float> d_zlutcmpscores; // [ zlutplanes * njobs ]

		uint localizeFlags; // Indicates whether kernels should be ran for building zlut, z computing, or QI
		Device* device;

		enum State {
			StreamIdle,
			StreamPendingExec,
			StreamExecuting
		};
		State state;
	};

	int numThreads;
	int batchSize;

	ImageLUTConfig imageLUTConfig;

	dim3 blocks(int workItems) { return dim3((workItems+numThreads-1)/numThreads); }
	dim3 blocks() {	return dim3((batchSize+numThreads-1)/numThreads); }
	dim3 threads() { return dim3(numThreads); }

	std::vector<Stream*> streams;
	std::list<LocalizationResult> results;
	int resultCount;
	LocMode_t localizeMode;
	Threads::Mutex resultMutex, jobQueueMutex;
	std::vector<Device*> devices;
	bool useTextureCache; // speed up using texture cache. 
	float gc_offsetFactor, gc_gainFactor;
	Threads::Mutex gc_mutex;

	QI qi;
	cudaDeviceProp deviceProp;
	ZLUTParams zlutParams;

	Threads::Handle *schedulingThread;
	Atomic<bool> quitScheduler;
	void SchedulingThreadMain();
	static void SchedulingThreadEntryPoint(void *param);

	template<typename TImageSampler> void ExecuteBatch(Stream *s);
	Stream* GetReadyStream(); // get a stream that not currently executing, and still has room for images
	void InitializeDeviceList();
	Stream* CreateStream(Device* device, int streamIndex);
	void CopyStreamResults(Stream* s);
	void StreamUpdateZLUTSize(Stream *s);
	void CPU_ApplyGainCorrection(Stream *s);

public:
	// Profiling
	struct KernelProfileTime {
		KernelProfileTime() {com=qi=imageCopy=zcompute=imap=getResults=0.0;}
		double com, qi, imageCopy, zcompute, imap, getResults;
	};
	KernelProfileTime time, cpu_time;
	int batchesDone;
	std::string deviceReport;
};


