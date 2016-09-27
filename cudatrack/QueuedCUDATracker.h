/*! \page CUDAReadings List of CUDA references

General introductions:
- David A. Patterson and John L. Hennessy. Computer Organization and Design, chapter Appendix A: Graphics and Computing GPUs. Morgan Kaufmann, 5th edition, 2013. \cite book:CUDA
- https://devblogs.nvidia.com/parallelforall/
	-# https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/
- http://www.graphics.stanford.edu/~hanrahan/talks/why/walk001.html

Specific optimimizations:
- http://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf
- http://cuda-programming.blogspot.nl/2013/02/texture-memory-in-cuda-what-is-texture.html
- https://devblogs.nvidia.com/parallelforall/
	-# https://devblogs.nvidia.com/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/
	-# https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
	-# https://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/
	-# https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/

Reference:
- http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- http://docs.nvidia.com/cuda/cufft/

CUDA Version related:
- https://devtalk.nvidia.com/default/topic/858507/cuda-setup-and-installation/fft-libraries-for-win32-cuda-7-0-missing-/
- https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
- http://docs.roguewave.com/totalview/8.14.1/html/index.html#page/User_Guides/totalviewug-about-cuda.31.4.html

*/
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

class CPUTracker;

template<typename T>
struct cudaImageList;
typedef cudaImageList<float> cudaImageListf;

/// Structure to hold further information relevant for a localization job. Mostly to enable additions without needing to change kernel calls.
struct LocalizationParams {
	int zlutIndex;		///< Index of the bead/zlut.
	int zlutPlane;		///< Plane number for LUT lookup. If <0 then, no zlut for this job. \b Not \b used.
};

/// Structure to group parameters to be passed to every kernel
struct BaseKernelParams {
	int njobs;						///< Number of jobs in the batch.
	LocalizationParams* locParams;	///< Additional localization parameters. Holds the ZLUT/bead index.
	float* imgmeans;				///< Array of image means.
	cudaImageListf images;			///< The images for this batch.
};

#include "QI.h"

/// Structure to group parameters about the ZLUT
struct ZLUTParams {
	/*! \brief Get the radial profile saved in the LUT for a particular bead and plane.

	\param [in] bead	The bead for which to get the profile.
	\param [in] plane	The plane to retrieve.
	*/
	CUBOTH float* GetRadialZLUT(int bead, int plane) { return img.pixelAddress(0, plane, bead); }
	float minRadius;		///< Radius in pixels of the starting point of the sampling area
	float maxRadius;		///< Maximum radial distance in pixels of the sampling area
	float* zcmpwindow;		///< The radial weights to use for the error curve calculation
	int angularSteps;		///< The number of angular steps used to generate the lookup table
	int planes;				///< The number of planes per lookup table
	cudaImageListf img;		///< The imagelist holding the LUTs
	CUBOTH int radialSteps() { return img.w; }	///< Number of radial steps in the lookup table
	float2* trigtable;		///< Array of precomputed radial spoke sampling points (cos,sin pairs)
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
};

/*! \brief CUDA implementation of the QueuedTracker interface.

Exploits the inherent parallel computing power of GPU hardware to increase the speed at which tracking can be performed.
Optimizations will be listed on a per-function basis.
Speeds of almost 40000 100x100 ROI/s with 5 %QI iterations and Z localization have been achieved on a setup with a GTX1080 card.

The currently implemented tracking algorithms are COM, Quadrant Interpolation and 2D Gaussian with Max-Likelihood estimation.

It will automatically use all available CUDA devices if using the \p QTrkCUDA_UseAll value for QTrkSettings::cuda_device.

Method:
- \ref ScheduleLocalization - Load images into host-side image buffer
- \ref SchedulingThreadMain - Scheduling thread executes any batch that is filled
- \ref ExecuteBatch - Running batch 
	-# Async copy host-side buffer to device
	-# Bind image to texture memory
	-# Run COM kernel
	-# QI loop: 
		-# Run QI kernels: Sample from texture into quadrant profiles
		-# Run CUFFT. Each iteration per axis does 2x forward FFT, and 1x backward FFT.
		-# Run QI kernel: Compute positions
	-# Compute ZLUT profiles
	-# Depending on localize flags:
		-# copy ZLUT profiles (for ComputeBuildZLUT flag)
		-# generate compare profile kernel + compute Z kernel (for ComputeZ flag)
	-# Unbind image from texture memory
	-# Async copy results to host

Thread-Safety:

We assume 2 threads concurrently accessing the tracker functions.
- Queueing thread: \ref ScheduleLocalization, \ref SetRadialZLUT, \ref GetRadialZLUT, \ref Flush, \ref IsIdle
- Fetching thread: \ref FetchResults, \ref GetResultCount, \ref ClearResults
- Mutexes:
	-# \ref jobQueueMutex - controlling access to state and jobs. 
		Used by ScheduleLocalization, scheduler thread, and GetQueueLen
	-# \ref resultMutex - controlling access to the results list, 
		Locked by the scheduler whenever results come available, and by calling threads when they run GetResults/Count

Issues:
- Due to FPU operations on texture coordinates, there are small numerical differences between localizations of the same image at a different position in the batch
*/
class QueuedCUDATracker : public QueuedTracker {
public:
	/*! \brief Initialize a QueuedCUDATracker instance.

	\param [in] cc The settings to be used.
	\param [in] batchSize The number of ROIs to be handled in one batch. Default (-1) calculates optimum. See \ref batchSize for optimization info.

	\throws runtime_error Init error: GPU does not support CUDA capability 2.0 or higher.
	\throws runtime_error Init error: Failed to create GPU streams.
	*/
	QueuedCUDATracker(const QTrkComputedConfig& cc, int batchSize=-1);

	/*! \brief Delete an instance of a CUDA Tracker. */
	~QueuedCUDATracker();

	/*! \brief Enable or disable the use of textures.

	Texture cache provides a significant speedup due to 2D L1 caching as compared to normal linear memory caching (\ref CUDAReadings).

	\param [in] useTextureCache Boolean to enable texture cache usage.
	*/
	void EnableTextureCache(bool useTextureCache) { this->useTextureCache=useTextureCache; }
	
	// Below are override fucntions which have been documented in QueuedTracker.h
	void SetLocalizationMode(LocMode_t locType) override;
	void ScheduleLocalization(void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) override;
	void ClearResults() override;

	void SetRadialZLUT(float* data,  int numLUTs, int planes) override; 
	void SetRadialWeights(float *zcmp) override;
	void GetRadialZLUT(float* data) override; // delete[] memory afterwards
	void GetRadialZLUTSize(int& count, int& planes, int &radialSteps) override;
	int FetchResults(LocalizationResult* results, int maxResults) override;

	void BeginLUT(uint flags) override;
	/*!
	\copydoc QueuedTracker::BuildLUT
	\note Uses the CPU tracker for LUT building!
	*/
	void BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, int plane, vector2f* known_pos = 0) override;
	void FinalizeLUT() override;

	void EnableRadialZLUTCompareProfile(bool enabled) {}
	void GetRadialZLUTCompareProfile(float* dst) {} // dst = [count * planes]
	
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

	/*! \brief Datatype to hold image lookup tables. */
	typedef Image4DMemory<float> ImageLUT;

protected:
	/*! \brief Structure to maintain data for each GPU. 
	
	Holds global, once-calculated values required by the algorithms.

	\todo Explicitly put this in global memory?
	*/
	struct Device {
		/*! \brief Initialize a GPU/CUDA Device.

		\param [in] index Index of the device pertaining to this instance.
		*/
		Device(int index) 
		{
			this->index=index; 
			radial_zlut=calib_offset=calib_gain=cudaImageListf::emptyList(); 
		}
		~Device(); 

		/*! \brief Copy the ZLUT data to device memory.

		\param [in] data		Pointer to the start of the ZLUT data.
		\param [in] radialsteps Number of radialsteps per LUT plane
		\param [in] planes		Number of planes per ZLUT.
		\param [in]	numLUTs		Number of ZLUTs in the dataset.
		*/
		void SetRadialZLUT(float *data, int radialsteps, int planes, int numLUTs);

		/*! \brief Set pixel calibration images.

		See \ref QueuedTracker::SetPixelCalibrationImages for more info.

		\param [in] offset		(2D) Array with the per-pixel offsets to use.
		\param [in] gain		(2D) Array with the per-pixel gain to use.
		\param [in] img_width	Width of the full image.
		\param [in]	img_height	Height of the full image.
		*/
		void SetPixelCalibrationImages(float* offset, float* gain, int img_width, int img_height);

		/*! \brief Set the radial weights used to calculate ZLUT plane match scores.

		The radial weights are used by \ref ZLUT_ComputeProfileMatchScores, if set.

		\param [in] zcmp Array with the weights to be used. Size has to equal QtrkComputedConfig::radialsteps.
		*/
		void SetRadialWeights(float* zcmp);
		
		cudaImageListf radial_zlut;				///< List in which to keep the lookup table images.
		cudaImageListf calib_offset;			///< List in which to keep the offset images.
		cudaImageListf calib_gain;				///< List in which to keep the gain images.
		device_vec<float> zcompareWindow;		///< Vector in which to keep the radial weights for error curve calculation.
		QI::DeviceInstance qi_instance;			///< Instance of global values used by the QI submodule.
		QI::DeviceInstance qalign_instance;		///< Instance of global values used by the QI submodule used to perform %QI alignment.
		device_vec<float2> zlut_trigtable;		///< Vector of pre-calculated 2D sampling points.
		int index;								///< Device index of the device this instance is located on.
	};

	/*! \brief Structure to maintain data for each stream. Shell around CUDA native streams.
	
	Typically, there are 4 streams per available GPU.

	Streams are used to discretize all GPU transactions (memory transfer to device, calculations, transfer from device) into
	bigger batches to increase efficiency. Each stream has its own job queue, pre-allocated memory for a whole batch, 
	and their batches can be executed individually from one another. On newer devices, streams can queue and 
	run their operations concurrently, leading to higher effective calculation speeds by overlapping memory 
	transfers and calculations. Host variables are maintained in 
	<a href="https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/">pinned memory</a> 
	to optimize transfer speeds.

	QueuedCUDATracker::ScheduleLocalization finds a currently available stream and adds the new job to its queue.
	When a stream's state is set to \ref StreamPendingExec, it is automatically executed by the scheduling thread \ref SchedulingThreadMain.
	*/
	struct Stream {
		/*! \brief Constructor. Don't use directly, use \ref CreateStream instead.

		\param [in] streamIndex Index used of mutex name.
		*/
		Stream(int streamIndex);

		/*! \brief Delete the stream instance. */
		~Stream();

		/*! \brief Test to see if the stream's batch is finished.
		
		\note Also always returns true when not in \ref StreamExecuting.

		\return Boolean flag indicating whether execution is done.
		*/
		bool IsExecutionDone();

		/*! \brief Prints the device and host allocated memory usage of a single stream to debug output. */
		void OutputMemoryUse();

		/*! \brief Get the number of jobs currently in this stream's queue.
		
		The maximum queue size is \ref batchSize.

		\return The number of jobs in the queue.
		*/
		int JobCount() { return jobs.size(); }
		
		pinned_array<float3> results;				///< 3D array in pinned host memory with the localization result values.
		pinned_array<float3> com;					///< 3D array in pinned host memory with the center of mass results.
		pinned_array<float> imgMeans;				///< Array in pinned host memory with the ROI intensity means.
		pinned_array<LocalizationParams> locParams;	///< Array in pinned host memory with additional localization parameters.
		device_vec<LocalizationParams> d_locParams; ///< Array in device memory with additional localization parameters.
		std::vector<LocalizationJob> jobs;			///< Vector of jobs. Filled with jobs then all jobs are executed at once and the vector is cleared.
		
		cudaImageListf images;						///< Image list of all images belonging to the queued jobs.
//		pinned_array<float, cudaHostAllocWriteCombined> hostImageBuf; // original image format pixel buffer
		pinned_array<float> hostImageBuf;			///< Buffer in pinned host memory holding the images for the image list \p images.
		Threads::Mutex imageBufMutex;				///< Mutex for accesses to the images in memory.

		// CUDA objects
		cudaStream_t stream; // Stream used			///< The actual CUDA stream instance for this shell.

		// Events
		cudaEvent_t localizationDone;				///< CUDA event used to determine when a batch is finished.
		// Events for profiling
		cudaEvent_t imageCopyDone;					///< CUDA event for profiling of image copies.
		cudaEvent_t comDone;						///< CUDA event for profiling of the center of mass algorithm.
		cudaEvent_t qiDone;							///< CUDA event for profiling of the quadrant interpolation algorithm.
		cudaEvent_t qalignDone;						///< CUDA event for profiling of the quadrant align algorithm.
		cudaEvent_t zcomputeDone;					///< CUDA event for profiling of the z localization.
		cudaEvent_t batchStart;						///< CUDA event to record the start of a batch.

		// Intermediate data
		device_vec<float3> d_resultpos;				///< 3D vector in device memory to hold intermediate results.
		device_vec<float3> d_com; // z is zero		///< 3D vector in device memory to hold COM results. COM only performs 2D localizations, so \p z is always 0.

		QI::StreamInstance qi_instance;				///< Linked stream of the QI submodule.
		QI::StreamInstance qalign_instance;			///< Linked stream of the QI submodule used to perform quadrant alignment.

		device_vec<float> d_imgmeans;				///< Vector in device memory to hold ROI means.
		device_vec<float> d_radialprofiles;			///< Vector in device memory to hold all calculated radial profiles. Size is [ radialsteps * njobs ].
		device_vec<float> d_zlutcmpscores;			///< Vector in device memory to hold all calculated error curves. Size is [ zlutplanes * njobs ].

		uint localizeFlags;							///< Flags for localization choices. See \ref LocalizeModeEnum.
		Device* device;								///< Reference to the device instance this stream should run on.

		/*! \brief Possible stream states.
		
		\todo Why is there no StreamDoneExec state?
		*/
		enum State {
			StreamIdle,			///< The Stream is idle and can accept more jobs. In other words, the queue is not full.
			StreamPendingExec,	///< The Stream is ready to be executed. That is, the queue is full and the batch is ready or \ref Flush was called.
			StreamExecuting		///< The Stream is currently active on the GPU and executing its batch.
		};
		State state;			///< The state flag for the stream.
	};

	/*!	\brief Amount of images to be sent at once per stream.

	Higher batchsize = higher speeds. 
	Reason why it's faster:
	1. More threads & blocks are created, allowing more efficient memory latency hiding by warp switching, or in other words, higher occupancy is achieved.
	2. Bigger batches are copied at a time, achieving higher effective PCIe bus bandwidth.
	*/
	int batchSize;

	/*! \brief Number of threads to use in a general thread block.
	
	Used by the \ref blocks and \ref threads functions to quickly calculate parameterspace covering kernel execution dimensions.
	*/
	int numThreads;

	/*! \brief Calculate the number of thread blocks of size \ref numThreads needed to have 1 thread per job. 
	
	Use in conjunction with \ref threads as: 
	\code
	kernel <<< blocks(), threads() >>> (a, b);
	\endcode

	\return CUDA 3D data for use in kernel calls.
	*/
	dim3 blocks() {	return dim3((batchSize+numThreads-1)/numThreads); }
	/*! \brief Calculate the number of thread blocks of size \ref numThreads needed to have 1 thread per \p workItems. 
	
	Use in conjunction with \ref threads as: 
	\code
	kernel <<< blocks(items), threads() >>> (a, b);
	\endcode

	\param [in] workItems Number of total threads needed.

	\return CUDA 3D data for use in kernel calls.
	*/
	dim3 blocks(int workItems) { return dim3((workItems+numThreads-1)/numThreads); }

	/*! \brief Get the CUDA native datatype with the threadblock dimensions to use.

	Use in conjunction with \ref blocks as: 
	\code
	kernel <<< blocks(), threads() >>> (a, b);
	\endcode

	\return CUDA 3D data for use in kernel calls.
	*/
	dim3 threads() { return dim3(numThreads); }

	std::vector<Stream*> streams;				///< Vector of usable streams.
	std::list<LocalizationResult> results;		///< Vector of completed results.
	int resultCount;							///< Number of results available.
	LocMode_t localizeMode;						///< Flags for localization choices. See \ref LocalizeModeEnum.
	Threads::Mutex resultMutex;					///< Mutex for result memory accesses.
	Threads::Mutex jobQueueMutex;				///< Mutex for job queue accesses.
	std::vector<Device*> devices;				///< Vector of device instances used.
	bool useTextureCache;						///< Flag to use texture cache. Default is true. Disable using \ref EnableTextureCache.
	float gc_offsetFactor;						///< Factor by which to scale the pixel calibration offset.
	float gc_gainFactor;						///< Factor by which to scale the gain.
	std::vector<float> gc_offset;				///< Vector with offsets used for pixel correction.
	std::vector<float> gc_gain;					///< Vector with gains used for pixel correction.
	Threads::Mutex gc_mutex;					///< Mutex for pixel calibration operations.
	uint zlut_build_flags;						///< Flags for ZLUT building. Not actually used yet.

	QI qi;										///< The QI instance used to perform the 2D localization using the quadrant interpolation algorithm.
	QI qalign;									///< Instance of QI used specifically for quadrant alignment.
	cudaDeviceProp deviceProp;					///< Variable used to save device properties obtained from the CUDA API.

	Threads::Handle *schedulingThread;			///< Handle to the scheduling thread for later reference.
	Atomic<bool> quitScheduler;					///< Thread shutdown flag with built-in thread safety (atomic).
	void SchedulingThreadMain();				///< Loop executed by the scheduling thread which executes threads when needed.
	static void SchedulingThreadEntryPoint(void *param); ///< Entry point for thread creation.

	/*! \brief Execute the queued batch on a stream.

	This entails the following steps:

	-# Async copy host-side buffer to device
	-# Bind image to texture memory
	-# Run COM kernel
	-# QI loop: 
		-# Run QI kernels: Sample from texture into quadrant profiles
		-# Run CUFFT. Each iteration per axis does 2x forward FFT, and 1x backward FFT.
		-# Run QI kernel: Compute positions
	-# Compute ZLUT profiles
	-# Depending on localize flags:
		-# copy ZLUT profiles (for ComputeBuildZLUT flag)
		-# generate compare profile kernel + compute Z kernel (for ComputeZ flag)
	-# Unbind image from texture memory
	-# Async copy results to host

	\param [in] s Pointer to the stream to execute.
	*/
	template<typename TImageSampler> 
	void ExecuteBatch(Stream *s);

	Stream* GetReadyStream();								///< Get a stream that is not currently executing, and still has room for images
	void InitializeDeviceList();							///< Build the list of devices to be used based on the QTrkSettings::cuda_device flag.
	
	/*! \brief Initialize a stream instance.

	\param [in] device		Pointer to the device instance this stream runs on.
	\param [in] streamIndex The number of this stream.
	*/
	Stream* CreateStream(Device* device, int streamIndex);

	/*! \brief Copy localization results from device to host memory. Also updates profiling times.

	\param [in] s The stream from which to copy results.
	*/
	void CopyStreamResults(Stream* s);

	/*! \brief Update zlut vector dimensions. Use when settings change.

	\param [in] s The stream for which to update.
	*/
	void StreamUpdateZLUTSize(Stream *s);

	/*! \brief Use the CPU-based tracker to apply the pixel calibrations.

	This is only used for LUT building because that entirely still runs on the CPU (because speed is not a limit).

	\param [in] trk			Instance of CPUTracker to use
	\param [in] beadIndex	Number of the ROI to scale
	*/
	void CPU_ApplyOffsetGain(CPUTracker* trk, int beadIndex);


public:
	/// Structure used to hold profiling data.
	struct KernelProfileTime {
		KernelProfileTime() {com=qi=imageCopy=zcompute=zlutAlign=getResults=0.0;}
		double com, qi, imageCopy, zcompute, zlutAlign, getResults;
	};
	KernelProfileTime time, cpu_time;
	int batchesDone;			///< Number of fully completed batches.

	std::string deviceReport;	///< String holding a human-readable description of used GPUs. Filled during \ref InitializeDeviceList.
};


