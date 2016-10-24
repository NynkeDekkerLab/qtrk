#pragma once

#include "threads.h"

#include "QueuedTracker.h"
#include "cpu_tracker.h"

#include <list>

/*! \brief CPU implementation of the QueuedTracker interface.

Creates and maintains multiple threads (one per available core by default)
and schedules localizations over them. Each thread has its own CPUTracker instance which
provides the actual calculations.
*/
class QueuedCPUTracker : public QueuedTracker {
public:
	/*! \brief Create a QueuedCPUTracker instance with specified configuration.
	
	\param [in] cc Computed config to use.

	\return A QueuedCPUTracker instance.
	*/
	QueuedCPUTracker(const QTrkComputedConfig& cc);

	/*! \brief Delete the instance and free related memory. */
	~QueuedCPUTracker();

	/*! \brief Start up the threads and initiate their [worker function](\ref QueuedCPUTracker::WorkerThreadMain).*/
	void Start();

	/*! \brief Enable or disable handling of jobs.
	
	\param [in] pause True stops new jobs from being started.
	*/
	void Break(bool pause);

	/*! \brief Generate a test image based on a the point spread function.

	Creates an ImageData object and calls \ref GenerateTestImage. Automatically uses ROI size from QtrkComputedConfig::width and QtrkComputedConfig::height.

	\param [out] dst		Pre-allocated float array in which to save the image.
	\param [in]  xp			X coordinate of the PSF center.
	\param [in]  yp			Y coordinate of the PSF center.
	\param [in]  size		Spread of the PSF. Can't be zero. Higher size increases radius of created diffraction pattern.
	\param [in]  SNratio	Signal to noise ratio.
	*/
	void GenerateTestImage(float* dst, float xp, float yp, float size, float SNratio);

	/*! \brief Get the used number of threads.

	\return The number of threads.
	*/
	int NumThreads() { return cfg.numThreads; }

	// QueuedTracker interface
	void SetLocalizationMode(LocMode_t lt) override;
	void SetRadialZLUT(float* data, int num_zluts, int planes) override;
	void GetRadialZLUT(float* zlut) override;
	void GetRadialZLUTSize(int& count ,int& planes, int& rsteps) override;
	void SetRadialWeights(float* rweights) override;

	/*! \brief Overload of \ref SetRadialWeights to use a vector.
	
	\param [in] weights Float vector with the radial weights.
	*/
	void SetRadialWeights(std::vector<float> weights) { SetRadialWeights(&weights[0]); }
	void ScheduleLocalization(void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) override;

	void EnableRadialZLUTCompareProfile(bool enabled);
	void GetRadialZLUTCompareProfile(float* dst); // dst = [count * planes]

	/*! \brief Test the planes saved in the lookup table against itself.

	Implemented for testing purposes. Writes the output to a CSV file.
	Can be used to inspect ZLUT quality.
	*/
	void ZLUTSelfTest();

	void BeginLUT(uint flags);
	void BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, int plane, vector2f* known_pos=0) override;
	void FinalizeLUT() override;

	void GetImageZLUTSize(int* dims);
	void GetImageZLUT(float* dst);
	bool SetImageZLUT(float* dst, float *radial_zlut, int* dims);

	void SetPixelCalibrationImages(float* offset, float* gain) override;
	void SetPixelCalibrationFactors(float offsetFactor, float gainFactor) override;

	int GetQueueLength(int *maxQueueLength=0) override;
	int FetchResults(LocalizationResult* dstResult, int maxResults) override;
	void ClearResults() override;
	void Flush() override { };

	bool IsIdle() override;
	int GetResultCount() override;

	/*! \brief Get the debug image of a tracker.

	\param [in] id		ID of the tracker (= threadID).
	\param [out] w		Reference to an int in which the image width will be stored.
	\param [out] h		Reference to an int in which the image height will be stored.
	\param [out] data	Pointer to a float array that will be filled with the image.

	\return True if the call was valid.
	*/
	bool GetDebugImage(int id, int *w, int*h, float** data);

	ConfigValueMap GetConfigValues() override;
	void SetConfigValue(std::string name, std::string value) override;

	std::string GetProfileReport() { return "CPU tracker currently has no profile reporting"; }

	/*! \brief Get a pointer to the starting pixel of the specified LUT.

	LUTs are saved in one large contiguous memory region. This function returns a reference to a specific LUT.

	\param [in] index Index of the requested LUT.

	\return Pointer to the LUT starting pixel
	*/
	float* GetZLUTByIndex(int index) { return &zluts[ index * (zlut_planes*cfg.zlut_radialsteps) ]; }
private:

	/*! \brief Structure to facilitate the use of threads. */
	struct Thread {
		Thread() { tracker=0; manager=0; thread=0;  mutex=0; }
		CPUTracker *tracker;			///< The tracker for this thread.
		Threads::Handle* thread;		///< Handle to the actual OS thread.
		QueuedCPUTracker* manager;		///< Parent QueuedCPUTracker instance.
		Threads::Mutex *mutex;			///< The mutex for this thread.

		void lock() { mutex->lock(); }		///< Lock this thread's mutex.
		void unlock(){ mutex->unlock(); }	///< Unlock this thread's mutex.
	};

	/*! \brief Structure around \ref LocalizationJob to link data and job. */
	struct Job {
		Job() { data=0; dataType=QTrkU8; }
		~Job() { delete[] data; }

		uchar* data;					///< Pointer to the location in memory where the ROI for this job is located.
		QTRK_PixelDataType dataType;	///< Data type of the image.
		LocalizationJob job;			///< The regular LocalizationJob job information.
	};

	LocMode_t localizeMode;				///< Localization settings.
	Threads::Mutex jobs_mutex;			///< Mutex for the job queue.
	Threads::Mutex jobs_buffer_mutex;	///< Mutex for the job buffer.
	Threads::Mutex results_mutex;		///< Mutex for the results.
	std::deque<Job*> jobs;				///< Queue to hold the jobs.

	/*! \brief Number of jobs in the queue.

	\note Why maintain this manually and not just use jobs.size()?
	*/
	int jobCount;						
	std::vector<Job*> jobs_buffer;		///< Stores memory. Enables reuse of allocated \ref Job memory after a job has been completed.
	std::deque<LocalizationResult> results; ///< Queue to store the localization results.
	int resultCount;					///< Number of results available.
	int maxQueueSize;					///< Maximum number of jobs in the queue.
	int jobsInProgress;					///< Number of jobs currently being processed.

	Threads::Mutex gc_mutex;			///< Image calibration mutex
	float *calib_gain;					///< \copydoc QueuedCUDATracker::gc_gain
	float *calib_offset;				///< \copydoc QueuedCUDATracker::gc_offset
	float gc_gainFactor;				///< \copydoc QueuedCUDATracker::gc_gainFactor
	float gc_offsetFactor;				///< \copydoc QueuedCUDATracker::gc_offsetFactor

	int downsampleWidth;				///< Width of ROIs after downsampling.
	int downsampleHeight;				///< Height of ROIs after downsampling.

	std::vector<Thread> threads;		///< Vector with active threads.

	/*! \brief Pointer to the first pixel of the Z lookup tables.

	All LUTs are saved in one big contiguous section of memory. 
	Calculate specific LUTs or planes based on their indexes. Order is [beadIndex][plane][step].
	See \ref GetZLUTByIndex.
	*/	
	float* zluts;
	float* zlut_cmpprofiles;			///< Array in which to save errorcurves if enabled with \ref EnableRadialZLUTCompareProfile.
	bool zlut_enablecmpprof;			///< Flag to save errorcurves. See \ref EnableRadialZLUTCompareProfile.
	int zlut_count;						///< Number of ZLUTs (= number of beads).
	int zlut_planes;					///< Number of planes per ZLUT.
	uint zlut_buildflags;				///< ZLUT build flags, set through \ref BeginLUT.

	std::vector<float> zcmp;				///< Scaling factors for the ZLUT algorithm.
	std::vector<float> qi_radialbinweights;	///< Scaling factors for the QI algorithm.
	
	void UpdateZLUTs();					///< Update the ZLUTs for all threads.

	int image_lut_dims[4];				///< Image LUT dimensions, 4D. [numBeads][numPlanes][height][width].
	int image_lut_nElem_per_bead;		///< Image LUT number of pixels in LUT per bead. Is Width * Height * Planes.
	int ImageLUTNumBeads() { return image_lut_dims[0]; }	///< Return the number of beads in the image LUT.
	int ImageLUTWidth() { return image_lut_dims[3]; }		///< Return the height of the image LUT.
	int ImageLUTHeight() { return image_lut_dims[2]; }		///< Return the width of the image LUT.
	float* image_lut;					///< Image LUT data pointer.
	float* image_lut_dz;				///< Image LUT first derivative. (???)
	float* image_lut_dz2;				///< Image LUT second derivative. (???)

	/*! \brief Get a specific image LUT or plane from an image LUT.

	\param [in] index The beadindex of the requested LUT.
	\param [in] plane Index of the requested plane. 0 by default.

	\return Pointer to the first element of the requested LUT or plane.
	*/
	float* GetImageLUTByIndex(int index, int plane=0) { 
		return &image_lut [ index * image_lut_nElem_per_bead + plane * (image_lut_dims[2]*image_lut_dims[3]) ]; 
	}

	bool quitWork;				///< Signal threads to stop their work.
	bool processJobs;			///< Flag for threads to continue processing jobs.
	bool dbgPrintResults;		///< Flag to enable easy printing of intermediate data.

	/*! \brief Flag job memory for reuse.

	See \ref jobs_buffer. 
	\warning Only call when a job is fully finished, including results copying.

	\param [in] j The job to free.
	*/
	void JobFinished(Job* j);

	/*! \brief Get the next job in the queue.

	\return The next job in the queue.
	*/
	Job* GetNextJob();

	/*! \brief Allocate job memory.

	Tries to re-use memory of finished jobs before allocating new memory.

	\return Pointer to a usable job instance.
	*/
	Job* AllocateJob();

	/*! \brief Add a job to the queue.

	\param [in] j The job to add.
	*/
	void AddJob(Job* j);

	/*! \brief Process a job.

	\param [in] th The thread on which to execute.
	\param [in] j The job to execute.
	*/
	void ProcessJob(Thread* th, Job* j);

	/*! \brief Copy the ROI to the tracker's memory.
	
	\param [in] trk The tracker for which to set the image.
	\param [in] job The job this tracker will execute. Image data is taken from the job.
	*/
	void SetTrackerImage(CPUTracker* trk, Job *job);

	/*! \brief Calibrate an image.
	
	Calibrates the image set on a tracker through \ref SetTrackerImage.

	\param [in] trk The tracker whose image to calibrate.
	\param [in] beadIndex The bead this image belongs to.
	*/
	void ApplyOffsetGain(CPUTracker* trk, int beadIndex);

	/*! \brief The loop executed by the threads.

	Will start execution of a new job whenever available and allowed.

	\param [in] arg Reference to the \ref QueuedCPUTracker::Thread that will execute this loop.
	*/
	static void WorkerThreadMain(void* arg);
};