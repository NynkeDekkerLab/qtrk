#pragma once

#include "threads.h"

#include "QueuedTracker.h"
#include "cpu_tracker.h"

#include <list>

class QueuedCPUTracker : public QueuedTracker {
public:
	QueuedCPUTracker(const QTrkComputedConfig& cc);
	~QueuedCPUTracker();
	void Start();
	void Break(bool pause);
	void GenerateTestImage(float* dst, float xp, float yp, float z, float photoncount);
	int NumThreads() { return cfg.numThreads; }

	// QueuedTracker interface
	void SetLocalizationMode(LocMode_t lt) override;
	void SetRadialZLUT(float* data, int num_zluts, int planes) override;
	void GetRadialZLUT(float* zlut) override;
	void GetRadialZLUTSize(int& count ,int& planes, int& rsteps) override;
	void SetRadialWeights(float* rweights) override;
	void ScheduleLocalization(void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) override;

	void BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, uint flags, int plane) override;
	void FinalizeLUT() override;

	void GetImageZLUTSize(int* dims);
	void GetImageZLUT(float* dst);
	void SetImageZLUT(float* dst, float *radial_zlut, int* dims, float *rweights=0);

	void SetPixelCalibrationImages(float* offset, float* gain) override;
	void SetPixelCalibrationFactors(float offsetFactor, float gainFactor) override;

	int GetQueueLength(int *maxQueueLength=0) override; // In queue + in progress
	int FetchResults(LocalizationResult* results, int maxResults) override;
	void ClearResults() override;
	void Flush() override { };

	bool IsIdle() override;
	int GetResultCount() override;
	bool GetDebugImage(int id, int *w, int*h, float** data);

	ConfigValueMap GetConfigValues() override;
	void SetConfigValue(std::string name, std::string value) override;

	std::string GetProfileReport() { return "CPU tracker currently has no profile reporting"; }

private:
	struct Thread {
		Thread() { tracker=0; manager=0; thread=0;  mutex=0; }
		CPUTracker *tracker;
		Threads::Handle* thread;
		QueuedCPUTracker* manager;
		Threads::Mutex *mutex;

		void lock() { mutex->lock(); }
		void unlock(){ mutex->unlock(); }
	};

	struct Job {
		Job() { data=0; dataType=QTrkU8; }
		~Job() { delete[] data; }

		uchar* data;
		QTRK_PixelDataType dataType;
		LocalizationJob job;
	};

	LocMode_t localizeMode;
	Threads::Mutex jobs_mutex, jobs_buffer_mutex, results_mutex;
	std::deque<Job*> jobs;
	int jobCount;
	std::vector<Job*> jobs_buffer; // stores memory
	std::deque<LocalizationResult> results;
	int resultCount;
	int maxQueueSize;
	int jobsInProgress;

	Threads::Mutex gc_mutex;
	float *calib_gain, *calib_offset, gc_gainFactor, gc_offsetFactor;

	std::vector<Thread> threads;
	float* zluts;
	int zlut_count, zlut_planes;
	std::vector<float> zcmp;
	float* GetZLUTByIndex(int index) { return &zluts[ index * (zlut_planes*cfg.zlut_radialsteps) ]; }
	void UpdateZLUTs();

	int image_lut_dims[4], image_lut_nElem_per_bead;
	int ImageLUTNumBeads() { return image_lut_dims[0]; }
	int ImageLUTWidth() { return image_lut_dims[3]; }
	int ImageLUTHeight() { return image_lut_dims[2]; }
	float* image_lut, *image_lut_dz, *image_lut_dz2;

	float* GetImageLUTByIndex(int index, int plane=0) { 
		return &image_lut [ index * image_lut_nElem_per_bead + plane * (image_lut_dims[2]*image_lut_dims[3]) ]; 
	}

	// signal threads to stop their work
	bool quitWork, processJobs, dbgPrintResults;

	void JobFinished(Job* j);
	Job* GetNextJob();
	Job* AllocateJob();
	void AddJob(Job* j);
	void ProcessJob(Thread* th, Job* j);

	static void WorkerThreadMain(void* arg);
};

