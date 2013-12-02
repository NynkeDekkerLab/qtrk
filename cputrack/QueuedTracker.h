
// Defines the interface for trackers (both CPU and CUDA)

#pragma once

#include "std_incl.h" 
#include "threads.h"
#include <map>


#include "qtrk_c_api.h"

struct ImageData;

// minimum number of samples for a profile radial bin. Below this the image mean will be used
#define MIN_RADPROFILE_SMP_COUNT 4


// Abstract tracker interface, implementated by QueuedCUDATracker and QueuedCPUTracker
class QueuedTracker
{
public:
	QueuedTracker();
	virtual ~QueuedTracker();

	virtual void SetLocalizationMode(LocMode_t locType) = 0;

	// These are per-bead! So both gain and offset are sized [width*height*numbeads], similar to ZLUT
	// result=gain*(pixel+offset)
	virtual void SetPixelCalibrationImages(float* offset, float* gain) = 0;
	virtual void SetPixelCalibrationFactors(float offsetFactor, float gainFactor) = 0;

	// Frame and timestamp are ignored by tracking code itself, but usable for the calling code
	// Pitch: Distance in bytes between two successive rows of pixels (e.g. address of (0,0) -  address of (0,1) )
	// ZlutIndex: Which ZLUT to use for ComputeZ/BuildZLUT
	virtual void ScheduleLocalization(void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) = 0;
	virtual void ClearResults() = 0;
	virtual void Flush() = 0; // stop waiting for more jobs to do, and just process the current batch

	// Schedule an entire frame at once, allowing for further optimizations
	virtual int ScheduleFrame(void *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo);
	
	// data can be zero to allocate ZLUT data. zcmp has to have 'zlut_radialsteps' elements
	virtual void SetRadialZLUT(float* data, int count, int planes, float* zcmp=0) = 0; 
	virtual void GetRadialZLUT(float* dst) = 0; 
	virtual void GetRadialZLUTSize(int& count, int& planes, int& radialsteps) = 0;

	// dims = [ count, planes, height, width ]  (Just like how the data is ordered)
	virtual void GetImageZLUTSize(int* dims) {}
	virtual void GetImageZLUT(float* dst) {}
	virtual void SetImageZLUT(float* dst, float *radial_zlut, int* dims, float *rweights=0) {}

	virtual void BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, uint flags, int plane) = 0;
	virtual void FinalizeLUT() = 0;
	
	virtual int GetResultCount() = 0;
	virtual int FetchResults(LocalizationResult* results, int maxResults) = 0;

	virtual int GetQueueLength(int *maxQueueLen=0) = 0;
	virtual bool IsIdle() = 0;

	virtual void SetConfigValue(std::string name, std::string value) = 0;
	typedef std::map<std::string, std::string> ConfigValueMap;
	virtual ConfigValueMap GetConfigValues() = 0;

	virtual std::string GetProfileReport() { return ""; }
	virtual std::string GetWarnings() { return ""; }

	virtual bool GetDebugImage(int ID, int *w, int *h, float** pData) { return false; } // deallocate result with delete[] 
	ImageData DebugImage(int ID);

	QTrkComputedConfig cfg;

	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, uint frame, uint timestamp, vector3f* initial, uint zlutIndex, uint zlutPlane);
};

void CopyImageToFloat(uchar* data, int width, int height, int pitch, QTRK_PixelDataType pdt, float* dst);
QueuedTracker* CreateQueuedTracker(const QTrkComputedConfig& cc);
void SetCUDADevices(int *devices, int numdev); // empty for CPU tracker



// Polynomial least-square fit weights, used for Z and QI fitting
// Changes to this require rebuild of code
#define QI_LSQFIT_WEIGHTS { 0.14f, 0.5f, 0.85f, 1.0f, 0.85f, 0.5f, 0.14f }
#define QI_LSQFIT_NWEIGHTS 7

#define ZLUT_LSQFIT_WEIGHTS { 0.5f, 0.85f, 1.0f, 0.85f, 0.5f }
#define ZLUT_LSQFIT_NWEIGHTS 5


inline int PDT_BytesPerPixel(QTRK_PixelDataType pdt) {
	const int pdtBytes[] = {1, 2, 4};
	return pdtBytes[(int)pdt];
}
