
// Defines the interface for trackers (both CPU and CUDA)

#pragma once

#include "std_incl.h" 
#include "threads.h"
#include <map>

struct ImageData;

// minimum number of samples for a profile radial bin. Below this the image mean will be used
#define MIN_RADPROFILE_SMP_COUNT 4

enum LocalizeModeEnum {
	// Flags for selecting 2D localization type
	LT_OnlyCOM = 0, // use only COM
	LT_XCor1D = 1, // COM+XCor1D
	LT_QI = 2, // COM+QI
	LT_Gaussian2D = 4, // 2D Gaussian localization
	LT_IMAP = 8, // Image Alignment In Parallel

	LT_2DMask = 15,
	LT_LocalizeZ = 16,
	LT_BuildRadialZLUT = 32,
	LT_NormalizeProfile = 64,
	LT_BuildImageLUT = 128,
	LT_Force32Bit = 0xffffffff
};

typedef int LocMode_t; // LocalizationModeEnum

enum QTRK_PixelDataType
{
	QTrkU8 = 0,
	QTrkU16 = 1,
	QTrkFloat = 2
};


#pragma pack(push, 1)
struct LocalizationJob {
	LocalizationJob() {
		frame=timestamp=zlutIndex=zlutPlane=0; 
	}
	LocalizationJob(uint frame, uint timestamp, uint zlutPlane, uint zlutIndex) :
		frame (frame), timestamp(timestamp), zlutPlane(zlutPlane), zlutIndex(zlutIndex) 
	{}
	uint frame, timestamp;
	int zlutIndex; // or bead#
	uint zlutPlane; // for ZLUT building
	vector3f initialPos;
};

// DONT CHANGE, Mapped to labview clusters!
struct LocalizationResult {
	LocalizationJob job;
	vector3f pos;
	vector2f pos2D() { return vector2f(pos.x,pos.y); }
	vector2f firstGuess;
	uint error;
};
// DONT CHANGE, Mapped to labview clusters (QTrkSettings.ctl)!
struct QTrkSettings {
	QTrkSettings() {
		width = height = 150;
		numThreads = -1;
		xc1_profileLength = 128;
		xc1_profileWidth = 32;
		xc1_iterations = 2;
		zlut_minradius = 5.0f;
		zlut_angular_coverage = 0.7f;
		zlut_radial_coverage = 3.0f;
		zlut_roi_coverage = 1.0f;
		qi_iterations = 4;
		qi_minradius = 5;
		qi_angular_coverage = 0.7f;
		qi_radial_coverage = 3.0f; 
		qi_roi_coverage = 1.0f;
		qi_angstep_factor = 1.0f;
		cuda_device = -2;
		com_bgcorrection = 0.0f;
		gauss2D_iterations = 6;
		gauss2D_sigma = 4;
	}
	int width, height;
	int numThreads;

#define QTrkCUDA_UseList -3   // Use list defined by SetCUDADevices
#define QTrkCUDA_UseAll -2
#define QTrkCUDA_UseBest -1
	// cuda_device < 0: use flags above
	// cuda_device >= 0: use as hardware device index
	int cuda_device;

	float com_bgcorrection; // 0.0f to disable

	float zlut_minradius;
	float zlut_radial_coverage;
	float zlut_angular_coverage;
	float zlut_roi_coverage; // maxradius = ROI/2*roi_coverage

	int qi_iterations;
	float qi_minradius;
	float qi_radial_coverage;
	float qi_angular_coverage;
	float qi_roi_coverage;
	float qi_angstep_factor;

	int xc1_profileLength;
	int xc1_profileWidth;
	int xc1_iterations;

	int gauss2D_iterations;
	float gauss2D_sigma;
};

struct ROIPosition
{
	int x,y; // top-left coordinates. ROI is [ x .. x+w ; y .. y+h ]
};
#pragma pack(pop)

// Parameters computed from QTrkSettings
struct QTrkComputedConfig : public QTrkSettings
{
	QTrkComputedConfig() {}
	QTrkComputedConfig(const QTrkSettings& base) { *((QTrkSettings*)this)=base; Update(); }
	void Update();

	// Computed from QTrkSettings
	int zlut_radialsteps;
	int zlut_angularsteps;
	float zlut_maxradius;
	
	int qi_radialsteps;
	int qi_angstepspq;
	float qi_maxradius;
};

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
	virtual void GetRadialZLUT(float* dst) = 0; // delete[] memory afterwards
	virtual void GetRadialZLUTSize(int& count, int& planes, int& radialsteps) = 0;

	// dims = [ count, planes, height, width ]  (Just like how the data is ordered)
	virtual void GetImageZLUTSize(int* dims) {}
	virtual void GetImageZLUT(float* dst) {}
	virtual void SetImageZLUT(float* dst, int* dims) {}

	enum BuildLUTMode {
		RadialLUT = 1, ImageLUT = 2
	};
	virtual void ProcessLUTImages(void* data, int pitch, QTRK_PixelDataType pdt, uint mode_flags, int plane) = 0;
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
