#pragma once

#include "dllmacros.h"

enum LocalizeModeEnum {
	// Flags for selecting 2D localization type
	LT_OnlyCOM = 0, // use only COM
	LT_XCor1D = 1, // COM+XCor1D
	LT_QI = 2, // COM+QI
	LT_Gaussian2D = 4, // 2D Gaussian localization
	LT_ZLUTAlign = 8, // XYZ Alignment with ZLUT

	LT_LocalizeZ = 16,
	LT_NormalizeProfile = 64,
	LT_ClearFirstFourPixels = 128,
	LT_FourierLUT = 256,
	LT_LocalizeZWeighted = 512,

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

// 24 bytes
struct LocalizationJob {
	LocalizationJob() {
		frame=timestamp=zlutIndex=0; 
	}
	LocalizationJob(uint frame, uint timestamp, uint zlutPlane, uint zlutIndex) :
		frame (frame), timestamp(timestamp), zlutIndex(zlutIndex) 
	{}
	uint frame, timestamp;   
	int zlutIndex; // or bead#
	vector3f initialPos;
};


// DONT CHANGE, Mapped to labview clusters!
// 13*4 = 52 bytes
struct LocalizationResult {
	LocalizationJob job; //24
	vector3f pos;
	vector2f pos2D() { return vector2f(pos.x,pos.y); }
	vector2f firstGuess; // COM pos
	uint error;
	float imageMean;
};
// DONT CHANGE, Mapped to labview clusters (QTrkSettings.ctl)!
struct QTrkSettings {
	QTrkSettings() {
		width = height = 150;
		numThreads = -1;
		xc1_profileLength = 128;
		xc1_profileWidth = 32;
		xc1_iterations = 2;
		zlut_minradius = 1.0f;
		zlut_angular_coverage = 0.7f;
		zlut_radial_coverage = 3.0f;
		zlut_roi_coverage = 1.0f;
		qi_iterations = 5;
		qi_minradius = 1;
		qi_angular_coverage = 0.7f;
		qi_radial_coverage = 3.0f; 
		qi_roi_coverage = 1.0f;
		qi_angstep_factor = 1.0f;
		cuda_device = -2;
		com_bgcorrection = 0.0f;
		gauss2D_iterations = 6;
		gauss2D_sigma = 4;
		downsample = 0;
		testRun = false;
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

	int downsample; // 0 = original, 1 = 1x (W=W/2,H=H/2)

	bool testRun;
};

struct ROIPosition
{
	int x,y; // top-left coordinates. ROI is [ x .. x+w ; y .. y+h ]
};


// Parameters computed from QTrkSettings
struct QTrkComputedConfig : public QTrkSettings
{
	QTrkComputedConfig() {}
	QTrkComputedConfig(const QTrkSettings& base) { *((QTrkSettings*)this)=base; Update(); }
	void Update();
	void WriteToLog();
	void WriteToFile();

	// Computed from QTrkSettings
	int zlut_radialsteps;
	int zlut_angularsteps;
	float zlut_maxradius;
	
	int qi_radialsteps;
	int qi_angstepspq;
	float qi_maxradius;
};

#pragma pack(pop)

class QueuedTracker;

CDLL_EXPORT QueuedTracker* DLL_CALLCONV QTrkCreateInstance(QTrkSettings *cfg);
CDLL_EXPORT void DLL_CALLCONV QTrkFreeInstance(QueuedTracker* qtrk);

// C API, mainly intended to allow binding to .NET
CDLL_EXPORT void DLL_CALLCONV QTrkSetLocalizationMode(QueuedTracker* qtrk, LocMode_t locType);

// Frame and timestamp are ignored by tracking code itself, but usable for the calling code
// Pitch: Distance in bytes between two successive rows of pixels (e.g. address of (0,0) -  address of (0,1) )
// ZlutIndex: Which ZLUT to use for ComputeZ/BuildZLUT
CDLL_EXPORT void DLL_CALLCONV QTrkScheduleLocalization(QueuedTracker* qtrk, void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV QTrkClearResults(QueuedTracker* qtrk);
CDLL_EXPORT void DLL_CALLCONV QTrkFlush(QueuedTracker* qtrk); // stop waiting for more jobs to do, and just process the current batch

// Schedule an entire frame at once, allowing for further optimizations
CDLL_EXPORT int DLL_CALLCONV QTrkScheduleFrame(QueuedTracker* qtrk, void *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo);
	
// data can be zero to allocate ZLUT data. zcmp has to have 'zlut_radialsteps' elements
CDLL_EXPORT void DLL_CALLCONV QTrkSetRadialZLUT(QueuedTracker* qtrk, float* data, int count, int planes, float* zcmp=0); 
CDLL_EXPORT void DLL_CALLCONV QTrkGetRadialZLUT(QueuedTracker* qtrk, float* dst);
CDLL_EXPORT void DLL_CALLCONV QTrkGetRadialZLUTSize(QueuedTracker* qtrk, int* count, int* planes, int* radialsteps);

CDLL_EXPORT void DLL_CALLCONV QTrkBuildLUT(QueuedTracker* qtrk, void* data, int pitch, QTRK_PixelDataType pdt, bool imageLUT, int plane);
CDLL_EXPORT void DLL_CALLCONV QTrkFinalizeLUT(QueuedTracker* qtrk);
	
CDLL_EXPORT int DLL_CALLCONV QTrkGetResultCount(QueuedTracker* qtrk);
CDLL_EXPORT int DLL_CALLCONV QTrkFetchResults(QueuedTracker* qtrk, LocalizationResult* results, int maxResults);

CDLL_EXPORT int DLL_CALLCONV QTrkGetQueueLength(QueuedTracker* qtrk, int *maxQueueLen);
CDLL_EXPORT bool DLL_CALLCONV QTrkIsIdle(QueuedTracker* qtrk);

CDLL_EXPORT void DLL_CALLCONV QTrkGetProfileReport(QueuedTracker* qtrk, char *dst, int maxStrLen);
CDLL_EXPORT void DLL_CALLCONV QTrkGetWarnings(QueuedTracker* qtrk, char *dst, int maxStrLen);

CDLL_EXPORT void DLL_CALLCONV QTrkGetComputedConfig(QueuedTracker* qtrk, QTrkComputedConfig* cfg);

