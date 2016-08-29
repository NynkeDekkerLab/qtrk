#pragma once

#include "dllmacros.h"

/// Flags for selecting localization type
enum LocalizeModeEnum {
	LT_OnlyCOM = 0,		///< Use only COM
	LT_XCor1D = 1,		///< COM+XCor1D
	LT_QI = 2,			///< COM+QI
	LT_Gaussian2D = 4,	///< 2D Gaussian localization
	LT_ZLUTAlign = 8,	///< XYZ Alignment with ZLUT

	LT_LocalizeZ = 16,
	LT_NormalizeProfile = 64,
	LT_ClearFirstFourPixels = 128,
	LT_FourierLUT = 256,
	LT_LocalizeZWeighted = 512,

	LT_Force32Bit = 0xffffffff
};

typedef int LocMode_t; // LocalizationModeEnum

/// Flags indicating the data type of image data
enum QTRK_PixelDataType
{
	QTrkU8 = 0,		///< 8 bit unsigned int
	QTrkU16 = 1,	///< 16 bit unsigned int
	QTrkFloat = 2	///< 64 bit float
};


#pragma pack(push, 1)

/*! \brief Structure for region of interest metadata. 

Compiled without padding to line up with LabVIEW alignment. Size is 24 bytes.
\warning Changing this requires changing of the linked LabVIEW cluster QTrkLocalizationJob.ctl.
\note Frame and timestamp are ignored by tracking code itself, but usable for the calling code.
*/
struct LocalizationJob {
	LocalizationJob() {
		frame=timestamp=zlutIndex=0; 
	}
	LocalizationJob(uint frame, uint timestamp, uint zlutPlane, uint zlutIndex) :
		frame (frame), timestamp(timestamp), zlutIndex(zlutIndex) 
	{}
	uint frame;				///< Frame number this ROI belongs to
	uint timestamp;			///< Time stamp of the frame
	int zlutIndex;			///< Bead number of this ROI. Used to get the right ZLUT from memory.
	vector3f initialPos;	///< Optional (Not used)
};

/*! \brief Structure for job results. 

Compiled without padding to line up with LabVIEW alignment. Size is 52 bytes.
\warning Changing this requires changing of the linked LabVIEW cluster QTrkLocalizationResult.ctl.
*/
struct LocalizationResult {
	LocalizationJob job;			///< Job metadata. See ::LocalizationJob.
	vector3f pos;					///< Final 3D position found. If no z localization was performed, the value of z will be 0.
	vector2f pos2D() { return vector2f(pos.x,pos.y); } ///< Final 2D position found.
	vector2f firstGuess;			///< (x,y) position found by the COM localization. Used as initial position for the subsequent algorithms.
	uint error;						///< Flag (boolean) indicating whether the ROI boundary was hit during localization. A 1 indicates a hit.
	float imageMean;				///< Average pixel value of the ROI.
};

/*! \brief Structure for the settings used by the algorithms implemented in QueuedTracker. 

Compiled without padding to line up with LabVIEW alignment.
\warning Changing this requires changing of the linked LabVIEW cluster QTrkSettings.ctl.
\note All settings are always defined, eventhough only one of the three 2D localization algorithms is used.
*/
struct QTrkSettings {
	QTrkSettings() { /// Set default values on initialization
		width = height = 100;
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
		//testRun = false; // CHANGED! Comment/remove when compiling for labview
	}
	int width;		///< Width of regions of interest to be handled. Typically equals QTrkSettings::height (square ROI).
	int height;		///< Height of regions of interest to be handled. Typically equals QTrkSettings::width (square ROI).
	
	/*! \brief Number of threads/streams to use. Defaults differ between CPU and GPU implementations.

	[CPU]: Default -1 chooses \a threads equal to the number of CPUs available. <br/>
	[CUDA]: Default -1 chooses 4 \a streams per available device.
	*/
	int numThreads; 

#define QTrkCUDA_UseList -3	
#define QTrkCUDA_UseAll -2		
#define QTrkCUDA_UseBest -1
	/*! \brief CUDA only. Flag for device selection.
	<table>
	<tr><td>cuda_device >= 0<td>Use as hardware device index.
	<tr><td>cuda_device < 0<td>Use flags below
	</table>

	<table>
	<tr><td>QTrkCUDA_UseList<td>-3<td>Use list defined by SetCUDADevices
	<tr><td>QTrkCUDA_UseAll<td>-2<td>Use all available devices. Default option.
	<tr><td>QTrkCUDA_UseBest<td>-1<td>Only use the best device.
	</table>
	*/
	int cuda_device;

	float com_bgcorrection;			///< Background correction factor for COM. Defines the number of standard deviations data needs to be away from the image mean to be taken into account.

	float zlut_minradius;			///< Distance in pixels from the bead center from which to start sampling profiles. Default 1.0.
	float zlut_radial_coverage;		///< Sampling points per radial pixel. Default 3.0.
	float zlut_angular_coverage;	///< Factor of the sampling perimeter to cover with angular sampling steps. Between 0 and 1, default 0.7.
	float zlut_roi_coverage;		///< Factor of the ROI to include in sampling. Between 0 and 1, default 1. Maxradius = ROI/2*roi_coverage.

	int qi_iterations;				///< Number of times to run the %QI algorithm, sampling around the last found position.
	float qi_minradius;				///< Distance in pixels from the bead center from which to start sampling profiles. Default 1.0.
	float qi_radial_coverage;		///< Sampling points per radial pixel. Default 3.0.
	float qi_angular_coverage;		///< Factor of the sampling perimeter to cover with angular sampling steps. Between 0 and 1, default 0.7.
	float qi_roi_coverage;			///< Factor of the ROI to include in sampling. Between 0 and 1, default 1. Maxradius = ROI/2*roi_coverage.
	float qi_angstep_factor;		///< Factor to reduce angular steps on lower iterations. Default 1.0 (no effect). Increase for faster early iterations but more image noise sensitivity.

	int xc1_profileLength;			///< Profile length for the cross correlation.
	int xc1_profileWidth;			///< Profile width for the cross correlation.
	int xc1_iterations;				///< Number of times to run the cross correlation algorithm.

	int gauss2D_iterations;			///< Number of times to run the 2D gaussian algorithm.
	float gauss2D_sigma;			///< Standard deviation to use in the 2D gaussian algorithm.

	int downsample;					///< Image downsampling factor. Applied before anything else. 0 = original, 1 = 1x (W=W/2,H=H/2).

	/*! \brief 	Flag to run a test run. 
	
	A test run dumps a lot of intermediate data to the disk for algorithm inspection (only %QI & ZLUT).
	\warning CHANGED compared to LabVIEW! Comment/remove when compiling for LabVIEW.
	\todo Add to LabVIEW cluster.
	*/
	bool testRun;					
};

/// Struct used to define the top-left corner position of an ROI within a frame. ROI is [ x .. x+w ; y .. y+h ].
struct ROIPosition
{
	int x,y;
};

/*! \brief Structure for derived settings computed from base settings in ::QTrkSettings. 

Compiled without padding to line up with LabVIEW alignment.
\warning Changing this requires changing of the linked LabVIEW cluster QTrkSettings.ctl.
\note All settings are always defined, eventhough only one of the three 2D localization algorithms is used.
*/
struct QTrkComputedConfig : public QTrkSettings
{
	QTrkComputedConfig() {}
	QTrkComputedConfig(const QTrkSettings& base) { *((QTrkSettings*)this)=base; Update(); }
	void Update();			///< Compute the derived settings
	void WriteToLog();		///< Write all settings to specified log file (Jelmer)
	void WriteToFile();		///< Write all settings to specified output file (Jordi, to combine with ::QTrkSettings.testRun)

	// Computed from QTrkSettings
	int zlut_radialsteps;	///< Number of radial steps to sample on.
	int zlut_angularsteps;	///< Number of angular steps to sample on.
	float zlut_maxradius;	///< Max radius in pixels of the sampling circle.
	
	int qi_radialsteps;		///< Number of radial steps to sample on.
	int qi_angstepspq;		///< Number of angular steps to sample on.
	float qi_maxradius;		///< Max radius in pixels of the sampling circle.
};

#pragma pack(pop)

class QueuedTracker;

/** \defgroup c_api API - C
\brief API functions available to a C or .NET program. 

These DLLs are compiled by the \a cputrack and \a cudatrack projects.
*/

/** \addtogroup c_api
	@{
*/

/*!\brief Create a QTrk instance and return a pointer to it. 

\param [in] cfg Pointer to the structure with the desired tracking settings.
\return Pointer to the created QTrk instance.
*/
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

/** @} */