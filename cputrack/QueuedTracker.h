/*! \mainpage Queued Tracker software
 *
 * \section intro_sec Introduction
 *
 * QueuedTracker, or QTrk in short, is an API that facilitates the 3 dimensional subpixel tracking of a magnetic bead in a Magnetic Tweezers (MT) setup.
 * The code found here generates a general purpose library in the form of a DLL that can be used from either .NET applications or LabVIEW. 
 * The [LabVIEW GUI and hardware control](https://github.com/NynkeDekkerLab/BeadTracker/tree/master), which is not included in this documentation, 
 has been created very specifically for the setups as designed and used in the [Nynke Dekker lab](http://nynkedekkerlab.tudelft.nl/).
 *
 * \section hist_sec History
 *
 * The MT setups are homebuilt devices for biological single molecule measurements. They have evolved over the years and so has the need
 * for the related software. Considering the framerate and number of pixels of state of the art cameras involved in the setups, the
 * requirements with regards to data handling are now very steep. A measurement running for a few hours generates hundreds of gigabytes worth
 * of image data. As such, the need arose to do the image analysis fast, in real-time. This software was created to do precisely that.
 * To ensure high speed data analysis, multiple algorithms have been implemented in multithreaded CPU and GPU (through CUDA) implementations, 
 * with a scheduling shell (QueuedCPUTracker and QueuedCUDATracker) and separate data gathering and saving thread (ResultManager).
 *
 * \section imple_sec Implementation
 * 
 * The goal is to find a 3 dimensional position of a bead from a microscopic image. A typical image of a single bead is displayed below:
 * \image html 00008153-s.jpg
 * To perform the tracking, specific algorithms exist and have been implemented. Currently the available options are:
 * <table>
 <tr><th>Algorithm						<th>Dimensions	<th>CPU										<th>CUDA								<th>Reference	<th>Notes
 <tr><td>Starting point					<td>			<td>QueuedCPUTracker::ProcessJob			<td>QueuedCUDATracker::ExecuteBatch		<td>			<td>Functions from which the algorithms are called dependent on settings.
 <tr><td>
 <tr><td>Center of Mass	(COM)			<td>x,y			<td>CPUTracker::ComputeMeanAndCOM			<td>::BgCorrectedCOM					<td>			<td>Always executed for first guess.
 <tr><td>1D Cross Correlation (XCor1D)	<td>x,y			<td>CPUTracker::ComputeXCorInterpolated		<td>Not implemented						<td>			<td>
 <tr><td>Quadrant Interpolation (%QI)	<td>x,y			<td>CPUTracker::ComputeQI					<td>::QI, QI::Execute					<td>			<td>Recommended algorithm. Optimized for speed and accuracy.
 <tr><td>2D Gaussian fit				<td>x,y			<td>CPUTracker::Compute2DGaussianMLE		<td>::G2MLE_Compute						<td>			<td>
 <tr><td>Lookup table (LUT)				<td>z			<td>CPUTracker::ComputeRadialProfile
														<br/>CPUTracker::LUTProfileCompare			<td>::ZLUT_RadialProfileKernel
																									<br/>::ZLUT_ComputeZ					<td>			<td>Only available method for z localization.
 </table>
 *
 * \section soft_sec Required software
 * 
 * To be able to compile the DLLs, you need:
 * - Visual Studio (2010)
 * - CUDA (6.5)
 *
 * To be able to use the CUDA DLLs, the cudart32_65.dll and cufft32_65.dll (or 64 bit versions if compiled with that) need to be known and accessible
 * by the system, i.e. they need to be in the same folder.
 * We are currently limited to CUDA 6.5 at the highest due to the fact that we have a 32 bit LabVIEW version on setups
 * and 32 bit cuFFT has been deprecated in newer CUDA versions.
 *
 * \section cred_sec Credits
 *
 * Original work by Jelmer Cnossen. Maintenance, testing, documentation and improvements by Jordi Wassenburg.
 */
#pragma once

#include "std_incl.h" 
#include "threads.h"
#include <map>


#include "qtrk_c_api.h"

template<typename T>
struct TImageData;
typedef TImageData<float> ImageData;
class CImageData;

/// Minimum number of samples for a profile radial bin. Below this the image mean will be used.
#define MIN_RADPROFILE_SMP_COUNT 4


/*! \brief Abstract tracker interface, implemented by QueuedCUDATracker and QueuedCPUTracker. */
class QueuedTracker
{
public:
	QueuedTracker();
	virtual ~QueuedTracker();

	/*! \brief Select which algorithm is to be used.
	
	\param [in] locType An integer used as a bitmask for settings based on ::LocalizeModeEnum.
	*/
	virtual void SetLocalizationMode(LocMode_t locType) = 0;

	// These are per-bead! So both gain and offset are sized [width*height*numbeads], similar to ZLUT
	// result=gain*(pixel+offset)
	virtual void SetPixelCalibrationImages(float* offset, float* gain) = 0;
	virtual void SetPixelCalibrationFactors(float offsetFactor, float gainFactor) = 0;

	/*! \brief Add a job to the queue to be processed. A job entails running the required algorithms on a single region of interest.

	\param [in] data	Pointer to the data. Type specified by [pdt].
	\param [in] pitch	Distance in bytes between two successive rows of pixels (e.g. address of (0,0) - address of (0,1)).
	\param [in] pdt		Type of [data], specified by ::QTRK_PixelDataType.
	\param [in] jobInfo Structure with metadata for the ROI to be handled. See ::LocalizationJob.
	*/
	virtual void ScheduleLocalization(void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) = 0;
	void ScheduleImageData(ImageData* data, const LocalizationJob *jobInfo); ///< Quick function to schedule a single ROI from an ::ImageData object.
	virtual void ClearResults() = 0; ///< Clear results.
	virtual void Flush() = 0; ///< Stop waiting for more jobs to do, and just process the current batch

	/*! \brief Schedule an entire frame at once, allowing for further optimizations.

	Queues each ROI within the frame seperatly with \ref ScheduleLocalization.

	\param [in] imgptr		Pointer to the top-left pixel of the frame.
	\param [in] pitch		Size in bytes of one row in memory.
	\param [in] width		Width of the frame in pixels.
	\param [in] height		Height of the frame in pixels.
	\param [in] positions	Array of \ref ROIPosition with the top-left pixel of each ROI to be queued.
	\param [in] numROI		Number of ROIs to handle within the frame.
	\param [in]	pdt			Data type of the image given through \p imgptr.
	\param [in] jobInfo		\ref LocalizationJob with information about the frame.

	\return Amount of ROIs queued.
	*/
	virtual int ScheduleFrame(void *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo);
	
	/*!  \brief Set the radial lookup tables to be used for z tracking.

	\p Data can be zero to allocate ZLUT data. 
	LUTs should have been created before by \ref BuildLUT, but not necessarily by the current instance as long as setting match.

	\param [in] data	Pointer to the start of the ZLUT data.
	\param [in]	count	Number of ZLUTs in the dataset.
	\param [in] planes	Number of planes per ZLUT.
	*/	
	virtual void SetRadialZLUT(float* data, int count, int planes) = 0;
	
	/*!  \brief Get the radial lookup tables used for z tracking.

	\param [in,out] dst	Pointer to the pre-allocated memory in which to save the data.
	*/
	virtual void GetRadialZLUT(float* dst) = 0; 

	/*! \brief Get the dimensions of the radial lookup table data.

	\param [in,out] count			Reference to pre-allocated int. Returns number of lookup tables.
	\param [in,out] planes			Reference to pre-allocated int. Returns number of planes per lookup table.
	\param [in,out] radialsteps		Reference to pre-allocated int. Returns number of steps per plane.
	*/
	virtual void GetRadialZLUTSize(int& count, int& planes, int& radialsteps) = 0;

	/*! \brief Set radial weights used for comparing LUT profiles.
	
	\param [in] zcmp Array of radial weights to use. \p zcmp has to have \p zlut_radialsteps elements.
	*/
	virtual void SetRadialWeights(float* zcmp) = 0;

	/*! \brief Set a flag to enable saving of error curves.

	Errors obtained by comparing a radial profile to a ZLUT will be kept in memory rather than destroyed. Only saves for one localization. 
	Error curve can be retreived by \ref GetRadialZLUTCompareProfile.

	\note Not implemented for CUDA.
	\param [in] enabled Flag (boolean) to save error curves. Default is false.
	*/
	virtual void EnableRadialZLUTCompareProfile(bool enabled) = 0;

	/*! \brief Get saved error curve.
	
	See \ref EnableRadialZLUTCompareProfile.
	\note Not implemented for CUDA.
	\param [in] dst Pointer to the pre-allocated memory in which to save the error curve. Size is \p count * \p planes.
	*/
	virtual void GetRadialZLUTCompareProfile(float* dst) = 0; // dst = [count * planes]

	/*! \brief Get the dimensions of the image lookup table data.

	\note Use of image LUT is currently not clear. Radial ZLUT is always used.

	\param [in,out] dims			Reference to pre-allocated int array. Returns [ count, planes, height, width ].
	*/
	virtual void GetImageZLUTSize(int* dims) {}

	/*!  \brief Get the image lookup tables used.

	\note Use of image LUT is currently not clear. Radial ZLUT is always used.

	\param [in,out] dst	Pointer to the pre-allocated memory in which to save the data.
	*/
	virtual void GetImageZLUT(float* dst) {}

	/*!  \brief Set the image lookup tables to be used for z tracking.

	\note Use of image LUT is currently not clear. Radial ZLUT is always used.

	\param [in] src			Pointer to the data for the image LUT.
	\param [in]	radial_zlut	Pointer to the data for the radial LUT.
	\param [in] dims		Array of dimension sizes for the image LUT. See \ref GetImageZLUTSize.
	*/
	virtual bool SetImageZLUT(float* src, float *radial_zlut, int* dims) { return false; }

#define BUILDLUT_IMAGELUT 1
#define BUILDLUT_FOURIER 2
#define BUILDLUT_NORMALIZE 4
#define BUILDLUT_BIASCORRECT 8
	virtual void BeginLUT(uint flags) = 0;
	virtual void BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, int plane, vector2f* known_pos=0) = 0;
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

	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, uint frame, uint timestamp, vector3f* initial, uint zlutIndex);
	void ComputeZBiasCorrection(int bias_planes, CImageData* result, int smpPerPixel, bool useSplineInterp);
	float ZLUTBiasCorrection(float z, int zlut_planes, int bead);
	void SetZLUTBiasCorrection(const CImageData& data); // w=zlut_planes, h=zlut_count
	CImageData *GetZLUTBiasCorrection();

protected:
	CImageData* zlut_bias_correction;
};

void CopyImageToFloat(uchar* data, int width, int height, int pitch, QTRK_PixelDataType pdt, float* dst);
QueuedTracker* CreateQueuedTracker(const QTrkComputedConfig& cc);
void SetCUDADevices(int *devices, int numdev); // empty for CPU tracker


// Polynomial least-square fit weights, used for Z and QI fitting
// Changes to this require rebuild of code
#define QI_LSQFIT_WEIGHTS { 0.14f, 0.5f, 0.85f, 1.0f, 0.85f, 0.5f, 0.14f }
#define QI_LSQFIT_NWEIGHTS 7

//#define ZLUT_LSQFIT_NWEIGHTS 5
//#define ZLUT_LSQFIT_WEIGHTS { 0.5f, 0.85f, 1.0f, 0.85f, 0.5f }

//#define ZLUT_LSQFIT_NWEIGHTS 3
//#define ZLUT_LSQFIT_WEIGHTS { 0.85f, 1.0f, 0.85f }

#define ZLUT_LSQFIT_NWEIGHTS 7
#define ZLUT_LSQFIT_WEIGHTS { 0.15f, 0.5f, 0.85f, 1.0f, 0.85f, 0.5f, 0.15f }

//#define ZLUT_LSQFIT_NWEIGHTS 9
//#define ZLUT_LSQFIT_WEIGHTS {0.4f, 0.5f, 0.7f, 0.9f, 1.0f, 0.9f, 0.7f, 0.5f, 0.4f }

inline int PDT_BytesPerPixel(QTRK_PixelDataType pdt) {
	const int pdtBytes[] = {1, 2, 4};
	return pdtBytes[(int)pdt];
}
