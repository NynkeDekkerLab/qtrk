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
 * \section algo_sec Algorithms
 * 
 * The goal is to find a 3 dimensional position of a bead from a microscopic image. A typical image of a single bead is displayed below:
 * \image html 00008153-s.jpg
 * To perform the tracking, specific algorithms exist and have been implemented. Currently the available options are:
 * <table>
 <tr><th>Algorithm						<th>Dimensions	<th>CPU										<th>CUDA								<th>Reference		<th>Notes
 <tr><td>Starting point					<td>			<td>QueuedCPUTracker::ProcessJob			<td>QueuedCUDATracker::ExecuteBatch		<td>				<td>Functions from which the algorithms are called dependent on settings.
 <tr><td>
 <tr><td>Center of Mass	(COM)			<td>x,y			<td>CPUTracker::ComputeMeanAndCOM			<td>::BgCorrectedCOM					<td>https://en.wikipedia.org/wiki/Image_moment<td>Always executed for first guess.
 <tr><td>1D Cross Correlation (XCor1D)	<td>x,y			<td>CPUTracker::ComputeXCorInterpolated		<td>Not implemented						<td>https://en.wikipedia.org/wiki/Cross-correlation<td>
 <tr><td>Quadrant Interpolation (%QI)	<td>x,y			<td>CPUTracker::ComputeQI					<td>::QI, QI::Execute					<td>\cite loen:QI	<td>Recommended algorithm. Optimized for speed and accuracy.
 <tr><td>2D Gaussian fit				<td>x,y			<td>CPUTracker::Compute2DGaussianMLE		<td>::G2MLE_Compute						<td>https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function<td>
 <tr><td>Lookup table (LUT)				<td>z			<td>CPUTracker::ComputeRadialProfile
														<br/>CPUTracker::LUTProfileCompare			<td>::ZLUT_RadialProfileKernel
																									<br/>::ZLUT_ComputeZ					<td>\cite loen:QI	<td>Only available method for z localization.
 </table>
 * Internally, %QI has grown to be the most used and verified algorithm. Usage of it is recommended for data equality.
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
 * \section disclaimer Disclaimer
 *
 * Code is provided as-is. Bugs may be encountered. Maintenance and support at the discretion of lab members.
 *
 * This code and documentation is intended for internal use and provided publicly in the interest of transparency.
 *
 * \section cred_sec Credits
 *
 * Original work by Jelmer Cnossen \cite rsi:cnos. Maintenance, testing, documentation and improvements by Jordi Wassenburg.
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


/*! \brief Abstract tracker interface, implemented by QueuedCUDATracker and QueuedCPUTracker. 

In general, it holds a queue of "jobs" (\ref LocalizationJob) to perform.
These jobs are created and automatically queued using \ref ScheduleLocalization(uchar*, int, QTRK_PixelDataType, uint, uint, vector3f*, uint).
Internally, a scheduling thread divides the work over multiple threads to speed up the calculations.
When jobs are completed, their results become available to retrieve from memory. Usage of the \ref ResultManager class is recommended to handle this process.

Queue and result related functionalities are exposed to an API, see \ref lab_API and \ref c_api.
*/
class QueuedTracker
{
public:
	QueuedTracker();
	virtual ~QueuedTracker();

	/*! \brief Select which algorithm is to be used.
	
	\param [in] locType An integer used as a bitmask for settings based on \ref LocalizeModeEnum.
	*/
	virtual void SetLocalizationMode(LocMode_t locType) = 0;

	/*! \brief Set the pixel calibration images.

	These images are used to scale the input image to get rid of background influences in the image.
	The values are per-pixel-per-ROI. Result = gain*(pixel+offset).

	\param [in] offset	Array of the offset values to use per pixel. Size and order is [width*height*numbeads].
	\param [in] gain	Array of gain values to use per pixel. Size and order is [width*height*numbeads].
	*/
	virtual void SetPixelCalibrationImages(float* offset, float* gain) = 0;

	/*! \brief Set the pixel calibration factors.

	The factors can be used to increase or decrease the effects of the images supplied through \ref SetPixelCalibrationImages for further finetuning.
	These only have an effect when an image is actually set through that function.

	\param [in] offsetFactor	Factor by which to scale the offset values.
	\param [in] gainFactor		Factor by which to scale the gain values.
	*/
	virtual void SetPixelCalibrationFactors(float offsetFactor, float gainFactor) = 0;

	/*! \brief Add a job to the queue to be processed. A job entails running the required algorithms on a single region of interest.

	If a localization can not be added to the queue because it is full, the thread will be put to sleep and periodically try again.
	
	\param [in] data	Pointer to the data. Type specified by \p pdt.
	\param [in] pitch	Distance in bytes between two successive rows of pixels (e.g. address of (0,0) - address of (0,1)).
	\param [in] pdt		Type of \p data, specified by ::QTRK_PixelDataType.
	\param [in] jobInfo Structure with metadata for the ROI to be handled. See \ref LocalizationJob.
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

	\param [out] dst	Pointer to the pre-allocated memory in which to save the data.
	*/
	virtual void GetRadialZLUT(float* dst) = 0; 

	/*! \brief Get the dimensions of the radial lookup table data.

	\param [out] count			Reference to pre-allocated int. Returns number of lookup tables.
	\param [out] planes			Reference to pre-allocated int. Returns number of planes per lookup table.
	\param [out] radialsteps		Reference to pre-allocated int. Returns number of steps per plane.
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

	\param [out] dims Reference to pre-allocated int array. Returns [ count, planes, height, width ].
	*/
	virtual void GetImageZLUTSize(int* dims) {}

	/*!  \brief Get the image lookup tables used.

	\note Use of image LUT is currently not clear. Radial ZLUT is always used.

	\param [out] dst	Pointer to the pre-allocated memory in which to save the data.
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
	/*! \brief Setup to begin building a lookup table. 

	Sets the flags by which the lookup table is built. Flags are defined in a uint bitmask format as:
	<table>
	<tr><th>Name				<th>Value	<th>Description
	<tr><td>					<td>0		<td>Default, no special flags.
	<tr><td>BUILDLUT_IMAGELUT	<td>1 (2^0)	<td>Build an image LUT.
	<tr><td>BUILDLUT_FOURIER	<td>2 (2^1) <td>Build a fourier LUT.
	<tr><td>BUILDLUT_NORMALIZE	<td>4 (2^2) <td>Normalize radial profiles. Irrelevant, since \ref FinalizeLUT always normalizes.
	<tr><td>BUILDLUT_BIASCORRECT<td>8 (2^3) <td>Enable bias correction. Currently only partly implemented, \b don't \b use.
	</table>

	\param [in] flags UINT interpreted as a series of bits to set settings.
	*/
	virtual void BeginLUT(uint flags) = 0;

	/*! \brief Add a new lookup table plane.

	Takes a stack of ROI images through \p data. Determines and adds the profile for each ROI to its respective LUT.

	\param [in] data		Pointer to the start of an image stack.
	\param [in] pitch		Width of the data in memory so offsets can be calculated.
	\param [in] pdt			Pixel data type for the data. See \ref QTRK_PixelDataType.
	\param [in] plane		The plane number the ROIs are taken for.
	\param [in] known_pos	Center position from which to start making the radial profile. A standard QI localization with applied settings is performed if 0 (default).
	*/
	virtual void BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, int plane, vector2f* known_pos=0) = 0;

	/*! \brief Finalize the lookup tables in memory.

	Normalizes the profiles for radial lookup tables and calculates derivates and adds boundary conditions for image LUTs.
	*/
	virtual void FinalizeLUT() = 0;
	
	/*! \brief Get the number of finished localization jobs (=results) available in memory.

	\returns The number of available results.
	*/
	virtual int GetResultCount() = 0;

	/*! \brief Fetch available results.

	\note Removes results from internal QueuedTracker memory.

	\param [in] results Array of pre-allocated \ref LocalizationResult to which to add the results.
	\param [in] maxResults Maximum number of results to fetch. Corresponds to maximum size of \p dstResult.

	\return Number of fetched results.
	*/
	virtual int FetchResults(LocalizationResult* results, int maxResults) = 0;

	/*! \brief Get the lengths of the queue of jobs to be handled.

	\param [out] maxQueueLen Pre-allocated integer that returns the maximum size of the queue if nonzero.

	\return Number of jobs currently being handled and in the queue.
	*/
	virtual int GetQueueLength(int *maxQueueLen=0) = 0;

	/*! \brief Test to see if the tracker is idle.

	That is, \ref GetQueueLength == 0.
	
	\return Boolean indicating if the tracker is idle.
	*/
	virtual bool IsIdle() = 0;
	
	/*! \brief Datastructure used to return additional settings in a string-string key-value pair mapping. 
	
	Currently only two settings are available: \p use_texturecache for CUDA and \p trace for CPU.
	*/
	typedef std::map<std::string, std::string> ConfigValueMap;

	/*! \brief Get the used additional configurations. */
	virtual ConfigValueMap GetConfigValues() = 0;

	/*! \brief Set an additional setting. 
	
	\param [in] name Name of the setting.
	\param [in] value Value of the setting.
	*/
	virtual void SetConfigValue(std::string name, std::string value) = 0;
	
	/*! \brief Get the output of performance profiling.

	Only implemented in CUDA at the moment.

	\return String with the parsed profiling output.
	*/
	virtual std::string GetProfileReport() { return ""; }

	/*! \brief Get a report of encountered errors.

	\note Placeholder function, no warnings are generated or returned anywhere for now.

	\return String with the parsed warning output.
	*/
	virtual std::string GetWarnings() { return ""; }

	/*! \brief Get the debug image for a specific thread.

	Debug image can be set in trackers by copying data into \p debugImage, for instance:
	\code
	#ifdef _DEBUG
		std::copy(srcImage, srcImage+width*height, debugImage);
	#endif
	\endcode
	\warning \p pData has to be cleared with delete[] in the calling function!

	\param [in] ID Thread number from which to grab the image.
	\param [out] w Pointer to an integer in which the image width will be stored.
	\param [out] h Pointer to an integer in which the image height will be stored.
	\param [out] pData Reference to where the data array will be saved.
	
	\return Boolean indicating if the debug image was succesfully returned.
	*/
	virtual bool GetDebugImage(int ID, int *w, int *h, float** pData) { return false; } // deallocate result with delete[] 
	
	/*! \brief Get the debug image as an \ref ImageData object.

	\param [in] ID Thread number from which to grab the image.

	\return An \ref ImageData instance with the debug image.
	*/
	ImageData DebugImage(int ID);

	/*! \brief The settings used by this instance of QueuedTracker.
	*/
	QTrkComputedConfig cfg;

	/*! \brief Add an image to the queue to be processed. Creates a job.

	Creates a job and then calls \ref ScheduleLocalization(void*, int, QTRK_PixelDataType, const LocalizationJob*).

	\param [in] data		Pointer to the data. Type specified by \p pdt.
	\param [in] pitch		Distance in bytes between two successive rows of pixels (e.g. address of (0,0) - address of (0,1)).
	\param [in] pdt			Type of \p data, specified by ::QTRK_PixelDataType.
	\param [in] frame		Frame number this localization belongs to.
	\param [in] timestamp	Timestamp for this job.
	\param [in] initial		Initial position for the algorithms requiring one. If none is specified, a COM track is performed to determine one.
	\param [in] zlutIndex	Number of the bead. Used to determine which ZLUT to use.
	*/
	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, uint frame, uint timestamp, vector3f* initial, uint zlutIndex);

	/** \defgroup zbias ZLUT Bias Correction
	\brief Functions related to correcting the bias in the z lookup table scheme.

	For more information, see Jelmer's paper on this software environment \cite rsi:cnos. This is not used on setups as far as I (Jordi) am aware,
	since it was still in a development stage when Jelmer left.
	*/

	/** \addtogroup zbias
		@{
	*/
	void ComputeZBiasCorrection(int bias_planes, CImageData* result, int smpPerPixel, bool useSplineInterp);
	float ZLUTBiasCorrection(float z, int zlut_planes, int bead);
	void SetZLUTBiasCorrection(const CImageData& data); // w=zlut_planes, h=zlut_count
	CImageData* GetZLUTBiasCorrection();

protected:
	CImageData* zlut_bias_correction;

	/** @} */
};

/*! \brief Copies image data from a generic QTRK_PixelDataType array to a float array.

\param [in] data	Array with the data in the \p pdt type.
\param [in] width	Width of the image.
\param [in] height	Height of the image.
\param [in] pitch	Width of the array in memory.
\param [in] pdt		\ref QTRK_PixelDataType specifier for \p data.
\param [in] dst		Pre-allocated float array in which to save the data.
*/
void CopyImageToFloat(uchar* data, int width, int height, int pitch, QTRK_PixelDataType pdt, float* dst);

/*! \brief Helper function to create a QueuedTracker instance.

Used to determine the creation of a CUDA or CPU instance through compiler definitions.
*/
QueuedTracker* CreateQueuedTracker(const QTrkComputedConfig& cc);

/*! \brief Set the list of devices to be used when \ref QTrkComputedConfig::cuda_device is set to \ref QTrkCUDA_UseList.

\note Empty for CPU tracker.

\param [in] devices Array with index numbers of devices to be used.
\param [in] numdev  Amount of devices to use (= length of \p devices).
*/
void SetCUDADevices(int *devices, int numdev);


/// Polynomial least-square fit weights, used for QI fitting
/// Changes to this require rebuild of code
#define QI_LSQFIT_WEIGHTS { 0.14f, 0.5f, 0.85f, 1.0f, 0.85f, 0.5f, 0.14f }
#define QI_LSQFIT_NWEIGHTS 7

/// Polynomial least-square fit weights, used for Z fitting
/// Changes to this require rebuild of code
//#define ZLUT_LSQFIT_NWEIGHTS 5
//#define ZLUT_LSQFIT_WEIGHTS { 0.5f, 0.85f, 1.0f, 0.85f, 0.5f }

//#define ZLUT_LSQFIT_NWEIGHTS 3
//#define ZLUT_LSQFIT_WEIGHTS { 0.85f, 1.0f, 0.85f }

#define ZLUT_LSQFIT_NWEIGHTS 7
#define ZLUT_LSQFIT_WEIGHTS { 0.15f, 0.5f, 0.85f, 1.0f, 0.85f, 0.5f, 0.15f }

//#define ZLUT_LSQFIT_NWEIGHTS 9
//#define ZLUT_LSQFIT_WEIGHTS {0.4f, 0.5f, 0.7f, 0.9f, 1.0f, 0.9f, 0.7f, 0.5f, 0.4f }

/*! \brief sizeof() equivalent for the \ref QTRK_PixelDataType.

\param [in] pdt The pixel data type used.
\returns The size in bytes of the specified datatype.
*/
inline int PDT_BytesPerPixel(QTRK_PixelDataType pdt) {
	const int pdtBytes[] = {1, 2, 4};
	return pdtBytes[(int)pdt];
}
