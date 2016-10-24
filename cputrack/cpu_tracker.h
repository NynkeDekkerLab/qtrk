#pragma once

#include "QueuedTracker.h"
#include "utils.h"
#include "scalar_types.h"
#include "kissfft.h"

/// Class to facilitate 1D cross correlation calculations.
class XCor1DBuffer {
public:
	/// Constructs the buffer and initiates the FFT workers.
	XCor1DBuffer(int xcorw) : fft_forward(xcorw, false), fft_backward(xcorw, true), xcorw(xcorw)
	{}

	kissfft<scalar_t> fft_forward;	///< Handle to forward FFT [kissfft](https://sourceforge.net/projects/kissfft/) instance.
	kissfft<scalar_t> fft_backward; ///< Handle to backward FFT [kissfft](https://sourceforge.net/projects/kissfft/) instance.
	int xcorw;						///< Width of the cross correlation.

	/*! \brief Calculates a cross correlation much like https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation.

	- Calculates the FFTs of both the normal and reverse profiles
	- Multiplies the FFT of the normal profile with the complex conjugate FFT of the reverse profile
	- Transforms the result back to obtain the cross correlation

	\param [in] prof		The original profile.
	\param [in] prof_rev	The reversed profile.
	\param [out] result		Pre-allocated array that will hold the cross correlation.
	*/
	void XCorFFTHelper(complex_t* prof, complex_t* prof_rev, scalar_t* result);
};

/*! \brief Class with all CPU algorithm implementations.

Has one ROI in its memory (\ref srcImage) and the correct member functions are called from its parent's QueuedCPUTracker::ProcessJob.
All functions work on the image in memory. 
Many of the settings that appear here come from \ref QTrkComputedConfig and are further explained there.
*/
class CPUTracker
{
public:
	int width;			///< ROI width.
	int height;			///< ROI height.
	int xcorw;			///< Cross correlation profile length.
	int trackerID;		///< ID of this tracker (= ID of thread it runs in).

	float *srcImage;	///< Memory region that holds the image on which tracking is to be performed.
	float *debugImage;	///< Memory in which an intermediate image can optionally be place and retrieved.
	float mean;			///< Mean intensity of the ROI. Calculated in \ref ComputeMeanAndCOM.
	float stdev;		///< Standard deviation of values in the ROI. Calculated in \ref ComputeMeanAndCOM.
#ifdef _DEBUG
	float maxImageValue;	///< Maximum intensity in the image.
#endif

	/*! \brief Pointer to the first data in the ZLUTs.

	All LUTs are saved in one big contiguous section of memory of size zlut_planes*zlut_count*zlut_res. 
	Calculate specific LUTs or planes based on their indexes: 
	\code zluts[index * (zlut_planes * zlut_res) + plane * zlut_res + r]. \endcode
	The ZLUT system stores 'zlut_count' number of 2D zlut's, so every bead can be tracked with its own unique ZLUT.
	*/
	float* zluts;
	bool zlut_memoryOwner;				///< Flag indicating if this instance is the owner of the zluts memory, or is it external. False in all normal operation.
	int zlut_planes;					///< Number of planes per ZLUT.
	int zlut_res;						///< ZLUT resolution = number of radial steps in a plane.
	int zlut_count;						///< Number of ZLUTs (= # beads) available. 
	float zlut_minradius;				///< Minimum radius in pixels to start sampling for a ZLUT profile.
	float zlut_maxradius;				///< Maximum radius in pixels of the ZLUT profile sampling circle.
	std::vector<float> zlut_radialweights;	///< Vector with the radialweights used by the error curve calculation.
	kissfft<scalar_t> *qa_fft_forward;	///< Handle to forward FFT [kissfft](https://sourceforge.net/projects/kissfft/) instance for quadrant align.
	kissfft<scalar_t> *qa_fft_backward; ///< Handle to backward FFT [kissfft](https://sourceforge.net/projects/kissfft/) instance for quadrant align.

	/*! \brief Flag to enable running a test run.
	
	A test run outputs a lot of intermediate data into an output path using \ref GetCurrentOutputPath.
	This can currently be used to analyse the particulars of the ZLUT algorithms.

	Output currently happens in \ref LUTProfileCompare.
	A [Matlab GUI](https://github.com/NynkeDekkerLab/TestDiag) has been created to read and analyse the created files.
	
	\warning No link to LabVIEW, test executables only.
	*/
	bool testRun;

	/*! \brief Get the start of the ZLUT of a specific bead.

	\param [in] index	The bead number for which to get the LUT.

	\return				A pointer to the start of the LUT of this bead.
	*/
	float* GetRadialZLUT(int index)  { return &zluts[zlut_res*zlut_planes*index]; }

	XCor1DBuffer* xcorBuffer;					///< The handle from which to perform cross correlation calculations.
	std::vector<vector2f> quadrantDirs;			///< Vector with the sampling points for a single quadrant (cos & sin pairs).
	int qi_radialsteps;							///< Number of radialsteps in the QI calculations.
	kissfft<scalar_t> *qi_fft_forward;			///< Handle to forward FFT [kissfft](https://sourceforge.net/projects/kissfft/) instance for QI.
	kissfft<scalar_t> *qi_fft_backward;			///< Handle to backward FFT [kissfft](https://sourceforge.net/projects/kissfft/) instance for QI.

	/// Class to facilitate 2D Fourier transforms.
	class FFT2D {
	public:
		kissfft<float> xfft;					///< Handle to FFT [kissfft](https://sourceforge.net/projects/kissfft/) instance for x.
		kissfft<float> yfft;					///< Handle to FFT [kissfft](https://sourceforge.net/projects/kissfft/) instance for y.
		std::complex<float> *cbuf;				///< Calculation buffer.
		/*! \brief Initialize the class.

		\param [in] w Image width.
		\param [in] h Image height.
		*/
		FFT2D(int w, int h) : xfft(w,false), yfft(h,false) {cbuf=new std::complex<float>[w*h]; }
		
		/*! \brief Free memory and delete the instance. */
		~FFT2D() { delete[] cbuf; }

		/*! \brief Apply the 2D FFT.

		\param [in,out] d		Input array of floats. Output will be put back in this array.
		*/
		void Apply(float* d);
	};
	FFT2D *fft2d;				///< Instance of \ref FFT2D to perform 2D FFTs.

	float& GetPixel(int x, int y) { return srcImage[width*y+x]; }	///< Get the image pixel greyscale value at point (x,y).
	int GetWidth() { return width; }								///< Get the width of the image.
	int GetHeight() { return height; }								///< Get the height of the image.

	/*! \brief Create an instance of CPUTracker.

	\param [in] w				Image width.
	\param [in] h				Image height.
	\param [in] xcorwindow		Cross correlation window size.
	\param [in] testRun			Flag for test run. See \ref QTrkSettings::testRun.
	*/
	CPUTracker(int w, int h, int xcorwindow=128, bool testRun = false);

	/*! \brief Destroy a CPUTracker instance. */
	~CPUTracker();

	/*! \brief Test to see if a circle extends outside of image boundaries.

	\param [in] center		The center of the circle to test.
	\param [in] radius		The radius of the circle.

	\return False if circle remains inside image boundaries.
	*/
	bool CheckBoundaries(vector2f center, float radius);

	/*! \brief Compute the cross correlation offsets and resulting position.

	See https://en.wikipedia.org/wiki/Cross-correlation for algorithm details.

	\param [in] initial			The initial position from which to start the algorithm.
	\param [in] iterations		The number of iterations to perform.
	\param [in] profileWidth	Width of the profiles to correlate.
	\param [in,out] boundaryHit	Pre-allocated bool that will be set to true if the image boundaries have been exceeded during sampling.

	\return The resulting position with applied offsets.
	*/
	vector2f ComputeXCorInterpolated(vector2f initial, int iterations, int profileWidth, bool& boundaryHit);

	/*! \brief Execute the quadrant interpolation algorithm.
	
	See \cite loen:QI for algorithm details. See \ref QTrkComputedConfig for more information on the settings.
	
	\param [in] initial						The initial position from which to start the algorithm.
	\param [in] iterations					The number of iterations to perform.
	\param [in] radialSteps					Amount of radial steps per quadrant profile.
	\param [in] angularStepsPerQuadrant		Amount of angular sampling steps per profile.
	\param [in] angStepIterationFactor		Angular step iteration factor, see \ref QTrkSettings::qi_angstep_factor.
	\param [in] minRadius					Minimum radius of the sampling area in pixels.
	\param [in] maxRadius					Maximum radius of the sampling area in pixels.
	\param [in,out] boundaryHit				Pre-allocated bool that will be set to true if the image boundaries have been exceeded during sampling.
	\param [in]	radialweights				Array of the radial weights to use.

	\return The resulting position with applied offsets.
	*/
	vector2f ComputeQI(vector2f initial, int iterations, int radialSteps, int angularStepsPerQuadrant, 
		float angStepIterationFactor, float minRadius, float maxRadius, bool& boundaryHit, float* radialweights=0);

	/// Structure to group results from the 2D Gaussian fit.
	struct Gauss2DResult {
		vector2f pos;			///< Found position.
		float I0;				///< Found fit parameter.
		float bg;				///< Found fit parameter.
	};

	/*! \brief Calculate a 2D Gaussian fit on the image.

	\param [in] initial				The initial position from which to start the algorithm.
	\param [in] iterations			The number of iterations to perform.
	\param [in] sigma				Expected standard deviation of the gaussian fit.

	\return The fitting results.
	*/
	Gauss2DResult Compute2DGaussianMLE(vector2f initial, int iterations, float sigma);

	/*! \brief %QI helper function to calculate the offset from concatenated profiles.

	\param [in] qi_profile			The concatenated profiles' FFT from which to obtain the offset.
	\param [in] nr					The number of radial steps per quadrant profile.
	\param [in] axisForDebug		Axis number used to save intermediate data.

	\return The calculated offset.
	*/
	scalar_t QI_ComputeOffset(complex_t* qi_profile, int nr, int axisForDebug);

	/*! \brief QA helper function to calculate the offset from concatenated profiles.

	\param [in] profile				The concatenated profiles' FFT from which to obtain the offset.
	\param [in] zlut_prof_fft		The ZLUT profiles' FFT from which to obtain the offset.
	\param [in] nr					The number of radial steps per quadrant profile.
	\param [in] axisForDebug		Axis number used to save intermediate data.

	\return The calculated offset.
	*/
	scalar_t QuadrantAlign_ComputeOffset(complex_t* profile, complex_t* zlut_prof_fft, int nr, int axisForDebug);

	/*! \brief Find a measure for the asymmetry of the ROI.

	The center of mass of each radial line (see further explanation of the %QI algorithm) is calculated. The standard deviation of these
	COMs is returned as a measure of the asymmetry.

	See \ref QTrkComputedConfig for more information on the settings.

	\param [in] center			The center of the sampling area.
	\param [in] radialSteps		The number of radial steps per spoke.
	\param [in] angularSteps	The number of angular steps to take.
	\param [in] minRadius		Minimum sampling radius in pixels.
	\param [in] maxRadius		Maximum sampling radius in pixels.
	\param [in] dstAngProf		Optional. Pre-allocated array to hold the calculated COMs.

	\return Standard deviation of the radial spokes' centers of mass.
	*/
	float ComputeAsymmetry(vector2f center, int radialSteps, int angularSteps, float minRadius, float maxRadius, float *dstAngProf=0);

	/*! \brief Set the image on which to perform the tracking.

	\param [in] srcImage	Array with the image data.
	\param [in] srcpitch	Width of one row of data in bytes (typically size(dataType)*imageWidth).
	*/
	template<typename TPixel> void SetImage(TPixel* srcImage, uint srcpitch);

	/*! \brief Set an image with 16 bit type.

	\param [in] srcImage	Array with the image data.
	\param [in] srcpitch	Width of one row of data in bytes (typically size(dataType)*imageWidth).
	*/
	void SetImage16Bit(ushort* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }

	/*! \brief Set an image with 8 bit type.

	\param [in] srcImage	Array with the image data.
	\param [in] srcpitch	Width of one row of data in bytes (typically size(dataType)*imageWidth).
	*/
	void SetImage8Bit(uchar* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }

	/*! \brief Set an image with float type.

	\param [in] srcImage	Array with the image data.
	*/
	void SetImageFloat(float* srcImage);

	/*! \brief Save the tracker's image to a jpg file.

	\param [in] filename	Name of the file. Can also be full path.
	*/
	void SaveImage(const char *filename);

	/*! \brief Calculate the center of mass of the image.

	Also calculates and saves the mean and standard deviation of the image.
	Always performed as first step in localization for a first guess.

	The moments are calculated as absolute values around the mean. 
	An optional background correction is possible by only taking into account outliers.

	\param [in] bgcorrection	Factor of standard deviation not to take into account for the center of mass calculation. Default is 0.
	*/
	vector2f ComputeMeanAndCOM(float bgcorrection=0.0f);

	/*! \brief Wrapper to compute a radial profile.

	\param [in,out]	dst				Pre-allocated float array of size \p radialSteps to return the profile.
	\param [in]		radialSteps		Number of radial steps in the profile.
	\param [in]		angularSteps	Number of angular steps in the profile.
	\param [in]		minradius		Minimum sampling circle radius in pixels.
	\param [in]		maxradius		Maximum sampling circle radius in pixels.
	\param [in]		center			The center around which to sample.
	\param [in]		crp				No clue, good luck Josko!
	\param [in,out]	boundaryHit		Optional. Bool set to indicate whether the image boundary has been hit during sampling.
	\param [in]		normalize		Optional. Normalize the radial profile?
	*/
	void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius, vector2f center, bool crp, bool* boundaryHit=0, bool normalize=true);
	
	/*! \brief Wrapper to compute a radial profile.

	\param [in,out]	dst				Pre-allocated float array of size \p radialSteps to return the profile.
	\param [in]		radialSteps		Number of radial steps in the profile.
	\param [in]		angularSteps	Number of angular steps in the profile.
	\param [in]		quadrant		Quadrant number. 0-3, 1 is upper right, progresses in counterclockwise order.
	\param [in]		minRadius		Minimum sampling circle radius in pixels.
	\param [in]		maxRadius		Maximum sampling circle radius in pixels.
	\param [in]		center			The center around which to sample.
	\param [in]		radialWeights	Optional. Array of radial weights by which to weigh the profile. 
	*/
	void ComputeQuadrantProfile(scalar_t* dst, int radialSteps, int angularSteps, int quadrant, float minRadius, float maxRadius, vector2f center, float* radialWeights=0);
	
	/*! \brief Helper function to calculate the Z position.

	Calculates the radial profile with \ref CPUTracker::ComputeRadialProfile and calls \ref CPUTracker::LUTProfileCompare to compare it to the LUT.

	\param [in] center				The center around which to sample.
	\param [in] angularSteps		Number of angular steps in the profile.
	\param [in] zlutIndex			Index of the ZLUT/bead number. 
	\param [in,out] boundaryHit		Optional. Bool set to indicate whether the image boundary has been hit during sampling.
	\param [in,out] profile			Optional. Pre-allocated array to retrieve the profile if so desired.
	\param [in,out] cmpprof			Optional. Pre-allocated array to retrieve the error curve if so desired..
	\param [in] normalizeProfile	Optional. Normalize the profile? Default is true.			
	*/
	float ComputeZ(vector2f center, int angularSteps, int zlutIndex, bool* boundaryHit=0, float* profile=0, float* cmpprof=0, bool normalizeProfile=true)
	{
		float* prof = profile ? profile : ALLOCA_ARRAY(float, zlut_res);
		ComputeRadialProfile(prof,zlut_res,angularSteps, zlut_minradius, zlut_maxradius, center, false, boundaryHit, normalizeProfile);
		return LUTProfileCompare(prof, zlutIndex, cmpprof, LUTProfMaxQuadraticFit);
	}
	
	/*! \brief Calculate a 2D fourier transform of the tracker's image.

	See \ref FFT2D for more information.
	*/
	void FourierTransform2D();

	/*! \brief Calculate the radial profile based on the fourier transform.

	Setting is available, not ever really used.

	\param [in,out]	dst				Pre-allocated float array of size \p radialSteps to return the profile.
	\param [in]		radialSteps		Number of radial steps in the profile.
	\param [in]		angularSteps	Number of angular steps in the profile.
	\param [in]		minradius		Minimum sampling circle radius in pixels.
	\param [in]		maxradius		Maximum sampling circle radius in pixels.
	*/
	void FourierRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius);

	/*! \brief Normalize an image.

	See \ref normalize.

	\param [in] image		Optional. The image to normalize. If a null pointer (default) is given, uses the source image.
	*/
	void Normalize(float *image=0);

	/*! \brief Tell the tracker where in memory the LUT is located.

	The LUT is a large contiguous memory area in which all LUTs are saved. See \ref QueuedCPUTracker::zluts.

	It is possible to make one CPUTracker instance the owner of the ZLUT memory if no other manager (like QueuedCPUTracker) is used.
	For this, use the \p copyMemory parameter.

	\param [in] data			Pointer to the first element of the ZLUT memory.
	\param [in] planes			The number of planes per lookup table.
	\param [in] res				Number of radial steps per plane.
	\param [in] num_zluts		The number of LUTs in memory.
	\param [in]	minradius		Starting sampling distance of the profiles in pixels.
	\param [in] maxradius		Ending sampling distance of the profiles in pixels.
	\param [in] copyMemory		Bool to indicate whether the data should be copied to locally managed memory.
	\param [in] useCorrelation	Deprecated. No effect, only there not to break calls.
	*/
	void SetRadialZLUT(float* data, int planes, int res, int num_zluts, float minradius, float maxradius, bool copyMemory, bool useCorrelation);
	
	/*! \brief Set the radial weights to be used for profile comparisons.

	\param [in] radweights		Array with the radial weights.

	*/
	void SetRadialWeights(float* radweights);

	/// Settings to compute the Z coordinate from the error curve.
	enum LUTProfileMaxComputeMode { 
		LUTProfMaxQuadraticFit,			///< Default. Use a 7-point quadratic fit around the error curve's maximum.
		LUTProfMaxSplineFit,			///< Compute a spline fit.
		LUTProfMaxSimpleInterp			///< Unimplemented.
	};

	/*! \brief Calculate the error curve from a specific profile and LUT.

	\param [out] errorcurve_dest		Pre-allocated output array of size \ref zlut_planes which will be filled with the error curve.
	\param [in] profile					Array with the profile to compare against the LUT.
	\param [in] zlut_sel				Pointer to the first element of the selected LUT.
	*/
	void CalculateErrorCurve(double* errorcurve_dest, float* profile, float* zlut_sel);

	/*! \brief Calculates an intermediate ZLUT profile between two planes.

	At each radial step, the value is linearly interpolated based on the two nearest saved profiles.
	Use to obtain estimated profiles at non-integer z values.

	\param [out] profile_dest			Pre-allocated output array of size \ref zlut_res which will be filled with the error curve.
	\param [in] z						The height at which to determine the expected profile.
	\param [in] zlutIndex				The index of the zlut/bead.
	*/
	void CalculateInterpolatedZLUTProfile(float* profile_dest, float z, int zlutIndex);

	/*! \brief WIP. Calculate the error flag between two profiles.

	\param [in] prof1		The first profile. Typically the measured profile.
	\param [in] prof2		The second profile. Typically a profile from the LUT.

	\return	The error value between the two profiles. If 'high', tracking was unreliable.
	*/
	float CalculateErrorFlag(double* prof1, double* prof2);

	/*! \brief Compare a profile to a LUT and calculate the optimum Z value for it.

	\param [in] profile			The profile to compare.
	\param [in]	zlutIndex		The number of the ZLUT to compare with.
	\param [in] cmpProf			Pre-allocated array in which to return the error curve. Not used if null pointer is passed.
	\param [in] maxPosMethod	Which method to use to determine the Z position from the error curve. See \ref LUTProfileMaxComputeMode.
	\param [in] fitcurve		Optional. Pre-allocated array in which to return the error curve fit. Not used if null pointer is passed.
	\param [in] maxPos			Optional. Pre-allocated integer in which the position of the error curve's maximum will be put.
	\param [in] frameNum		Optional. The number of the frame this track job belongs to. Only used in testRun.
	*/
	float LUTProfileCompare(float* profile, int zlutIndex, float* cmpProf, LUTProfileMaxComputeMode maxPosMethod, float* fitcurve=0, int *maxPos=0, int frameNum = 0);
	
	/*! \brief Compare a profile to a LUT and calculate the optimum Z value for it using an initial estimate.

	\warning Specific use unknown, only called when \ref LT_LocalizeZWeighted is enabled. Probably was used to test the effect of different weights and fit methods.

	\param [in] rprof			The profile to compare.
	\param [in] zlutIndex		The number of the ZLUT to compare against.
	\param [in] z_estim			An initial guess for the z position.
	*/
	float LUTProfileCompareAdjustedWeights(float* rprof, int zlutIndex, float z_estim);

	/*! \brief Get the debug image.

	The debug image can be used to store intermediate results of image processing and retrieved using this function.

	\return A pointer to the array with the debug image in it.
	*/
	float* GetDebugImage() { return debugImage; }

	/*! \brief Scale the input image with the background calibration values.

	See \ref QueuedTracker::SetPixelCalibrationImages() for more information.

	\param [in] offset				Array with per-pixel offsets.
	\param [in] gain				Array with per-pixel gains.
	\param [in] offsetFactor		Factor by which to scale the offsets.
	\param [in] gainFactor			Factor by which to scale the gains.
	*/
	void ApplyOffsetGain(float *offset, float *gain, float offsetFactor, float gainFactor);

	/*! \brief Initializes the fft handles.

	Readies \ref qi_fft_forward and \ref qi_fft_backward for use.

	\param [in] nsteps The number of radial steps per profile. FFT length is 2 * nsteps.
	*/
	void AllocateQIFFTs(int nsteps);

	/*! \brief Perform the quadrant align algorithm.

	See \ref LT_ZLUTAlign.

	\param [in] initial							The initial, 3 dimensional position.
	\param [in] beadIndex						The index of the bead/zlut.
	\param [in] angularStepsPerQuadrant			Number of angular steps per quadrant.
	\param [in] boundaryHit						Pre-allocated bool to return whether the image boundary has been hit.

	\return	The updated 3D position.
	*/
	vector3f QuadrantAlign(vector3f initial, int beadIndex, int angularStepsPerQuadrant, bool& boundaryHit);
};

template<typename TPixel>
void CPUTracker::SetImage(TPixel* data, uint pitchInBytes)
{
	uchar* bp = (uchar*)data;

	for (int y=0;y<height;y++) {
		for (int x=0;x<width;x++) {
			srcImage[y*width+x] = ((TPixel*)bp)[x];
		}
		bp += pitchInBytes;
	}

	mean=0.0f;
}
