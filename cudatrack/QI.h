#pragma once

/// Datatype for representation of complex values. 2D Float.
typedef float2 qicomplex_t;

/// Structure to hold QI settings. See \ref QTrkComputedConfig for more info on most settings.
struct QIParams {
	float minRadius;
	float maxRadius;
	int radialSteps;
	int iterations;
	int maxAngularSteps;	///< Maximum amount of angular steps. Used to enable usage of \ref QTrkSettings::qi_angstep_factor.
	float2* cos_sin_table;	///< Reference to \ref QI::DeviceInstance::qi_trigtable for use in kernels. Radial sampling points.
};

/*! \brief Class to specifically perform quadrant interpolation calculations on the GPU.

Maintains all memory and functions related to %QI to keep QueuedCUDATracker more clean.
*/
class QI
{
public:
	/*! \brief Contains all QI memory space that is allocated per stream. */
	struct StreamInstance {
		device_vec<float> d_shiftbuffer;			///< Calculation buffer. Size is \ref batchSize * \ref qi_FFT_length.
		device_vec<float2> d_QIprofiles;			///< Memory to hold the concatenated X and Y profiles generated from the quadrant profiles. In-place forward and backward FFT is performed in this memory. Size is \ref batchSize * (2 axes) * (2 * \ref QIParams::radialSteps).
		device_vec<float2> d_QIprofiles_reverse;	///< Memory to hold the reversed X and Y profiles. In-place forward FFT is performed in this memory. Size is \ref batchSize * (2 axes) * (2 * \ref QIParams::radialSteps).
		device_vec<float> d_quadrants;				///< Memory to hold the quadrant profiles. Size is \ref batchSize * (4 quadrants) * \ref QIParams::radialSteps.
#if defined(GPU_DEBUG) || defined(DOXYGEN)
		/*! \brief Extra device vector specifically for arbitrary debug output
		GPU_DEBUG is defined at the top of gpu_utils.h

		USAGE:
		Fill with any kind of data in a function
		Output in any host code to a csv file using
		\ref DbgOutputVectorToFile.
		*/
		device_vec<float> d_DebugOutput;
#endif
		cufftHandle fftPlan; ///< Handle to make calls to [CUFFT](http://docs.nvidia.com/cuda/cufft/). A single CUFFT plan can be used for both forward and inverse transforms.

		/*! \brief Delete the stream instance.
		
		\bug Why aren't the device_vec instances deleted?
		*/
		~StreamInstance() {
			cufftDestroy(fftPlan);
		}

		/*! \brief Return the total size of memory in bytes used for QI by each stream.
		
		\return Memory size in bytes.
		*/
		int memsize() {
			size_t fftSize;
			cufftGetSize(fftPlan,&fftSize);
			return d_QIprofiles.memsize() + d_QIprofiles_reverse.memsize() + d_quadrants.memsize() + fftSize;
		}

		cudaStream_t stream; ///< Reference to the parent stream, managed by \ref QueuedCUDATracker::Stream.
	};

	/*! \brief Contains all QI memory space that is allocated per device (shared between streams). */
	struct DeviceInstance
	{
		device_vec<float2> qi_trigtable;	///< Sampling points for radial resampling. Calculated once in \ref InitDevice.	
		QIParams * d_qiparams;				///< Instance of common QI parameters.
		device_vec<float> d_radialweights;	///< The radial weights used in ZLUT profile calculation.

		/// Free the device QI memory.
		~DeviceInstance() { cudaFree(d_qiparams); }
		/// Instantiate the device QI memory.
		DeviceInstance() { d_qiparams=0; }
	};

	/*! \brief Execute a batch of QI calculations.
	Runs \ref Iterate the correct number of times and recalculates the number of radial spokes for every iteration,
	based on QTrkComputedConfig::qi_angstep_factor.

	\param [in] p			Reference to \ref BaseKernelParams with parameters for this call.
	\param [in] cfg			Reference to the settings to use.
	\param [in] s			The QI::StreamInstance to use.
	\param [in] d			The QI::DeviceInstance to use.
	\param [in] initial		Vector with the initial positions from which to start the radial samplings on the first iteration.
	\param [out] output		Vector that will be filled with the final results.
	*/
	template<typename TImageSampler>
	void Execute (BaseKernelParams& p, const QTrkComputedConfig& cfg, StreamInstance* s, DeviceInstance* d, device_vec<float3>* initial, device_vec<float3> *output);

	/*! \brief Ready a device for %QI calculations.

	Calculate required parameters and fill a \ref DeviceInstance.

	\param [in,out] d		The instance to be initialized.
	\param [in]		cc		The configuration used for the algorithm.
	*/
	void InitDevice(DeviceInstance* d, QTrkComputedConfig& cc);

	/*! \brief Ready a stream for %QI calculations.

	Calculate required parameters and fill a \ref StreamInstance.

	\param [in,out] s			The instance to be initialized.
	\param [in]		cc			The configuration used for the algorithm.
	\param [in]		stream		The normal CUDA stream this %QI Stream will relate to.
	\param [in]		batchSize	The calculation batch size. See \ref batchSize.
	*/
	void InitStream(StreamInstance* s, QTrkComputedConfig& cc, cudaStream_t stream, int batchSize);

	/*! \brief Initialize this %QI instance.

	Copy relevant settings to local datastructures.

	\param [in]		cfg			The configuration used for the algorithm.
	\param [in]		batchSize	The calculation batch size. See \ref batchSize.
	*/
	void Init(QTrkComputedConfig& cfg, int batchSize);

private:
	/*! \brief Perform one %QI iteration.

	This is where the actual algorithm is executed. See \cite loen:QI for details.

	The kernels are called in the following order:
	- \ref QI_ComputeQuadrants - Calculate the 4 quadrant profiles
	- \ref QI_QuadrantsToProfiles - Convert the quadrant profiles into the concatenated and their respective reverse profiles
	- Forward FFT ([CUFFT](http://docs.nvidia.com/cuda/cufft/)) - Calculate the fourier transforms of the profiles and reverse profiles
	- \ref QI_MultiplyWithConjugate - Multiply the transforms to calculate their autocorrelation 
	- Reverse FFT ([CUFFT](http://docs.nvidia.com/cuda/cufft/)) - Transform the multiplied transforms back to the time-domain autocorrelation
	- \ref QI_OffsetPositions - Calculate and apply the X and Y shifts from the autocorrelations

	\param [in]		p				KernelParamaters to use.
	\param [in]		initial			The initial positions to use as sampling centers.
	\param [out]	output			Vector with the resulting positions.
	\param [in]		s				The stream to execute.
	\param [in]		d				The device on which to execute.
	\param [in]		angularSteps	The number of angular steps to use.
	*/
	template<typename TImageSampler>
	void Iterate(BaseKernelParams& p, device_vec<float3>* initial, device_vec<float3>* output, StreamInstance *s, DeviceInstance* d, int angularSteps);
	
	QIParams params;		///< Structure with settings relevant to quadrant interpolation.
	
	/*! \brief Parameter for length required for arrays going into FFT. 2 * radial steps.

	Used to be nearest power of two, but since switching to cuFFT, this is not needed for speed optimization anymore.
	*/
	int qi_FFT_length;	
	int batchSize;			///< See \ref QueuedCUDATracker::batchSize. Local copy.
	int numThreads;			///< See \ref QueuedCUDATracker::numThreads.
	
	/// Same function as \ref QueuedCUDATracker::blocks.
	dim3 blocks(int njobs) { return dim3((njobs+numThreads-1)/numThreads); }
	/// Same function as \ref QueuedCUDATracker::threads.
	dim3 threads() { return dim3(numThreads); }
};