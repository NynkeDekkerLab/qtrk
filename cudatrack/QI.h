
#pragma once


// Types used by QI algorithm
typedef float qivalue_t;
typedef float2 qicomplex_t;

struct QIParams {
	float minRadius, maxRadius;
	int radialSteps, iterations, trigtablesize;
	float2* cos_sin_table;
};

class QI
{
public:
	// Contains all QI memory space that is allocated per stream
	struct StreamInstance {
		device_vec<float> d_shiftbuffer; // [QI_fftlength * njobs] ComputeMaxInterp temp space
		device_vec<float2> d_QIprofiles;
		device_vec<float2> d_QIprofiles_reverse;
		device_vec<float> d_quadrants;
		cufftHandle fftPlan; // a CUFFT plan can be used for both forward and inverse transforms

		~StreamInstance() {
			cufftDestroy(fftPlan);
		}

		int memsize() {
			return d_QIprofiles.memsize() + d_QIprofiles_reverse.memsize() + d_quadrants.memsize();
		}
		cudaStream_t stream;
	};
	// Contains all QI memory space that is allocated per device (shared between streams)
	struct DeviceInstance
	{
		device_vec<float2> qi_trigtable;
		QIParams * d_qiparams;

		~DeviceInstance() { cudaFree(d_qiparams); }
		DeviceInstance() { d_qiparams=0; }
	};

	void Execute (BaseKernelParams& p, const QTrkComputedConfig& cfg, StreamInstance* s, DeviceInstance* d, device_vec<float3>* initial, device_vec<float3> *output, bool useTexCache);

	void InitDevice(DeviceInstance*d,  QTrkComputedConfig& cc);
	void InitStream(StreamInstance* s, QTrkComputedConfig& cc, cudaStream_t stream, int batchSize);
	void Init(QTrkComputedConfig& cfg, int batchSize);

private:
	template<typename TImageSampler>
	void Iterate(BaseKernelParams& p, device_vec<float3>* initial, device_vec<float3>* output, StreamInstance *s, DeviceInstance* d, int angularSteps);
	
	QIParams params;

	// QI profiles need to have power-of-two dimensions. qiProfileLen stores the closest power-of-two value that is bigger than cfg.qi_radialsteps
	int qi_FFT_length;
	int batchSize;
	int numThreads;
	
	dim3 blocks(int njobs) { return dim3((njobs+numThreads-1)/numThreads); }
	dim3 threads() { return dim3(numThreads); }
};



