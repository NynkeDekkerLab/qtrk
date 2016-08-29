
#include "std_incl.h"
#include "labview.h"

#include "QueuedTracker.h"

/** \addtogroup lab_functions
	@{
*/
enum QueueFrameFlags {
	QFF_Force32Bit = 0x7fffffff
};

#pragma pack(push,1)
struct CUDADeviceInfo 
{
	LStrHandle name;
	int clockRate;
	int multiProcCount;
	int major, minor;
};
#pragma pack(pop)

#if defined(CUDA_TRACK) || defined(DOXYGEN)
static bool CheckCUDAErrorLV(cudaError err, ErrorCluster* e)
{
	if (err != cudaSuccess) {
		const char* errstr = cudaGetErrorString(err);
		FillErrorCluster(kAppErrorBase, SPrintf("CUDA error: %s", errstr).c_str(), e);
		return false;
	}
	return true;
}
#endif
/** @} */

/** \defgroup lab_API API - LabVIEW
\brief API functions available to LabVIEW. 

These DLLs are compiled by the \a lvcputrack and \a lvcudatrack projects.
*/

/** \addtogroup lab_API
	@{
*/

/** \defgroup lab_API_general General
\ingroup lab_API
\brief General API functions available to LabVIEW. 
*/

/** \addtogroup lab_API_general
	@{
*/
CDLL_EXPORT QueuedTracker* DLL_CALLCONV qtrk_create(QTrkSettings* settings, LStrHandle warnings, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_destroy(QueuedTracker* qtrk, ErrorCluster* error);
CDLL_EXPORT void DLL_CALLCONV	qtrk_free_all();

CDLL_EXPORT void				qtrk_set_localization_mode(QueuedTracker* qtrk, uint locType, ErrorCluster* e);
CDLL_EXPORT void				qtrk_find_beads(uint8_t* image, int pitch, int w,int h, int* smpCornerPos, int roi, float imgRelDist, float acceptance, LVArray2D<uint32_t> **output);
/** @} */

/** \defgroup lab_API_luts ZLUT
\ingroup lab_API
\brief ZLUT API functions available to LabVIEW. 
*/

/** \addtogroup lab_API_luts
	@{
*/
CDLL_EXPORT void DLL_CALLCONV	qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut, LVArray<float>** zcmpWindow, int normalize, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_get_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pzlut, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_set_image_lut(QueuedTracker* qtrk, LVArrayND<float,4>** imageLUT,  LVArray3D<float>** radialZLUT, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_get_image_lut(QueuedTracker* qtrk, LVArrayND<float, 4>** imageLUT, ErrorCluster* e);
CDLL_EXPORT void				qtrk_build_lut_plane(QueuedTracker* qtrk, LVArray3D<float> **data, uint flags, int plane, ErrorCluster* err);
CDLL_EXPORT void				qtrk_finalize_lut(QueuedTracker* qtrk, ErrorCluster *e);
/** @} */

/** \defgroup lab_API_queueing Queueing
\ingroup lab_API
\brief Queue API functions available to LabVIEW. 

Use to queue images to be handled by the tracker.
*/

/** \addtogroup lab_API_queueing
	@{
*/
CDLL_EXPORT int					qtrk_get_queue_len(QueuedTracker* qtrk, int* maxQueueLen, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_queue_u16(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<ushort>** data, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV	qtrk_queue_u8(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<uchar>** data, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV	qtrk_queue_float(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<float>** data, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV	qtrk_queue_pitchedmem(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV	qtrk_queue_array(QueuedTracker* qtrk,  ErrorCluster* error,LVArray2D<uchar>** data,uint pdt, const LocalizationJob *jobInfo);
CDLL_EXPORT uint DLL_CALLCONV	qtrk_queue_frame(QueuedTracker* qtrk, uchar* image, int pitch, int w,int h, uint pdt, ROIPosition* pos, int numROI, const LocalizationJob *pJobInfo, QueueFrameFlags flags, ErrorCluster* e);
/** @} */

/** \defgroup lab_API_results Results
\ingroup lab_API
\brief Result API functions available to LabVIEW. 
*/

/** \addtogroup lab_API_results
	@{
*/
CDLL_EXPORT int					qtrk_resultcount(QueuedTracker* qtrk, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_clear_results(QueuedTracker* qtrk, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_flush(QueuedTracker* qtrk, ErrorCluster* e);
CDLL_EXPORT int DLL_CALLCONV	qtrk_get_results(QueuedTracker* qtrk, LocalizationResult* results, int maxResults, int sortByID, ErrorCluster* e);
CDLL_EXPORT int DLL_CALLCONV	qtrk_idle(QueuedTracker* qtrk, ErrorCluster* e);
/** @} */

/** \defgroup lab_API_debug Debugging
\ingroup lab_API
\brief Debug API functions available to LabVIEW. 
*/

/** \addtogroup lab_API_debug
	@{
*/
CDLL_EXPORT void DLL_CALLCONV	qtrk_set_logfile_path(const char* path);
CDLL_EXPORT void DLL_CALLCONV	qtrk_get_computed_config(QueuedTracker* qtrk, QTrkComputedConfig* cc, ErrorCluster *err);
CDLL_EXPORT void DLL_CALLCONV	qtrk_set_pixel_calib_factors(QueuedTracker* qtrk, float offsetFactor, float gainFactor, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_set_pixel_calib(QueuedTracker* qtrk, LVArray3D<float>** offset, LVArray3D<float>** gain, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrk_dump_memleaks();
CDLL_EXPORT void				qtrk_get_profile_report(QueuedTracker* qtrk, LStrHandle str);
CDLL_EXPORT void				qtrk_enable_zlut_cmpprof(QueuedTracker* qtrk, bool enable, ErrorCluster* e);
CDLL_EXPORT void				qtrk_get_zlut_cmpprof(QueuedTracker* qtrk, LVArray2D<float> ** output, ErrorCluster* e);
CDLL_EXPORT int					qtrk_get_debug_image(QueuedTracker* qtrk, int id, LVArray2D<float>** data, ErrorCluster* e);
CDLL_EXPORT void				qtrk_compute_fisher(LVArray2D<float> **lut, QTrkSettings* cfg, vector3f* pos, LVArray2D<float> ** fisherMatrix, LVArray2D<float> ** inverseMatrix, vector3f* xyzVariance, int Nsamples, float maxPixelValue);
CDLL_EXPORT void				qtrk_test_array_passing(int n, LVArray<float>** flt1D, LVArray2D<float>** flt2D, LVArray<int>** int1D, LVArray2D<int>** int2D);
CDLL_EXPORT void				qtrk_simulate_tracking(QueuedTracker* qtrk, int nsmp, int beadIndex, vector3f* centerPos, vector3f* range, vector3f *outBias, vector3f* outScatter, float photonsPerWell, ErrorCluster* e);

CDLL_EXPORT void DLL_CALLCONV	qtrk_generate_gaussian_spot(LVArray2D<float>** image, vector2f* pos, float sigma, float I0, float Ibg, int applyNoise);
CDLL_EXPORT void DLL_CALLCONV	qtrk_generate_image_from_lut(LVArray2D<float>** image, LVArray2D<float>** lut, float *LUTradii, vector2f* position, float z, float M, float sigma_noise);
/** @} */

/** \defgroup lab_API_zbias Z Bias
\ingroup lab_API
\brief Z bias API functions available to LabVIEW. 

See also \ref zbias.
*/

/** \addtogroup lab_API_zbias
	@{
*/
CDLL_EXPORT void				qtrk_compute_zlut_bias_table(QueuedTracker* qtrk, int bias_planes, LVArray2D<float>** lvresult, int smpPerPixel, int useSplineInterp,ErrorCluster* e);
CDLL_EXPORT void				qtrk_set_zlut_bias_table(QueuedTracker* qtrk, LVArray2D<float>** biastbl, ErrorCluster* e);
/** @} */

#if defined(CUDA_TRACK) || defined(DOXYGEN)
/** \defgroup lab_API_cuda CUDA
\ingroup lab_API
\brief Cuda API functions available to LabVIEW. 
*/

/** \addtogroup lab_API_cuda
	@{
*/
CDLL_EXPORT int DLL_CALLCONV	qtrkcuda_device_count(ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV	qtrkcuda_set_device_list(LVArray<int>** devices);
CDLL_EXPORT void				qtrkcuda_get_device(int device, CUDADeviceInfo *info, ErrorCluster* e);
CDLL_EXPORT void				qtrkcuda_enable_texture_cache(QueuedTracker* qtrk, int enable, ErrorCluster* e);
/** @} */
#endif

/** @} */