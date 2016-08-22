using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using System.Text;

namespace QTrkDotNet
{
    public struct QTrkSettings
    {
	    int width, height;
	    int numThreads;
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

    public class QTrk
    {
        public const string DllName = "qtrkcuda.dll";

        [DllImport(DllName)] public static extern IntPtr QTrkCreateInstance(QTrkSettings *cfg);
        [DllImport(DllName)] public static extern void QTrkFreeInstance(QueuedTracker* qtrk);

        // C API, mainly intended to allow binding to .NET
        [DllImport(DllName)] public static extern void QTrkSetLocalizationMode(QueuedTracker* qtrk, LocMode_t locType);

        // Frame and timestamp are ignored by tracking code itself, but usable for the calling code
        // Pitch: Distance in bytes between two successive rows of pixels (e.g. address of (0,0) -  address of (0,1) )
        // ZlutIndex: Which ZLUT to use for ComputeZ/BuildZLUT
        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkScheduleLocalization(QueuedTracker* qtrk, void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo);
        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkClearResults(QueuedTracker* qtrk);
        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkFlush(QueuedTracker* qtrk); // stop waiting for more jobs to do, and just process the current batch

        // Schedule an entire frame at once, allowing for further optimizations
        [DllImport(DllName)] public static extern int DLL_CALLCONV QTrkScheduleFrame(QueuedTracker* qtrk, void *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo);
	
        // data can be zero to allocate ZLUT data. zcmp has to have 'zlut_radialsteps' elements
        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkSetRadialZLUT(QueuedTracker* qtrk, float* data, int count, int planes, float* zcmp=0); 
        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkGetRadialZLUT(QueuedTracker* qtrk, float* dst);
        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkGetRadialZLUTSize(QueuedTracker* qtrk, int* count, int* planes, int* radialsteps);

        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkBuildLUT(QueuedTracker* qtrk, void* data, int pitch, QTRK_PixelDataType pdt, bool imageLUT, int plane);
        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkFinalizeLUT(QueuedTracker* qtrk);
	
        [DllImport(DllName)] public static extern int DLL_CALLCONV QTrkGetResultCount(QueuedTracker* qtrk);
        [DllImport(DllName)] public static extern int DLL_CALLCONV QTrkFetchResults(QueuedTracker* qtrk, LocalizationResult* results, int maxResults);

        [DllImport(DllName)] public static extern int DLL_CALLCONV QTrkGetQueueLength(QueuedTracker* qtrk, int *maxQueueLen);
        [DllImport(DllName)] public static extern bool DLL_CALLCONV QTrkIsIdle(QueuedTracker* qtrk);

        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkGetProfileReport(QueuedTracker* qtrk, char *dst, int maxStrLen);
        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkGetWarnings(QueuedTracker* qtrk, char *dst, int maxStrLen);

        [DllImport(DllName)] public static extern void DLL_CALLCONV QTrkGetComputedConfig(QueuedTracker* qtrk, QTrkComputedConfig* cfg);


    }
}
