/*
Labview API for CPU tracker
*/
#include "std_incl.h"
#include <Windows.h>
#undef min
#undef max

#include "random_distr.h"
#include "labview.h"
#include "cpu_tracker.h"



CDLL_EXPORT CPUTracker* DLL_CALLCONV create_tracker(uint w, uint h, uint xcorw)
{
	try {
		Sleep(300);
		return new CPUTracker(w,h,xcorw);
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
		return 0;
	}
}

CDLL_EXPORT void DLL_CALLCONV destroy_tracker(CPUTracker* tracker)
{
	try {
		delete tracker;
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}


CDLL_EXPORT void DLL_CALLCONV compute_com(CPUTracker* tracker, float* out)
{
	vector2f com = tracker->ComputeMeanAndCOM();
	out[0] = com.x;
	out[1] = com.y;
}

CDLL_EXPORT int DLL_CALLCONV compute_xcor(CPUTracker* tracker, vector2f* position, int iterations, int profileWidth)
{
	bool boundaryHit;
	*position = tracker->ComputeXCorInterpolated(*position, iterations, profileWidth, boundaryHit);

	return boundaryHit ? 1 : 0;
}


CDLL_EXPORT int DLL_CALLCONV compute_qi(CPUTracker* tracker, vector2f* position, int iterations, int radialSteps, int angularStepsPerQ, float minRadius, float maxRadius, LVArray<float>** radialweights)
{
	bool boundaryHit;

	float* rw = 0;
	if (radialweights && (*radialweights)->dimSize == radialSteps)
		rw = (*radialweights)->elem;

	*position = tracker->ComputeQI(*position, iterations, radialSteps, angularStepsPerQ, 1, minRadius,maxRadius, boundaryHit, rw);
	return boundaryHit ? 1 : 0;
}


CDLL_EXPORT void DLL_CALLCONV set_image_from_memory(CPUTracker* tracker, LVArray2D<uchar>** pData, ErrorCluster* error)
{
	LVArray2D<uchar>* data = *pData;
	if (data->dimSizes[0] != tracker->GetWidth() || data->dimSizes[1] != tracker->GetHeight()) {
		ArgumentErrorMsg(error, "Given image has invalid dimensions");
		return;
	}
	tracker->SetImage8Bit( data->elem, tracker->GetWidth() );
}



CDLL_EXPORT void DLL_CALLCONV set_image_u8(CPUTracker* tracker, LVArray2D<uchar>** pData, ErrorCluster* error)
{
	LVArray2D<uchar>* data = *pData;
	if (data->dimSizes[0] != tracker->GetWidth() || data->dimSizes[1] != tracker->GetHeight()) {
		ArgumentErrorMsg(error, "Given image has invalid dimensions");
		return;
	}
	tracker->SetImage8Bit( data->elem, tracker->GetWidth() );
}

CDLL_EXPORT void DLL_CALLCONV set_image_u16(CPUTracker* tracker, LVArray2D<ushort>** pData, ErrorCluster* error)
{
	LVArray2D<ushort>* data = *pData;
	if (data->dimSizes[0] != tracker->GetWidth() || data->dimSizes[1] != tracker->GetHeight()) {
		ArgumentErrorMsg(error, "Given image has invalid dimensions");
		return;
	}
	tracker->SetImage16Bit( data->elem, tracker->GetWidth()*sizeof(ushort) );
}

CDLL_EXPORT void DLL_CALLCONV set_image_float(CPUTracker* tracker, LVArray2D<float>** pData, ErrorCluster* error)
{
	LVArray2D<float>* data = *pData;
	if (data->dimSizes[0] != tracker->GetWidth() || data->dimSizes[1] != tracker->GetHeight()) {
		ArgumentErrorMsg(error, "Given image has invalid dimensions");
		return;
	}
	tracker->SetImageFloat( data->elem );
}


CDLL_EXPORT float DLL_CALLCONV compute_z(CPUTracker* tracker, float* center, int angularSteps, int zlut_index, uint *error, LVArray<float>** profile, int* bestIndex, LVArray<float>** errorCurve)
{
	bool boundaryHit=false;
	if (profile) 
		ResizeLVArray(profile, tracker->zlut_res);

	if (errorCurve) {
		ResizeLVArray(errorCurve, tracker->zlut_planes);
	}

	int maxPos;
	float* prof = profile ? (*profile)->elem : ALLOCA_ARRAY(float, tracker->zlut_res);
	tracker->ComputeRadialProfile(prof,tracker->zlut_res,angularSteps, tracker->zlut_minradius, tracker->zlut_maxradius,*(vector2f*) center, false, &boundaryHit, true);
	float z= tracker->LUTProfileCompare(prof, zlut_index, errorCurve ? (*errorCurve)->elem : 0, CPUTracker::LUTProfMaxQuadraticFit, 0, &maxPos);

	if (bestIndex) *bestIndex=maxPos;

	if (error)
		*error = boundaryHit?1:0;
	return z;
}


CDLL_EXPORT void DLL_CALLCONV get_debug_img_as_array(CPUTracker* tracker, LVArray2D<float>** pdbgImg)
{
	float* src = tracker->GetDebugImage();
	if (src) {
		ResizeLVArray2D(pdbgImg, tracker->GetHeight(), tracker->GetWidth());
		LVArray2D<float>* dst = *pdbgImg;

	//	dbgout(SPrintf("copying %d elements to Labview array\n", len));
		for (int i=0;i<dst->numElem();i++)
			dst->elem[i] = src[i];
	}
}

CDLL_EXPORT void DLL_CALLCONV compute_crp(CPUTracker* tracker, LVArray<float>** result, int radialSteps, float *radii, float* center, uint* boundaryHit, LVArray2D<float>** crpmap)
{
}


CDLL_EXPORT float DLL_CALLCONV compute_asymmetry(CPUTracker* tracker, LVArray<float>** result, int radialSteps, float *radii, float* center, uint* boundaryHit)
{
	LVArray<float>* dst = *result;
	bool bhit = false;
	float asym = tracker->ComputeAsymmetry(*(vector2f*)center, radialSteps, dst->dimSize, radii[0], radii[1], dst->elem);
	if (boundaryHit) *boundaryHit = bhit ? 1 : 0;
	return asym;
}


CDLL_EXPORT void DLL_CALLCONV compute_radial_profile(CPUTracker* tracker, LVArray<float>** result, int angularSteps, float *radii, float* center, uint* boundaryHit)
{
	LVArray<float>* dst = *result;
	bool bhit = false;
	tracker->ComputeRadialProfile(&dst->elem[0], dst->dimSize, angularSteps, radii[0], radii[1], *(vector2f*)center, false, &bhit);

	if (boundaryHit) *boundaryHit = bhit ? 1 : 0;
}




CDLL_EXPORT void DLL_CALLCONV set_ZLUT(CPUTracker* tracker, LVArray3D<float>** pZlut, float *radii, int angular_steps, bool useCorrelation, LVArray<float>** radialweights, bool normalize)
{
	LVArray3D<float>* zlut = *pZlut;

	int numLUTs = zlut->dimSizes[0];
	int planes = zlut->dimSizes[1];
	int res = zlut->dimSizes[2];

	if (normalize) {
		NormalizeZLUT( zlut->elem , numLUTs, planes, res);
	}

	tracker->SetRadialZLUT(zlut->elem, planes, res, numLUTs, radii[0], radii[1], true, useCorrelation);
	if (radialweights)
		tracker->SetRadialWeights( ((*radialweights)->dimSize>0) ? (*radialweights)->elem : 0);
}

CDLL_EXPORT void DLL_CALLCONV get_ZLUT(CPUTracker* tracker, int zlutIndex, LVArray2D<float>** dst)
{
	 float* zlut = tracker->GetRadialZLUT(zlutIndex);
	 ResizeLVArray2D(dst, tracker->zlut_planes, tracker->zlut_res);
	 std::copy(zlut, zlut+(tracker->zlut_planes*tracker->zlut_res), (*dst)->elem);
}


CDLL_EXPORT void DLL_CALLCONV generate_test_image(LVArray2D<float> **img, float xp, float yp, float size, float photoncount)
{
	try {
		ImageData data((*img)->elem, (*img)->dimSizes[1],(*img)->dimSizes[0]);
		GenerateTestImage(data, xp, yp, size, photoncount);
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}




CDLL_EXPORT void DLL_CALLCONV generate_image_from_lut(LVArray2D<float>** image, LVArray2D<float>** lut, 
					float *LUTradii, vector3f* position, float pixel_max, int useSplineInterp, int samplesPerPixel)
{
	ImageData img((*image)->elem, (*image)->dimSizes[1], (*image)->dimSizes[0]);
	ImageData zlut((*lut)->elem, (*lut)->dimSizes[1], (*lut)->dimSizes[0]);

	GenerateImageFromLUT(&img, &zlut, LUTradii[0], LUTradii[1], *position, useSplineInterp!=0, samplesPerPixel);
	if (pixel_max>0) {
		img.normalize();
		ApplyPoissonNoise(img, pixel_max);
	}
}

