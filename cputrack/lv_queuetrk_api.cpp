/**
\file
Labview API for the functionality in QueuedTracker.h
*/
#include "std_incl.h"
#include "utils.h"
#include "labview.h"
#include "QueuedTracker.h"
#include "threads.h" 

#include "lv_qtrk_api.h"

#include "ResultManager.h"
#include "FisherMatrix.h"
#include "BeadFinder.h"

static Threads::Mutex trackerListMutex;
static std::vector<QueuedTracker*> trackerList;

CDLL_EXPORT void DLL_CALLCONV qtrk_free_all()
{
	trackerListMutex.lock();
	DeleteAllElems(trackerList);
	trackerListMutex.unlock();
}

void SetLVString (LStrHandle str, const char *text)
{
	int msglen = strlen(text);
	MgErr err = NumericArrayResize(uB, 1, (UHandle*)&str, msglen);
	if (!err)
	{
		MoveBlock(text, LStrBuf(*str), msglen);
		LStrLen(*str) = msglen;
	}
}

std::vector<std::string> LVGetStringArray(int count, LStrHandle *str)
{
	std::vector<std::string> result(count);

	for (int x=0;x<count;x++) {
		uChar *val = LHStrBuf(str[x]);
		int len = LHStrLen (str[x]);
		result[x]=std::string((const char*)val, (size_t)len );
	}
	return result;
}

MgErr FillErrorCluster(MgErr err, const char *message, ErrorCluster *error)
{
	if (err)
	{
		error->status = LVBooleanTrue;
		error->code = err;
		SetLVString ( error->message, message );
	}
	return err;
}

void ArgumentErrorMsg(ErrorCluster* e, const std::string& msg) 
{
	FillErrorCluster(mgArgErr, msg.c_str(), e);
}

bool ValidateTracker(QueuedTracker* tracker, ErrorCluster* e, const char *funcname)
{
	if (std::find(trackerList.begin(), trackerList.end(), tracker)==trackerList.end()) {
		ArgumentErrorMsg(e, SPrintf("QTrk C++ function %s called with invalid tracker pointer: %p", funcname, tracker));
		return false;
	}
	return true;
}

CDLL_EXPORT int qtrk_get_debug_image(QueuedTracker* qtrk, int id, LVArray2D<float>** data, ErrorCluster* e)
{
	int w,h;
	if (ValidateTracker(qtrk, e, "qtrk_get_debug_image")) {
		float* img;
		if (qtrk->GetDebugImage(id, &w,&h, &img)) {
			ResizeLVArray2D(data, h, w);
			for (int i=0;i<w*h;i++) (*data)->elem[i]=img[i];
			delete[] img;
			return 1;
		}
	}
	return 0;
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_logfile_path(const char* path)
{
	dbgsetlogfile(path);
}

CDLL_EXPORT void DLL_CALLCONV qtrk_get_computed_config(QueuedTracker* qtrk, QTrkComputedConfig* cc, ErrorCluster *err)
{
	if (ValidateTracker(qtrk, err, "get computed config"))
		*cc = qtrk->cfg;
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut, LVArray<float>** zcmpWindow, int normalize, ErrorCluster* e)
{
	LVArray3D<float>* zlut = *pZlut;

	if (ValidateTracker(tracker,e, "set ZLUT")) {

		int numLUTs = zlut->dimSizes[0];
		int planes = zlut->dimSizes[1];
		int res = zlut->dimSizes[2];

		dbgprintf("Setting ZLUT size: %d beads, %d planes, %d radialsteps\n", numLUTs, planes, res);

		float* zcmp = 0;
		if (zcmpWindow && (*zcmpWindow)->dimSize > 0) {
			if ( (*zcmpWindow)->dimSize != res)
				ArgumentErrorMsg(e, SPrintf("Z Compare window should have the same resolution as the ZLUT (%d elements)", res));
			else
				zcmp = (*zcmpWindow)->elem;
		}

		if (numLUTs * planes * res == 0) {
			tracker->SetRadialZLUT(0, 0, 0);
		} else {
			if (res != tracker->cfg.zlut_radialsteps)
				ArgumentErrorMsg(e, SPrintf("set_ZLUT: 3rd dimension should have size of zlut_radialsteps (%d instead of %d)", tracker->cfg.zlut_radialsteps, res));
			else {

				if (normalize) {
					NormalizeZLUT( zlut->elem, numLUTs, planes, res );
				}

				tracker->SetRadialZLUT(zlut->elem, numLUTs, planes);
				tracker->SetRadialWeights(zcmp);
			}
		}
	}
}

CDLL_EXPORT void DLL_CALLCONV qtrk_get_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pzlut, ErrorCluster* e)
{
	if (ValidateTracker(tracker, e, "get ZLUT")) {
		int dims[3];

		tracker->GetRadialZLUTSize(dims[0], dims[1], dims[2]);
		if(dims[0]*dims[1]*dims[2]>0) {
			ResizeLVArray3D(pzlut, dims[0], dims[1], dims[2]);
			tracker->GetRadialZLUT( (*pzlut)->elem );
		}
	}
}

CDLL_EXPORT void DLL_CALLCONV qtrk_get_image_lut(QueuedTracker* qtrk, LVArrayND<float, 4>** imageLUT, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "get_image_lut")) {

		int dims[4];
		qtrk->GetImageZLUTSize(dims);
		ResizeLVArray(imageLUT, dims);

		if ( (*imageLUT)->numElem () > 0 ){
			qtrk->GetImageZLUT( (*imageLUT)->elem );
		}
	}
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_image_lut(QueuedTracker* qtrk, LVArrayND<float,4>** imageLUT,  LVArray3D<float>** radialZLUT, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "set_image_lut")) {
		if ( (*imageLUT)->numElem () == 0 ){
			qtrk->SetImageZLUT (0, 0, 0);
		}
		else {
			qtrk->SetImageZLUT( (*imageLUT)->elem, (*radialZLUT)->elem, (*imageLUT)->dimSizes );
		}
	}
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_pixel_calib_factors(QueuedTracker* qtrk, float offsetFactor, float gainFactor, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "set pixel calib factors")) {
		qtrk->SetPixelCalibrationFactors(offsetFactor, gainFactor);
	}
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_pixel_calib(QueuedTracker* qtrk, LVArray3D<float>** offset, LVArray3D<float>** gain, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "set pixel calibration images")) {
		int count, planes, radialSteps;
		qtrk->GetRadialZLUTSize(count, planes, radialSteps);

		float *offset_data = 0, *gain_data = 0;

		if((*offset)->dimSizes[0] != 0) {
			if (qtrk->cfg.width != (*offset)->dimSizes[2] || qtrk->cfg.height != (*offset)->dimSizes[1]) {
				ArgumentErrorMsg(e, SPrintf("set_pixel_calib: Offset images passed with invalid dimension (%d,%d)", (*offset)->dimSizes[2], (*offset)->dimSizes[1]));
				return;
			}
			if (count != (*offset)->dimSizes[0]) {
				ArgumentErrorMsg(e, SPrintf("set_pixel_calib: Expecting offset to have %d images (same as ZLUT). %d given", count, (*offset)->dimSizes[0]));
				return;
			}
			offset_data = (*offset)->elem;
		}

		if((*gain)->dimSizes[0] != 0) {
			if (qtrk->cfg.width != (*gain)->dimSizes[2] || qtrk->cfg.height != (*gain)->dimSizes[1]) {
				ArgumentErrorMsg(e, SPrintf("set_pixel_calib: Gain images passed with invalid dimension (%d,%d)", (*gain)->dimSizes[2], (*gain)->dimSizes[1]));
				return;
			}
			if (count != (*gain)->dimSizes[0]) {
				ArgumentErrorMsg(e, SPrintf("set_pixel_calib: Expecting gain to have %d images (same as ZLUT). %d given", count, (*gain)->dimSizes[0]));
				return;
			}
			gain_data = (*gain)->elem;
		}

		qtrk->SetPixelCalibrationImages(offset_data, gain_data);
	}
}

CDLL_EXPORT QueuedTracker* qtrk_create(QTrkSettings* settings, LStrHandle warnings, ErrorCluster* e)
{
	QueuedTracker* tracker = 0;
	try {
		QTrkComputedConfig cc(*settings);
		cc.WriteToLog();

		tracker = CreateQueuedTracker(cc);

		std::string w = tracker->GetWarnings();
		if (!w.empty()) SetLVString(warnings, w.c_str());

		trackerListMutex.lock();
		trackerList.push_back(tracker);
		trackerListMutex.unlock();
	} catch(const std::runtime_error &exc) {
		FillErrorCluster(kAppErrorBase, exc.what(), e );
	}
	return tracker;
}

CDLL_EXPORT void qtrk_destroy(QueuedTracker* qtrk, ErrorCluster* error)
{
	trackerListMutex.lock();

	auto pos = std::find(trackerList.begin(),trackerList.end(),qtrk);
	if (pos == trackerList.end()) {
		ArgumentErrorMsg(error, SPrintf( "Trying to call qtrk_destroy with invalid qtrk pointer %p", qtrk));
		qtrk = 0;
	}
	else
		trackerList.erase(pos);

	trackerListMutex.unlock();

	if(qtrk) delete qtrk;
}

template<typename T>
bool CheckImageInput(QueuedTracker* qtrk, LVArray2D<T> **data, ErrorCluster  *error)
{
	if (!data) {
		ArgumentErrorMsg(error, "Image data array is empty");
		return false;
	} else if( (*data)->dimSizes[1] != qtrk->cfg.width || (*data)->dimSizes[0] != qtrk->cfg.height ) {
		ArgumentErrorMsg(error, SPrintf( "Image data array has wrong size (%d,%d). Should be: (%d,%d)", (*data)->dimSizes[1], (*data)->dimSizes[0], qtrk->cfg.width, qtrk->cfg.height));
		return false;
	}
	return true;
}

CDLL_EXPORT void qtrk_queue_u16(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<ushort>** data, const LocalizationJob *jobInfo)
{
	if (CheckImageInput(qtrk, data, error)) 
	{
		qtrk->ScheduleLocalization( (uchar*)(*data)->elem, sizeof(ushort)*(*data)->dimSizes[1], QTrkU16, jobInfo);
	}
}

CDLL_EXPORT void qtrk_queue_u8(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<uchar>** data, const LocalizationJob *jobInfo)
{
	if (CheckImageInput(qtrk, data, error))
	{
#ifdef _DEBUG
	//dbgprintf("Job: 8bit image, frame %d, bead %d\n", jobInfo->frame, jobInfo->zlutIndex);
#endif

		qtrk->ScheduleLocalization( (*data)->elem, sizeof(uchar)*(*data)->dimSizes[1], QTrkU8, jobInfo);
	}
}

CDLL_EXPORT void qtrk_queue_float(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<float>** data, const LocalizationJob *jobInfo)
{
	if (CheckImageInput(qtrk, data, error))
	{
		qtrk->ScheduleLocalization( (uchar*) (*data)->elem, sizeof(float)*(*data)->dimSizes[1], QTrkFloat, jobInfo);
	}
}


CDLL_EXPORT void qtrk_queue_pitchedmem(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, const LocalizationJob *jobInfo)
{
	qtrk->ScheduleLocalization(data, pitch, (QTRK_PixelDataType)pdt, jobInfo);
}

CDLL_EXPORT void qtrk_queue_array(QueuedTracker* qtrk,  ErrorCluster* error,LVArray2D<uchar>** data,uint pdt, const LocalizationJob *jobInfo)
{
	uint pitch;

	if (pdt == QTrkFloat) 
		pitch = sizeof(float);
	else if(pdt == QTrkU16) 
		pitch = 2;
	else pitch = 1;

	if (!CheckImageInput(qtrk, data, error))
		return;

	pitch *= (*data)->dimSizes[1]; // LVArray2D<uchar> type works for ushort and float as well
//	dbgprintf("zlutindex: %d, zlutplane: %d\n", zlutIndex,zlutPlane);
	qtrk_queue_pitchedmem(qtrk, (*data)->elem, pitch, pdt, jobInfo);
}

CDLL_EXPORT uint qtrk_read_timestamp(uchar* image, int w, int h)
{
	if (w*h<4) return 0;

	uint ts;
	uchar *timestamp = (uchar*)&ts;
	// Assume little endian only
	for (int i=0;i<4;i++)
		timestamp[i] = image[i];
	return ts;
}

CDLL_EXPORT uint qtrk_queue_frame(QueuedTracker* qtrk, uchar* image, int pitch, int w,int h, 
	uint pdt, ROIPosition* pos, int numROI, const LocalizationJob *pJobInfo, QueueFrameFlags flags, ErrorCluster* e)
{
	LocalizationJob jobInfo = *pJobInfo;
	int nQueued;
	if ( (nQueued=qtrk->ScheduleFrame(image, pitch, w,h, pos, numROI, (QTRK_PixelDataType)pdt, &jobInfo)) != numROI) {
		ArgumentErrorMsg(e, SPrintf( "Not all ROIs (%d out of %d) were queued. Check image borders vs ROIs.", nQueued, numROI));
	}
	return jobInfo.timestamp;
}

CDLL_EXPORT void qtrk_clear_results(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "clear results")) {
		qtrk->ClearResults();
	}
}

// Accepts data as 3D image [numbeads]*[width]*[height]
CDLL_EXPORT void qtrk_build_lut_plane(QueuedTracker* qtrk, LVArray3D<float> **data, uint flags, int plane, ErrorCluster* err)
{
	if (ValidateTracker(qtrk, err, "build_lut_plane")) {
		if ((*data)->dimSizes[2] != qtrk->cfg.width || (*data)->dimSizes[1] != qtrk->cfg.height) {
			ArgumentErrorMsg(err, SPrintf("Invalid size: %dx%d. Expecting %dx%d\n", (*data)->dimSizes[2], (*data)->dimSizes[1], qtrk->cfg.width, qtrk->cfg.height));
			return;
		}

		int cnt,planes,rsteps;
		qtrk->GetRadialZLUTSize(cnt, planes, rsteps);

		if ((*data)->dimSizes[0] != cnt) {
			ArgumentErrorMsg(err, SPrintf("Invalid number of images given (%d). Expecting %d", (*data)->dimSizes[0], cnt));
			return;
		}

		qtrk->BuildLUT( (*data)->elem, sizeof(float*) * qtrk->cfg.width, QTrkFloat, plane);
	}
}

CDLL_EXPORT void qtrk_finalize_lut(QueuedTracker* qtrk, ErrorCluster *e)
{
	if (ValidateTracker(qtrk, e, "finalize_lut"))
		qtrk->FinalizeLUT();
}


CDLL_EXPORT int qtrk_get_queue_len(QueuedTracker* qtrk, int* maxQueueLen, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "fullqueue"))
		return qtrk->GetQueueLength(maxQueueLen);
	return 0;
}

CDLL_EXPORT int qtrk_resultcount(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "resultcount")) {
		return qtrk->GetResultCount();
	} 
	return 0;
}

CDLL_EXPORT void qtrk_flush(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "flush")) {
		qtrk->Flush();
	}
}

CDLL_EXPORT int qtrk_get_results(QueuedTracker* qtrk, LocalizationResult* results, int maxResults, int sortByID, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "get_results")) {
		int resultCount = qtrk->FetchResults(results, maxResults);

		if (sortByID) {
			std::sort(results, results+resultCount, [](decltype(*results) a, decltype(*results) b) { return a.job.frame<b.job.frame; } );
		}

		return resultCount;
	} 
	return 0;
}

CDLL_EXPORT void qtrk_get_zlut_cmpprof(QueuedTracker* qtrk, LVArray2D<float> ** output, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "get zlut compare profiles"))
	{
		int cnt,planes,rsteps;
		qtrk->GetRadialZLUTSize(cnt,planes,rsteps);
		ResizeLVArray2D(output, cnt, planes);

		qtrk->GetRadialZLUTCompareProfile((*output)->elem);
	}
}

CDLL_EXPORT void qtrk_enable_zlut_cmpprof(QueuedTracker* qtrk, bool enable, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "enable zlut cmpprof"))
		qtrk->EnableRadialZLUTCompareProfile(enable);
}


CDLL_EXPORT void qtrk_set_localization_mode(QueuedTracker* qtrk, uint locType, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "set_localization_mode")) {
		qtrk->SetLocalizationMode( (LocMode_t)locType );
	}
}

CDLL_EXPORT int qtrk_idle(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "is_idle"))
		return qtrk->IsIdle() ? 1 : 0;
	return 0;
}

CDLL_EXPORT void qtrk_compute_zlut_bias_table(QueuedTracker* qtrk, int bias_planes, LVArray2D<float>** lvresult, int smpPerPixel, int useSplineInterp,ErrorCluster* e)
{
	if(ValidateTracker(qtrk, e,"compute_zlut_bias_table")) {
		CImageData result;
		qtrk->ComputeZBiasCorrection(bias_planes, &result, smpPerPixel, useSplineInterp!=0);

		ResizeLVArray2D(lvresult, result.h, result.w);
		result.copyTo ( (*lvresult)->elem );
	}
}

CDLL_EXPORT void qtrk_set_zlut_bias_table(QueuedTracker* qtrk, LVArray2D<float>** biastbl, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e,"set zlut bias table")) {
		int numbeads,planes,radialsteps;
		qtrk->GetRadialZLUTSize(numbeads, planes, radialsteps);

		if ((*biastbl)->dimSizes[1] != numbeads) {
			ArgumentErrorMsg(e, SPrintf( "Bias table should be [numbeads] high and [biasplanes] wide. Expected #beads=%d", numbeads) );
		}
		else {
			qtrk->SetZLUTBiasCorrection( ImageData( (*biastbl)->elem, (*biastbl)->dimSizes[0], (*biastbl)->dimSizes[1] ) );
		}
	}
}

CDLL_EXPORT void DLL_CALLCONV qtrk_generate_gaussian_spot(LVArray2D<float>** image, vector2f* pos, float sigma, float I0, float Ibg, int applyNoise)
{
	ImageData img((*image)->elem, (*image)->dimSizes[1], (*image)->dimSizes[0]);

	GenerateGaussianSpotImage(&img, *pos, sigma, I0, Ibg);

	if (applyNoise)
		ApplyPoissonNoise(img,1);
}

CDLL_EXPORT void DLL_CALLCONV qtrk_generate_image_from_lut(LVArray2D<float>** image, LVArray2D<float>** lut, 
					float *LUTradii, vector2f* position, float z, float M, float sigma_noise)
{
	ImageData img((*image)->elem, (*image)->dimSizes[1], (*image)->dimSizes[0]);
	ImageData zlut((*lut)->elem, (*lut)->dimSizes[1], (*lut)->dimSizes[0]);

	vector3f pos (position->x, position->y, z);
	GenerateImageFromLUT(&img, &zlut, LUTradii[0], LUTradii[1], pos);
	//img.normalize();
	if(sigma_noise>0)
		ApplyGaussianNoise(img, sigma_noise);
}


CDLL_EXPORT void qtrk_dump_memleaks()
{
#ifdef USE_MEMDBG
	_CrtDumpMemoryLeaks();
#endif
}

CDLL_EXPORT void qtrk_get_profile_report(QueuedTracker* qtrk, LStrHandle str)
{
	SetLVString(str, qtrk->GetProfileReport().c_str());
}


CDLL_EXPORT void qtrk_compute_fisher(LVArray2D<float> **lut, QTrkSettings* cfg, vector3f* pos, LVArray2D<float> ** fisherMatrix, 
		LVArray2D<float> ** inverseMatrix, vector3f* xyzVariance, int Nsamples, float maxPixelValue)
{
	QTrkComputedConfig cc (*cfg);
	ImageData lutImg( (*lut)->elem, (*lut)->dimSizes[1], (*lut)->dimSizes[0] );
	SampleFisherMatrix fm(maxPixelValue);
	Matrix3X3 mat = fm.ComputeAverageFisher(*pos, Nsamples, vector3f(1,1,1), vector3f(1,1,1)*0.001f, cfg->width, cfg->height, [&](ImageData&out, vector3f pos) {
		GenerateImageFromLUT(&out, &lutImg, cc.zlut_minradius,cc.zlut_maxradius, pos);
	});
	
	if (fisherMatrix) {
		ResizeLVArray2D( fisherMatrix, 3, 3);
		for (int i=0;i<9;i++)
			(*fisherMatrix)->elem[i] = mat[i];
	}

	if (inverseMatrix) {
		Matrix3X3 inv = mat.Inverse();
		ResizeLVArray2D( inverseMatrix, 3, 3);
		for (int i=0;i<9;i++)
			(*inverseMatrix)->elem[i] = inv[i];
	}

	if (xyzVariance)
		*xyzVariance = mat.Inverse().diag();
}


CDLL_EXPORT void qtrk_find_beads(uint8_t* image, int pitch, int w,int h, int* smpCornerPos, int roi, float imgRelDist, float acceptance, LVArray2D<uint32_t> **output)
{
	BeadFinder::Config cfg;
	cfg.img_distance = imgRelDist;
	cfg.roi = roi;
	cfg.similarity = acceptance;
	auto results = BeadFinder::Find(image, pitch, w,h, smpCornerPos[0], smpCornerPos[1], &cfg);

	ResizeLVArray2D(output, results.size(), 2);
	for (int i=0;i<results.size();i++)
	{
		(*output)->get(i, 0) = results[i].x;
		(*output)->get(i, 1) = results[i].y;
	}
}


CDLL_EXPORT void qtrk_test_array_passing(int n, LVArray<float>** flt1D, LVArray2D<float>** flt2D, LVArray<int>** int1D, LVArray2D<int>** int2D)
{
	for (int i=0;i<(*int2D)->dimSizes[0];i++)
	{
		for (int j=0;j<(*int2D)->dimSizes[1];j++)
		{
			dbgprintf("%d\t", (*int2D)->get(i, j));
		}
		dbgprintf("\n");
	}

	ResizeLVArray(flt1D, n);
	ResizeLVArray(int1D, n);
	ResizeLVArray2D(flt2D, n/2,n);
	ResizeLVArray2D(int2D, n/2,n);
	for (int i=0;i<n;i++) {
		(*int1D)->elem[i]=(i+1)*i;
		(*flt1D)->elem[i]=sqrtf(i);
		for (int j=0;j<n/2;j++) {
			(*int2D)->get(j, i) = j*2+i;
			(*flt2D)->get(j, i) = j*2+i;
		}
	}
}

CDLL_EXPORT void qtrk_simulate_tracking(QueuedTracker* qtrk, int nsmp, int beadIndex, vector3f* centerPos, vector3f* range, vector3f *outBias, vector3f* outScatter, float photonsPerWell, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "qtrk_simulate_tracking")) {

		int nZLUT, nPlanes, nRadialSteps;
		qtrk->GetRadialZLUTSize(nZLUT, nPlanes, nRadialSteps);

		float* lut = new float[nZLUT*nPlanes*nRadialSteps];
		qtrk->GetRadialZLUT(lut);

		ImageData img = ImageData::alloc(qtrk->cfg.width,qtrk->cfg.height);
		ImageData zlut;
		zlut.data = & lut [nRadialSteps*nPlanes*beadIndex];
		zlut.w = nRadialSteps;
		zlut.h = nPlanes;

		// Generate images
		std::vector<vector3f> positions(nsmp);
		for (int i=0;i<nsmp;i++) {
			vector3f pos = *centerPos + *range * vector3f(rand_uniform<float>(), rand_uniform<float>(), rand_uniform<float>());
			positions[i]=pos;
			GenerateImageFromLUT( &img, &zlut, qtrk->cfg.zlut_minradius, qtrk->cfg.zlut_maxradius, pos);
			ApplyPoissonNoise(img, photonsPerWell);
			qtrk->ScheduleLocalization((uchar*)img.data, sizeof(float)*img.w, QTrkFloat,i,i,0,beadIndex);
		}

		img.free();

		qtrk->Flush();
		while (!qtrk->IsIdle());

		LocalizationResult* results=new LocalizationResult[nsmp];
		qtrk->FetchResults(results, nsmp);
		vector3f sumBias, sumScatter;
		for (int i=0;i<nsmp;i++) {
			vector3f truepos = positions [ results[i].job.frame ];
			vector3f diff = results[i].pos - truepos;
			sumBias += diff;
		}
		vector3f meanBias = sumBias / nsmp;
		for (int i=0;i<nsmp;i++) {
			vector3f truepos = positions [ results[i].job.frame ];
			vector3f diff = results[i].pos - truepos;
			diff -= meanBias;
			sumScatter += diff*diff;
		}
		*outScatter = sqrt (sumScatter / nsmp);
		*outBias = sumBias;
	}
}


#if defined(CUDA_TRACK) || defined(DOXYGEN)

#include "cuda_runtime.h"

CDLL_EXPORT void qtrkcuda_set_device_list(LVArray<int>** devices)
{
	SetCUDADevices( (*devices)->elem, (*devices)->dimSize );
}

CDLL_EXPORT int qtrkcuda_device_count(ErrorCluster* e) 
{
	int c;
	if (CheckCUDAErrorLV(cudaGetDeviceCount(&c), e)) {
		return c;
	}
	return 0;
}

CDLL_EXPORT void qtrkcuda_get_device(int device, CUDADeviceInfo *info, ErrorCluster* e)
{
	cudaDeviceProp prop;
	if (CheckCUDAErrorLV(cudaGetDeviceProperties(&prop, device), e)) {
		info->multiProcCount = prop.multiProcessorCount;
		info->clockRate = prop.clockRate;
		info->major = prop.major;
		info->minor = prop.minor;
		SetLVString(info->name, prop.name);
	}
}

CDLL_EXPORT void qtrkcuda_enable_texture_cache(QueuedTracker* qtrk, int enable, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "enable_texture_cache")) {
		qtrk->SetConfigValue("use_texturecache", enable?"1":"0");
	}
}

#else // Generate empty functions to prevent labview crashes

CDLL_EXPORT int qtrkcuda_device_count(ErrorCluster* e) { return 0; }
CDLL_EXPORT void qtrkcuda_get_device(int device, void *info, ErrorCluster* e) {}

#endif

