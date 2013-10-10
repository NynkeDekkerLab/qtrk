/// --------------------------------------------------------------------------------------
// ResultManager

#include "std_incl.h"
#include "labview.h"
#include "ResultManager.h"
#include "utils.h"
#include <unordered_set>

static  std::unordered_set<ResultManager*> rm_instances;

static bool ValidRM(ResultManager* rm, ErrorCluster* err)
{
	if(rm_instances.find(rm) == rm_instances.end()) {
		ArgumentErrorMsg(err, "Invalid ResultManager instance passed.");
		return false;
	}
	return true;
}

CDLL_EXPORT void DLL_CALLCONV rm_destroy_all()
{
	DeleteAllElems(rm_instances);
}

CDLL_EXPORT ResultManager* DLL_CALLCONV rm_create(const char *file, const char *frameinfo, ResultManagerConfig* cfg, LStrHandle* names)
{
	std::vector<std::string> colNames;
	
	if (names) colNames = LVGetStringArray(cfg->numFrameInfoColumns, names);
	ResultManager* rm = new ResultManager(file, frameinfo, cfg, colNames);

	rm_instances.insert(rm);
	return rm;
}

CDLL_EXPORT void DLL_CALLCONV rm_set_tracker(ResultManager* rm, QueuedTracker* qtrk, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		rm->SetTracker(qtrk);
	}
}

CDLL_EXPORT void DLL_CALLCONV rm_destroy(ResultManager* rm, ErrorCluster  *err)
{
	if (ValidRM(rm, err)) {
		rm_instances.erase(rm);
		delete rm;
	}
}

CDLL_EXPORT void DLL_CALLCONV rm_store_frame_info(ResultManager* rm, int frame, double timestamp, float* cols, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
#ifdef _DEBUG
		dbgprintf("rm_store_frame_info: frame=%d, ts=%f\n", frame,timestamp);
#endif
		rm->StoreFrameInfo(frame, timestamp, cols);
	} 
}

CDLL_EXPORT int DLL_CALLCONV rm_getbeadresults(ResultManager* rm, int start, int numFrames, int bead, LocalizationResult* results, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		if (bead < 0 || bead >= rm->Config().numBeads)
			ArgumentErrorMsg(err,SPrintf( "Invalid bead index: %d. Accepted range: [0-%d]", bead, rm->Config().numBeads));
		else
			return rm->GetBeadPositions(start,start+numFrames,bead,results);
	}
	return 0;
}


CDLL_EXPORT void DLL_CALLCONV rm_getframecounters(ResultManager* rm, int* startFrame, int* lastSaveFrame, 
				int* endFrame, int *capturedFrames, int *localizationsDone, int *lostFrames, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {

		auto r = rm->GetFrameCounters();

		if (startFrame) *startFrame = r.startFrame;
		if (lastSaveFrame) *lastSaveFrame = r.lastSaveFrame;
		if (endFrame) * endFrame = r.processedFrames;
		if (capturedFrames) *capturedFrames = r.capturedFrames;
		if (localizationsDone) *localizationsDone = r.localizationsDone;
		if (lostFrames) *lostFrames = r.lostFrames;
	}
}


CDLL_EXPORT void DLL_CALLCONV rm_flush(ResultManager* rm, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		rm->Flush();
	}
}

CDLL_EXPORT int DLL_CALLCONV rm_getresults(ResultManager* rm, int startFrame, int numFrames, LocalizationResult* results, ErrorCluster* err)
{
	if (ValidRM(rm,err)) {
		return rm->GetResults(results, startFrame, numFrames);
	}
	return 0;
}

CDLL_EXPORT void DLL_CALLCONV rm_removebead(ResultManager* rm, int bead, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		if (rm->GetTracker()) {
			ArgumentErrorMsg(err, "Cannot discard bead results while tracking in progress");
		}

		rm->RemoveBeadResults(bead);
	}
}



CDLL_EXPORT void DLL_CALLCONV rm_getconfig(ResultManager* rm, ResultManagerConfig* cfg, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		*cfg = rm->Config();
	}
}
