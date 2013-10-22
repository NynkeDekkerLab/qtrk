#include "std_incl.h"
#include "QueuedTracker.h"
#include "utils.h"

void QTrkComputedConfig::Update()
{
	int roi = width / 2;

	zlut_maxradius = roi*zlut_roi_coverage;
	float zlut_perimeter = 2*3.141593f*zlut_maxradius;
	zlut_angularsteps = zlut_perimeter*zlut_angular_coverage;
	zlut_radialsteps = (zlut_maxradius-zlut_minradius)*zlut_radial_coverage;

	qi_maxradius = roi*qi_roi_coverage;
	float qi_perimeter = 2*3.141593f*qi_maxradius;
	qi_angstepspq = qi_perimeter*qi_angular_coverage/4;
	qi_radialsteps = (qi_maxradius-qi_minradius)*qi_radial_coverage;

	qi_radialsteps = NearestPowerOf2(qi_radialsteps);
}

QueuedTracker::QueuedTracker()
{
}

QueuedTracker::~QueuedTracker()
{
}


void QueuedTracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, uint frame, uint timestamp, vector3f* initial, uint zlutIndex, uint zlutPlane)
{
	LocalizationJob j;
	j.frame= frame;
	j.timestamp = timestamp;
	if (initial) j.initialPos = *initial;
	j.zlutIndex = zlutIndex;
	j.zlutPlane = zlutPlane;
	ScheduleLocalization(data,pitch,pdt,&j);
}



ImageData QueuedTracker::DebugImage(int ID)
{
	ImageData img;
	GetDebugImage(ID, &img.w, &img.h, &img.data);
	return img;
}



int QueuedTracker::ScheduleFrame(void *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo)
{
	uchar* img = (uchar*)imgptr;
	int bpp = PDT_BytesPerPixel(pdt);
	int count=0;
	for (int i=0;i<numROI;i++){
		ROIPosition& pos = positions[i];

		if (pos.x < 0 || pos.y < 0 || pos.x + cfg.width > width || pos.y + cfg.height > height) {
			dbgprintf("Skipping ROI %d. Outside of image.\n", i);
			continue;
		}

		uchar *roiptr = &img[pitch * pos.y + pos.x * bpp];
		LocalizationJob job = *jobInfo;
		job.zlutIndex = i + jobInfo->zlutIndex; // used as offset
		ScheduleLocalization(roiptr, pitch, pdt, &job);
		count++;
	}
	return count;
}

