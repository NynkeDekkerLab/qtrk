#include "std_incl.h"
#include "QueuedTracker.h"
#include "utils.h"
#include "cpu_tracker.h"

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

void QTrkComputedConfig::WriteToLog()
{
#define WRITEVAR(v) dbgprintf("Setting: %s set to %g\n", #v, (float) v)
	WRITEVAR(width);
	WRITEVAR(height);
	WRITEVAR(numThreads);
	WRITEVAR(cuda_device);
	WRITEVAR(com_bgcorrection);
	WRITEVAR(zlut_minradius);
	WRITEVAR(zlut_radial_coverage);
	WRITEVAR(zlut_angular_coverage);
	WRITEVAR(zlut_roi_coverage); 
	WRITEVAR(qi_iterations);
	WRITEVAR(qi_minradius);
	WRITEVAR(qi_radial_coverage);
	WRITEVAR(qi_angular_coverage);
	WRITEVAR(qi_roi_coverage);
	WRITEVAR(qi_angstep_factor);
	WRITEVAR(xc1_profileLength);
	WRITEVAR(xc1_profileWidth);
	WRITEVAR(xc1_iterations);
	WRITEVAR(gauss2D_iterations);
	WRITEVAR(gauss2D_sigma);
	WRITEVAR(zlut_radialsteps);
	WRITEVAR(zlut_angularsteps);
	WRITEVAR(zlut_maxradius);
	WRITEVAR(qi_radialsteps);
	WRITEVAR(qi_angstepspq);
	WRITEVAR(qi_maxradius);
	WRITEVAR(downsample);
#undef WRITEVAR
}

void QTrkComputedConfig::WriteToFile()
{
	std::string path = GetCurrentOutputPath(false);
	FILE* f = fopen(SPrintf("%s\\UsedSettings.txt",path.c_str()).c_str(),"w+");

#define WRITEVAR(v) fprintf_s(f,"Setting: %s set to %g\n", #v, (float) v)
	WRITEVAR(width);
	WRITEVAR(height);
	WRITEVAR(numThreads);
	WRITEVAR(cuda_device);
	WRITEVAR(com_bgcorrection);
	WRITEVAR(zlut_minradius);
	WRITEVAR(zlut_radial_coverage);
	WRITEVAR(zlut_angular_coverage);
	WRITEVAR(zlut_roi_coverage); 
	WRITEVAR(qi_iterations);
	WRITEVAR(qi_minradius);
	WRITEVAR(qi_radial_coverage);
	WRITEVAR(qi_angular_coverage);
	WRITEVAR(qi_roi_coverage);
	WRITEVAR(qi_angstep_factor);
	WRITEVAR(xc1_profileLength);
	WRITEVAR(xc1_profileWidth);
	WRITEVAR(xc1_iterations);
	WRITEVAR(gauss2D_iterations);
	WRITEVAR(gauss2D_sigma);
	WRITEVAR(zlut_radialsteps);
	WRITEVAR(zlut_angularsteps);
	WRITEVAR(zlut_maxradius);
	WRITEVAR(qi_radialsteps);
	WRITEVAR(qi_angstepspq);
	WRITEVAR(qi_maxradius);
	WRITEVAR(downsample);
#undef WRITEVAR

	fclose(f);
}

QueuedTracker::QueuedTracker()
{
	zlut_bias_correction=0;
}

QueuedTracker::~QueuedTracker()
{
}

void QueuedTracker::ScheduleImageData(ImageData* data, const LocalizationJob* job)
{
	ScheduleLocalization(data->data, data->pitch(), QTrkFloat, job);
}

void QueuedTracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, uint frame, uint timestamp, vector3f* initial, uint zlutIndex)
{
	LocalizationJob j;
	j.frame= frame;
	j.timestamp = timestamp;
	if (initial) j.initialPos = *initial;
	j.zlutIndex = zlutIndex;
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

float QueuedTracker::ZLUTBiasCorrection(float z, int zlut_planes, int bead)
{
	if (!zlut_bias_correction)
		return z;

	/*
        bias = d(r,4);
        measured_pos = d(r,1) + bias;
        pos = measured_pos;
        for k=1:2
            guess_bias = interp1(d(r,1), bias, pos);
            pos = measured_pos - guess_bias;
        end
		*/

	// we have to reverse the bias table: we know that true_z + bias(true_z) = measured_z, but we can only get bias(measured_z)
	// It seems that one can iterate towards the right position:
	
	float pos = z;
	for (int k=0;k<4;k++) {
		float tblpos = pos / (float)zlut_planes * zlut_bias_correction->w;
		float bias = zlut_bias_correction->interpolate1D(bead, tblpos);
		pos = z - bias;
	}
	return pos;
}
void QueuedTracker::ComputeZBiasCorrection(int bias_planes, CImageData* result, int smpPerPixel, bool useSplineInterp)
{
	int count,zlut_planes,radialsteps;
	GetRadialZLUTSize(count, zlut_planes, radialsteps);
	float* zlut_data = new float[count*zlut_planes*radialsteps];
	GetRadialZLUT(zlut_data);

	std::vector<float> qi_rweights = ComputeRadialBinWindow(cfg.qi_radialsteps);
	std::vector<float> zlut_rweights = ComputeRadialBinWindow(cfg.zlut_radialsteps);

	if (zlut_bias_correction)
		delete zlut_bias_correction;

	zlut_bias_correction = new CImageData(bias_planes, count);

	parallel_for(count*bias_planes, [&](int job) {
	//for (int job=0;job<count*bias_planes;job++) {
		int bead = job/bias_planes;
		int plane = job%bias_planes;
		
		float *zlut_ptr = &zlut_data[ bead * (zlut_planes*radialsteps) ];
		CPUTracker trk (cfg.width,cfg.height);
		ImageData zlut(zlut_ptr, radialsteps, zlut_planes);
		trk.SetRadialZLUT(zlut.data, zlut.h, zlut.w, 1, cfg.zlut_minradius,cfg.zlut_maxradius, false, false);
		trk.SetRadialWeights(&zlut_rweights[0]);

		vector3f pos(cfg.width/2,cfg.height/2, plane /(float) bias_planes * zlut_planes );
		ImageData img = ImageData::alloc(cfg.width,cfg.height);
		GenerateImageFromLUT(&img, &zlut, cfg.zlut_minradius, cfg.zlut_maxradius, pos, useSplineInterp,smpPerPixel);

		bool bhit;
		trk.SetImageFloat(img.data);
		vector2f com = trk.ComputeMeanAndCOM();
		vector2f qi = trk.ComputeQI(com, 2, cfg.qi_radialsteps, cfg.qi_angstepspq, cfg.qi_angstep_factor, cfg.qi_minradius, cfg.qi_maxradius, bhit, &qi_rweights[0]);
		float z = trk.ComputeZ(qi, cfg.zlut_angularsteps, 0);
		zlut_bias_correction->at(plane, bead) = z - pos.z;

		//trk.ComputeRadialProfile(
		img.free();
		if ((job%(count*bias_planes/10)) == 0) 
			dbgprintf("job=%d\n", job);
//	}
	});

	if (result)
		*result = *zlut_bias_correction;
}

void QueuedTracker::SetZLUTBiasCorrection(const CImageData& bc)
{
	if (zlut_bias_correction) delete zlut_bias_correction;
	zlut_bias_correction = new CImageData(bc);
}

CImageData* QueuedTracker::GetZLUTBiasCorrection()
{
	if (zlut_bias_correction) return new CImageData(*zlut_bias_correction);
	return 0;
}

