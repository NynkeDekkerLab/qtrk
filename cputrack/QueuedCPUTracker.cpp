#include "std_incl.h"
#include "QueuedCPUTracker.h"
#include <float.h>
#include <functional> 
#include "DebugResultCompare.h"

#ifndef CUDA_TRACK
QueuedTracker* CreateQueuedTracker(const QTrkComputedConfig& cc){
	return new QueuedCPUTracker(cc);
}
void SetCUDADevices(int* dev, int numdev)
{
}
#endif
bool QueuedCPUTracker::IsIdle()
{
	return GetQueueLength() == 0;
}

int QueuedCPUTracker::GetResultCount()
{
	results_mutex.lock();
	int rc = resultCount;
	results_mutex.unlock();
	return rc;
}

void QueuedCPUTracker::ClearResults()
{
	results_mutex.lock();
	resultCount = 0;
	results.clear();
	results_mutex.unlock();
}

void QueuedCPUTracker::JobFinished(QueuedCPUTracker::Job* j)
{
	jobs_buffer_mutex.lock();
	jobs_buffer.push_back(j);
	jobs_buffer_mutex.unlock();

	jobs_mutex.lock();
	jobsInProgress--;
	jobs_mutex.unlock();
}

QueuedCPUTracker::Job* QueuedCPUTracker::GetNextJob()
{
	QueuedCPUTracker::Job *j = 0;
	jobs_mutex.lock();
	if (!jobs.empty()) {
		j = jobs.front();
		jobs.pop_front();
		jobCount --;
		jobsInProgress ++;
	}
	jobs_mutex.unlock();
	return j;
}

QueuedCPUTracker::Job* QueuedCPUTracker::AllocateJob()
{
	QueuedCPUTracker::Job *j;
	jobs_buffer_mutex.lock();
	if (!jobs_buffer.empty()) {
		j = jobs_buffer.back();
		jobs_buffer.pop_back();
	} else 
		j = new Job;
	jobs_buffer_mutex.unlock();
	return j;
}

void QueuedCPUTracker::AddJob(Job* j)
{
	jobs_mutex.lock();
	jobs.push_back(j);
	jobCount++;
	jobs_mutex.unlock();
}

int QueuedCPUTracker::GetQueueLength(int *maxQueueLength)
{
	int jc;
	jobs_mutex.lock();
	jc = jobCount + jobsInProgress;
	jobs_mutex.unlock();

	if (maxQueueLength) 
		*maxQueueLength = this->maxQueueSize;

	return jc;
}

QueuedCPUTracker::QueuedCPUTracker(const QTrkComputedConfig& cc) 
	: jobs_mutex("jobs"), jobs_buffer_mutex("jobs_buffer"), results_mutex("results")
{
	if(0){//cc.testRun){
		std::string folder = GetCurrentOutputPath();
		if(GetFileAttributesA(folder.c_str()) & FILE_ATTRIBUTE_DIRECTORY)
			CreateDirectory((LPCTSTR)folder.c_str(),NULL);
	}
	
	/// Copy the configuration.
	cfg = cc;
	quitWork = false;

	/// If \ref QTrkComputedConfig::numThreads is not specified (<0), select one thread per CPU.
	if (cfg.numThreads < 0) {
		cfg.numThreads = Threads::GetCPUCount();
		dbgprintf("Using %d threads\n", cfg.numThreads);
	} 

	/// Queue length is 250 per thread, min 500.
	maxQueueSize = std::max(2,cfg.numThreads) * 250;
	jobCount = 0;
	resultCount = 0;

	zlut_cmpprofiles = 0;
	zlut_enablecmpprof = false;
	zluts = 0;
	zlut_count = zlut_planes = 0;
	processJobs = false;
	jobsInProgress = 0;
	dbgPrintResults = false;

	/// Calculate the radial profile weights from the amount of steps.
	qi_radialbinweights = ComputeRadialBinWindow(cfg.qi_radialsteps);
	
	calib_gain = calib_offset = 0;
	gc_gainFactor = gc_offsetFactor = 1.0f;
	/// Initialize to only COM localizations.
	localizeMode = LT_OnlyCOM;

	for (int i=0;i<4;i++) image_lut_dims[i]=0;
	image_lut_nElem_per_bead=0;
	image_lut = 0;
	image_lut_dz = image_lut_dz2 = 0;

	downsampleWidth = cfg.width >> cfg.downsample;
	downsampleHeight = cfg.height >> cfg.downsample;

	/// Invoke the \ref Start function.
	Start();
}

QueuedCPUTracker::~QueuedCPUTracker()
{
	// wait for threads to finish
	quitWork = true;

	for (unsigned int k = 0; k<threads.size(); k++) {
		delete threads[k].mutex;
		Threads::WaitAndClose(threads[k].thread);
		delete threads[k].tracker;
	}

	// free job memory
	DeleteAllElems(jobs);
	DeleteAllElems(jobs_buffer);

	delete[] calib_gain;
	delete[] calib_offset;
	if (zluts) delete[] zluts;
	if (zlut_cmpprofiles) delete[] zlut_cmpprofiles;
	if (image_lut) delete[] image_lut;
	if (image_lut_dz) delete[] image_lut_dz;
	if (image_lut_dz2) delete[] image_lut_dz2;
}

void QueuedCPUTracker::SetPixelCalibrationImages(float* offset, float* gain)
{
	if (zlut_count > 0) {
		int nelem = cfg.width*cfg.height*zlut_count;

		if (calib_gain == 0 && gain)  {
			calib_gain = new float[nelem];
			memcpy(calib_gain, gain, sizeof(float)*nelem);
		}
		else if (calib_gain && gain == 0) {
			delete[] calib_gain;
			calib_gain = 0;
		}

		if (calib_offset == 0 && offset) {
			calib_offset = new float[nelem];
			memcpy(calib_offset, offset, sizeof(float)*nelem);
		}
		else if (calib_offset && offset == 0) {
			delete[] calib_offset;
			calib_offset = 0;
		}

#ifdef _DEBUG
		std::string path = GetLocalModulePath();
		for (int i=0;i<zlut_count;i++) {
			if(calib_gain) FloatToJPEGFile( SPrintf("%s/gain-bead%d.jpg", path.c_str(), i).c_str(), &calib_gain[cfg.width*cfg.height*i], cfg.width,cfg.height);
			if(calib_offset) FloatToJPEGFile( SPrintf("%s/offset-bead%d.jpg", path.c_str(), i).c_str(), &calib_offset[cfg.width*cfg.height*i], cfg.width,cfg.height);
		}
#endif
	}
}

void QueuedCPUTracker::SetPixelCalibrationFactors(float offsetFactor, float gainFactor)
{
	gc_mutex.lock();
	gc_gainFactor = gainFactor;
	gc_offsetFactor = offsetFactor;
	gc_mutex.unlock();
}


void QueuedCPUTracker::Break(bool brk)
{
	processJobs = !brk;
}


void QueuedCPUTracker::Start()
{
	quitWork = false;

	threads.resize(cfg.numThreads);
	for (int k=0;k<cfg.numThreads;k++) {

		threads[k].mutex = new Threads::Mutex();
#if 0
		threads[k].mutex->name = SPrintf("thread%d", k);
		threads[k].mutex->trace = true;
#endif
		threads[k].tracker = new CPUTracker(downsampleWidth, downsampleHeight, cfg.xc1_profileLength, false);//cfg.testRun);
		threads[k].manager = this;
		threads[k].tracker->trackerID = k;
	}

	for (unsigned int k = 0; k<threads.size(); k++) {
		threads[k].thread = Threads::Create(WorkerThreadMain, &threads[k]);
		Threads::SetBackgroundPriority(threads[k].thread, true);
	}

	processJobs = true;
}

void QueuedCPUTracker::WorkerThreadMain(void* arg)
{
	Thread* th = (Thread*)arg;
	QueuedCPUTracker* this_ = th->manager;

	while (!this_->quitWork) {
		Job* j = 0;
		if (this_->processJobs) 
			j = this_->GetNextJob();

		if (j) {
			this_->ProcessJob(th, j);
			this_->JobFinished(j);
		} else {
			Threads::Sleep(1);
		}
	}
	//dbgprintf("Thread %p ending.\n", arg);
}

void QueuedCPUTracker::ApplyOffsetGain(CPUTracker* trk, int beadIndex)
{
	if (calib_offset || calib_gain) {
		int index = cfg.width*cfg.height*beadIndex;

		//gc_mutex.lock();
		//float gf = gc_gainFactor, of = gc_offsetFactor;
		//gc_mutex.unlock();
		float gf=1,of=1;

		trk->ApplyOffsetGain(calib_offset ? &calib_offset[index] : 0, calib_gain ? &calib_gain[index] : 0, of, gf);
//		if (j->job.frame%100==0)
	}
}

void QueuedCPUTracker::ProcessJob(QueuedCPUTracker::Thread *th, Job* j)
{
	/// Choose the CPUTracker instance based on the specified thread.
	CPUTracker* trk = th->tracker;
	th->lock();

	/// Set the tracker's image memory.
	SetTrackerImage(trk, j);

	//FloatToJPEGFile("dbg.jpg",(float*)j->data,cfg.width,cfg.height);

	if (localizeMode & LT_ClearFirstFourPixels) {
		trk->srcImage[0]=trk->srcImage[1]=trk->srcImage[2]=trk->srcImage[3]=0;
	}

	/// Calibrate the images if needed.
	ApplyOffsetGain(trk, j->job.zlutIndex);

//	dbgprintf("Job: id %d, bead %d\n", j->id, j->zlut);

	LocalizationResult result={};
	result.job = j->job;

	/// Always compute the COM.
	vector2f com = trk->ComputeMeanAndCOM(cfg.com_bgcorrection);
	result.imageMean = trk->mean;

	if (_isnan(com.x) || _isnan(com.y))
		com = vector2f(cfg.width/2,cfg.height/2);

	bool boundaryHit = false;

	/// Compute 1DXCor, QI or 2DGaussian based on settings.
	if (localizeMode & LT_XCor1D) {
		result.firstGuess = com;
		vector2f resultPos = trk->ComputeXCorInterpolated(com, cfg.xc1_iterations, cfg.xc1_profileWidth, boundaryHit);
		result.pos.x = resultPos.x;
		result.pos.y = resultPos.y;
	} else if (localizeMode & LT_QI ){ 
		result.firstGuess = com;
		vector2f resultPos = trk->ComputeQI(com, cfg.qi_iterations, cfg.qi_radialsteps, cfg.qi_angstepspq, cfg.qi_angstep_factor, cfg.qi_minradius, cfg.qi_maxradius, boundaryHit, &qi_radialbinweights[0]);
		result.pos.x = resultPos.x;
		result.pos.y = resultPos.y;
	} else if (localizeMode & LT_Gaussian2D) {
		result.firstGuess = com;
		CPUTracker::Gauss2DResult gr = trk->Compute2DGaussianMLE(com, cfg.gauss2D_iterations, cfg.gauss2D_sigma);
		vector2f xy = gr.pos;
		result.pos = vector3f(xy.x,xy.y,0.0f);
	} else {
		result.firstGuess.x = result.pos.x = com.x;
		result.firstGuess.y = result.pos.y = com.y;
	}

	/// Compute Z profile and Z position if Z localization is requested.
	bool normalizeProfile = (localizeMode & LT_NormalizeProfile)!=0;
	if(localizeMode & LT_LocalizeZ) {
		float* prof=ALLOCA_ARRAY(float,cfg.zlut_radialsteps);

		for (int i=0;i< ((localizeMode & LT_ZLUTAlign) ? 5 : 1) ; i++) {
			if (localizeMode & LT_FourierLUT) {
				trk->FourierRadialProfile(prof,cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius);
			} else {
				trk->ComputeRadialProfile(prof,cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, result.pos2D(), false, &boundaryHit, normalizeProfile );
			}

			float *cmpprof = 0;

			if (zlut_enablecmpprof)
				cmpprof = &zlut_cmpprofiles[j->job.zlutIndex*zlut_planes];

			if (i > 0) {
				// update with Quadrant Align
				result.pos = trk->QuadrantAlign(result.pos, j->job.zlutIndex, cfg.qi_angstepspq, boundaryHit);
			}
			result.pos.z = trk->LUTProfileCompare(prof, j->job.zlutIndex, cmpprof, CPUTracker::LUTProfMaxQuadraticFit,(float*)0,(int*)0,j->job.frame);
			//dbgprintf("[%d] x=%f, y=%f, z=%f\n", i, result.pos.x,result.pos.y,result.pos.z);
			
		}

		if (localizeMode & LT_LocalizeZWeighted) {
			result.pos.z = trk->LUTProfileCompareAdjustedWeights(prof, j->job.zlutIndex, result.pos.z);
		}

		if (zlut_bias_correction)
			result.pos.z = ZLUTBiasCorrection(result.pos.z, zlut_planes, j->job.zlutIndex);
	}

	if(dbgPrintResults)
		dbgprintf("fr:%d, bead: %d: x=%f, y=%f, z=%f\n",result.job.frame, result.job.zlutIndex, result.pos.x, result.pos.y, result.pos.z);

	th->unlock();

	result.error = boundaryHit ? 1 : 0;
	/// Add the results to the available results.
	results_mutex.lock();
	results.push_back(result);
	resultCount++;
	results_mutex.unlock();
}

void QueuedCPUTracker::SetRadialZLUT(float* data, int num_zluts, int planes)
{
//	jobs_mutex.lock();
//	results_mutex.lock();
	if (zlut_bias_correction)
		delete zlut_bias_correction;

	if (zluts) delete[] zluts;
	int res = cfg.zlut_radialsteps;
	int total = num_zluts*res*planes;
	if (total > 0) {
		zluts = new float[planes*res*num_zluts];
		std::fill(zluts,zluts+(planes*res*num_zluts), 0.0f);
		zlut_planes = planes;
		zlut_count = num_zluts;
		if(data)
			std::copy(data, data+(planes*res*num_zluts), zluts);
	}
	else
		zluts = 0;

	if (zlut_cmpprofiles) delete[] zlut_cmpprofiles;
	zlut_cmpprofiles = new float [num_zluts*planes];

	UpdateZLUTs();
//	results_mutex.unlock();
//	jobs_mutex.unlock();
}

void QueuedCPUTracker::SetRadialWeights(float *rweights)
{
	if (rweights)
		zcmp.assign(rweights, rweights + cfg.zlut_radialsteps);
	else
		zcmp.clear();

	for (unsigned int i = 0; i<threads.size(); i++){
		threads[i].lock();
		threads[i].tracker->SetRadialWeights(rweights);
		threads[i].unlock();
	}

}

void QueuedCPUTracker::UpdateZLUTs()
{
	for (unsigned int i = 0; i<threads.size(); i++){
		threads[i].lock();
		threads[i].tracker->SetRadialZLUT(zluts, zlut_planes, cfg.zlut_radialsteps, zlut_count, cfg.zlut_minradius, cfg.zlut_maxradius, false, false);
		threads[i].unlock();
	}
}

void QueuedCPUTracker::GetRadialZLUTSize(int &count, int& planes, int &rsteps)
{
	count = zlut_count;
	planes = zlut_planes;
	rsteps = cfg.zlut_radialsteps;
}

void QueuedCPUTracker::GetRadialZLUT(float *zlut)
{
	int nElem = zlut_planes*cfg.zlut_radialsteps*zlut_count;
	if (nElem>0) {
		results_mutex.lock();
		memcpy(zlut, zluts, sizeof(float)* nElem);
		results_mutex.unlock();
	}
}

void QueuedCPUTracker::ScheduleLocalization(void* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo)
{
	if (processJobs) { /// \bug So if process is stopped, it'll just queue regardless of queue sizes?
		while(maxQueueSize != 0 && GetQueueLength () >= maxQueueSize)
			Threads::Sleep(5);
	}

	Job* j = AllocateJob();
	int dstPitch = PDT_BytesPerPixel(pdt) * cfg.width;

	if(!j->data || j->dataType != pdt) {
		if (j->data) delete[] j->data;
		j->data = new uchar[dstPitch * cfg.height];
	}
	for (int y=0; y<cfg.height; y++)
		memcpy(&j->data[dstPitch*y], & ((uchar*)data)[pitch*y], dstPitch);

	j->dataType = pdt;
	j->job = *jobInfo;

	AddJob(j);
}

int QueuedCPUTracker::FetchResults(LocalizationResult* dstResults, int maxResults)
{
	int numResults = 0;
	results_mutex.lock();
	while (numResults < maxResults && !results.empty()) {
		dstResults[numResults++] = results.back();
		results.pop_back();
		resultCount--;
	}
	results_mutex.unlock();
	return numResults;
}

void QueuedCPUTracker::GenerateTestImage(float* dst, float xp,float yp, float size, float SNratio)
{
	ImageData img(dst,cfg.width,cfg.height);
	::GenerateTestImage(img,xp,yp,size,SNratio);
}

bool QueuedCPUTracker::GetDebugImage(int id, int *w, int *h,float** data)
{
	if (id >= 0 && id < (int)threads.size()) {
		threads[id].lock();

		*w = cfg.width;
		*h = cfg.height;

		*data = new float [cfg.width*cfg.height];
		memcpy(*data, threads[id].tracker->GetDebugImage(), sizeof(float)* cfg.width*cfg.height);
		
		threads[id].unlock();
		return true;
	}

	return false;
}

QueuedTracker::ConfigValueMap QueuedCPUTracker::GetConfigValues()
{
	ConfigValueMap cvm;
	cvm["trace"] = dbgPrintResults ? "1" : "0";
	return cvm;
}


void QueuedCPUTracker::SetConfigValue(std::string name, std::string value)
{
	if (name == "trace")
		dbgPrintResults = !!atoi(value.c_str());
}

void QueuedCPUTracker::SetLocalizationMode(LocMode_t lt)
{
	while (!IsIdle());
	localizeMode = lt;
}

void QueuedCPUTracker::GetImageZLUTSize(int* dims)
{
	for (int i=0;i<4;i++)
		dims[i]=image_lut_dims[i];
}


void QueuedCPUTracker::GetImageZLUT(float* dst) 
{
	if (image_lut) {
		memcpy(dst, image_lut, sizeof(float)*image_lut_nElem_per_bead*image_lut_dims[0]);
	}
}

void QueuedCPUTracker::EnableRadialZLUTCompareProfile(bool enabled)
{
	zlut_enablecmpprof=enabled;
}

// dst = [count * planes]
void QueuedCPUTracker::GetRadialZLUTCompareProfile(float* dst)
{
	if (zlut_cmpprofiles){ 
		for (int i=0;i<zlut_count*zlut_planes;i++)
			dst[i]=zlut_cmpprofiles[i];
	}
}

bool QueuedCPUTracker::SetImageZLUT(float* src, float *radial_lut, int* dims)
{
	if (image_lut)  {
		delete[] image_lut;
		image_lut=0;
	}

	for (int i=0;i<4;i++)
		image_lut_dims[i]=dims[i];

	image_lut_nElem_per_bead = dims[1]*dims[2]*dims[3];

	if (image_lut_nElem_per_bead > 0 && dims[0] > 0) {
		image_lut = new float [image_lut_nElem_per_bead*image_lut_dims[0]];
		memset(image_lut, 0, sizeof(float)*image_lut_nElem_per_bead*image_lut_dims[0]);
	}

	if (src) {
		memcpy(image_lut, src, sizeof(float)*image_lut_nElem_per_bead*image_lut_dims[0]);
	}

	SetRadialZLUT(radial_lut, dims[0], dims[1]);

	return true; // returning true indicates this implementation supports ImageLUT
}

void QueuedCPUTracker::BeginLUT(uint flags)
{
	zlut_buildflags = flags;
}

void QueuedCPUTracker::BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, int plane, vector2f* known_pos)
{
	parallel_for(zlut_count,[&] (int i) {
//	for(int i=0;i<zlut_count;i++) {
		CPUTracker trk (cfg.width,cfg.height);
		void *img_data = (uchar*)data + pitch * cfg.height * i;

		if (pdt == QTrkFloat) {
			trk.SetImage((float*)img_data, pitch);
		} else if (pdt == QTrkU8) {
			trk.SetImage8Bit((uchar*)img_data, pitch);
		} else {
			trk.SetImage16Bit((ushort*)img_data,pitch);
		}
		ApplyOffsetGain(&trk, i);

		vector2f pos;

		if (known_pos) {
			pos = known_pos[i];
		} else {
			vector2f com = trk.ComputeMeanAndCOM();
			bool bhit;
			pos = trk.ComputeQI(com, cfg.qi_iterations, cfg.qi_radialsteps, cfg.qi_angstepspq, cfg.qi_angstep_factor, cfg.qi_minradius, cfg.qi_maxradius, bhit);
			//dbgprintf("BuildLUT() COMPos: %f,%f, QIPos: x=%f, y=%f\n", com.x,com.y, pos.x, pos.y);
		}
		if (zlut_buildflags & BUILDLUT_IMAGELUT) {
			int h=ImageLUTHeight(), w=ImageLUTWidth();
			float* lut_dst = &image_lut[ i * image_lut_nElem_per_bead + w*h* plane ];

			vector2f ilut_scale(1,1);
			float startx = pos.x - w/2*ilut_scale.x;
			float starty = pos.y - h/2*ilut_scale.y;

			for (int y=0;y<h;y++) {
				for (int x=0;x<w;x++) {

					float px = startx + x*ilut_scale.x;
					float py = starty + y*ilut_scale.y;

					bool outside=false;
					float v = Interpolate(trk.srcImage, trk.width, trk.height, px, py, &outside);
					lut_dst[y*w+x] += v - trk.mean;
				}
			}
		}

		float *bead_zlut=GetZLUTByIndex(i);
		float *tmp = new float[cfg.zlut_radialsteps];

		if (zlut_buildflags  & BUILDLUT_FOURIER){
			trk.FourierRadialProfile(tmp, cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius);
			if (plane==0) {
				for (int i=0;i<trk.width*trk.height;i++)
					trk.srcImage[i]=sqrtf(trk.srcImage[i]);
				trk.SaveImage("freqimg.jpg");
			}
		} else {
			trk.ComputeRadialProfile(tmp, cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, pos, false);
		}
	//	WriteArrayAsCSVRow("rlut-test.csv", tmp, cfg.zlut_radialsteps, plane>0);
		for(int i=0;i<cfg.zlut_radialsteps;i++) 
			bead_zlut[plane*cfg.zlut_radialsteps+i] += tmp[i];
		delete[] tmp;
	} );
}

void QueuedCPUTracker::FinalizeLUT()
{
	// normalize radial LUT?

	if (zluts) {
		for (int i=0;i<zlut_count*zlut_planes;i++) {

		//	WriteArrayAsCSVRow("finalize-lut.csv", &zluts[cfg.zlut_radialsteps*i], cfg.zlut_radialsteps, i>0);
			NormalizeRadialProfile(&zluts[cfg.zlut_radialsteps*i], cfg.zlut_radialsteps);

		}
	}

	int w = ImageLUTWidth();
	int h = ImageLUTHeight();

	if (w * h > 0) {

		image_lut_dz = new float [image_lut_nElem_per_bead * ImageLUTNumBeads()];
		image_lut_dz2 = new float [image_lut_nElem_per_bead * ImageLUTNumBeads()];

		// Compute 1st and 2nd order derivatives
		for (int i=0;i<image_lut_dims[0];i++) {
			for (int z=1;z<image_lut_dims[1]-1;z++) {
				float *img = &image_lut[ image_lut_nElem_per_bead * i + w*h*z ]; // current plane
				float *imgL = &image_lut[ image_lut_nElem_per_bead * i + w*h*(z-1) ]; // one plane below
				float *imgU = &image_lut[ image_lut_nElem_per_bead * i + w*h*(z+1) ]; // one plane above

				float *img_dz = &image_lut_dz[ image_lut_nElem_per_bead * i + w*h*z ];
				float *img_dz2 = &image_lut_dz2[ image_lut_nElem_per_bead * i + w*h*z ];

				// Numerical approx of derivatives..
				for (int y=0;y<h;y++) {
					for (int x=0;x<w;x++) {
						const float h = 1.0f;
						img_dz[y*w+x] = 0.5f * ( imgU[y*w+x] - imgL[y*w+x] ) / h;
						img_dz2[y*w+x] = (imgU[y*w+x] - 2*img[y*w+x] + imgL[y*w+x]) / (h*h);
					}
				}
			}
			for (int k=0;k<w*h;k++) {
				// Top and bottom planes are simply copied from the neighbouring planes
				image_lut_dz[ image_lut_nElem_per_bead * i + w*h*0 + k] = image_lut_dz[ image_lut_nElem_per_bead * i + w*h*1 + k];
				image_lut_dz[ image_lut_nElem_per_bead * i + w*h*(image_lut_dims[1]-1) + k] = image_lut_dz[ image_lut_nElem_per_bead * i + w*h*(image_lut_dims[1]-2) + k];
				image_lut_dz2[ image_lut_nElem_per_bead * i + w*h*0 + k] = image_lut_dz2[ image_lut_nElem_per_bead * i + w*h*1 + k];
				image_lut_dz2[ image_lut_nElem_per_bead * i + w*h*(image_lut_dims[1]-1) + k] = image_lut_dz2[ image_lut_nElem_per_bead * i + w*h*(image_lut_dims[1]-2) + k];
			}
		}
	}
}

void QueuedCPUTracker::ZLUTSelfTest()
{
	if (zluts){
		/*bool compEnabled = zlut_enablecmpprof;
		if(!compEnabled)
			EnableRadialZLUTCompareProfile(true);*/


		threads[0].lock();
		CPUTracker* trk = threads[0].tracker;
		
		for (int ii=0;ii<zlut_count;ii++) {
			//trk.SetRadialZLUT(&zluts[cfg.zlut_radialsteps*zlut_planes*ii],zlut_planes,cfg.zlut_radialsteps,zlut_count,cfg.zlut_minradius,cfg.zlut_maxradius,false,false);
			
			float* curZLUT = GetZLUTByIndex(ii);			
			for(int plane=0;plane<zlut_planes;plane++){
				float* cmpprof = new float[zlut_planes];
				trk->LUTProfileCompare(&curZLUT[plane*cfg.zlut_radialsteps],ii,cmpprof,CPUTracker::LUTProfMaxQuadraticFit);
				WriteArrayAsCSVRow("D:\\TestImages\\zlutSelfTest.csv",cmpprof,zlut_planes,true);
				delete[] cmpprof;
			}			
		}

		threads[0].unlock();
		
		//EnableRadialZLUTCompareProfile(compEnabled);
	}
}

void QueuedCPUTracker::SetTrackerImage(CPUTracker* trk, Job* j)
{
	if (j->dataType == QTrkU8) {
		trk->SetImage8Bit(j->data, cfg.width);
	} else if (j->dataType == QTrkU16) {
		trk->SetImage16Bit((ushort*)j->data, cfg.width*2);
	} else {
		trk->SetImageFloat((float*)j->data);
	}
}