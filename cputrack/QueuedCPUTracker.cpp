#include "std_incl.h"
#include "QueuedCPUTracker.h"
#include <float.h>
#include <functional>

#include "DebugResultCompare.h"

#ifndef CUDA_TRACK
QueuedTracker* CreateQueuedTracker(const QTrkComputedConfig& cc){ 
	return new QueuedCPUTracker(cc);
}
void SetCUDADevices(int* dev, int ndev) {
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
	cfg = cc;
	quitWork = false;

	if (cfg.numThreads < 0) {
		cfg.numThreads = Threads::GetCPUCount();
		dbgprintf("Using %d threads\n", cfg.numThreads);
	} 

	maxQueueSize = std::max(2,cfg.numThreads) * 50;
	jobCount = 0;
	resultCount = 0;

	zluts = 0;
	zlut_count = zlut_planes = 0;
	processJobs = false;
	jobsInProgress = 0;
	dbgPrintResults = false;
	
	calib_gain = calib_offset = 0;
	gc_gainFactor = gc_offsetFactor = 1.0f;
	localizeMode = LT_OnlyCOM;

	for (int i=0;i<4;i++) image_lut_dims[i]=0;
	image_lut_nElem_per_bead=0;
	image_lut = 0;
	image_lut_dz = image_lut_dz2 = 0;

	zlutAlignRootFinder = RF_Secant;

	Start();
}

QueuedCPUTracker::~QueuedCPUTracker()
{
	// wait for threads to finish
	quitWork = true;

	for (int k=0;k<threads.size();k++) {
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
	if (image_lut) delete[] image_lut;
	if (image_lut_dz) delete[] image_lut_dz;
	if (image_lut_dz2) delete[] image_lut_dz2;
}

void QueuedCPUTracker::SetPixelCalibrationImages(float* offset, float* gain)
{
	if (zlut_count > 0) {
		int nelem = cfg.width*cfg.height*zlut_count;

		if (calib_gain == 0) {
			calib_gain = new float[nelem];
			calib_offset = new float[nelem];
		}

		memcpy(calib_gain, gain, sizeof(float)*nelem);
		memcpy(calib_offset, offset, sizeof(float)*nelem);

#ifdef _DEBUG
		std::string path = GetLocalModulePath();
		for (int i=0;i<zlut_count;i++) {
			FloatToJPEGFile( SPrintf("%s/gain-bead%d.jpg", path.c_str(), i).c_str(), &calib_gain[cfg.width*cfg.height*i], cfg.width,cfg.height);
			FloatToJPEGFile( SPrintf("%s/offset-bead%d.jpg", path.c_str(), i).c_str(), &calib_offset[cfg.width*cfg.height*i], cfg.width,cfg.height);
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
		threads[k].tracker = new CPUTracker(cfg.width, cfg.height, cfg.xc1_profileLength);
		threads[k].manager = this;
		threads[k].tracker->trackerID = k;
	}

	for (int k=0;k<threads.size();k++) {
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

void QueuedCPUTracker::ProcessJob(QueuedCPUTracker::Thread *th, Job* j)
{
	CPUTracker* trk = th->tracker;
	th->lock();

	if (j->dataType == QTrkU8) {
		trk->SetImage8Bit(j->data, cfg.width);
	} else if (j->dataType == QTrkU16) {
		trk->SetImage16Bit((ushort*)j->data, cfg.width*2);
	} else {
		trk->SetImageFloat((float*)j->data);
	}

	if (localizeMode & LT_ClearFirstFourPixels) {
		trk->srcImage[0]=trk->srcImage[1]=trk->srcImage[2]=trk->srcImage[3]=0;
	}

	if (calib_offset && calib_gain) {
		int index = cfg.width*cfg.height*j->job.zlutIndex;

		gc_mutex.lock();
		float gf = gc_gainFactor, of = gc_offsetFactor;
		gc_mutex.unlock();

		trk->ApplyOffsetGain(&calib_offset[index], &calib_gain[index], of, gf);
//		if (j->job.frame%100==0)
	}

//	dbgprintf("Job: id %d, bead %d\n", j->id, j->zlut);

	LocalizationResult result={};
	result.job = j->job;

	vector2f com = trk->ComputeMeanAndCOM(cfg.com_bgcorrection);

	if (_isnan(com.x) || _isnan(com.y))
		com = vector2f(cfg.width/2,cfg.height/2);

	bool boundaryHit = false;

	if (localizeMode & LT_XCor1D) {
		result.firstGuess = com;
		vector2f resultPos = trk->ComputeXCorInterpolated(com, cfg.xc1_iterations, cfg.xc1_profileWidth, boundaryHit);
		result.pos.x = resultPos.x;
		result.pos.y = resultPos.y;
	} else if (localizeMode & LT_QI ){ 
		result.firstGuess = com;
		vector2f resultPos = trk->ComputeQI(com, cfg.qi_iterations, cfg.qi_radialsteps, cfg.qi_angstepspq, cfg.qi_angstep_factor, cfg.qi_minradius, cfg.qi_maxradius, boundaryHit);
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

	bool normalizeProfile = (localizeMode & LT_NormalizeProfile)!=0;
	if(localizeMode & LT_LocalizeZ) {
		if (localizeMode & LT_FourierLUT) {
			trk->FourierTransform2D();
			result.pos.z = trk->ComputeZ(vector2f(cfg.width/2,cfg.height/2), cfg.zlut_angularsteps, j->job.zlutIndex, false, &boundaryHit, 0, 0, normalizeProfile );
		} else 
			result.pos.z = trk->ComputeZ(result.pos2D(), cfg.zlut_angularsteps, j->job.zlutIndex, false, &boundaryHit, 0, 0, normalizeProfile );
	} else if (localizeMode & LT_BuildRadialZLUT && (j->job.zlutIndex >= 0 && j->job.zlutIndex < zlut_count)) {
		float* zlut = GetZLUTByIndex(j->job.zlutIndex);
		float* rprof = ALLOCA_ARRAY(float, cfg.zlut_radialsteps);
		trk->ComputeRadialProfile(rprof, cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, result.pos2D(), false, &boundaryHit, normalizeProfile);
		float* dstprof = &zlut[j->job.zlutPlane * cfg.zlut_radialsteps];
		for (int i=0;i<cfg.zlut_radialsteps;i++) 
			dstprof[i] += rprof[i];
	}

	if (localizeMode & LT_ZLUTAlign ){
		result.pos = ZLUTAlign(th, j->job, result.pos);
	}

	if(dbgPrintResults)
		dbgprintf("fr:%d, bead: %d: x=%f, y=%f, z=%f\n",result.job.frame, result.job.zlutIndex, result.pos.x, result.pos.y, result.pos.z);

	th->unlock();

	result.error = boundaryHit ? 1 : 0;

	results_mutex.lock();
	results.push_back(result);
	resultCount++;
	results_mutex.unlock();
}

vector3f QueuedCPUTracker::ZLUTAlign(QueuedCPUTracker::Thread *th, const LocalizationJob& job, vector3f pos)
{
	CPUTracker* trk = th->tracker;
	vector3d rpos;

	if (zlutAlignRootFinder == RF_GradientDescent) {
		for (int i=0;i<400;i++) {
			vector3d d;
			//float k=1.0f/sqrtf(1+i);
			rpos = trk->ZLUTAlignGradientStep (pos, job.zlutIndex, &d, vector3d(0.02f,0.02f,0.1f), vector3d(1e-3,1e-3,1e-3));
			if (th == &this->threads[0]) dbgprintf("dXYZ[%d]: %f, %f, %f. at %f, %f, %f\n", i, d.x,d.y,d.z, rpos.x,rpos.y,rpos.z);
		}
	} else if (zlutAlignRootFinder == RF_Secant) {
		rpos = trk->ZLUTAlignSecantMethod (pos, job.zlutIndex,10, vector3f(2e-3,2e-3,2e-3));
	} else if (zlutAlignRootFinder == RF_NewtonRaphson) {
		for (int i=0;i<5;i++) {
			vector3d d;
			rpos = trk->ZLUTAlignNewtonRaphsonIndependentStep (pos, job.zlutIndex, &d, vector3f(2e-3,2e-3,2e-3));
			if (th == &this->threads[0]) dbgprintf("dXYZ[%d]: %f, %f, %f. at %f, %f, %f\n", i, d.x,d.y,d.z, rpos.x,rpos.y,rpos.z);
		}
	}

	return rpos;
}

void QueuedCPUTracker::SetRadialZLUT(float* data, int num_zluts, int planes)
{
//	jobs_mutex.lock();
//	results_mutex.lock();

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
}

void QueuedCPUTracker::UpdateZLUTs()
{
	for (int i=0;i<threads.size();i++){
		threads[i].lock();
		threads[i].tracker->SetRadialZLUT(zluts, zlut_planes, cfg.zlut_radialsteps, zlut_count, cfg.zlut_minradius, cfg.zlut_maxradius, cfg.zlut_angularsteps, false, false, zcmp.empty() ? 0 : &zcmp[0]);
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
	if (processJobs) {
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


void QueuedCPUTracker::GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount)
{
	ImageData img(dst,cfg.width,cfg.height);
	::GenerateTestImage(img,xp,yp,z,photoncount);
}



bool QueuedCPUTracker::GetDebugImage(int id, int *w, int *h,float** data)
{
	if (id >= 0 && id < threads.size()) {
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


typedef std::map<QueuedCPUTracker::ZLUTAlignRootFinder, std::string> ZLUTAlignRootFinderNameMap;
ZLUTAlignRootFinderNameMap& rootFinderNames() {
	static ZLUTAlignRootFinderNameMap names;
	if (names.empty()) {
		names[QueuedCPUTracker::RF_NewtonRaphson]="nr";
		names[QueuedCPUTracker::RF_NewtonRaphson3D]="nr3d";
		names[QueuedCPUTracker::RF_Secant]="secant";
		names[QueuedCPUTracker::RF_GradientDescent]="grdesc";
	}
	return names;
}


QueuedTracker::ConfigValueMap QueuedCPUTracker::GetConfigValues()
{
	ConfigValueMap cvm;
	cvm["trace"] = dbgPrintResults ? "1" : "0";
	cvm["za_rootfinder"] = rootFinderNames()[zlutAlignRootFinder];
	return cvm;
}


void QueuedCPUTracker::SetConfigValue(std::string name, std::string value)
{
	if (name == "trace")
		dbgPrintResults = !!atoi(value.c_str());
	if (name == "za_rootfinder") {
		bool found=false;
		for (auto i = rootFinderNames().begin(); i != rootFinderNames().end(); ++i)
			if (i->second == value) {
				zlutAlignRootFinder = i->first;
				break;
			}
		if (!found) {
			dbgprintf("SetConfigValue(): Unknown ZLUTAlign root finder method: %s\n", value.c_str());
		}
	}
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

	return true; // returning true indicates this implementation support ImageLUT
}


void QueuedCPUTracker::BuildLUT(void* data, int pitch, QTRK_PixelDataType pdt, uint flags, int plane)
{
	parallel_for(zlut_count, [&] (int i) {
		CPUTracker trk (cfg.width,cfg.height);
		void *img_data = (uchar*)data + pitch * cfg.height * i;

		if (pdt == QTrkFloat) {
			trk.SetImage((float*)img_data, pitch);
		} else if (pdt == QTrkU8) {
			trk.SetImage8Bit((uchar*)img_data, pitch);
		} else {
			trk.SetImage16Bit((ushort*)img_data,pitch);
		}

		vector2f com = trk.ComputeMeanAndCOM();
		bool bhit;
		vector2f qipos = trk.ComputeQI(com, cfg.qi_iterations, cfg.qi_radialsteps, cfg.qi_angstepspq, cfg.qi_angstep_factor, cfg.qi_minradius, cfg.qi_maxradius, bhit);

		dbgprintf("BuildLUT() COMPos: %f,%f, QIPos: x=%f, y=%f\n", com.x,com.y, qipos.x, qipos.y);

		if (flags & BUILDLUT_IMAGELUT) {
			int h=ImageLUTHeight(), w=ImageLUTWidth();
			float* lut_dst = &image_lut[ i * image_lut_nElem_per_bead + w*h* plane ];

			vector2f ilut_scale(1,1);
			float startx = qipos.x - w/2*ilut_scale.x;
			float starty = qipos.y - h/2*ilut_scale.y;

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

		if (flags & BUILDLUT_FOURIER){
			trk.FourierTransform2D();
			trk.ComputeRadialProfile(tmp, cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, vector2f(cfg.width/2,cfg.height/2), false, 0, true);
		}
		else {
			trk.ComputeRadialProfile(tmp, cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, qipos, false, 0, true);
		}
	//	WriteArrayAsCSVRow("rlut-test.csv", tmp, cfg.zlut_radialsteps, plane>0);
		for(int i=0;i<cfg.zlut_radialsteps;i++) 
			bead_zlut[plane*cfg.zlut_radialsteps+i] += tmp[i];
		delete[] tmp;
	});
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


