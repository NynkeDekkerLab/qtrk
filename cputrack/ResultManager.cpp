#include "std_incl.h"
#include "ResultManager.h"
#include "utils.h"

#define BINFILE_VERSION 3

TextResultFile::TextResultFile(const char *fn, bool write)
{
	f = fopen(fn, write?"w":"r");
}

void TextResultFile::LoadRow(std::vector<vector3f>& pos)
{

}

void TextResultFile::SaveRow(std::vector<vector3f>& pos)
{
}

void BinaryResultFile::LoadRow(std::vector<vector3f>& pos)
{

}

void BinaryResultFile::SaveRow(std::vector<vector3f>& pos)
{
}


ResultManager::FrameCounters::FrameCounters()
{
	startFrame = 0;
	lastSaveFrame = 0;
	processedFrames = 0;
	capturedFrames = 0;
	localizationsDone = 0;
	lostFrames = 0;
	fileError = 0;
}

ResultManager::ResultManager(const char *outfile, const char* frameInfoFile, ResultManagerConfig *cfg, std::vector<std::string> colnames)
{
	config = *cfg;
	outputFile = outfile;
	this->frameInfoFile = frameInfoFile;

	qtrk = 0;

	thread = Threads::Create(ThreadLoop, this);
	quit=false;

	if (!outputFile.empty())
		remove(outfile);
	if (!this->frameInfoFile.empty())
		remove(frameInfoFile);

	frameInfoNames = colnames;

	if (config.binaryOutput) {
		WriteBinaryFileHeader();
	}

	dbgprintf("Allocating ResultManager with %d beads, %d motor columns and writeinterval %d\n", cfg->numBeads, cfg->numFrameInfoColumns, cfg->writeInterval);
}

void ResultManager::WriteBinaryFileHeader()
{
	FILE* f = fopen(outputFile.c_str(), "wb");
	if (!f) {
		throw std::runtime_error( SPrintf("Unable to open file %s", outputFile.c_str()));
	}

	// Write file header
	int version = BINFILE_VERSION;
	fwrite(&version, sizeof(int), 1, f);
	fwrite(&config.numBeads, sizeof(int), 1, f);
	fwrite(&config.numFrameInfoColumns, sizeof(int), 1, f);
	long data_offset_pos = ftell(f);
	int tmp=1234;
	fwrite(&tmp,sizeof(int), 1,f);
	for (int i=0;i<config.numFrameInfoColumns;i++) {
		auto& n = frameInfoNames[i];
		fwrite(n.c_str(), n.length()+1, 1, f);
	}
	long data_offset = ftell(f);
	dbgprintf("frame data offset: %d\n", data_offset);
	fseek(f, data_offset_pos, SEEK_SET);
	fwrite(&data_offset, sizeof(long), 1, f);
	
	dbgprintf("writing %d beads and %d frame-info columns into file %s\n", config.numBeads, config.numFrameInfoColumns, outputFile.c_str());
	fclose(f);
}

ResultManager::~ResultManager()
{
	quit = true;
	Threads::WaitAndClose(thread);

	DeleteAllElems(frameResults);
}


void ResultManager::StoreResult(LocalizationResult *r)
{
	if (CheckResultSpace(r->job.frame)) {

		LocalizationResult scaled = *r;
		// Make roi-centered pos
		scaled.pos = scaled.pos - vector3f( qtrk->cfg.width*0.5f, qtrk->cfg.height*0.5f, 0);
		scaled.pos = ( scaled.pos + config.offset ) * config.scaling;
		FrameResult* fr = frameResults[r->job.frame - cnt.startFrame];
		fr->results[r->job.zlutIndex] = scaled;
		fr->count++;

		// Advance processedFrames, either because measurements have been completed or because frames have been lost
		while (cnt.processedFrames - cnt.startFrame < (int) frameResults.size() && 
			(frameResults[cnt.processedFrames-cnt.startFrame]->count == config.numBeads || cnt.capturedFrames - cnt.processedFrames > 1000 ))
		{
			if (frameResults[cnt.processedFrames-cnt.startFrame]->count < config.numBeads)
				cnt.lostFrames ++;

			cnt.processedFrames ++;
		}

		cnt.localizationsDone ++;
	} else
		cnt.lostFrames++;
}

void ResultManager::WriteBinaryResults()
{
	if (outputFile.empty())
		return;

	FILE* f = fopen(outputFile.c_str(), "ab");
	if (!f) {
		dbgprintf("ResultManager::WriteBinaryResult() Unable to open file %s\n", outputFile.c_str());
		cnt.fileError ++;
		return;
	}

	for (int j=cnt.lastSaveFrame; j<cnt.processedFrames;j++)
	{
		auto fr = frameResults[j-cnt.startFrame];
		if (f) {
			fwrite(&j, sizeof(uint), 1, f);
			fwrite(&fr->timestamp, sizeof(double), 1, f);
			fwrite(&fr->frameInfo[0], sizeof(float), config.numFrameInfoColumns, f);
			for (int i=0;i<config.numBeads;i++) 
			{
				LocalizationResult *r = &fr->results[i];
				fwrite(&r->pos, sizeof(vector3f), 1, f);
			}
			for (int i=0;i<config.numBeads;i++)
				fwrite(&fr->results[i].error, sizeof(int), 1, f);
			for (int i=0;i<config.numBeads;i++) {
				fwrite(&fr->results[i].imageMean, sizeof(float), 1, f);
			}
		}
	}
	fclose(f);
}

void ResultManager::WriteTextResults()
{
	FILE* f = outputFile.empty () ? 0 : fopen(outputFile.c_str(), "a");
	FILE* finfo = frameInfoFile.empty() ? 0 : fopen(frameInfoFile.c_str(), "a");

	for (int k=cnt.lastSaveFrame; k<cnt.processedFrames;k++)
	{
		auto fr = frameResults[k-cnt.startFrame];
		if (f) {
			fprintf(f,"%d\t%f\t", k, fr->timestamp);
			for (int i=0;i<config.numBeads;i++) 
			{
				LocalizationResult *r = &fr->results[i];
				fprintf(f, "%.7f\t%.7f\t%.7f\t", r->pos.x,r->pos.y,r->pos.z);
			}
			fputs("\n", f);
		}
		if (finfo) {
			fprintf(finfo,"%d\t%f\t", k, fr->timestamp);
			for (int i=0;i<config.numFrameInfoColumns;i++)
				fprintf(finfo, "%.7f\t", fr->frameInfo[i]);
			fputs("\n", finfo);
		}
	}
	if(finfo) fclose(finfo);
	if(f) fclose(f);

}

void ResultManager::Write()
{
	resultMutex.lock();
	if (config.binaryOutput)
		WriteBinaryResults();
	else
		WriteTextResults();

	dbgprintf("Saved frame %d to %d\n", cnt.lastSaveFrame, cnt.processedFrames);
	cnt.lastSaveFrame = cnt.processedFrames;

	resultMutex.unlock();
}

QueuedTracker* ResultManager::GetTracker()
{
	trackerMutex.lock();
	QueuedTracker* trk = qtrk;
	trackerMutex.unlock();
	return trk;
}

void ResultManager::SetTracker(QueuedTracker *qtrk)
{
	trackerMutex.lock();
	this->qtrk = qtrk;
	trackerMutex.unlock();
}

bool ResultManager::Update()
{
	trackerMutex.lock();

	if (!qtrk) {
		trackerMutex.unlock();
		return 0;
	}

	const int NResultBuf = 40;
	LocalizationResult resultbuf[NResultBuf];

	int count = qtrk->FetchResults( resultbuf, NResultBuf );

	resultMutex.lock();
	for (int i=0;i<count;i++)
		StoreResult(&resultbuf[i]);

	trackerMutex.unlock();

	if (cnt.processedFrames - cnt.lastSaveFrame >= config.writeInterval) {
		Write();
	}

	if (config.maxFramesInMemory>0 && frameResults.size () > config.maxFramesInMemory) {

		int del = frameResults.size()-config.maxFramesInMemory;

		if (cnt.processedFrames < cnt.startFrame) {
			// write away any results that might be in there, unfinished localizations will be zero.
			int lost = cnt.startFrame-cnt.processedFrames;
			cnt.processedFrames += lost;
			cnt.lostFrames += lost;

			Write();
		}

		dbgprintf("Removing %d frames from memory\n", del);
		
		for (int i=0;i<del;i++)
			delete frameResults[i];
		frameResults.erase(frameResults.begin(), frameResults.begin()+del);

		cnt.startFrame += del;
	}
	resultMutex.unlock();

	return count>0;
}

void ResultManager::ThreadLoop(void *param)
{
	ResultManager* rm = (ResultManager*)param;

	while(true) {
		if (!rm->Update())
			Threads::Sleep(20);

		if (rm->quit)
			break;
	}
}

int ResultManager::GetBeadPositions(int startfr, int end, int bead, LocalizationResult* results)
{
	int count = end-startfr;

	resultMutex.lock();
	if (end > cnt.processedFrames)
		end = cnt.processedFrames;

	int start = startfr - cnt.startFrame;
	if (start < 0) start = 0;
	if (count > cnt.processedFrames-cnt.startFrame)
		count = cnt.processedFrames-cnt.startFrame;

	for (int i=0;i<count;i++){
		results[i] = frameResults[i+start]->results[bead];
	}
	resultMutex.unlock();

	return count;
}


void ResultManager::Flush()
{
	trackerMutex.lock();
	if (qtrk) qtrk->Flush();
	trackerMutex.unlock();

	Update();

	resultMutex.lock();

	Write();

	// Dump stats about unfinished frames for debugging
#ifdef _DEBUG
	for (uint i=0;i<frameResults.size();i++) {
		FrameResult *fr = frameResults[i];
//		dbgprintf("Frame %d. TS: %f, Count: %d\n", i, fr->timestamp, fr->count);
		if (fr->count != config.numBeads) {
			for (int j=0;j<fr->results.size();j++) {
				if( fr->results[j].job.frame == 0)
					dbgprintf("%d, ", j );
			}
			dbgprintf("\n");
		}
	}
#endif
	resultMutex.unlock();
}


ResultManager::FrameCounters ResultManager::GetFrameCounters()
{
	resultMutex.lock();
	FrameCounters c = cnt;
	resultMutex.unlock();
	return c;
}

int ResultManager::GetResults(LocalizationResult* results, int startFrame, int numFrames)
{
	resultMutex.lock();

	if (startFrame >= cnt.startFrame && numFrames+startFrame <= cnt.processedFrames)  {
		for (int f=0;f<numFrames;f++) {
			int index = f + startFrame - cnt.startFrame;
			for (int j=0;j<config.numBeads;j++)
				results[config.numBeads*f+j] = frameResults[index]->results[j];
		}
	}
	resultMutex.unlock();

	return numFrames;
}

bool ResultManager::CheckResultSpace(int fr)
{
	if(fr < cnt.startFrame)
		return false; // already removed, ignore

	if (fr > cnt.processedFrames + 20000) {

		dbgprintf("ResultManager: Ignoring suspiciously large frame number (%d).\n", fr);

		return false;
	}

	while (fr >= cnt.startFrame + frameResults.size()) {
		frameResults.push_back (new FrameResult( config.numBeads, config.numFrameInfoColumns));
	}
	return true;
}

void ResultManager::StoreFrameInfo(int frame, double timestamp, float* columns)
{
	resultMutex.lock();

	if (CheckResultSpace(frame)) {
		FrameResult* fr = frameResults[frame-cnt.startFrame];		

		if (!fr->hasFrameInfo) {
			fr->timestamp = timestamp;
			for(int i=0;i<config.numFrameInfoColumns;i++)
				fr->frameInfo[i]=columns[i];
			++cnt.capturedFrames;
			fr->hasFrameInfo=true;
		}
	}

	resultMutex.unlock();
}


int ResultManager::GetFrameCount()
{
	resultMutex.lock();
	int nfr = cnt.capturedFrames;
	resultMutex.unlock();
	return nfr;
}

bool ResultManager::RemoveBeadResults(int bead)
{
	// TODO: We need to modify the saved data file

	for (uint i=0;i<frameResults.size();i++) {
		auto fr = frameResults[i];

		fr->count--;
		fr->results.erase(fr->results.begin()+bead);
	}

	return true;
}

