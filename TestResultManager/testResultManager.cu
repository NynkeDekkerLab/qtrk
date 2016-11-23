#include <iostream>
#include <string>
#include <ctime>
#include <chrono> 
#include "omp.h"
#include <stdio.h>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../cputrack/ResultManager.h"
#include "../cputrack/QueuedTracker.h" 
#include "../cputrack/QueuedCPUTracker.h" 
#include "../cudatrack/QueuedCUDATracker.h" 
#include "../cputrack-test/SharedTests.h"
/*
This file tests the Result Manager. It is pretty straightforward:
- Build the Result Manager as LabView would
- Overload it with Data
- Test whether it loses frames during overload



*/
int main(int argc, char* argv[])
{
	auto timeStart = std::chrono::high_resolution_clock::now();
	fprintf(stderr, "Note: Initialising ndlab/test/ResultManager (%d arguments).\n", argc);

	//number of images/frames
	const int N = (int)atoi(argv[1]); //4 beads, 24 images; 6 frames per image?

	fprintf(stderr, "Testing ResultManager with %d images. %s \n", N, argv[1]);

	const char* file = "./ResultManagerData.txt";
	const char * frameinfo = "./ResultManagerFrameInfo.txt";
	ResultManagerConfig * cfg = new ResultManagerConfig;
	cfg->numBeads = 4;
	cfg->numFrameInfoColumns = 0;
	cfg->scaling = vector3f(1.0, 1.0, 1.0);
	cfg->offset = vector3f(0.0, 0.0, 0.0);
	cfg->writeInterval = 2500;
	cfg->maxFramesInMemory = 0;// 100000;
	cfg->binaryOutput = false;

	std::vector< std::string > colNames;
	std::string testName("Hey now");
	colNames.push_back(testName);

	// The GodFather manages your results.
	ResultManager* manager = new ResultManager(file, frameinfo, cfg, colNames);
	//The QueuedCPUTracker instance is required to retrieve the results. It needs settings.

	QTrkComputedConfig settings;
	settings.qi_iterations = 2;
	settings.zlut_minradius = 1;
	settings.qi_minradius = 1;
	settings.width = settings.height = 100;
	settings.Update();

	//Let's load some image data.
	auto data = ReadJPEGFile("exp.jpg");

	QueuedTracker * qtrk;
	if (argc == 3)
	{
		std::string argTracker = std::string(argv[2]);
		if (argTracker == "gpu")
		{
			fprintf(stderr, "Using CUDA tracker (GPU).\n");
			qtrk = new QueuedCUDATracker(settings);
		}
		else
		{
			fprintf(stderr, "Using CPU tracker (CPU).\n");
			qtrk = new QueuedCPUTracker(settings);
		}
	}
	//localization Mode QI tracker
	auto modeQI = (LocMode_t)(LT_QI | LT_NormalizeProfile | LT_LocalizeZ);

	qtrk->SetLocalizationMode(modeQI);


	std::vector<LocalizationJob> jobs;

	int frame = 0;
	for (int i = 0; i < N; i++)
	{
		if (i % 4 == 0 && i != 0)
		{
			frame++;
		}
		//Make a localization job (batch of calculations)
		LocalizationJob job(frame, 0, 0, 0);
		job.zlutIndex = i % 4;
		jobs.push_back(job);
		qtrk->ScheduleImageData(&data, &job);
	}
	manager->SetTracker(qtrk);
	//Process images (using Flush because Start is CPU only)
	qtrk->Flush();
	auto timeTrack = std::chrono::high_resolution_clock::now();
	while (manager->GetFrameCounters().localizationsDone < N)
	{
		auto timeEnd = std::chrono::high_resolution_clock::now();
		auto microSeconds = (int)std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeTrack).count();
		if (microSeconds > 100000)
		{
			auto sinceStart = (int)std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
			auto counters = manager->GetFrameCounters();

			double time = sinceStart / 1000000;
			fprintf(stderr, "Update[%.3f]: %d Localisations performed.\n", time, counters.localizationsDone);
			timeTrack = timeEnd;
		}
	}

	//Assign tracker

	//Assign frame
	float somefloat = 0.0;
	for (int i = 0; i < N / 4; i++)
	{
		manager->StoreFrameInfo(i, i, &somefloat);
	}
	//Pointer that will be filled with results
	std::vector<LocalizationResult> results;

	vector3f startPosition = { 0.0, 0.0, 0.0 };
	vector2f initialGuess = { 50.0, 50.0 };

	for (int i = 0; i < N; i++)
	{
		results.push_back({ jobs.at(i), startPosition, initialGuess, 0.0, 0.0 });
	}

	//Fill results array
	manager->Flush();
	//Wait untill all localizations have been performed.

	timeTrack = std::chrono::high_resolution_clock::now();
	while (manager->GetFrameCounters().lastSaveFrame  < N / 4)
	{
		auto timeEnd = std::chrono::high_resolution_clock::now();
		auto microSeconds = (int)std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeTrack).count();
		if (microSeconds > 100000)
		{
			auto sinceStart = (int)std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
			auto counters = manager->GetFrameCounters();

			double time = sinceStart / 1000000;
			fprintf(stderr, "Update[%.3f]: %d frames saved.\n", time, counters.lastSaveFrame);
			timeTrack = timeEnd;
		}
	}

	auto counters = manager->GetFrameCounters();
	auto getResults = manager->GetResults(results.data(), 0, N / cfg->numBeads);
	fprintf(stderr, "ResultManager results (%d) :\n", getResults);
	fprintf(stderr, "\t frame\t bead\t x\t y\t z\n");
	for (unsigned int i = (results.size() - 25)>0 ? (results.size() - 25) : 0; i < results.size(); i++)
	{
		auto result = results[i];
		fprintf(stderr, "\t%d\t%d\t%.3f\t%.3f\t%.3f\n", result.job.frame, i % 4, result.pos.x, result.pos.y, result.pos.z);

	}
	//Report final information

	printf("Frame counters:\n\t Started at %d, processed %d, finished on %d\n", counters.startFrame, counters.processedFrames, counters.lastSaveFrame);
	printf("\tCaptured %d, localizations %d, lostFrames %d, file error %d.\n", counters.capturedFrames, counters.localizationsDone, counters.lostFrames, counters.fileError);
	//report time, end program 
	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto microSeconds = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
	fprintf(stderr, "Note: Elapsed time %ld microseconds. \n", (int)microSeconds);
	return 0;
}
