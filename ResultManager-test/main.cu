#include "std_incl.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <stdio.h>
#include <iostream>
#include <string>
#include <ctime> 
#include "omp.h"
#include <stdio.h>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <boost/tokenizer.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp> 
#include <boost/filesystem/operations.hpp> 
#include <boost/chrono.hpp> 
#include <memory>
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
	//At the end, we report the time taken.
	auto timeStart = boost::chrono::high_resolution_clock::now();
	//Just a reminder.
	fprintf(stderr, "Note: Initialising ndlab/test/ResultManager (%d arguments).\n", argc);
	//Otherwise, it will throw errors.
	if (argc != 3)
	{
		fprintf(stderr, "You have to give in N, the number of images (multiple of 4) and \n\t gpu or cpu.\n");
		return 0;
	}
	//Number of images, which are inputted as if there are 4 beads (which really doesn't matter).
	const int N = (int)atoi(argv[1]); //4 beads, 24 images; 6 frames per image?
	//Always report back.
	fprintf(stderr, "Testing ResultManager with %d images. %s \n", N, argv[1]);
	//Where to safe data/frameinfo. 
	const char* file = "./ResultManagerData.txt";
	const char * frameinfo = "./ResultManagerFrameInfo.txt";
	// Better to use modern smart pointers. Config settings are largely taken from older tests.
	std::shared_ptr<ResultManagerConfig> cfg = std::make_shared<ResultManagerConfig>();
	cfg->numBeads = 4;
	cfg->numFrameInfoColumns = 0;
	cfg->scaling = vector3f(1.0f, 1.0f, 1.0f);
	cfg->offset = vector3f(0.0f, 0.0f, 0.0f);
	cfg->writeInterval = 2500;
	cfg->maxFramesInMemory = 0;// 100000;
	cfg->binaryOutput = false;

	std::vector< std::string > colNames;
	std::string testName("Hey now");
	colNames.push_back(testName);

	fprintf(stderr, "Allocating Result Manager now. \n");
	// The manager is an instance of the Result Manager class. Surprise.
	std::shared_ptr<ResultManager> manager = std::make_shared<ResultManager>(file, frameinfo, cfg.get(), colNames);
	//The QueuedCPUTracker instance is required to retrieve the results. It needs settings.
	QTrkComputedConfig settings;
	settings.qi_iterations = 5;
	settings.zlut_minradius = 1;
	settings.qi_minradius = 1;
	settings.width = settings.height = 100;
	settings.Update();
	//Experimental file to use for test.
	std::string fileName = "./exp.jpg";
	//Make sure the file exists; otherwise, it will throw a really weird error.
	bool fileExists = boost::filesystem::exists(fileName);

	if (!fileExists)
	{
		fprintf(stderr, "File %s not found; is it in the directory of the executable?\n\n", fileName.c_str());
		return 0;
	}

	//Let's load some image data.
	auto data = ReadJPEGFile(fileName.c_str());

	std::shared_ptr<QueuedTracker> qtrk;
	//Make sure we are using the right tracker (as requested)
	if (argc == 3)
	{
		std::string argTracker = std::string(argv[2]);
		if (argTracker == "gpu")
		{
			fprintf(stderr, "Using CUDA tracker (GPU).\n");
			std::shared_ptr<QueuedCUDATracker> cudaTracker = std::make_shared<QueuedCUDATracker>(settings);
			cudaTracker->EnableTextureCache(true);

			qtrk = cudaTracker;
		}
		else if(argTracker == "cpu")
		{
			fprintf(stderr, "Using CPU tracker (CPU).\n");
			qtrk = std::make_shared<QueuedCPUTracker>(settings);
		}
		else
		{
			fprintf(stderr, "No tracker specified. Choose either cpu or gpu.\n");
			return 0;
		}
	}
	else
	{
		fprintf(stderr, "Faulty arguments. Your mother was a hamster, %d th of her name.", argc);
		return 0;
	}
	//localization Mode QI tracker
	auto modeQI = (LocMode_t)(LT_QI | LT_NormalizeProfile | LT_LocalizeZ);

	qtrk->SetLocalizationMode(modeQI);

	//Make the calculations
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
		job.zlutIndex = i % 4; //actually, frame number
		jobs.push_back(job);
		qtrk->ScheduleImageData(&data, &job);
	}
	manager->SetTracker(qtrk.get());
	//Process images (using Flush because Start is CPU only)
	qtrk->Flush();
	//Report progress
	int i = 0;
	while (manager->GetFrameCounters().localizationsDone < N)
	{
		if (i > 100000)
		{
			auto counters = manager->GetFrameCounters();
			fprintf(stderr, "Update: %d Localisations performed.\n", counters.localizationsDone);
			i = 0;
		}
		i++;
	}
	 
	//Assign frame
	float somefloat = 0.0;
	for (int i = 0; i < N / 4; i++)
	{
		manager->StoreFrameInfo(i, i, &somefloat);
	}
	//Pointer that will be filled with results
	std::vector<LocalizationResult> results;

	vector3f startPosition(0.0f, 0.0f, 0.0f);
	vector2f initialGuess(45.0f, 50.0f);
	//Allocate the 'array'  that will hold the results.
	for (int i = 0; i < N; i++)
	{
		LocalizationResult currentResult;
		currentResult.job = jobs.at(i);
		currentResult.pos = startPosition;
		currentResult.firstGuess = initialGuess;
		currentResult.error = 0;
		currentResult.imageMean = 0.0f;
		results.push_back(currentResult);
	}

	//Fill results array
	manager->Flush();
	//Wait untill all localizations have been performed.

	i = 0;
	while (manager->GetFrameCounters().lastSaveFrame  < N / 4)
	{
		if (i > 100000)
		{
			auto counters = manager->GetFrameCounters();

			fprintf(stderr, "Update[%.3f]: %d frames saved.\n", i, counters.lastSaveFrame);
		}
	}
	//Get the results
	auto counters = manager->GetFrameCounters();
	auto getResults = manager->GetResults(results.data(), 0, N / cfg->numBeads);
	//Report results
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
	auto timeEnd = boost::chrono::high_resolution_clock::now();
	auto microSeconds = boost::chrono::duration_cast<boost::chrono::microseconds>(timeEnd - timeStart).count();
	fprintf(stderr, "Note: Elapsed time %ld microseconds. \n", microSeconds);
	return 0;
}

