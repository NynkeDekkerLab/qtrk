#pragma once

#include "../cputrack/std_incl.h"
#include "../cputrack/QueuedCPUTracker.h"


template<typename T> T sq(T x) { return x*x; }
template<typename T> T distance(T x, T y) { return sqrt(x*x+y*y); }

float distance(vector2f a,vector2f b);

enum OutputModes{
	Console = 1,
	Files = 2,
	Images = 4
};

class outputter{
public:
	outputter(int mode = 1);
	~outputter();
	void outputString(std::string out, bool ConsoleOnly = false);
	void outputImage(ImageData img, std::string filename = "UsedImage");
	
	template<typename T>
	void outputArray(T* arr, int size){
		std::ostringstream out;
		for(int ii=0;ii<size;ii++){
			out << "[" << ii << "] : " << arr[ii] << "\n";
		}
		outputString(out.str());
	}

	void newFile(std::string filename, const char* mode = "w+");
private:
	void init(int mode);
	struct outputModes{
		bool File;
		bool Console;
		bool Images;
	};
	outputModes modes;
	FILE* outputFile;
	std::string folder;
};

ImageData CropImage(ImageData img, int x, int y, int w, int h);
ImageData ResizeImage(ImageData img, int factor);
ImageData AddImages(ImageData img1, ImageData img2, vector2f displacement);
ImageData GaussMask(ImageData img, float sigma);
ImageData SkewImage(ImageData img, int fact);

void GetOuterEdges(float* out,int size, ImageData img);
float BackgroundMedian(ImageData img);
float BackgroundStdDev(ImageData img);
float BackgroundRMS(ImageData img);