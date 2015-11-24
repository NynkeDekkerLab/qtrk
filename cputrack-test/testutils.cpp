#include <time.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "testutils.h"

float distance(vector2f a,vector2f b) { return distance(a.x-b.x,a.y-b.y); }

bool DirExists(const std::string& dirName_in)
{
  DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
  if (ftyp == INVALID_FILE_ATTRIBUTES)
    return false;  //something is wrong with your path!

  if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
    return true;   // this is a directory!

  return false;    // this is not a directory!
}

int NumFilesInDir(const std::string& dirName_in)
{
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	std::string dirName = dirName_in + "*";
	hFind = FindFirstFile(dirName.c_str(),&FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		printf ("FindFirstFile failed (%d)\n", GetLastError());
		return 0;
	} 
	int out = -1;
	while(FindNextFile(hFind,&FindFileData)){
		out++;
	}
	return out;
}

int NumJpgInDir(const std::string& dirName_in)
{
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	std::string dirName = dirName_in + "*";
	hFind = FindFirstFile(dirName.c_str(),&FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		printf ("FindFirstFile failed (%d)\n", GetLastError());
		return 0;
	} 
	int out = -1;
	while(FindNextFile(hFind,&FindFileData)){
		std::string name = FindFileData.cFileName;
		if(name.find(".jpg")!=name.npos) {
			out++;
		}
	}
	return out;
}

outputter::outputter(int mode)	{ init(mode); }

outputter::~outputter(){
	if(modes.File && outputFile != NULL)
		fclose(outputFile);	
}

void outputter::outputString(std::string out, bool ConsoleOnly){
	if(modes.Console || ConsoleOnly){
		std::cout << out << std::endl;
	}

	if(modes.File && !ConsoleOnly){
		if(!outputFile)
			newFile("OutputFile");
		fprintf_s(outputFile,"%s\n",out.c_str());
	}
}

void outputter::outputImage(ImageData img, std::string filename){
	if(modes.Images){
		std::string file = folder + filename + ".jpg";
		FloatToJPEGFile(file.c_str(),img.data,img.w,img.h);
	}
}

void outputter::newFile(std::string filename, const char* mode){
	if(modes.File){
		if(outputFile)
			fclose(outputFile);
		std::string outfile = folder + filename + ".txt";
		outputFile = fopen(outfile.c_str(),mode);
	}
}

void outputter::init(int mode){
	modes.Console	= (mode & Console) != 0;
	modes.File		= (mode & Files) != 0;
	modes.Images	= (mode & Images) != 0;
	
	outputFile	= NULL;

	if(!modes.Console && !modes.File){
		modes.Console = true;
		printf_s("No output mode selected, using console by default.\n");
	}

	if(modes.File || modes.Images){
		char date[14];
		GetFormattedTimeString(date);
		folder = "D:\\TestImages\\TestOutput\\" + std::string(date) + "\\";
		CreateDirectory((LPCTSTR)folder.c_str(),NULL);
	}
}

ImageData CropImage(ImageData img, int x, int y, int w, int h)
{
	ImageData croppedImg = ImageData::alloc(w,h);

	if( x < 0 || y < 0 || x + w > img.w || y + h > img.h){
		return img;
	}

	for(int x_i = x; x_i < x+w; x_i++){
		for(int y_i = y; y_i < y+h; y_i++){
			croppedImg.at(x_i-x,y_i-y) = img.at(x_i,y_i);
		}
	}
	return croppedImg;
}

ImageData ResizeImage(ImageData img, int factor)
{
	ImageData resizedImg = ImageData::alloc(img.w*factor,img.h*factor);
	
	for(int x_i=0;x_i<img.w;x_i++){
		for(int y_i=0;y_i<img.h;y_i++){
			for(int x_fact=0;x_fact<factor;x_fact++){
				for(int y_fact=0;y_fact<factor;y_fact++){
					resizedImg.at(x_i*factor+x_fact,y_i*factor+y_fact) = img.at(x_i,y_i);
				}
			}
		}
	}
	return resizedImg;
}

ImageData AddImages(ImageData img1, ImageData img2, vector2f displacement)
{
	ImageData addedImg = ImageData::alloc(img1.w,img1.h);

	for(int x_i=0;x_i<img1.w;x_i++){
		for(int y_i=0;y_i<img2.h;y_i++){
			if(x_i-displacement.x > 0 && x_i-displacement.x < img1.w && y_i-displacement.y > 0 && y_i-displacement.y < img1.h) {
				addedImg.at(x_i,y_i) = ( img1.at(x_i,y_i) + img2.at(x_i-displacement.x,y_i-displacement.y) )/2;
			} else {
				addedImg.at(x_i,y_i) = img1.at(x_i,y_i);
			}
		}
	}
	return addedImg;
}

ImageData GaussMask(ImageData img, float sigma) 
{
	ImageData gaussImg = ImageData::alloc(img.w,img.h);
	vector2f centre = vector2f(img.w/2,img.h/2);
	for(int x_i=0;x_i<img.w;x_i++){
		for(int y_i=0;y_i<img.h;y_i++){
			float gaussfact = expf(- (x_i-centre.x)*(x_i-centre.x) / (2*sigma*sigma) - (y_i-centre.y)*(y_i-centre.y) / (2*sigma*sigma));
			gaussImg.at(x_i,y_i) = img.at(x_i,y_i)*gaussfact;
		}
	}

	return gaussImg;
}

ImageData SkewImage(ImageData img, float fact) 
{
	ImageData skewImg = ImageData::alloc(img.w,img.h);
	vector2f centre = vector2f(img.w/2,img.h/2);
	float median  = BackgroundMedian(img);
	float stddev  = BackgroundStdDev(img);
	float maxskew = median*stddev;
	for(int x_i=0;x_i<img.w;x_i++){
		for(int y_i=0;y_i<img.h;y_i++){
			int diagonalOffset = (y_i - x_i*img.h/img.w);
			float skew = fact*((float)diagonalOffset/img.h)*maxskew;
			skewImg.at(x_i,y_i) = img.at(x_i,y_i)+skew;
		}
	}

	return skewImg;
}

void GetOuterEdges(float* out,int size, ImageData img){
	int x,y=0;
	for(int ii = 0; ii < size; ii++){
		if(ii < img.w){ // Top
			x = ii;
			y = 0;
		}
		else if(ii < img.w + img.h - 1){ // Right
			x = img.w-1;
			y = ii-(img.w-1);
		}
		else if(ii < img.w * 2 + img.h - 2){ // Bottom
			y = img.h-1;
			x = ii-(img.h+img.w-1);
		}
		else{ // Left
			x = 0;
			y = ii-(img.h+img.w*2-3);
		}
		out[ii] = img.at(x,y);
	}
}

float BackgroundMedian(ImageData img){
	int size = img.w * 2 + img.h * 2 - 4;
	float* outeredge = new float[size];
	GetOuterEdges(outeredge,size,img);
	std::sort(outeredge,outeredge+size);
	float median;
	if(size % 2 == 0)
		median = (outeredge[(int)(size/2-1)] + outeredge[(int)(size/2+1)])/2;
	else
		median = outeredge[size/2];
	delete[] outeredge;
	return median;
}

float BackgroundStdDev(ImageData img){
	int size = img.w * 2 + img.h * 2 - 4;
	float* outeredge = new float[size];
	GetOuterEdges(outeredge,size,img);
	float stddev = ComputeStdDev(outeredge,size);
	delete[] outeredge;
	return stddev;
}

float BackgroundRMS(ImageData img){
	int size = img.w * 2 + img.h * 2 - 4;
	float* outeredge = new float[size];
	GetOuterEdges(outeredge,size,img);
	float sqsum = 0.0f;
	for(int ii = 0; ii < size; ii++){
		sqsum += outeredge[ii]*outeredge[ii];
	}
	delete[] outeredge;
	return sqrt(1/(float)size*sqsum);
}
