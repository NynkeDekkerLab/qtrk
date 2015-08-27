#include "std_incl.h"
#include <cstdarg>
#include <functional>
#include "utils.h"
#include <Windows.h>
#undef min
#undef max
#include <string>
#include <complex>

#include "random_distr.h"
#include "LsqQuadraticFit.h"
#include "QueuedTracker.h"
#include "threads.h"
#include "CubicBSpline.h"
#include "time.h"
#include <tchar.h>

std::string GetCurrentOutputPath(bool ext)
{
	std::string base = "D:\\TestImages\\TestOutput\\";
	std::string search = base + "*";
	LPCSTR w_folder = _T(search.c_str());
	
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	
	hFind = FindFirstFile(w_folder,&FindFileData);
	std::string dirName;
	while(FindNextFile(hFind,&FindFileData))
		dirName = FindFileData.cFileName;
	if(ext)
		return SPrintf("%s%s\\%s",base.c_str(),dirName.c_str(),"ZLUTDiag\\");
	else
		return SPrintf("%s%s",base.c_str(),dirName.c_str());
}

void GetFormattedTimeString(char* output)
{
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	sprintf(output, "%02d%02d%02d-%02d%02d%02d",timeinfo->tm_year-100,timeinfo->tm_mon+1,timeinfo->tm_mday,timeinfo->tm_hour,timeinfo->tm_min,timeinfo->tm_sec);
}

static std::string logFilename;

void dbgsetlogfile(const char*path)
{
	logFilename = path;
}

std::string GetLocalModuleFilename()
{
#ifdef WIN32
	char path[256];
	HMODULE hm = NULL;

    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR) &GetLocalModuleFilename, &hm);

    GetModuleFileNameA(hm, path, sizeof(path));
	return path;
#else
	#error GetLocalModuleName() not implemented for this platform
#endif
}

PathSeperator::PathSeperator(std::string fullpath)
{
	int filenameEnd = fullpath.size();
	int filenameStart = 0;
	for (int i = fullpath.size()-1; i>=0; i--) {
		if (fullpath[i] == '.' && extension.empty()) {
			extension = fullpath.substr(i+1);
			filenameEnd = i;
		}
		if (fullpath[i] == '/' || fullpath[i] == '\\')  {
			directory = fullpath.substr(0,i+1);
			filenameStart = i+1;
			break;
		}
	}
	filename = fullpath.substr(filenameStart, filenameEnd - filenameStart);
}

void WriteToLog(const char *str)
{
	if (logFilename.empty()) {
		auto ps = PathSeperator(GetLocalModuleFilename());
		logFilename = ps.directory + ps.filename + "-log.txt";
	}

	if (str) {
		FILE* f = fopen(logFilename.c_str(), "a");
		if (f) {
			fputs(str,f);
			fclose(f);
		}
	}
}

std::string GetDirectoryFromPath(std::string fullpath)
{
	for (int i=fullpath.size()-1;i>=0;i--) {
		if (fullpath[i] == '/' || fullpath[i] == '\\') {
			return fullpath.substr(0, i);
		}
	}
	return "";
}

std::string GetLocalModulePath()
{
	std::string dllpath = GetLocalModuleFilename();
	return GetDirectoryFromPath(dllpath);
}

	
std::string file_ext(const char *f){
	int l=strlen(f)-1;
	while (l > 0) {
		if (f[l] == '.')
			return &f[l+1];
		l--;
	}
	return "";
}


std::string SPrintf(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);

	va_end(ap);
	return buf;
}

void dbgout(const std::string& s) {
	OutputDebugString(s.c_str());
	printf(s.c_str());
	WriteToLog(s.c_str());
}

void dbgprintf(const char *fmt,...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);
	OutputDebugString(buf);
	fputs(buf,stdout);
	WriteToLog(buf);

	va_end(ap);
}

void GenerateTestImage(ImageData& img, float xp, float yp, float size, float SNratio)
{
	float S = 1.0f/sqrt(size);
	for (int y=0;y<img.h;y++) {
		for (int x=0;x<img.w;x++) {
			float X = x - xp;
			float Y = y - yp;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = sinf(r/(5*S)) * expf(-r*r*S*0.001f);
			img.at(x,y)=v;
		}
	}

	if (SNratio>0) {
		ApplyGaussianNoise(img, 1.0f/SNratio);
	}
	img.normalize();
}



void ComputeCRP(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius,
	vector2f center, ImageData* img, float paddingValue, float*crpmap)
{
	vector2f* radialDirs = (vector2f*)ALLOCA(sizeof(vector2f)*angularSteps);
	for (int j=0;j<angularSteps;j++) {
		float ang = 2*3.141593f*j/(float)angularSteps;
		radialDirs[j] = vector2f(cosf(ang), sinf(ang) );
	}

	for (int i=0;i<radialSteps;i++)
		dst[i]=0.0f;

	float* map = crpmap ? crpmap : (float*)ALLOCA(sizeof(float)*radialSteps*angularSteps);
	float* com = (float*)ALLOCA(sizeof(float)*angularSteps);

	float rstep = (maxradius-minradius) / radialSteps;
	float comsum = 0.0f;
	for (int a=0;a<angularSteps;a++) {
		float r = minradius;
		float sum = 0.0f, moment=0.0f;
		for (int i=0;i<radialSteps; i++) {
			float x = center.x + radialDirs[a].x * r;
			float y = center.y + radialDirs[a].y * r;
			float v = img->interpolate(x,y);
			r += rstep;
			map[a*radialSteps+i] = v;
			sum += v;
			moment += i*v;
		}
		com[a] = moment/sum;
		comsum += com[a];
	}
	float avgcom = comsum/angularSteps;
	float totalrmssum2 = 0.0f;
	for (int i=0;i<radialSteps; i++) {
		double sum = 0.0f;
		for (int a=0;a<angularSteps;a++) {
			float shift = com[a]-avgcom;
			sum += map[a*radialSteps+i];
		}
		dst[i] = sum/angularSteps-paddingValue;
		totalrmssum2 += dst[i]*dst[i];
	}
	double invTotalrms = 1.0f/sqrt(totalrmssum2/radialSteps);
	for (int i=0;i<radialSteps;i++) {
		dst[i] *= invTotalrms;
	}
}

// One-dimensional background corrected center-of-mass
float ComputeBgCorrectedCOM1D(float *data, int len, float cf)
{
	float sum=0, sum2=0;
	float moment=0;

	for (int x=0;x<len;x++) {
		float v = data[x];
		sum += v;
		sum2 += v*v;
	}

	float invN = 1.0f/len;
	float mean = sum * invN;
	float stdev = sqrtf(sum2 * invN - mean * mean);
	sum = 0.0f;

	for(int x=0;x<len;x++)
	{
		float v = data[x];
		v = std::max(0.0f, fabs(v-mean)-cf*stdev);
		sum += v;
		moment += x*v;
	}
	return moment / (float)sum;
}

void NormalizeRadialProfile(scalar_t * prof, int rsteps)
{
	double sum=0.0f;
	for (int i=0;i<rsteps;i++)
		sum += prof[i];

	float mean =sum/rsteps;
	double rmssum2 = 0.0;

	for (int i=0;i<rsteps;i++) {
		prof[i] -= mean;
		rmssum2 += prof[i]*prof[i];
	}
	double invTotalrms = 1.0f/sqrt(rmssum2/rsteps);
	for (int i=0;i<rsteps;i++)
		prof[i] *= invTotalrms;
		
/*
	scalar_t minVal = prof[0];
	for (int i=0;i<rsteps;i++)
		if(prof[i]<minVal) minVal=prof[i]; 

	float rms=0;
	for (int i=0;i<rsteps;i++) {
		prof[i]-=minVal;
		rms += prof[i]*prof[i];
	}
	rms=1.0f/sqrt(rms);
	for (int i=0;i<rsteps;i++) {
		prof[i]*=rms;
	}*/
}


void NormalizeZLUT(float* zlut ,int numBeads, int planes, int radialsteps)
{
	for(int i=0;i<numBeads;i++)
		for (int j=0;j<planes;j++)
			NormalizeRadialProfile(&zlut[radialsteps*planes*i + radialsteps*j], radialsteps);
}

void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius,
	vector2f center, ImageData* img, float mean, bool normalize)
{
	vector2f* radialDirs = (vector2f*)ALLOCA(sizeof(vector2f)*angularSteps);
	for (int j=0;j<angularSteps;j++) {
		float ang = 2*3.141593f*j/(float)angularSteps;
		radialDirs[j] = vector2f(cosf(ang), sinf(ang));
	}

	for (int i=0;i<radialSteps;i++)
		dst[i]=0.0f;

//	center.x += 0.5f;
//	center.y += 0.5f;

	bool trace=false;
	float rstep = (maxradius-minradius) / radialSteps;
	int totalsmp = 0;
	for (int i=0;i<radialSteps; i++) {
		double sum = 0.0f;

		int nsamples = 0;
		float r = minradius+rstep*i;
		for (int a=0;a<angularSteps;a++) {
			float x = center.x + radialDirs[a].x * r;
			float y = center.y + radialDirs[a].y * r;
			bool outside;
			float v = img->interpolate(x,y, &outside);
			if (!outside) {
				sum += v;
				nsamples++;
			}
		}

		if (trace) {
			dbgprintf("%f,[%d]; ", sum, nsamples);
		}

		dst[i] = nsamples > MIN_RADPROFILE_SMP_COUNT ? sum/nsamples : mean;
	}
	if(trace)
		dbgprintf("\n");

	if (normalize) 
		NormalizeRadialProfile(dst, radialSteps);
}


inline float sq(float x) { return x*x; }

void GenerateImageFromLUT(ImageData* image, ImageData* zlut, float minradius, float maxradius, vector3f pos, bool splineInterp, int oversampleSubdiv)
{
//	lut.w = radialcov * ( (image->w/2 * roicov ) - minradius );
//	lut.w = radialcov * ( maxradius - minradius );

	float radialcov = zlut->w / (maxradius-minradius);
	float* zinterp = (float*)ALLOCA(zlut->w * sizeof(float));

	if (splineInterp) {
		int iz = std::max(1, std::min(zlut->h-3, (int)pos.z));
		float weights[4];
		float fz = pos.z-iz;
		ComputeBSplineWeights(weights, fz);
		// Interpolate ZLUT using B-spline weights
		for (int r=0;r<zlut->w;r++) {
			float zlutv = 0;
			for (int i=0;i<4;i++)
				zlutv += weights[i] * zlut->at(r, i-1+iz);
			zinterp[r] = zlutv;
		}
	}
	else {
		// The two Z planes to interpolate between
		int iz = (int)pos.z;
		if (iz < 0) 
			zinterp = zlut->data;
		else if (iz>=zlut->h-1)
			zinterp = &zlut->data[ (zlut->h-1)*zlut->w ];
		else {
			float* zlut0 = &zlut->data [ (int)pos.z * zlut->w ]; 
			float* zlut1 = &zlut->data [ ((int)pos.z + 1) * zlut->w ];
			zinterp = (float*)ALLOCA(sizeof(float)*zlut->w);
			for (int r=0;r<zlut->w;r++) 
				zinterp[r] = Lerp(zlut0[r], zlut1[r], pos.z-iz);
		}
	}

	int oversampleWidth=oversampleSubdiv,oversampleHeight=oversampleSubdiv;
	float oxstep = 1.0f / oversampleWidth;
	float oystep = 1.0f / oversampleHeight;

	for (int y=0;y<image->h;y++)
		for (int x=0;x<image->w;x++) 
		{
			float s = 0.0f;

			for (int ox=0;ox<oversampleWidth;ox++)
				for (int oy=0;oy<oversampleHeight;oy++) {

					float X = x+(ox+0.5f)*oxstep - pos.x - 0.5f;
					float Y = y+(oy+0.5f)*oystep - pos.y - 0.5f;

					float pixr = sqrtf(X*X+Y*Y);
					float r = (pixr - minradius) * radialcov;

					if (r > zlut->w-2)
						r = zlut->w-2;
					if (r < 0) r = 0;

					int i=(int)r;
					s += Lerp(zinterp[i], zinterp[i+1], r-i);
				}
		
			image->at(x,y) = s/(oversampleWidth*oversampleHeight); 
		}
}


void GenerateGaussianSpotImage(ImageData* img, vector2f pos, float sigma, float I0, float Ibg)
{
    float edenom = 1/sqrt(2*sigma*sigma);
	for (int y=0;y<img->h;y++)
		for(int x=0;x<img->w;x++) {
			float DeltaX = 0.5f * erf( (x-pos.x + .5f) * edenom ) - 0.5f * erf((x-pos.x - .5f) * edenom);
			float DeltaY = 0.5f * erf( (y-pos.y + .5f) * edenom ) - 0.5f * erf((y-pos.y - .5f) * edenom);
			img->at(x,y) = Ibg + I0 * DeltaX * DeltaY;
		}
}

void ApplyPoissonNoise(ImageData& img, float poissonMax, float maxval)
{
	/*auto f = [&] (int y) {
		for (int x=0;x<img.w;x++) {
			img.at(x,y) = rand_poisson<float>(factor*img.at(x,y));
		}
	};

	ThreadPool<int, std::function<void (int index)> > pool(f);

	for (int y=0;y<img.h;y++) {
		pool.AddWork(y);
	}
	pool.WaitUntilDone();*/

	float ratio = maxval / poissonMax;

	for (int x=0;x<img.numPixels();x++) {
		img[x] = (int)(rand_poisson<float>(poissonMax*img[x]) * ratio );
	}
}

void ApplyGaussianNoise(ImageData& img, float sigma)
{
	for (int k=0;k<img.numPixels();k++) {
		float v = img.data[k] + sigma * rand_normal<float>();
		if (v<0.0f) v= 0.0f;
		img.data[k]=v;
	}
}

std::vector< std::vector<float> > ReadCSV(const char *filename, char sep)
{
	std::list< std::vector <float> > data;

	FILE *f=fopen(filename,"r");
	if (f) {
		std::string buf;
		std::vector<float> vals;
		while (!feof(f)) {
			char c=fgetc(f);
			if (c == sep || c=='\n') {
				vals.push_back( atof(buf.c_str()) );
				buf.clear();
			}
			if (c == '\n') {
				data.push_back( vals );
				vals.clear();
			}

			if (c != sep && c!= '\n')
				buf+=c;
		}
		fclose(f);
	}
	
	std::vector<std::vector<float> > r;
	r.reserve(data.size());
	r.insert(r.begin(), data.begin(), data.end());
	return r;
}


std::vector<vector3f> ReadVector3CSV( const char *file, char sep )
{
	auto data=ReadCSV(file ,sep);

	std::vector<vector3f> r(data.size());

	for (int i=0;i<data.size();i++){
		r[i]=vector3f(data[i][0],data[i][1],data[i][2]);
	}
	return r;
}


void WriteTrace(std::string filename, vector3f* results, int nResults)
{
	FILE *f = fopen(filename.c_str(), "w");

	if (!f) {
		throw std::runtime_error(SPrintf("Can't open %s", filename.c_str()));
	}

	for (int i=0;i<nResults;i++)
	{
		fprintf(f, "%.7f\t%.7f\t%.7f\n", results[i].x, results[i].y, results[i].z);
	}

	fclose(f);
}

void WriteArrayAsCSVRow(const char *file, float* d, int len, bool append)
{
	FILE *f = fopen(file, append?"a":"w");
	if(f) {
		for (int i=0;i<len;i++)
			fprintf(f, "%.7f\t", d[i]);

		fprintf(f, "\n");
		fclose(f);
	}
}

void WriteImageAsCSV(const char* file, float* d, int w,int h, const char* labels[])
{
	FILE* f = fopen(file, "w");

	if (f) {

		if (labels) {
			for (int i=0;i<w;i++) {
				fprintf(f, "%s;\t", labels[i]);
			}
			fputs("\n", f);
		}

		for (int y=0;y<h;y++) {
			for (int x=0;x<w;x++)
			{
				fprintf(f, "%.10f", d[y*w+x]);
				if(x<w-1) fputs("\t", f); 
			}
			fprintf(f, "\n");
		}

		fclose(f);
	}
	else
		dbgprintf("WriteImageAsCSV: Unable to open file %s\n", file);
}


void WriteComplexImageAsCSV(const char* file, std::complex<float>* d, int w,int h, const char* labels[])
{
	FILE* f = fopen(file, "w");

	if (!f) {
		dbgprintf("WriteComplexImageAsCSV: Unable to open file %s\n", file);
		return;
	}

	if (labels) {
		for (int i=0;i<w;i++) {
			fprintf(f, "%s;\t", labels[i]);
		}
		fputs("\n", f);
	}

	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++)
		{
			float i=d[y*w+x].imag();
			fprintf(f, "%f%+fi", d[y*w+x].real(), i);
			if(x<w-1) fputs("\t", f); 
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

std::vector<uchar> ReadToByteBuffer(const char *filename)
{
	FILE *f = fopen(filename, "rb");

	if (!f)
		throw std::runtime_error(SPrintf("%s was not found", filename));

	fseek(f, 0, SEEK_END);
	int len = ftell(f);
	fseek(f, 0, SEEK_SET);

	std::vector<uchar> buf(len);
	fread(&buf[0], 1,len, f);

	fclose(f);
	return buf;
}


ImageData ReadJPEGFile(const char*fn)
{
	int w, h;
	uchar* imgdata;
	std::vector<uchar> jpgdata = ReadToByteBuffer(fn);
	ReadJPEGFile(&jpgdata[0], jpgdata.size(), &imgdata, &w,&h);

	float* fbuf = new float[w*h];
	for (int x=0;x<w*h;x++)
		fbuf[x] = imgdata[x]/255.0f;
	delete[] imgdata;

	return ImageData(fbuf,w,h);
}


void CopyImageToFloat(uchar* data, int width, int height, int pitch, QTRK_PixelDataType pdt, float* dst)
{
	if (pdt == QTrkU8) {
		for (int y=0;y<height;y++) {
			for (int x=0;x<width;x++)
				dst[x] = data[x];
			data += pitch;
			dst += width;
		}
	} else if(pdt == QTrkU16) {
		for (int y=0;y<height;y++) {
			ushort* u = (ushort*)data;
			for (int x=0;x<width;x++)
				dst[x] = u[x];
			data += pitch;
			dst += width;
		}
 	} else {
		for (int y=0;y<height;y++) {
			float* fsrc = (float*)data;
			for( int x=0;x<width;x++)
				dst[x] = fsrc[x];
			data += pitch;
			dst += width;
		}
	}
}



double GetPreciseTime()
{
	uint64_t freq, time;

	QueryPerformanceCounter((LARGE_INTEGER*)&time);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	return (double)time / (double)freq;
}




int NearestPowerOf2(int v)
{
	int r=1;
	while (r < v) 
		r *= 2;
	if ( fabsf(r-v) < fabsf(r/2-v) )
		return r;
	return r/2;
}

int NearestPowerOf3(int v)
{
	int r=1;
	while (r < v) 
		r *= 3;
	if ( fabsf(r-v) < fabsf(r/3-v) )
		return r;
	return r/3;
}


std::vector<float> ComputeRadialBinWindow(int rsteps)
{
	std::vector<float> wnd(rsteps);
	for (int i=0;i<rsteps;i++) {
		float x = i/(float)rsteps;
		float t2 = 0.05f; 
		float t1 = 0.01f;
		float fall = 1.0f-expf(-sq(1-x)/t2);
		float rise = 1.0f-expf(-sq(x)/t1);
		wnd[i] = sqrtf(fall*rise*x*2);
	}
	return wnd;
}


ImageData ReadLUTFile(const char *lutfile)
{
	PathSeperator sep(lutfile);
	if(sep.extension == "jpg") {
		return ReadJPEGFile(lutfile);
	}
	else {
		std::string fn = lutfile;
		fn = std::string(fn.begin(), fn.begin()+fn.find('#'));
		std::string num( ++( sep.extension.begin() + sep.extension.find('#') ), sep.extension.end());
		int lutIndex = atoi(num.c_str());

		int nbeads, nplanes, nsteps;
		FILE *f = fopen(fn.c_str(), "rb");

		if (!f)
			throw std::runtime_error("Can't open " + fn);

		fread(&nbeads, 4, 1, f);
		fread(&nplanes, 4, 1, f);
		fread(&nsteps, 4, 1, f);


		fseek(f, 12 + 4* (nsteps*nplanes * lutIndex), SEEK_SET);
		ImageData lut = ImageData::alloc(nsteps,nplanes);
		fread(lut.data, 4, nsteps*nplanes,f);
		fclose(f);
		lut.normalize();
		return lut;
	}
}

