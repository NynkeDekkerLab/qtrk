#pragma once
#include <functional>
#include "LsqQuadraticFit.h"
#include "threads.h"
#include "random_distr.h"

// Generate fisher matrix based using GenerateImageFromLUT
class SampleFisherMatrix 
{
	float maxValue;

	// imgpx = ( imgpx - imgmx ) / (2 *delta)
	template<typename T>
	void ImgDeriv(TImageData<T>& imgpx, TImageData<T>& imgmx, T delta)
	{
		T inv2d = 1.0f/(2*delta);
		for (int y=0;y<imgpx.w*imgpx.h;y++)
			imgpx[y] = (imgpx[y] - imgmx[y]) * inv2d;
	}

	template<typename T>
	double FisherElem(TImageData<T>& mu, TImageData<T>& muderiv1, TImageData<T>& muderiv2)
	{
		double sum = 0;
		for (int y=0;y<mu.h;y++) {
			for (int x=0;x<mu.w;x++)
				sum += (double)muderiv1.at(x,y) * (double) muderiv2.at(x,y) / (1e-5f + (double)mu.at(x,y));
		}
		return sum;
	}

public:
	SampleFisherMatrix(float maxValue)
	{
		this->maxValue = maxValue;
	}

	Matrix3X3 Compute(vector3f pos, vector3f delta, ImageData& lut, int w,int h, float zlutMinRadius,float zlutMaxRadius)
	{
		return Compute(pos, delta, w,h,[&](ImageData& out, vector3f pos) {
			ImageData tmp = ImageData::alloc(w,h);
			GenerateImageFromLUT(&tmp, &lut, zlutMinRadius, zlutMaxRadius, pos);
			out.set(tmp);
			tmp.free();
		});
	}

	Matrix3X3 Compute(vector3f pos, vector3f delta, int w, int h, std::function<void(ImageData& out, vector3f pos)> imageGenerator)
	{
		static int t=0;

		// compute derivatives
		auto mu = ImageData::alloc(w,h);
		imageGenerator(mu, pos);

		float maxImg = mu.mean();
	
		auto imgpx = ImageData::alloc(w,h);
		auto imgpy = ImageData::alloc(w,h);
		auto imgpz = ImageData::alloc(w,h);
		imageGenerator(imgpx, vector3f(pos.x+delta.x, pos.y,pos.z));
		imageGenerator(imgpy, vector3f(pos.x, pos.y+delta.y,pos.z));
		imageGenerator(imgpz, vector3f(pos.x, pos.y,pos.z+delta.z));

//		WriteImageAsCSV("imgpx.csv", imgpx.data, imgpx.w, imgpx.h);
//		WriteImageAsCSV("imgpy.csv", imgpy.data, imgpx.w, imgpx.h);
//		WriteImageAsCSV("imgpz.csv", imgpz.data, imgpx.w, imgpx.h);

		auto imgmx = ImageData::alloc(w,h);
		auto imgmy = ImageData::alloc(w,h);
		auto imgmz = ImageData::alloc(w,h);
		imageGenerator(imgmx, vector3f(pos.x-delta.x, pos.y,pos.z));
		imageGenerator(imgmy, vector3f(pos.x, pos.y-delta.y,pos.z));
		imageGenerator(imgmz, vector3f(pos.x, pos.y,pos.z-delta.z));
		//WriteImageAsCSV("imgmx.csv", imgmx.data, imgpx.w, imgpx.h);
		//WriteImageAsCSV("imgmy.csv", imgmy.data, imgpx.w, imgpx.h);
		//WriteImageAsCSV("imgmz.csv", imgmz.data, imgpx.w, imgpx.h);

		ImgDeriv(imgpx, imgmx, delta.x);
		ImgDeriv(imgpy, imgmy, delta.y);
		ImgDeriv(imgpz, imgmz, delta.z);

//		WriteJPEGFile("mu_dx.jpg", imgpx);
//		WriteJPEGFile("mu_dy.jpg", imgpy);
//		WriteJPEGFile("mu_dz.jpg", imgpz);
		t++;

		double Ixx = FisherElem(mu, imgpx, imgpx);
		double Iyy = FisherElem(mu, imgpy, imgpy);
		double Izz = FisherElem(mu, imgpz, imgpz);
		double Ixy = FisherElem(mu, imgpx, imgpy);
		double Ixz = FisherElem(mu, imgpx, imgpz);
		double Iyz = FisherElem(mu, imgpy, imgpz);

		Matrix3X3 fisher;
		fisher(0,0) = Ixx; fisher(0,1) = Ixy; fisher(0,2) = Ixz;
		fisher(1,0) = Ixy; fisher(1,1) = Iyy; fisher(1,2) = Iyz;
		fisher(2,0) = Ixz; fisher(2,1) = Iyz; fisher(2,2) = Izz;
		fisher *= maxValue / maxImg;

		imgpx.free(); imgpy.free(); imgpz.free(); 
		imgmx.free(); imgmy.free(); imgmz.free(); 
		mu.free();
		return fisher;
	}

	Matrix3X3 ComputeAverageFisher(vector3f pos, int Nsamples, vector3f sampleRange, vector3f delta, int w,int h, std::function<void(ImageData& out, vector3f pos)> imggen)
	{
		Matrix3X3* results = new Matrix3X3[Nsamples];

		auto f = [&] (int index) {
			vector3f smpPos = pos + sampleRange*vector3f(rand_uniform<float>()-0.5f,rand_uniform<float>()-0.5f, rand_uniform<float>());
			//vector3f smpPos = pos + ( vector3f(rand_normal<float>(), rand_normal<float>(), rand_normal<float>()) * initialStDev);
			results[index] = Compute(smpPos, delta, w,h, imggen);
	//		dbgprintf("%d done.\n", index);
		};

		if (0) { 
			ThreadPool<int, std::function<void (int index)> > pool(f);

			for (int i=0;i<Nsamples;i++)
				pool.AddWork(i);
			pool.WaitUntilDone();
		} else {
			for (int i=0;i<Nsamples;i++)
				f(i);
		}
		Matrix3X3 accum;
		for (int i=0;i<Nsamples;i++) 
			accum += results[i];
		delete[] results;

		Matrix3X3 matrix = accum;
		matrix *= 1.0f/Nsamples;
		return matrix;
	}
};
