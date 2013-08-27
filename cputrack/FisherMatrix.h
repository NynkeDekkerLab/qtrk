#pragma once
#include <functional>
#include "LsqQuadraticFit.h"
#include "threads.h"
#include "random_distr.h"

// Builds a fisher matrix for a localization ZLUT
class LUTFisherMatrix
{
	static inline float sq(float v) { return v*v; }

public:
	LUTFisherMatrix(float* lut, int radialsteps, int planes, int w,int h, float zlutMin, float zlutMax, float lutIntensityScale )  {
		this->lut = lut;
		this->radialsteps = radialsteps;
		this->planes = planes;
		profile = new float [radialsteps];
		dzProfile = new float [radialsteps];
		drProfile = new float [radialsteps];

		this->zlutMin = zlutMin;
		this->zlutMax = zlutMax;
		this->lutIntensityScale = lutIntensityScale;
		roiW = w;
		roiH = h;
	}
	~LUTFisherMatrix() {
		delete[] profile;
		delete[] dzProfile;
		delete[] drProfile;
	}
	

	void Compute(vector3f pos, int Nsamples, vector3f initialStDev)
	{
		Matrix3X3* results = new Matrix3X3[Nsamples];

		ThreadPool<int, std::function<void (int index)> > pool(
		[&] (int index) {
			vector3f smpPos = pos + ( vector3f(rand_normal<float>(), rand_normal<float>(), rand_normal<float>()) * initialStDev);
			results[index] = ComputeFisherSample(smpPos);
	//		dbgprintf("%d done.\n", index);
		});

		for (int i=0;i<Nsamples;i++)
			pool.AddWork(i);
		pool.WaitUntilDone();

		Matrix3X3 accum;
		for (int i=0;i<Nsamples;i++) 
			accum += results[i];
		delete[] results;

		// Compute inverse
		matrix = accum;
		matrix *= 1.0f/Nsamples;
		inverse = matrix.Inverse();
	}

	Matrix3X3 ComputeFisherSample(vector3f pos)
	{
		InterpolateProfile(pos.z);

		// Subpixel grid size
		const int SW=4;
		const int SH=4;
		const float xstep = 1.0f/SW;
		const float ystep = 1.0f/SH;

		int smp=0;
		double Izz=0, Ixx=0 , Iyy=0, Ixy=0, Ixz=0, Iyz=0;
		numPixels = 0;

		for (int py=0;py<roiH;py++) {
			for (int px=0;px<roiW;px++) {
				float xOffset = px + 0.5f/SW;
				float yOffset = py + 0.5f/SH;

				// 4x4 grid over each pixel, to approximate the integral
				for (int sy=0;sy<SW;sy++) {
					for (int sx=0;sx<SW;sx++) {
						float x = xOffset + sx*xstep;
						float y = yOffset + sy*ystep;

						float difx = pos.x-x, dify = pos.y-y;
						
						float r = sqrtf(difx*difx + dify*dify);
						// du/dx = d/dr * u(r) * 2x/r
						
						float dudr, dudz;
						float u = SampleProfile(r, dudr, dudz);
						float dudx = dudr * difx/(r+1e-9f);
						float dudy = dudr * dify/(r+1e-9f);
						
						// Ixx = 1/sigma^4 * ( du/dx )^2
						// Ixx = 1 / u * ( du/dx ) ^ 2  (Poisson case)

						float invU = 1/u;
						Ixx += invU * dudx*dudx;
						Iyy += invU * dudy*dudy;
						Ixy += invU * dudx*dudy;
						Ixz += invU * dudx*dudz;
						Iyz += invU * dudy*dudz;
						Izz += invU * dudz*dudz;
					}
				}
			}
		}

		Matrix3X3 m = Matrix3X3( 
			vector3f(Ixx, Ixy, Ixz),
			vector3f(Ixy, Iyy, Iyz),
			vector3f(Ixz, Iyz, Izz));

		m *= 1.0f/(SW*SH);
		return m;
	}

	float Quadratic3PointFit(float *y, float x, float& dydx)
	{
		float a = (y[0] - y[1]  -y[1]+y[2]) * 0.5f;
		float b = (y[2]-y[1]) - a;// = (y[1]-y[0]) + a, 
		float c = y[1];
	  // y  = a * x^2+ b * x + c

		dydx = 2*x*a+b;
		return a*x*x+b*x+c;
	}

	float SampleProfile(float r, float& deriv, float &dz)
	{
		int rounded = (int)(r+0.5f);
		int rs = rounded-1;
		float xval[] = { -1, 0, 1 };

		if (rs < 0) rs = 0;
		if (rs + 3 > radialsteps) rs = radialsteps-3;

		float x = r-rounded; 
/*		LsqSqQuadFit<float> u_fit(3, xval, &profile[rs]);
		LsqSqQuadFit<float> dudz_fit(3, xval, &dzProfile[rs]);

		dz = dudz_fit.compute(x);

		deriv = u_fit.computeDeriv(x);
		return u_fit.compute(x);*/

		dz = Quadratic3PointFit(&dzProfile[rs], x, deriv);
		return Quadratic3PointFit(&profile[rs], x, deriv);
	}

	// Compute profile
	void InterpolateProfile(float z)
	{
		int iz = (int)z;
		iz = std::max(0, std::min(planes-2, iz));
		float *prof0 = &lut[iz*radialsteps];
		float *prof1 = &lut[(iz+1)*radialsteps];

		profileMaxValue = 0;
		for (int r=0;r<radialsteps;r++) {
			profile[r] = prof0[r] + (prof1[r] - prof0[r]) * (z-iz);
			if(profile[r] > profileMaxValue) profileMaxValue=profile[r];
		}

		// Compute derivative
		const int NumZPlanes = 5;
		float zplanes[NumZPlanes];
		float prof[NumZPlanes];
		int minZ = std::max(iz-NumZPlanes/2, 0);
		int maxZ = std::min(iz+(NumZPlanes-NumZPlanes/2), planes);

		for(int r=0;r<radialsteps;r++) {
			for (int p=0;p<maxZ-minZ;p++) {
				zplanes[p]=p;
				prof[p]=LUT(p+minZ,r);
			}

			LsqSqQuadFit<float> qfit(NumZPlanes, zplanes, prof);
//			dzProfile[r] = prof1[r] - prof0[r]; // TODO: interpolate this stuff
			dzProfile[r] = qfit.computeDeriv(z-minZ);

			if (lutIntensityScale > 0.0f) {
				dzProfile[r] *= lutIntensityScale;
				profile[r] *= lutIntensityScale;
			}
		}

		if (lutIntensityScale>0.0f)
			profileMaxValue *= lutIntensityScale;
	}

	float LUT(int pl, int r) {
		return lut[pl*radialsteps+r];
	}

	int radialsteps, planes;
	float* lut;
	float zlutMin,zlutMax, lutIntensityScale;
	int roiW, roiH;

	float* profile;
	float* dzProfile; // derivative of profile wrt Z plane
	float *drProfile;

	Matrix3X3 matrix;
	Matrix3X3 inverse;
	int numPixels;
	float profileMaxValue;

	vector3f MinVariance() { return vector3f(inverse(0,0), inverse(1,1), inverse(2,2)); }
};
