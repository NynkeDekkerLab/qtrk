#pragma once

#include "LsqQuadraticFit.h"


// Builds a fisher matrix for a localization ZLUT
class LUTFisherMatrix
{
	static inline float sq(float v) { return v*v; }
public:

	LUTFisherMatrix(float* lut, int radialsteps, int planes)  {
		this->lut = lut;
		this->radialsteps = radialsteps;
		this->planes = planes;
		profile = new float [radialsteps];
		dzProfile = new float [radialsteps];
		drProfile = new float [radialsteps];
	}
	~LUTFisherMatrix() {
		delete[] profile;
		delete[] dzProfile;
		delete[] drProfile;
	}

	void Compute(int w, int h, vector3f pos, float zlutMinRadius, float zlutMaxRadius, float lutIntensityScale)
	{
		InterpolateProfile(pos.z, lutIntensityScale);

		// Subpixel grid size
		const int SW=4;
		const int SH=4;
		const float xstep = 1.0f/SW;
		const float ystep = 1.0f/SH;

		int smp=0;
		double Izz=0, Ixx=0 , Iyy=0, Ixy=0, Ixz=0, Iyz=0;
		numPixels = 0;

		for (int py=0;py<h;py++) {
			for (int px=0;px<w;px++) {
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

						this->numPixels++;
					}
				}
			}
		}

		matrix = Matrix3X3( 
			vector3f(Ixx, Ixy, Ixz),
			vector3f(Ixy, Iyy, Iyz),
			vector3f(Ixz, Iyz, Izz));

		matrix *= 1.0f/(SW*SH);

		// Compute inverse 
		inverse = matrix.Inverse();
	}

	float SampleProfile(float r, float& deriv, float &dz)
	{
		int rounded = (int)(r+0.5f);
		int rs = rounded-1;
		float xval[] = { -1, 0, 1 };

		if (rs < 0) rs = 0;
		if (rs + 3 > radialsteps) rs = radialsteps-3;

		LsqSqQuadFit<float> u_fit(3, xval, &profile[rs]);
		LsqSqQuadFit<float> dudz_fit(3, xval, &dzProfile[rs]);

		float x = r-rounded; 
		dz = dudz_fit.compute(x);

		deriv = u_fit.computeDeriv(x);
		return u_fit.compute(x);
	}

	// Compute profile
	void InterpolateProfile(float z, float lutIntensityScale)
	{
		int iz = (int)z;
		iz = std::max(0, std::min(planes-2, iz));
		float *prof0 = &lut[iz*radialsteps];
		float *prof1 = &lut[(iz+1)*radialsteps];

		profileMaxValue = 0;
		for (int r=0;r<radialsteps;r++) {
			profile[r] = prof0[r] + (prof1[r] - prof0[r]) * (z-iz);
			profileMaxValue = std::max(profileMaxValue, profile[r]);
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

	float* profile;
	float* dzProfile; // derivative of profile wrt Z plane
	float *drProfile;

	Matrix3X3 matrix;
	Matrix3X3 inverse;
	int numPixels;
	float profileMaxValue;

	vector3f MinVariance() { return vector3f(inverse(0,0), inverse(1,1), inverse(2,2)); }
};
