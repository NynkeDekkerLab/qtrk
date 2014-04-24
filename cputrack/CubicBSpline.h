#pragma once

#ifndef CUDA_SUPPORTED_FUNC
	#define CUDA_SUPPORTED_FUNC
#endif

/*
Cubic B-spline. Has continuous derivative when knots are distinct
	  B(i - 1) = 1/6 * (1 - t) ^ 3
	  B(i + 0) = 1/6 * (3*t^3 - 6*t^2 + 4)
	  B(i + 1) = 1/6 * (-3*t^3 + 3*t^2 + 3t + 1)
	  B(i + 2) = 1/6 * t^3
*/

inline void CUDA_SUPPORTED_FUNC ComputeBSplineWeights(float w[], float t)
{
	float t2=t*t;
	float t3=t2*t;
	float omt = 1-t;
	w[0] = 1.0f/6 * omt*omt*omt;
	w[1] = 1.0f/6 * (3*t3 - 6*t2 + 4);
	w[2] = 1.0f/6 * (-3*t3 + 3*t2 + 3*t + 1);
	w[3] = 1.0f/6 * t3;
}
/*
	  B(i - 1) = 1/6 * (1 - t) ^ 3
	  B(i + 0) = 1/6 * (3*t^3 - 6*t^2 + 4)
	  B(i + 1) = 1/6 * (-3*t^3 + 3*t^2 + 3t + 1)
	  B(i + 2) = 1/6 * t^3
		
		B'(i - 1) = 1/6 * -3 * (1-t) ^ 2				= -1/2 * (1-t) ^ 2
	  B'(i + 0) = 1/6 * (3*3*t^2 - 6*2*t)			= 1/2 * (3t^2 - 4t)
	  B'(i + 1) = 1/6 * (-3 * 3 * t^2 + 3*2*t + 3)	= 1/2 * (-3t^2 + 2t + 1)
    B'(i + 2) = 1/6 * 3 * t^2						= 1/2 * t^2

		B''(i - 1) = -1/2 * (1-t) ^ 2 =         -1/2 * -2 * (1-t)  = 1-t
	  B''(i + 0) = 1/2 * (3t^2 - 4t) =				1/2 * (6*t - 4)    = 3t - 2
	  B''(i + 1) = 1/2 * (-3t^2 + 2t + 1)	=		1/2 * (-6*t + 2) = -3t + 1
    B''(i + 2) = 1/2 * t^2 =								1/2 * 2 * t			= t
		*/
template<typename T>
void CUDA_SUPPORTED_FUNC ComputeBSplineDerivatives(float t, T* k, T& deriv, T& deriv2)
{
	float t2=t*t;
	float t3=t2*t;
	float omt=1-t;

	deriv = 0.5f * (-omt*omt * k[0] + (3*t2 - 4*t) * k[1] + (-3*t2 + 2*t + 1) * k[2] + t2 * k[3]);
	deriv2 = omt * k[0] + (3*t - 2) * k[1] + (-3*t + 1) * k[2] + t * k[3];
}



template<typename T>
float CUDA_SUPPORTED_FUNC ComputeSplineFitMaxPos(T* data, int len)
{
	// find the maximum by doing following the spline gradient
	int iMax=0;
	T vMax=data[0];
	for (int k=1;k<len;k++) {
		if (data[k]>vMax) {
			vMax = data[k];
			iMax = k;
		}
	}
	if (iMax < 1) iMax=1;
	if (len < 4) return iMax;
		
	// x(t) = w(0, t) * p(0) + w(1, t) * p(1) + w(2, t) * p(2) + w(3, t) * p(3)
	// x'(t) = w'(0, t) * p(0) + w'(1, t) * p(1) + w'(2, t) * p(2) + w'(3, t) * p(3)
	// x''(t) = w''(0, t) * p(0) + w''(1, t) * p(1) + w'(2, t) * p(2) + w'(3, t) * p(3)

	T x = iMax;
	for(int it=0;it<5;it++) {
		if (x < 1 )x = 1;
		if (x >= len-2) x = len-2;
		int i = (int)x;
		T t=x-i;
		T w,w2;
		ComputeBSplineDerivatives(t, &data[i-1], w, w2);
		T dx=w/w2;
//		dbgprintf("[%d]: x=%f, w=%f, w2=%f. dx=%f\n", it, x,w,w2, -dx);
		x -= dx;
	}
	if (x<0) x=0;
	if (x>=len-1) x=len-1;
	return x;
}


