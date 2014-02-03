#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <crtdbg.h>
#include "memdbg.h"
#include <cstdint>
#include <string>
#include <deque>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <complex>



#pragma pack(push, 4)
struct vector2f {
	vector2f() {x=y=0.0f; }
	vector2f(float X,float Y) { x=X;y=Y; }
	float x,y;

	static vector2f random(vector2f center, float R);
};

template<typename T>
struct vector3 {
	vector3() { x=y=z=0.0f; }
	template<typename Tx, typename Ty, typename Tz>
	vector3(Tx X,Ty Y,Tz Z) { x=X; y=Y; z=Z; }
	template<typename Tc> 
	vector3(const vector3<Tc>& o) : x(o.x),y(o.y),z(o.z) {}
	T x,y,z;

	vector3 operator*(const vector3& o) const {
		return vector3(x*o.x,y*o.y,z*o.z);
	}
	vector3 operator*(T a) const {
		return vector3(x*a,y*a,z*a);
	}
	friend vector3 operator*(T a, const vector3& b) { 
		return b*a;
	}
	vector3 operator+(const vector3& o) const {
		return vector3(x+o.x,y+o.y,z+o.z);
	}
	vector3& operator+=(const vector3& o) {
		x+=o.x; y+=o.y; z+=o.z; return *this;
	}
	vector3& operator-=(const vector3& o) {
		x-=o.x; y-=o.y; z-=o.z; return *this;
	}
	vector3 operator-(const vector3& o) const {
		return vector3(x-o.x,y-o.y,z-o.z);
	}
	vector3 operator-() const {
		return vector3(-x,-y,-z);
	}
	vector3& operator*=(const vector3& o) { 
		x*=o.x; y*=o.y; z*=o.z;
		return *this;
	}
	vector3& operator*=(T a) {
		x*=a; y*=a; z*=a;
		return *this;
	}
	vector3& operator/=(T a) {
		x/=a; y/=a; z/=a;
		return *this;
	}
	vector3 operator/(T a) {
		return vector3(x/a,y/a,z/a);
	}
	T length() {
		return sqrtf(x*x+y*y+z*z);
	}
	template<typename T>
	friend vector3 operator/(T a, vector3<T> b) {
		return vector3<T>(a/b.x,a/b.y,a/b.z);
	}
};

template<typename T>
inline vector3<T> sqrt(const vector3<T>& a) { return vector3<T>(sqrt(a.x),sqrt(a.y),sqrt(a.z)); }

typedef vector3<float> vector3f;
typedef vector3<double> vector3d;

#pragma pack(pop)


#define _CRT_SECURE_NO_WARNINGS

#ifdef _MSC_VER
#pragma warning(disable: 4244) // conversion from 'int' to 'float', possible loss of data
#endif


typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned long ulong;
typedef unsigned char uchar;

/*
 * Portable definition for SNPRINTF, VSNPRINTF, STRCASECMP and STRNCASECMP
 */
#ifdef _MSC_VER
	#if _MSC_VER > 1310
		#define SNPRINTF _snprintf_s
		#define VSNPRINTF _vsnprintf_s
	#else
		#define SNPRINTF _snprintf
		#define VSNPRINTF _vsnprintf
	#endif
	#define STRCASECMP _stricmp
	#define STRNCASECMP _strnicmp
	#define ALLOCA(size) _alloca(size) // allocates memory on stack
#else
	#define STRCASECMP strcasecmp
	#define STRNCASECMP strncasecmp
	#define SNPRINTF snprintf
	#define VSNPRINTF vsnprintf
	#define ALLOCA(size) alloca(size)
#endif
#define ALLOCA_ARRAY(T, N) ((T*)ALLOCA(sizeof(T) * N))

#include "dllmacros.h"
