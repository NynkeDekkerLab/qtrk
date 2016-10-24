#pragma once

#define _STDINT_H
// WOW, including this causes a labview crash on DLL unloading!! (Only if vision is not actually ran)
// #include "nivision.h"
#include "extcode.h"
#include "niimaq.h"
#include <complex>

/** \defgroup lab_functions LabVIEW datatypes and helper functions
\brief Definitions of datatypes and helper functions required for communication with LabVIEW.

Wrappers around LabVIEW's cintools function calls and data types. 
lv_prolog.h and lv_epilog.h set up the correct alignment for LabVIEW data.
See https://zone.ni.com/reference/en-XX/help/371361L-01/lvexcodeconcepts/configuring_the_clf_node/ for some more information.
*/

/** \addtogroup lab_functions
	@{
*/

/* lv_prolog.h and lv_epilog.h set up the correct alignment for LabVIEW data. */
#include "lv_prolog.h"

typedef struct {
	LVBoolean status;
	int32 code;
	LStrHandle message;
} ErrorCluster;

/// LabVIEW Array template
template<typename T>
struct LVArray {
	int32_t dimSize;
	T elem[1];
};
//typedef LVArray<float> **ppFloatArray;

/// LabVIEW 2D Array template
template<typename T>
struct LVArray2D {
	int32_t dimSizes[2];
	T elem[1];

	T & xy(int col, int row) {
		return elem[row*dimSizes[1]+col];
	}
	T& get(int row, int col) {
		return elem[row*dimSizes[1]+col];
	}
	int numElem() { return dimSizes[0]*dimSizes[1]; }
};
//typedef LVArray2D<float> **ppFloatArray2;

/// LabVIEW 3D Array template
template<typename T>
struct LVArray3D {
	int32_t dimSizes[3];
	T elem[1];

	int numElem() { return dimSizes[0]*dimSizes[1]*dimSizes[2]; }
};

/// LabVIEW N dimensions array template
template<typename T, int N>
struct LVArrayND {
	int32_t dimSizes[N];
	T elem[1];

	int numElem() { 
		int n = dimSizes[0];
		for (int i=1;i<N;i++) n*=dimSizes[i];
		return n; 
	}
};

#ifndef DOXYGEN // Hide these structs from documentation (why not a single enum?)

// Compile-time map of C++ types to Labview DataType codes
// https://zone.ni.com/reference/en-XX/help/371361J-01/lvexcode/numericarrayresize/
template<typename T> struct LVDataType {};
template<> struct LVDataType<float> { enum { code=9 }; };
template<> struct LVDataType<double> { enum { code=10 }; };
template<> struct LVDataType<int8_t> { enum { code=1 }; };
template<> struct LVDataType<int16_t> { enum { code=2 }; };
template<> struct LVDataType<int32_t> { enum { code=3 }; };
template<> struct LVDataType<int64_t> { enum { code=4 }; };
template<> struct LVDataType<uint8_t> { enum { code=5 }; };
template<> struct LVDataType<uint16_t> { enum { code=6 }; };
template<> struct LVDataType<uint32_t> { enum { code=7 }; };
template<> struct LVDataType<uint64_t> { enum { code=8 }; };
template<> struct LVDataType<std::complex<float> > { enum { code=0xc }; };
template<> struct LVDataType<std::complex<double> > { enum { code=0xd }; };

#endif

template<typename T>
void ResizeLVArray2D(LVArray2D<T>**& d, int rows, int cols) 
{
	if (NumericArrayResize(LVDataType<T>::code, 2, (UHandle*)&d, sizeof(T)*rows*cols) != mgNoErr)
		throw std::runtime_error( SPrintf("NumericArrayResize(2D array, %d, %d) returned error.", rows,cols));
	(*d)->dimSizes[0] = rows;
	(*d)->dimSizes[1] = cols;
}

template<typename T>
void ResizeLVArray3D(LVArray3D<T>**& d, int depth, int rows, int cols) 
{
	if (NumericArrayResize(LVDataType<T>::code, 3, (UHandle*)&d, sizeof(T)*rows*cols*depth) != mgNoErr)
		throw std::runtime_error( SPrintf("NumericArrayResize(3D array, %d, %d, %d) returned error.", depth,rows,cols));

	(*d)->dimSizes[0] = depth;
	(*d)->dimSizes[1] = rows;
	(*d)->dimSizes[2] = cols;
}

template<typename T, int N>
void ResizeLVArray(LVArrayND<T, N>**& d, int* dims) 
{
	for (int i=0;i<N;i++)
		(*d)->dimSizes[i]=dims[i];
	NumericArrayResize(LVDataType<T>::code, N, (UHandle*)&d, sizeof(T)*(*d)->numElem());
}

template<typename T>
void ResizeLVArray(LVArray<T>**& d, int elems) 
{
	if (NumericArrayResize(LVDataType<T>::code, 1, (UHandle*)&d, sizeof(T)*elems) != mgNoErr)
		throw std::runtime_error( SPrintf("NumericArrayResize(1D array, %d) returned error.", elems));
	(*d)->dimSize = elems;
}

#include "lv_epilog.h"

void ArgumentErrorMsg(ErrorCluster* e, const std::string& msg);

void SetLVString (LStrHandle str, const char *text);
MgErr FillErrorCluster(MgErr err, const char *message, ErrorCluster *error);
std::vector<std::string> LVGetStringArray(int count, LStrHandle *str);

class QueuedTracker;
/*! \brief Verify the referenced tracker is a valid QueuedTracker instance.

\param [in] tracker		The instance to verify.
\param [in] e			The LabVIEW error cluster for this call.
\param [in] funcname	The name of the desired function.

\return True if the supplied pointer can be used.
*/
bool ValidateTracker(QueuedTracker* tracker, ErrorCluster* e, const char *funcname);
/** @} */