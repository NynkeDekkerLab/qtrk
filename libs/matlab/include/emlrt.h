/*
 * @(#)emlrt.h    generated by: makeheader 5.1.5  Mon Aug  4 18:35:53 2008
 *
 *		built from:	../../src/include/copyright.h
 *				../../src/include/pragma_interface.h
 *				Alias.cpp
 *				AllocCheck.cpp
 *				Assign.cpp
 *				BoundsCheck.cpp
 *				Create.cpp
 *				CreateArray.cpp
 *				Destroy.cpp
 *				Gateway.cpp
 *				InitArray.cpp
 *				RefCount.cpp
 *				RunTimeErrors.cpp
 *				Serialize.cpp
 *				Signals.cpp
 *				TypeCheck.cpp
 *				VSArray.cpp
 */

#if defined(_MSC_VER) || __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ > 3)
#pragma once
#endif

#ifndef emlrt_h
#define emlrt_h


/*
 * Copyright 1984-2003 The MathWorks, Inc.
 * All Rights Reserved.
 */



/* Copyright 2003-2006 The MathWorks, Inc. */

/* Only define EXTERN_C if it hasn't been defined already. This allows
 * individual modules to have more control over managing their exports.
 */
#ifndef EXTERN_C

#ifdef __cplusplus
  #define EXTERN_C extern "C"
#else
  #define EXTERN_C extern
#endif

#endif


/*
 * MATLAB INTERNAL USE ONLY :: Create an mxArray alias
 */
EXTERN_C const mxArray *
emlrtAlias(const mxArray *in);


/*
 * MATLAB INTERNAL USE ONLY :: Create a persistent mxArray alias
 */
EXTERN_C const mxArray *
emlrtAliasP(const mxArray *in);


/*
 * MATLAB INTERNAL USE ONLY :: Create an mxArray vector alias
 */
EXTERN_C void 
emlrtAliases(int32_T nlhs, const mxArray **plhs, const mxArray **prhs);

 
/*
 * MATLAB INTERNAL USE ONLY :: MEX function context
 */
typedef struct {
    boolean_T   bFirstTime;
    boolean_T   bInitialized;
}   emlrtContext;


/*
 * MATLAB INTERNAL USE ONLY :: Simulation hysteresis suppression
 */
EXTERN_C void
emlrtShs(const void *a, const int b);


/*
 * MATLAB INTERNAL USE ONLY :: Verify default fimath
 */
EXTERN_C void
emlrtCheckDefaultFimathR2008b(const mxArray **ctFimath);


/*
 * MATLAB INTERNAL USE ONLY :: Query first-time sentinal
 */
EXTERN_C boolean_T
emlrtFirstTime(emlrtContext *aMAInfo);


/*
 * MATLAB INTERNAL USE ONLY :: Clear mxArray allocation count
 */
EXTERN_C void
emlrtClearAllocCount(emlrtContext *aMAInfo, boolean_T bM, uint32_T iL, const char* ctDTO);


/*
 * MATLAB INTERNAL USE ONLY :: Update mxArray allocation count
 */
EXTERN_C void
emlrtUpdateAllocCount(int32_T delta);


/*
 * MATLAB INTERNAL USE ONLY :: Update persistent mxArray allocation count
 */
EXTERN_C void
emlrtUpdateAllocCountP(int32_T delta);


/*
 * MATLAB INTERNAL USE ONLY :: Check mxArray allocation count
 */
EXTERN_C void
emlrtCheckAllocCount(void);


/*
 * MATLAB INTERNAL USE ONLY :: Assign to an mxArray
 */
EXTERN_C void 
emlrtAssign(const mxArray **lhs, const mxArray *rhs);


/*
 * MATLAB INTERNAL USE ONLY :: Assign to a persistent mxArray
 */
EXTERN_C void 
emlrtAssignP(const mxArray **lhs, const mxArray *rhs);


/*
 * MATLAB INTERNAL USE ONLY :: Assign to a vector of mxArrays
 */
EXTERN_C void 
emlrtAssigns(int32_T nlhs, const mxArray **plhs, const mxArray **prhs);

 
/*
 * MATLAB INTERNAL USE ONLY :: Array bounds check parameters
 */
typedef struct {
    int32_T     iFirst;
    int32_T     iLast;
    int32_T     lineNo;
    int32_T     colNo;
    const char *aName;
    const char *fName;
    const char *pName;
    int32_T     checkKind; /* see src/cg_ir/classic/cg_node.hpp::CG_BoundsCheckKindEnum */
}   emlrtBCInfo;


/*
 * MATLAB INTERNAL USE ONLY :: Array bounds check
 */
EXTERN_C int32_T 
emlrtBoundsCheck(int32_T indexValue, emlrtBCInfo *aInfo);


/*
 * MATLAB INTERNAL USE ONLY :: Create an mxArray string from a C string
 */
EXTERN_C const mxArray *
emlrtCreateString(const char *in);


/*
 * MATLAB INTERNAL USE ONLY :: Create an mxArray string from a single char
 */
EXTERN_C const mxArray *
emlrtCreateString1(char c);


/*
 * MATLAB INTERNAL USE ONLY :: Create a struct matrix mxArray
 */
EXTERN_C const mxArray *
emlrtCreateStructMatrix(int32_T m, int32_T n, int32_T nfields, const char **field_names);


/*
 * MATLAB INTERNAL USE ONLY :: Create a struct matrix mxArray
 */
EXTERN_C const mxArray *
emlrtCreateStructArray(int32_T ndim, int32_T *pdim, int32_T nfields, const char **field_names);


/*
 * MATLAB INTERNAL USE ONLY :: Add a field to a struct matrix mxArray
 */
EXTERN_C const mxArray *
emlrtAddField(const mxArray *mxStruct, const mxArray *mxField, const char *fldName, int index);


/*
 * MATLAB INTERNAL USE ONLY :: Get a field from a struct matrix mxArray
 */
EXTERN_C const mxArray *
emlrtGetField(const mxArray *mxStruct, int aIndex, const char *fldName);


/*
 * MATLAB INTERNAL USE ONLY :: Create a numeric matrix mxArray
 */
EXTERN_C const mxArray *
emlrtCreateNumericMatrix(int32_T m, int32_T n, int32_T classID, int32_T nComplexFlag);


/*
 * MATLAB INTERNAL USE ONLY :: Create a numeric matrix mxArray
 */
EXTERN_C const mxArray *
emlrtCreateNumericArray(int32_T ndim, void *pdim, int32_T classID, int32_T nComplexFlag);


/*
 * MATLAB INTERNAL USE ONLY :: Create a scaled numeric matrix mxArray
 */
EXTERN_C const mxArray *
emlrtCreateScaledNumericArrayR2008b(int32_T ndim, void *pdim, int32_T classID, int32_T nComplexFlag, int32_T aScale);


/*
 * MATLAB INTERNAL USE ONLY :: Create a double scalar mxArray
 */
EXTERN_C const mxArray *
emlrtCreateDoubleScalar(real_T in);


/*
 * MATLAB INTERNAL USE ONLY :: Create a logical matrix mxArray
 */
EXTERN_C const mxArray *
emlrtCreateLogicalMatrix(int32_T m, int32_T n);


/*
 * MATLAB INTERNAL USE ONLY :: Create a logical matrix mxArray
 */
EXTERN_C const mxArray *
emlrtCreateLogicalArray(int32_T ndim, int32_T *dims);


/*
 * MATLAB INTERNAL USE ONLY :: Create a logical scalar mxArray
 */
EXTERN_C const mxArray *
emlrtCreateLogicalScalar(boolean_T in);


/*
 * MATLAB INTERNAL USE ONLY :: Create a character array mxArray
 */
EXTERN_C const mxArray *
emlrtCreateCharArray(int32_T ndim, int32_T *dims);


/*
 * MATLAB INTERNAL USE ONLY :: Create a 2-D character array mxArray
 */
EXTERN_C const mxArray *
emlrtCreateCharArray2(int32_T m, int32_T n);


/*
 * MATLAB INTERNAL USE ONLY :: Create a numerictype mxArray
 */
EXTERN_C const mxArray *
 emlrtCreateNumericType(boolean_T issigned, int32_T wordlength, real_T bias, real_T slope, int32_T fixedexponent);


/*
 * MATLAB INTERNAL USE ONLY :: Create a FI mxArray from a value mxArray
 */
EXTERN_C const mxArray *emlrtCreateFI(const mxArray *fimath, const mxArray *ntype, const char *fitype, const mxArray *fival);


/*
 * MATLAB INTERNAL USE ONLY :: Get the intarray from a FI mxArray
 */
EXTERN_C const mxArray *emlrtImportFiIntArrayR2008b(const mxArray *aFiMx);


/*
 * MATLAB INTERNAL USE ONLY :: Create an array of mxArrays
 */
EXTERN_C const mxArray * 
emlrtCreateArray(const mxArray **ppa, int_T am, int_T an);


/*
 * MATLAB INTERNAL USE ONLY :: Destroy an mxArray
 */
EXTERN_C void
emlrtDestroyArray(const mxArray **pa);


/*
 * MATLAB INTERNAL USE ONLY :: Destroy a vector of mxArrays
 */
EXTERN_C void
emlrtDestroyArrays(int32_T narrays, const mxArray **parrays);


/*
 * MATLAB INTERNAL USE ONLY :: Call out to MATLAB
 */
EXTERN_C const mxArray *
emlrtCallMATLAB(int32_T nlhs, const mxArray **plhs, int32_T nrhs, const mxArray **prhs, const char *cmd, boolean_T tmp);


/*
 * MATLAB INTERNAL USE ONLY :: Initialize a character mxArray
 */
EXTERN_C void 
emlrtInitCharArray(int32_T n, const mxArray *a, char *s);


/*
 * MATLAB INTERNAL USE ONLY :: Initialize a logical mxArray
 */
EXTERN_C void 
emlrtInitLogicalArray(int32_T n, const mxArray *a, boolean_T *b);


/*
 * MATLAB INTERNAL USE ONLY :: Export a numeric mxArray
 */
EXTERN_C void
emlrtExportNumericArrayR2008b(const mxArray *aOut, void *aInData, int32_T aElementSize);


/*
 * MATLAB INTERNAL USE ONLY :: Create a shallow copy of an mxArray
 */
EXTERN_C const mxArray *
emlrtCreateReference(const mxArray *pa);

 
/*
 * MATLAB INTERNAL USE ONLY :: Array bounds check parameters
 */
typedef struct {
    int32_T     lineNo;
    int32_T     colNo;
    const char *fName;
    const char *pName;
}   emlrtRTEInfo;


/*
 * MATLAB INTERNAL USE ONLY :: Division by zero error
 */
EXTERN_C void 
emlrtDivisionByZeroError(void);


/*
 * MATLAB INTERNAL USE ONLY :: Division by zero error
 */
EXTERN_C void 
emlrtDivisionByZeroErrorR2008a(const emlrtRTEInfo *aInfo);


/*
 * MATLAB INTERNAL USE ONLY :: Integer overflow error
 */
EXTERN_C void 
emlrtIntegerOverflowErrorR2008a(const emlrtRTEInfo *aInfo);


/*
 * MATLAB INTERNAL USE ONLY :: Terminate serializing
 */
EXTERN_C void
emlrtSerializeTerminate(void);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize the sizes of a variable sized matrix.
 */
EXTERN_C void
emlrtSerializeVSizes(uint32_T aNumDimensions, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize the sizes of a variable sized matrix.
 */
EXTERN_C void
emlrtDeserializeVSizes(uint32_T aNumDimensions, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a double
 */
EXTERN_C void
emlrtDeserializeDouble(uint32_T numElements, real_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size double
 */
EXTERN_C void
emlrtDeserializeVDouble(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a double
 */
EXTERN_C void
emlrtSerializeDouble(uint32_T numElements, real_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size double
 */
EXTERN_C void
emlrtSerializeVDouble(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex double
 */
EXTERN_C void
emlrtDeserializeCDouble(uint32_T numElements, creal_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex double
 */
EXTERN_C void
emlrtDeserializeCVDouble(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex double
 */
EXTERN_C void
emlrtSerializeCDouble(uint32_T numElements, creal_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex double
 */
EXTERN_C void
emlrtSerializeCVDouble(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a single
 */
EXTERN_C void
emlrtDeserializeSingle(uint32_T numElements, real32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size single
 */
EXTERN_C void
emlrtDeserializeVSingle(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex single
 */
EXTERN_C void
emlrtDeserializeCSingle(uint32_T numElements, creal32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex single
 */
EXTERN_C void
emlrtDeserializeCVSingle(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a single
 */
EXTERN_C void
emlrtSerializeSingle(uint32_T numElements, real32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size single
 */
EXTERN_C void
emlrtSerializeVSingle(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex single
 */
EXTERN_C void
emlrtSerializeCSingle(uint32_T numElements, creal32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex single
 */
EXTERN_C void
emlrtSerializeCVSingle(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a signed, 8-bit integer
 */
EXTERN_C void
emlrtDeserializeSint8(uint32_T numElements, int8_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size signed, 8-bit integer
 */
EXTERN_C void
emlrtDeserializeVSint8(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a signed, 8-bit integer
 */
EXTERN_C void
emlrtSerializeSint8(uint32_T numElements, int8_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size signed, 8-bit integer
 */
EXTERN_C void
emlrtSerializeVSint8(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex signed, 8-bit integer
 */
EXTERN_C void
emlrtDeserializeCSint8(uint32_T numElements, cint8_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex signed, 8-bit integer
 */
EXTERN_C void
emlrtDeserializeCVSint8(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex signed, 8-bit integer
 */
EXTERN_C void
emlrtSerializeCSint8(uint32_T numElements, cint8_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex signed, 8-bit integer
 */
EXTERN_C void
emlrtSerializeCVSint8(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize an unsigned, 8-bit integer
 */
EXTERN_C void
emlrtDeserializeUint8(uint32_T numElements, uint8_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size unsigned, 8-bit integer
 */
EXTERN_C void
emlrtDeserializeVUint8(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize an unsigned, 8-bit integer
 */
EXTERN_C void
emlrtSerializeUint8(uint32_T numElements, uint8_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size unsigned, 8-bit integer
 */
EXTERN_C void
emlrtSerializeVUint8(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex unsigned, 8-bit integer
 */
EXTERN_C void
emlrtDeserializeCUint8(uint32_T numElements, cuint8_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex unsigned, 8-bit integer
 */
EXTERN_C void
emlrtDeserializeCVUint8(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex unsigned, 8-bit integer
 */
EXTERN_C void
emlrtSerializeCUint8(uint32_T numElements, cuint8_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex unsigned, 8-bit integer
 */
EXTERN_C void
emlrtSerializeCVUint8(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a signed, 16-bit integer
 */
EXTERN_C void
emlrtDeserializeSint16(uint32_T numElements, int16_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size signed, 16-bit integer
 */
EXTERN_C void
emlrtDeserializeVSint16(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a signed, 16-bit integer
 */
EXTERN_C void
emlrtSerializeSint16(uint32_T numElements, int16_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size signed, 16-bit integer
 */
EXTERN_C void
emlrtSerializeVSint16(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex signed, 16-bit integer
 */
EXTERN_C void
emlrtDeserializeCSint16(uint32_T numElements, cint16_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex signed, 16-bit integer
 */
EXTERN_C void
emlrtDeserializeCVSint16(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex signed, 16-bit integer
 */
EXTERN_C void
emlrtSerializeCSint16(uint32_T numElements, cint16_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex signed, 16-bit integer
 */
EXTERN_C void
emlrtSerializeCVSint16(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize an unsigned, 16-bit integer
 */
EXTERN_C void
emlrtDeserializeUint16(uint32_T numElements, uint16_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size unsigned, 16-bit integer
 */
EXTERN_C void
emlrtDeserializeVUint16(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize an unsigned, 16-bit integer
 */
EXTERN_C void
emlrtSerializeUint16(uint32_T numElements, uint16_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size unsigned, 16-bit integer
 */
EXTERN_C void
emlrtSerializeVUint16(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex unsigned, 16-bit integer
 */
EXTERN_C void
emlrtDeserializeCUint16(uint32_T numElements, cuint16_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex unsigned, 16-bit integer
 */
EXTERN_C void
emlrtDeserializeCVUint16(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex unsigned, 16-bit integer
 */
EXTERN_C void
emlrtSerializeCUint16(uint32_T numElements, cuint16_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex unsigned, 16-bit integer
 */
EXTERN_C void
emlrtSerializeCVUint16(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a signed, 32-bit integer
 */
EXTERN_C void
emlrtDeserializeSint32(uint32_T numElements, int32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size signed, 32-bit integer
 */
EXTERN_C void
emlrtDeserializeVSint32(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a signed, 32-bit integer
 */
EXTERN_C void
emlrtSerializeSint32(uint32_T numElements, int32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size signed, 32-bit integer
 */
EXTERN_C void
emlrtSerializeVSint32(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex signed, 32-bit integer
 */
EXTERN_C void
emlrtDeserializeCSint32(uint32_T numElements, cint32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex signed, 32-bit integer
 */
EXTERN_C void
emlrtDeserializeCVSint32(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex signed, 32-bit integer
 */
EXTERN_C void
emlrtSerializeCSint32(uint32_T numElements, cint32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex signed, 32-bit integer
 */
EXTERN_C void
emlrtSerializeCVSint32(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize an unsigned, 32-bit integer
 */
EXTERN_C void
emlrtDeserializeUint32(uint32_T numElements, uint32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size unsigned, 32-bit integer
 */
EXTERN_C void
emlrtDeserializeVUint32(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize an unsigned, 32-bit integer
 */
EXTERN_C void
emlrtSerializeUint32(uint32_T numElements, uint32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size unsigned, 32-bit integer
 */
EXTERN_C void
emlrtSerializeVUint32(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex unsigned, 32-bit integer
 */
EXTERN_C void
emlrtDeserializeCUint32(uint32_T numElements, cuint32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex unsigned, 32-bit integer
 */
EXTERN_C void
emlrtDeserializeCVUint32(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex unsigned, 32-bit integer
 */
EXTERN_C void
emlrtSerializeCUint32(uint32_T numElements, cuint32_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex unsigned, 32-bit integer
 */
EXTERN_C void
emlrtSerializeCVUint32(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a signed, 64-bit integer
 */
EXTERN_C void
emlrtDeserializeSint64(uint32_T numElements, void *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size signed, 64-bit integer
 */
EXTERN_C void
emlrtDeserializeVSint64(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a signed, 64-bit integer
 */
EXTERN_C void
emlrtSerializeSint64(uint32_T numElements, void *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size signed, 64-bit integer
 */
EXTERN_C void
emlrtSerializeVSint64(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex signed, 64-bit integer
 */
EXTERN_C void
emlrtDeserializeCSint64(uint32_T numElements, void *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex signed, 64-bit integer
 */
EXTERN_C void
emlrtDeserializeCVSint64(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex signed, 64-bit integer
 */
EXTERN_C void
emlrtSerializeCSint64(uint32_T numElements, void *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex signed, 64-bit integer
 */
EXTERN_C void
emlrtSerializeCVSint64(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize an unsigned, 64-bit integer
 */
EXTERN_C void
emlrtDeserializeUint64(uint32_T numElements, void *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size unsigned, 64-bit integer
 */
EXTERN_C void
emlrtDeserializeVUint64(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize an unsigned, 64-bit integer
 */
EXTERN_C void
emlrtSerializeUint64(uint32_T numElements, void *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size unsigned, 64-bit integer
 */
EXTERN_C void
emlrtSerializeVUint64(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a complex unsigned, 64-bit integer
 */
EXTERN_C void
emlrtDeserializeCUint64(uint32_T numElements, void *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size complex unsigned, 64-bit integer
 */
EXTERN_C void
emlrtDeserializeCVUint64(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a complex unsigned, 64-bit integer
 */
EXTERN_C void
emlrtSerializeCUint64(uint32_T numElements, void *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size complex unsigned, 64-bit integer
 */
EXTERN_C void
emlrtSerializeCVUint64(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a char
 */
EXTERN_C void
emlrtDeserializeChar(uint32_T numElements, char_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size char
 */
EXTERN_C void
emlrtDeserializeVChar(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a char
 */
EXTERN_C void
emlrtSerializeChar(uint32_T numElements, char_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size char
 */
EXTERN_C void
emlrtSerializeVChar(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a logical
 */
EXTERN_C void
emlrtDeserializeLogical(uint32_T numElements, boolean_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Deserialize a variable-size logical
 */
EXTERN_C void
emlrtDeserializeVLogical(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a logical
 */
EXTERN_C void
emlrtSerializeLogical(uint32_T numElements, boolean_T *d);


/*
 * MATLAB INTERNAL USE ONLY :: Serialize a variable-size logical
 */
EXTERN_C void
emlrtSerializeVLogical(uint32_T aNumDimensions, void *aData, void *aSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Initialize serializing
 */
EXTERN_C void
emlrtSerializeInitialize(boolean_T isDeserialize, boolean_T isVerification, const char *projectName, uint32_T aCheckSumLen, uint8_T *aChecksum);


/*
 * MATLAB INTERNAL USE ONLY :: Check for Ctrl+C (break)
 */
EXTERN_C void
emlrtBreakCheck(void);


/*
 * MATLAB INTERNAL USE ONLY :: Check the class of an mxArray
 */
EXTERN_C void 
emlrtCheckClass(char *msgName, const mxArray *pa, const char *className);


/*
 * MATLAB INTERNAL USE ONLY :: Check the complexness of an mxArray
 */
EXTERN_C void 
emlrtCheckSparse(char *msgName, const mxArray *pa, boolean_T sparse);


/*
 * MATLAB INTERNAL USE ONLY :: Check the size, class and complexness of an mxArray
 */
EXTERN_C void 
emlrtCheckBuiltIn(char *msgName, const mxArray *pa, 
                  char *className, boolean_T complex, uint32_T nDims, void *pDims);


/*
 * MATLAB INTERNAL USE ONLY :: Check the size, class and complexness of a variable-size mxArray
 */
EXTERN_C void 
emlrtCheckVsBuiltInR2008b(char *msgName, const mxArray *pa, 
                    char *className, boolean_T complex, uint32_T nDims, void *pDims, 
                    boolean_T *aDynamic, int32_T *aOutSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Check the type of a FI mxArray
 */
EXTERN_C void 
emlrtCheckFI(char *aMsgName, const mxArray *aFi, boolean_T aComplex, uint32_T aNDims, void *aVDims, 
             const mxArray *aFimath, const mxArray *aNumericType);


/*
 * MATLAB INTERNAL USE ONLY :: Check the type of a variable-size FI mxArray
 */
EXTERN_C void 
emlrtCheckVsFiR2008b(char *aMsgName, const mxArray *aFi, boolean_T aComplex, uint32_T aNDims, void *aVDims, 
                     boolean_T *aDynamic, int32_T *aOutSizes,
                     const mxArray *aFimath, const mxArray *aNumericType);


/*
 * MATLAB INTERNAL USE ONLY :: Check the type of a static-size struct mxArray
 */
EXTERN_C void 
emlrtCheckStruct(char *msgName, const mxArray *s, int32_T nFields, const char **fldNames,
                 uint32_T nDims, void *pDims);


/*
 * MATLAB INTERNAL USE ONLY :: Check the type of a variable-size struct mxArray
 */
EXTERN_C void
emlrtCheckVsStructR2008b(char *msgName, const mxArray *s, int32_T nFields, const char **fldNames,
                   uint32_T nDims, void *pDims, 
                   boolean_T *aDynamic, int32_T *aOutSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Set the actual size of a variable-size array
 */
extern void 
emlrtSetVsSizesR2008b(const mxArray *pa, uint32_T nDimsMax, int32_T *aOutSizes);


/*
 * MATLAB INTERNAL USE ONLY :: Import an mxArray
 */
EXTERN_C void
emlrtImportArrayR2008b(const mxArray *aIn, void *aOutData, int32_T aElementSize);


/*
 * MATLAB INTERNAL USE ONLY :: Import an mxArray
 */
EXTERN_C void
emlrtImportVsArrayR2008b(const mxArray *aIn, void *aOutData, int32_T aElementSize, uint32_T nDimsMax, int32_T *aOutSizes);

#endif /* emlrt_h */
