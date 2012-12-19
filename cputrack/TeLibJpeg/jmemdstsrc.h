/************************************************************************************
TerraLib - a library for developing GIS applications.
Copyright � 2001-2007 INPE and Tecgraf/PUC-Rio.

This code is part of the TerraLib library.
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

You should have received a copy of the GNU Lesser General Public
License along with this library.

The authors reassure the license terms regarding the warranties.
They specifically disclaim any warranties, including, but not limited to,
the implied warranties of merchantability and fitness for a particular purpose.
The library provided hereunder is on an "as is" basis, and the authors have no
obligation to provide maintenance, support, updates, enhancements, or modifications.
In no event shall INPE and Tecgraf / PUC-Rio be held liable to any party for direct,
indirect, special, incidental, or consequential damages arising out of the use
of this library and its documentation.
*************************************************************************************/
/*! \file jmemdst.h
    \brief This file complements the jpeglib source in order to allow the writting of a JPEG data to a buffer in memory
	\note THIS IS FOR INTERNAL USE. DO NOT USE IT DIRECTLY.
*/
#ifndef  __TERRALIB_INTERNAL_DRIVER_JPEGLIB_MEMDST_H
#define  __TERRALIB_INTERNAL_DRIVER_JPEGLIB_MEMDST_H

#include "jpeglib.h"
 
#ifdef __cplusplus
extern "C" {
#endif

/*! Expanded data destination object for memory buffer output 
	\note THIS IS FOR INTERNAL USE. DO NOT USE IT DIRECTLY.
*/
typedef struct 
{
  struct jpeg_destination_mgr pub;	/* public fields */
  JOCTET **pTargetData;				/* memory buffer for jpeg output */
  unsigned int *pNumBytes;			/* number of bytes in the buffer */
  unsigned int initialDataSize;		/* size of the initially allocated buffer*/
  int bufferPreallocated;			/* boolean indicating whether buffer was previously allocated */
  int bufferSizeChanged;			/* boolean indicating whether buffer was changed inside the routine*/ 
} mem_destination_mgr;

typedef mem_destination_mgr * mem_dest_ptr;

/*! try initial buffer size of 1M */
#define OUTPUT_BUF_SIZE  ((unsigned int)1048576)

//! Function to allows the writting of JPEG data to a memory buffer
GLOBAL(void)
j_mem_dest(j_compress_ptr cinfo, void **pTargetData,	unsigned int *pNumBytes);

// Expanded data source object for memory buffer input
typedef struct
{
	struct jpeg_source_mgr pub;   
	unsigned char* buffer;
	unsigned int   bufsize;
} my_source_mgr;

typedef my_source_mgr* my_src_ptr;

//! Function to allows the reading of JPEG data from a memory buffer
GLOBAL(void)
j_mem_src (j_decompress_ptr cinfo, unsigned char* buffer, unsigned int bufsize);
 
#ifdef __cplusplus
}
#endif


#endif   
