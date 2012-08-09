//file: ocllips.h

// OpenCL-LIPS data types, structs, function protorypes
// and OpenCL kernel source code

// (c) 2011- University of Oulu, Finland
// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
// Licensed under FreeBSD license. See file LICENSE for details.

#ifndef __OCLLIPS_H
#define __OCLLIPS_H

#include<R.h>
#include<Rinternals.h>

#ifdef __APPLE__
#include<OpenCL/cl.h>
#else
#include<CL/cl.h>
#endif

// *************************************
// ***   ocllips structures          ***
// *************************************


// For splitting 64-bit addresses into two int's
typedef union _split_t
{
	int II[2];
	long longValue;
} addr;


// Single precision real
typedef struct _socllips
{	
	// User given parameters in lips_init
	int numCols; // Number of columns/unknowns
	int numRHS;  // Number of right hand sides, i.e. number of columns in the measurement
	int sizeBuffer;  // Number of rows in the rotation buffer
	int sizeWorkgroup; // OpenCL workgroup size (should be of form 2^n)
	
	float zThreshold; // Threshold value for zero. This could also be user given parameter
	
	// Internal changing parameters
	int numTotRows; // Number of data rows fed into problem
	int numRmatRows; // Number of rows in the target matrix
	int numBufferRows; // Number of data rows currently in the rotation buffer
	long flops; // Number of floating point operations made		
	
	// Constant parameters depending on user given parameters
	int numRmatCols; // number of columns in the R matrix
	int sizeRmat; // Total size (cols x rows) of R matrix
	int sizeBufferMat; // Total size of the buffer matrix
	
	// Device buffers
	cl_mem dRmat; // OpenCL buffer for R matrix
	cl_mem dBufferMat; // OpenCL buffer for buffer matrix
	
	// OpenCL stuff
	cl_platform_id* platform_id;
	cl_device_id* device_id;
	cl_context* context;
	cl_command_queue* commandqueue;
	cl_program* kernel_program;
	
	cl_kernel* fullRotKernel;
	cl_kernel* partRotKernel;
} sOcllips;


typedef struct _cocllips
{	
	// User given parameters in lips_init
	int numCols; // Number of columns/unknowns
	int numRHS;  // Number of right hand sides, i.e. number of columns in the measurement
	int sizeBuffer;  // Number of rows in the rotation buffer
	int sizeWorkgroup; // OpenCL workgroup size (should be of form 2^n)
	
	float zThreshold; // Threshold value for zero. This could also be user given parameter
	
	// Internal changing parameters
	int numTotRows; // Number of data rows fed into problem
	int numRmatRows; // Number of rows in the target matrix
	int numBufferRows; // Number of data rows currently in the rotation buffer
	long flops; // Number of floating point operations made		
	
	// Constant parameters depending on user given parameters
	int numRmatCols; // number of columns in the R matrix
	int sizeRmat; // Total size (cols x rows) of R matrix
	int sizeBufferMat; // Total size of the buffer matrix
	
	// Device buffers
	cl_mem dRmat_r; // OpenCL buffer for R matrix (real part)
	cl_mem dRmat_i; // OpenCL buffer for R matrix (imag part)	
	cl_mem dBufferMat_r; // OpenCL buffer for buffer matrix (real part)
	cl_mem dBufferMat_i; // OpenCL buffer for buffer matrix (imag part)
	
	// OpenCL stuff
	cl_platform_id* platform_id;
	cl_device_id* device_id;
	cl_context* context;
	cl_command_queue* commandqueue;
	cl_program* kernel_program;
	
	cl_kernel* fullRotKernel;
	cl_kernel* partRotKernel;
} cOcllips;






// ***************************************
// ***   ocllips function prototypes   ***
// ***************************************
SEXP sInitOcllips( SEXP, SEXP, SEXP, SEXP);
SEXP sKillOcllips(SEXP);
SEXP sRotateOcllips(SEXP, SEXP, SEXP);
SEXP sGetDataOcllips(SEXP);

SEXP cInitOcllips(SEXP, SEXP, SEXP, SEXP);
SEXP cKillOcllips(SEXP);
SEXP cRotateOcllips(SEXP,SEXP,SEXP,SEXP);
SEXP cGetDataOcllips(SEXP);

SEXP cbacksolve(SEXP,SEXP);


#endif
