//file: rlips.h

// RLIPS C data types, structs, function prototypes

// (c) 2011- University of Oulu, Finland
// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
// Licensed under FreeBSD license. See file LICENSE for details.

#ifndef __RLIPS_H
#define __RLIPS_H

#include<R.h>
#include<Rinternals.h>

#ifdef __APPLE__
#include<OpenCL/cl.h>
#else
#include<CL/cl.h>
#endif

// *************************************
// ***   Rlips structures          ***
// *************************************


// For splitting 64-bit addresses into two int's
typedef union _split_t
{
	int II[2];
	long longValue;
} addr;


// RLIPS data structures

// Single precision real
typedef struct _sRlips
{	
	// User given parameters (in rlips.init)
	int numCols; // Number of columns/unknowns
	int numRHS;  // Number of right hand sides, i.e. number of columns in the measurement
	int sizeBuffer;  // Number of rows in the rotation buffer
	int sizeWorkgroup; // OpenCL workgroup size (should be of form 2^n)
	

	// Internal parameters
	float zThreshold; // Threshold value for zero. This could also be user given parameter,
					  // but is not at the moment.
	int numTotRows; // Total number of data rows fed into problem
	int numRmatRows; // Number of rows currently in the target matrix
	int numBufferRows; // Number of data rows currently in the rotation buffer
	long flops; // Number of floating point operations made	(NOT USED)	
	
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
} sRlips;


typedef struct _cRlips
{	
	// User given parameters (in rlips.init)
	int numCols; // Number of columns/unknowns
	int numRHS;  // Number of right hand sides, i.e. number of columns in the measurement
	int sizeBuffer;  // Number of rows in the rotation buffer
	int sizeWorkgroup; // OpenCL workgroup size (should be of form 2^n)
	

	
	// Internal parameters
	float zThreshold; // Threshold value for zero. This could also be user given parameter,
					  // but is not at the moment.
	int numTotRows; // Number of data rows fed into problem
	int numRmatRows; // Number of rows currently in the target matrix
	int numBufferRows; // Number of data rows currently in the rotation buffer
	long flops; // Number of floating point operations made	(NOT USED)
		
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
} cRlips;






// ***************************************
// ***   Rlips function prototypes   ***
// ***************************************
SEXP sInitRlips( SEXP, SEXP, SEXP, SEXP);
SEXP sKillRlips(SEXP);
SEXP sRotateRlips(SEXP, SEXP, SEXP);
SEXP sGetDataRlips(SEXP);

SEXP cInitRlips(SEXP, SEXP, SEXP, SEXP);
SEXP cKillRlips(SEXP);
SEXP cRotateRlips(SEXP,SEXP,SEXP,SEXP);
SEXP cGetDataRlips(SEXP);

SEXP cbacksolve(SEXP,SEXP);


#endif
