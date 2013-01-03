//file: rlips.c

// R Linear Inverse Problem Solver
// Version 0.5
// 
// This version contains only OpenCL versions of single precision real
// and complex routines.
//

// (c) 2011- University of Oulu, Finland
// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
// Licensed under FreeBSD license. See file LICENSE for details.



#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<R.h>
#include<Rinternals.h>

#ifdef MAC
#include<OpenCL/opencl.h>
#endif
#ifdef LINUX
#include<CL/opencl.h>
#endif

#include "rlips.h"
#include "rotations.h"
#include "kernelsources.h"


// Index of a row-major stored upper triangular matrix
// A	row
// B	column
// C	number of columns
#define ridx(A,B,C) ((B)+(A)*(2*(C)-(A)-1)/2)

// Index of a row-major stored matrix
// A	row
// B	column
// C	number of columns
#define yidx(A,B,C) ((A)*(C)+(B))

#define min(A,B) (((A)<=(B))?(A):(B))
#define max(A,B) (((A)>(B))?(A):(B))


/*

	Routines for single precision real problems

*/



/*
Initialize a new RLIPS problem and structure
Arguments:
	NCOLS		Number of columns in the theory matrix, i.e. number of unknowns
	NRHS		Number of columns in the measurement matrix
	NBUF		Rotation buffer size, i.e. number of data rows stored into a buffer
				matrix
	BLOCKSIZE	OpenCL work group size, should currently be a multiple of 16.
				The optimal value depends on the GPU used.
				
Returns a integer vector of length 2. The vector contains the (64bit) address of the
initialized structure.
*/

SEXP sInitRlips(SEXP NCOLS, SEXP NRHS, SEXP NBUF, SEXP BLOCKSIZE)
{
	// Definitions required by R's .Call functionality
	SEXP ref;
	PROTECT(ref = allocVector(INTSXP,2));
	NCOLS = coerceVector(NCOLS,INTSXP);
	NRHS = coerceVector(NRHS,INTSXP);
	NBUF = coerceVector(NBUF,INTSXP);
	BLOCKSIZE = coerceVector(BLOCKSIZE,INTSXP);
	int ncols = INTEGER(NCOLS)[0];
	int nrhs = INTEGER(NRHS)[0];
	int nbuf = INTEGER(NBUF)[0];
	int blocksize = INTEGER(BLOCKSIZE)[0];
 
	
	// Allocate new sRlips struct
	sRlips * restrict K;
	K = (sRlips *)malloc(sizeof(sRlips));
	
	
	// Allocate OpenCL structures in K
	K->platform_id = malloc(sizeof(cl_platform_id));
	K->device_id = malloc(sizeof(cl_device_id));
	K->context = malloc(sizeof(cl_context));
	K->commandqueue = malloc(sizeof(cl_command_queue));
	K->kernel_program = malloc(sizeof(cl_program));
	K->fullRotKernel = malloc(sizeof(cl_kernel));
	K->partRotKernel = malloc(sizeof(cl_kernel));
	
	
	// Set the user provided parameters
	K->numCols = ncols;
	K->numRHS = nrhs;
	K->sizeBuffer = nbuf;
	K->sizeWorkgroup = blocksize;
	
	
	// Column size of OpenCL buffers (smallest multiple of workgroup size that
	// contains both theory matrix columns and measurements)
	K->numRmatCols = (ncols + nrhs + 32 - 1) / 32 * 32;

	
	// Numbers which absolute value is smaller than zThreshold are 
	// considered as zeroes. Hard coded at the moment. Could/should be
	// a user provided parameter in the future.
	K->zThreshold = 1.0E-8f;
	
	// Set initial values for status parameters
	K->numTotRows = 0;		// Total number of data rows fed into the problem 
	K->numBufferRows = 0;	// Number of rows currently in the buffer 
	K->numRmatRows = 0;		// Number of rows currently in the target matrix
	K->flops = 0L;			// Number of floating point operations used. This
							// is currently not used!

		
	// Set OpenCL buffersizes
	K->sizeRmat = K->numRmatCols * K->numCols;
	K->sizeBufferMat = K->numRmatCols * K->sizeBuffer;



	
	// ********************************************
	// Initialize OpenCL
	cl_int error;
	
	// Get first OpenCL platform
	error = clGetPlatformIDs(1,K->platform_id,NULL);
	if (error != CL_SUCCESS)
	{
		printf("Did not get OpenCL platform! Error code %d. Exiting sInitRlips.\n", 
				error);
		exit(1);
	}
	
	// Ask for one GPU
	error = clGetDeviceIDs(*K->platform_id,CL_DEVICE_TYPE_GPU,1,K->device_id,NULL);
	if (error != CL_SUCCESS)
	{
		printf("Did not get OpenCL device! Error code %d. Exiting sInitRlips.\n", 
				error);
		exit(1);
	}
	
	// Create OpenCL context
	*K->context = clCreateContext(0,1,K->device_id,NULL,NULL,&error);
	if (error != CL_SUCCESS)
	{
		printf("Did not create OpenCL context! Error code %d. Exiting sInitRlips.\n", 
				error);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);
	}
	
	// Create command queue
	*K->commandqueue = clCreateCommandQueue(*K->context,*K->device_id,0,&error);
	if (error != CL_SUCCESS)
	{
		printf("Did not create OpenCL command queue! Error code %d. Exiting sInitRlips.\n", 
				error);
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);
	}
	
	// Create kernel program (KernelSource in kernelsources.h)
	*K -> kernel_program = 
		  clCreateProgramWithSource(
						 *K->context,
						 1,
						 (const char **)&sKernelSource,
						 NULL,
						 &error);
						 
	if (!*K->kernel_program)
	{
		printf("Could not create compute program! Exiting sInitRlips.\n");
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);
	}
	
	// Build kernel executable
	error = clBuildProgram(*K->kernel_program,0,NULL,"-w",NULL,NULL);
	if (error != CL_SUCCESS)
	{
		printf("Error code: %d\n",error);
		size_t len;
		char buffer[2048];
		
		printf("Failed to build program executable! Exiting sInitRlips.\n");
		
		clGetProgramBuildInfo(*K->kernel_program,
							  *K->device_id,
							  CL_PROGRAM_BUILD_LOG,
							  sizeof(buffer),
							  buffer,
							  &len);
		
		printf("%s\n",buffer);
		
		if (*K->kernel_program) clReleaseProgram(*K->kernel_program);
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);
	}
	
	// Create the kernel functions
	*K->fullRotKernel = clCreateKernel(*K->kernel_program,"s_full_rotations",&error);
	cl_int error2;
	*K->partRotKernel = clCreateKernel(*K->kernel_program,"s_partial_rotations",&error2);
	if (error != CL_SUCCESS || error2 != CL_SUCCESS)
	{
		printf("Could not create kernel! Error codes: %d, %d. Exiting sInitRlips.\n",
				error,error2);
		if(*K->fullRotKernel) clReleaseKernel(*K->fullRotKernel);
		if(*K->partRotKernel) clReleaseKernel(*K->partRotKernel);
		if (*K->kernel_program) clReleaseProgram(*K->kernel_program);
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);		
	}	
		
	// Create OpenCL buffer for target matrix
	K->dRmat = clCreateBuffer(*K->context,CL_MEM_READ_WRITE,
							  sizeof(float) * K->sizeRmat,NULL,&error);

	if (error != CL_SUCCESS || error2 != CL_SUCCESS)
	{
		printf("Could not create OpenCL data buffers! Exiting sInitRlips.\n");
		if(*K->fullRotKernel) clReleaseKernel(*K->fullRotKernel);
		if(*K->partRotKernel) clReleaseKernel(*K->partRotKernel);
		if (*K->kernel_program) clReleaseProgram(*K->kernel_program);
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);		
	}	
		
	// Construct address and return
	addr D;
	long *q;
	q = (long *)K;
	
	D.longValue = (long)q;
	
	INTEGER(ref)[0] = D.II[0];
	INTEGER(ref)[1] = D.II[1];
	
	UNPROTECT(1);	

	return(ref);
	
}

///////////////////////////////////////////////////////////////////////////



// Free allocated Rlips and all OpenCL structures associated to it.
// Arguments:
//		REF		Integer vector of leh two containing the address of RLIPS structure

SEXP sKillRlips(SEXP REF)
{
	// Definitions required by R .Call functionality
	REF = coerceVector(REF,INTSXP);
	
	// Construct address
	addr D;
	D.II[0] = INTEGER(REF)[0];
	D.II[1] = INTEGER(REF)[1];
	
	sRlips *K;
	K = (sRlips *)D.longValue;

	// Free/release/deallocate all OpenCL structures associated to this rlips
	// structure
	if(*K->fullRotKernel) clReleaseKernel(*K->fullRotKernel);
	if(*K->partRotKernel) clReleaseKernel(*K->partRotKernel);
	if (*K->kernel_program) clReleaseProgram(*K->kernel_program);
	if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
	if (*K->context) clReleaseContext(*K->context);	
	
	if (K->dRmat) clReleaseMemObject(K->dRmat);
	
	free(K);

	// Return nothing (can not use void with .Call)	
	return R_NilValue;	
}

///////////////////////////////////////////////////////////////////////////

/*
	RotateRlips
	Takes dataBuffer containing theory matrix rows and measurements and
	sends it to GPU device and makes the rotations in GPU device

	Arguments:
		REF					Integer vector containing the address of the RLIPS structure
		DOUBLE_DATABUFFER	Double vector containing the data in row-major order
		BUFFERROWS			Integer containing the number of data rows in
							DOUBLE_DATABUFFER vector
*/	

SEXP sRotateRlips(SEXP REF, SEXP DOUBLE_DATABUFFER, SEXP BUFFERROWS)
{
	// Definitions required by R .Call system
	REF = coerceVector(REF,INTSXP);
	
	double *double_dataBuffer = REAL(DOUBLE_DATABUFFER);
	BUFFERROWS = coerceVector(BUFFERROWS,INTSXP);
	int bufferRows = INTEGER(BUFFERROWS)[0];

	// Construct address
	addr D;
	D.II[0] = INTEGER(REF)[0];
	D.II[1] = INTEGER(REF)[1];
	
	sRlips *K;
	K = (sRlips *)D.longValue;
	
	// Check that the number of bufferRows does not exceed the device buffer size
	// This should not happen in any situation.
	if (bufferRows > K->sizeBuffer)
	{
		printf("Too many data rows to rotate! Buffer has %d rows. You tried to rotate %d rows.\nRotations not done!\n",K->sizeBuffer,bufferRows);
		return R_NilValue;
	}
	
	// Rotate only, if there is something to rotate
	if (bufferRows > 0)
	{
		int i;
		cl_int error, err1, error2;
		int rowsToRotate, numColumns, fRow, fCol, numRows1, numRows2;
		int stage, totalStages, firstRow, firstCol, numRotations, dRmatOffset, n;
		size_t localSize, globalSize;
		
		
		// Allocate float vector as data buffer
		float __attribute__ ((aligned (32))) *dataBuffer;
		dataBuffer = malloc(sizeof(float) * bufferRows * K->numRmatCols);
		
		// Copy the given double values as float values to the new buffer.
		// NB: This is actually faster than casting the data as floats in the R side!
		for (i=0 ; i< bufferRows * K->numRmatCols ; i++)
		{
			dataBuffer[i] = (float) double_dataBuffer[i];
		}
		
		
		// Move data buffer into device
		// Note that we create and copy at the same time
		K->dBufferMat = clCreateBuffer(*K->context,
									   CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
									   sizeof(float) * K->numRmatCols * bufferRows,
									   dataBuffer,
									   &error2); 
		
					
		// Are there any full rotations to be made, i.e. does the target matrix
		// already have rotated rows in it?
		if (K->numRmatRows > 0)
		{
			// Is R matrix already full, i.e all consecutive rotations will be full
			// rotations?
			if (K->numRmatRows >= K->numCols)
			{
				// Rotate the whole buffer at once
				rowsToRotate = bufferRows;
				
				// Rotate through all columns
				numColumns = K->numCols;
				
				// Set the rotation start row and column
				fRow = 0;
				fCol = 0;

				// Call full rotations
				sFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
			
			}
			// else: will R matrix become full with this buffer? If so, divide the
			// rotations in two parts:
			// 1. Rotate fully and partially enough rows to make the target matrix full
			// 2. Rotate the remaining rows fully into the target matrix
			else if (K->numRmatRows + bufferRows > K->numCols)
			{	
				// Divide the rows into two parts
				numRows1 = K->numCols - K->numRmatRows;
				numRows2 = bufferRows - numRows1;
				
				// Rotate first numRows1 rows
				rowsToRotate = numRows1;
				
				// Rotate numRmatRows columns
				numColumns = K->numRmatRows;
				
				// Set starting row and column
				fRow = 0;
				fCol = 0;
				
				// Rotate first numRows1 rows fully
				sFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				// Rotate first numRows1 rows
				rowsToRotate = numRows1;
				
				// Rotate numRows1 columns
				numColumns = numRows1;
				
				// Set starting row and column
				fRow = 0;
				fCol = K->numRmatRows;
				
				// Rotate partially
				sPartialRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				// Rotate the remaining rows fully
				rowsToRotate = numRows2;
				numColumns = K->numCols;
				fRow = numRows1;
				fCol = 0;
				
				// Full rotations
				sFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
			}
			// else: after rotating the buffer, the target matrix remains non-full,
			// i.e. rotate fully first numRmatRows rows and then rotate partially
			// the remaining rows.  
			else
			{
				rowsToRotate = bufferRows;
				numColumns = K->numRmatRows;
				fRow = 0;
				fCol = 0;
				
				// Full rotations
				sFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				rowsToRotate = bufferRows;
				numColumns = bufferRows;
				fRow = 0;
				fCol = K->numRmatRows;
				
				// Partial rotations
				sPartialRotations(K,rowsToRotate,numColumns,fRow,fCol);
			
			}
		}
		// else: There are no rows in the target matrix, i.e. these are the first
		// rotations. 
		else
		{
			// If there are more buffer rows than there are rows in the target matrix,
			// first rotate numRmatCols rows partially to form full target matrix
			// and then rotate the remaining rows fully into the target matrix
			if (bufferRows > K->numCols)
			{
				// Rotate the first numRmatCols rows partially
				rowsToRotate = K->numCols;
				numColumns = K->numCols;
				fRow = 0;
				fCol = 0;
				
				// Partial rotations
				sPartialRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				// Rotate the remaining rows fully
				rowsToRotate = bufferRows - K->numCols;
				numColumns = K->numCols;
				fRow = K->numCols;
				fCol = 0;
				
				// Full rotations
				sFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				
			}
			else
			// Target matrix is empty and there are not enough rows in the buffer
			// to make the target matrix full. Rotate the whole buffer into 
			// the target matrix partially
			{
				rowsToRotate = bufferRows;
				numColumns = bufferRows;
				fRow = 0;
				fCol = 0;
			
				// Partial rotations
				sPartialRotations(K,rowsToRotate,numColumns,fRow,fCol);
			}
		}
		
		// Update internal parameters
		K->numTotRows += bufferRows;
		K->numRmatRows += bufferRows;
		// Number of rows in the target matrix can not be over numRmatCols
		if (K->numRmatRows > K->numCols) K->numRmatRows = K->numCols;
		
		// Release OpenCL buffer
		clReleaseMemObject(K->dBufferMat);
	}

	// Return nothing to R			
	return R_NilValue;
}


/////////////////////////////////////////////////////////////////////////////////


// GetDataRlips
// Fetches R matrix from the GPU device and sends it back to R.
SEXP sGetDataRlips(SEXP REF)
{
	REF = coerceVector(REF,INTSXP);
	SEXP DOUBLE_DATABUFFER;
	//DOUBLE_DATABUFFER = coerceVector(DOUBLE_DATABUFFER,REALSXP);
	
	//double *double_dataBuffer = REAL(DOUBLE_DATABUFFER);
	//DATAROWS = coerceVector(DATAROWS,INTSXP);
	//int dataRows = INTEGER(DATAROWS)[0];
	
	// Construct address
	addr D;
	D.II[0] = INTEGER(REF)[0];
	D.II[1] = INTEGER(REF)[1];
	
	sRlips *K;
	K = (sRlips *)D.longValue;
	
	
	
	cl_int error;
	
	float __attribute__ ((aligned (32))) *dataBuffer;
	dataBuffer = malloc(sizeof(float) * K->sizeRmat);
	
	PROTECT(DOUBLE_DATABUFFER = allocVector(REALSXP, K->sizeRmat));
	
	// Read dRmat from device to dataBuffer
	error = clEnqueueReadBuffer(*K->commandqueue,K->dRmat,CL_TRUE,0,sizeof(float) * K->sizeRmat,dataBuffer,0,NULL,NULL);
	if (error != CL_SUCCESS)
	{
		printf("Could not read buffer from device!\n");
		//dataRows = 0;
		return R_NilValue;
	}
	
	int i;
	for (i=0 ; i < K->sizeRmat ; i ++)
	{
		REAL(DOUBLE_DATABUFFER)[i] = (double) dataBuffer[i];
	}
	
	// Return to R the total number of rows in R matrix.
	//dataRows = K->numTotRows;
	free(dataBuffer);
	
	UNPROTECT(1);
	
	return DOUBLE_DATABUFFER;
}


// Single precision complex


//void cInitRlips(int *ref, int *ncols, int *nrhs, int *nbuf, int *blocksize)
SEXP cInitRlips(SEXP NCOLS, SEXP NRHS, SEXP NBUF, SEXP BLOCKSIZE)
{

	SEXP ref;
	PROTECT(ref = allocVector(INTSXP,2));
	NCOLS = coerceVector(NCOLS,INTSXP);
	NRHS = coerceVector(NRHS,INTSXP);
	NBUF = coerceVector(NBUF,INTSXP);
	BLOCKSIZE = coerceVector(BLOCKSIZE,INTSXP);
	//intref = AS_INTEGER(REF);
	int ncols = INTEGER(NCOLS)[0];
	int nrhs = INTEGER(NRHS)[0];
	int nbuf = INTEGER(NBUF)[0];
	int blocksize = INTEGER(BLOCKSIZE)[0];

	cRlips *K;
	
	// Allocate new sRlips struct
	K = (cRlips *)malloc(sizeof(cRlips));
	
	// Allocate OpenCL structures in K
	K->platform_id = malloc(sizeof(cl_platform_id));
	K->device_id = malloc(sizeof(cl_device_id));
	K->context = malloc(sizeof(cl_context));
	K->commandqueue = malloc(sizeof(cl_command_queue));
	K->kernel_program = malloc(sizeof(cl_program));
	K->fullRotKernel = malloc(sizeof(cl_kernel));
	K->partRotKernel = malloc(sizeof(cl_kernel));
	
	// Set the user provided parameters
	K->numCols = ncols;
	K->numRHS = nrhs;
	K->sizeBuffer = nbuf;
	K->sizeWorkgroup = blocksize;
	
	// Column size of OpenCL buffers (smallest multiple of workgroup size that
	// contains both theory matrix columns and measurements)
	//K->numRmatCols = (*ncols + *nrhs + *blocksize - 1) / *blocksize * *blocksize;
	K->numRmatCols = (ncols + nrhs + 32 - 1) / 32 * 32;
	
	// Numbers whose absolute value is smaller than zThreshold are 
	// considered as zeroes
	K->zThreshold = 1.0E-8f;
	
	// Set initial values for status parameters
	K->numTotRows = 0;
	K->numBufferRows = 0;
	K->numRmatRows = 0;
	K->flops = 0L;
		
	// Set OpenCL buffersizes
	K->sizeRmat = K->numRmatCols * K->numCols;
	K->sizeBufferMat = K->numRmatCols * K->sizeBuffer;


//	printf("Trying to initialize OpenCL\n");
	
	// ********************************************
	// Initialize OpenCL
	cl_int error;
	
	// Get first OpenCL platform
	error = clGetPlatformIDs(1,K->platform_id,NULL);
	if (error != CL_SUCCESS)
	{
		printf("Did not get OpenCL platform! Error code %d. Exiting sInitRlips.\n", error);
		exit(1);
	}
	
	// Ask for one GPU
	error = clGetDeviceIDs(*K->platform_id,CL_DEVICE_TYPE_GPU,1,K->device_id,NULL);
	if (error != CL_SUCCESS)
	{
		printf("Did not get OpenCL device! Error code %d. Exiting sInitRlips.\n", error);
		exit(1);
	}
	
	// Create OpenCL context
	*K->context = clCreateContext(0,1,K->device_id,NULL,NULL,&error);
	if (error != CL_SUCCESS)
	{
		printf("Did not create OpenCL context! Error code %d. Exiting sInitRlips.\n", error);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);
	}
	
	// Create command queue
	*K->commandqueue = clCreateCommandQueue(*K->context,*K->device_id,0,&error);
	if (error != CL_SUCCESS)
	{
		printf("Did not create OpenCL command queue! Error code %d. Exiting sInitRlips.\n", error);
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);
	}
	
	// Create kernel program (KernelSource in Rlips.h)
	*K->kernel_program = clCreateProgramWithSource(*K->context,1,(const char **)&cKernelSource,NULL,&error);
	if (!*K->kernel_program)
	{
		printf("Could not create compute program! Exiting sInitRlips.\n");
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);
	}
	
	// Build kernel executable
	error = clBuildProgram(*K->kernel_program,0,NULL,"-w",NULL,NULL);
	if (error != CL_SUCCESS)
	{
		printf("Error code: %d\n",error);
		size_t len;
		char buffer[1024*100];
		
		printf("Failed to build program executable! Exiting sInitRlips.\n");
		
		clGetProgramBuildInfo(*K->kernel_program,*K->device_id,CL_PROGRAM_BUILD_LOG,sizeof(buffer),buffer,&len);
		
		printf("Log length: %d\n%s\n",(int) len,buffer);
		
		if (*K->kernel_program) clReleaseProgram(*K->kernel_program);
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);
	}
	
	// Create the kernel functions
	*K->fullRotKernel = clCreateKernel(*K->kernel_program,"c_full_rotations",&error);
	cl_int error2;
	*K->partRotKernel = clCreateKernel(*K->kernel_program,"c_partial_rotations",&error2);
	if (error != CL_SUCCESS || error2 != CL_SUCCESS)
	{
		printf("Could not create kernel! Error codes: %d, %d. Exiting sInitRlips.\n",error,error2);
		if(*K->fullRotKernel) clReleaseKernel(*K->fullRotKernel);
		if(*K->partRotKernel) clReleaseKernel(*K->partRotKernel);
		if (*K->kernel_program) clReleaseProgram(*K->kernel_program);
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);		
	}	
	
//	printf("Trying to create OpenCL buffers\n");
	
	// Create OpenCL buffers

	//printf("K->sizeRmat: %d\n",K->sizeRmat);
	K->dRmat_r = clCreateBuffer(*K->context,CL_MEM_READ_WRITE,sizeof(float) * K->sizeRmat,NULL,&error);
	K->dRmat_i = clCreateBuffer(*K->context,CL_MEM_READ_WRITE,sizeof(float) * K->sizeRmat,NULL,&error2);	
	//K->dBufferMat = clCreateBuffer(*K->context,CL_MEM_READ_WRITE,sizeof(float) * K->sizeBufferMat,NULL,&error2);
	if (error != CL_SUCCESS || error2 != CL_SUCCESS)
	{
		printf("Could not create OpenCL data buffers! Error codes: %d, %d. Exiting sInitRlips.\n",error,error2);
		if(*K->fullRotKernel) clReleaseKernel(*K->fullRotKernel);
		if(*K->partRotKernel) clReleaseKernel(*K->partRotKernel);
		if (*K->kernel_program) clReleaseProgram(*K->kernel_program);
		if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
		if (*K->context) clReleaseContext(*K->context);
		exit(1);		
	}	
	
	// Print info on max. workgroup size
	//size_t maxsize;
	//error = clGetKernelWorkGroupInfo(*K->partRotKernel, *K->device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
    //                                                  sizeof(size_t), &maxsize, NULL);
    //printf("Preferred work group size is %ld\n",maxsize);                                                

//	printf("Trying to create address\n");
	
	// Construct address and return
	addr D;
	long *q;
	q = (long *)K;
	
	D.longValue = (long)q;
	
	//ref[0] = D.II[0];
	//ref[1] = D.II[1];
	
	INTEGER(ref)[0] = D.II[0];
	INTEGER(ref)[1] = D.II[1];

	UNPROTECT(1);

	return(ref);
	
}


SEXP cKillRlips(SEXP REF)
{
	
	REF = coerceVector(REF,INTSXP);
	
	// Construct address
	addr D;
	D.II[0] = INTEGER(REF)[0];
	D.II[1] = INTEGER(REF)[1];
	
	cRlips *K;
	K = (cRlips *)D.longValue;

	if(*K->fullRotKernel) clReleaseKernel(*K->fullRotKernel);
	if(*K->partRotKernel) clReleaseKernel(*K->partRotKernel);
	if (*K->kernel_program) clReleaseProgram(*K->kernel_program);
	if (*K->commandqueue) clReleaseCommandQueue(*K->commandqueue);
	if (*K->context) clReleaseContext(*K->context);	
	
	if (K->dRmat_r) clReleaseMemObject(K->dRmat_r);
	if (K->dRmat_i) clReleaseMemObject(K->dRmat_i);	
	//if (K->dBufferMat_r) clReleaseMemObject(K->dBufferMat_r);
	//if (K->dBufferMat_i) clReleaseMemObject(K->dBufferMat_i);
	
	free(K);
	//ref[0] = 0;
	//ref[1] = 0;
	
	return R_NilValue;	
}



//void cRotateRlips(int *ref, double *double_dataBuffer_r, double *double_dataBuffer_i, int *bufferRows)
SEXP cRotateRlips(SEXP REF, SEXP DOUBLE_DATABUFFER_R, SEXP DOUBLE_DATABUFFER_I, SEXP BUFFERROWS)
{

	REF = coerceVector(REF,INTSXP);
	
	double *double_dataBuffer_r = REAL(DOUBLE_DATABUFFER_R);
	double *double_dataBuffer_i = REAL(DOUBLE_DATABUFFER_I);
	BUFFERROWS = coerceVector(BUFFERROWS,INTSXP);
	int bufferRows = INTEGER(BUFFERROWS)[0];

	// Construct address
	addr D;
	D.II[0] = INTEGER(REF)[0];
	D.II[1] = INTEGER(REF)[1];
	// Construct address
	//addr D;
	//D.II[0] = ref[0];
	//D.II[1] = ref[1];
	
	cRlips *K;
	K = (cRlips *)D.longValue;
	
	// Check that the number of bufferRows does not exceed the device buffer size
	if (bufferRows > K->sizeBuffer)
	{
		printf("Too many data rows to rotate! Buffer has %d rows. You tried to rotate %d rows.\nRotations not done!\n",K->sizeBuffer,bufferRows);
		return R_NilValue;
	}
	
	// Make sure there is something to rotate
	if (bufferRows > 0)
	{
		int i;
		cl_int error;
		int rowsToRotate, numColumns, fRow, fCol, numRows1, numRows2;
		// Variables used in inc files (Fix when changing to functions!)
		int stage, totalStages, firstRow, firstCol, numRotations, dRmatOffset, n;
		cl_int err1;
		size_t localSize, globalSize;
		
		
		// Let's see if this does the trick...
		float __attribute__ ((aligned (32))) *dataBuffer_r;
		float __attribute__ ((aligned (32))) *dataBuffer_i;
		
		dataBuffer_r = malloc(sizeof(float) * bufferRows * K->numRmatCols);
		dataBuffer_i = malloc(sizeof(float) * bufferRows * K->numRmatCols);
		
		for (i=0 ; i< bufferRows * K->numRmatCols ; i++)
		{
			dataBuffer_r[i] = (float) double_dataBuffer_r[i];
			dataBuffer_i[i] = (float) double_dataBuffer_i[i];
		}
		
		
		
		cl_int error2;
		// Move data buffer into device
		//error = clEnqueueWriteBuffer(*K->commandqueue,K->dBufferMat,CL_FALSE,0,
		//			sizeof(float) * K->numRmatCols * *bufferRows,dataBuffer,0,NULL,NULL);
		K->dBufferMat_r = clCreateBuffer(*K->context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(float) * K->numRmatCols * bufferRows,dataBuffer_r,&error2); 
		K->dBufferMat_i = clCreateBuffer(*K->context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(float) * K->numRmatCols * bufferRows,dataBuffer_i,&error2); 
					
		// Are there any full rotations to be made?
		if (K->numRmatRows > 0)
		{
			// Is R matrix already full?
			if (K->numRmatRows >= K->numCols)
			{
				rowsToRotate = bufferRows;
				//numColumns = K->numRmatCols;
				numColumns = K->numCols;
				fRow = 0;
				fCol = 0;
				
				//#include "rot_full_c.inc"
				cFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
			
			}
			// else: will R matrix become full with this buffer?
			else if (K->numRmatRows + bufferRows > K->numCols)
			{	
				numRows1 = K->numCols - K->numRmatRows;
				numRows2 = bufferRows - numRows1;
				
				rowsToRotate = numRows1;
				numColumns = K->numRmatRows;
				fRow = 0;
				fCol = 0;
				
				//#include "rot_full_c.inc"
				cFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				rowsToRotate = numRows1;
				numColumns = numRows1;
				fRow = 0;
				fCol = K->numRmatRows;
				
				//#include "rot_partial_c.inc"
				cPartialRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				rowsToRotate = numRows2;
				//numColumns = K->numRmatCols;
				numColumns = K->numCols;
				fRow = numRows1;
				fCol = 0;
				
				//#include "rot_full_c.inc"
				cFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
			}
			// else: first do normal full rotations and then partial rotations
			else
			{
				rowsToRotate = bufferRows;
				numColumns = K->numRmatRows;
				fRow = 0;
				fCol = 0;
				
				//#include "rot_full_c.inc"
				cFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				//clFinish(*K->commandqueue);
				rowsToRotate = bufferRows;
				numColumns = bufferRows;
				fRow = 0;
				fCol = K->numRmatRows;
				
				//#include "rot_partial_c.inc"
				cPartialRotations(K,rowsToRotate,numColumns,fRow,fCol);
			
			}
		}
		// else: do just partial rotations. This is only done if R matrix is empty
		else
		{
			if (bufferRows > K->numCols)
			{
				rowsToRotate = K->numCols;
				numColumns = K->numCols;
				fRow = 0;
				fCol = 0;
				
				//#include "rot_partial.inc"
				cPartialRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				rowsToRotate = bufferRows - K->numCols;
				numColumns = K->numCols;
				fRow = K->numCols;
				fCol = 0;
				
				//#include "rot_full.inc"
				cFullRotations(K,rowsToRotate,numColumns,fRow,fCol);
				
				
			}
			else
			{
				rowsToRotate = bufferRows;
				numColumns = bufferRows;
				fRow = 0;
				fCol = K->numRmatRows;
			
				//#include "rot_partial_c.inc"
				cPartialRotations(K,rowsToRotate,numColumns,fRow,fCol);
			}
		
		}
		
		// Update internal parameters
		K->numTotRows += bufferRows;
		K->numRmatRows += bufferRows;
		if (K->numRmatRows > K->numCols) K->numRmatRows = K->numCols;
		
		free(dataBuffer_r);
		free(dataBuffer_i);
		clReleaseMemObject(K->dBufferMat_r);
		clReleaseMemObject(K->dBufferMat_i);


		return R_NilValue;
		
		
	}
}


SEXP cGetDataRlips(SEXP REF)
{
	REF = coerceVector(REF,INTSXP);
	SEXP DOUBLE_DATABUFFER;
	
	//double *double_dataBuffer_r = REAL(DOUBLE_DATABUFFER_R);
	//double *double_dataBuffer_i = REAL(DOUBLE_DATABUFFER_I);
	//DATAROWS = coerceVector(DATAROWS,INTSXP);
	//int dataRows = INTEGER(DATAROWS)[0];
	
	// Construct address
	addr D;
	D.II[0] = INTEGER(REF)[0];
	D.II[1] = INTEGER(REF)[1];
	
	cRlips *K;
	K = (cRlips *)D.longValue;
	
	cl_int error;
	
	float __attribute__ ((aligned (32))) *dataBuffer_r;
	float __attribute__ ((aligned (32))) *dataBuffer_i;
	dataBuffer_r = malloc(sizeof(float) * K->sizeRmat);
	dataBuffer_i = malloc(sizeof(float) * K->sizeRmat);
	
	PROTECT(DOUBLE_DATABUFFER = allocVector(REALSXP, K->sizeRmat * 2));
	
	// Read dRmat from device to dataBuffer
	error = clEnqueueReadBuffer(*K->commandqueue,K->dRmat_r,CL_TRUE,0,sizeof(float) * K->sizeRmat,dataBuffer_r,0,NULL,NULL);
	error = clEnqueueReadBuffer(*K->commandqueue,K->dRmat_i,CL_TRUE,0,sizeof(float) * K->sizeRmat,dataBuffer_i,0,NULL,NULL);
	if (error != CL_SUCCESS)
	{
		printf("Could not read buffer from device!\n");
		//dataRows = 0;
		return R_NilValue;
	}
	
	int i;
	for (i=0 ; i < K->sizeRmat ; i++)
	{
		REAL(DOUBLE_DATABUFFER)[2*i] = (double) dataBuffer_r[i];
		REAL(DOUBLE_DATABUFFER)[2*i+1] = (double) dataBuffer_i[i];
	}
	
	// Return to R the total number of rows in R matrix.
	//*dataRows = K->numTotRows;
	
	UNPROTECT(1);
	
	return DOUBLE_DATABUFFER;
}


