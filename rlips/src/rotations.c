//file: rot_full.c

// Rlips internal functions for full rotations


// (c) 2011- University of Oulu, Finland
// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
// Licensed under FreeBSD license. See file LICENSE for details.

#include "rotations.h"
#include<stdio.h>


// Create index for upper triangular matrix
// stored in row-major order
// A row
// B column
// C number of columns in matrix
#define ridx(A,B,C) ((B)+(A)*(2*(C)-(A)-1)/2)

// Create index for rectangular matrix
// stored in row-major order
// A row
// B column
// C number of columns in matrix
#define yidx(A,B,C) ((A)*(C)+(B))

// Standard definitions for min and max
#define min(A,B) (((A)<=(B))?(A):(B))
#define max(A,B) (((A)>(B))?(A):(B))



/*
sFullRotations
=============

Performs full rotations for between target
and buffer matrices

Arguments:

	K				Real RLIPS structure
	rowsToRotate	Number of rows to be rotated
	numColumns		Number of columns to be rotated
	fRow			First row to be rotated
	fCol			First column to be rotated
*/
void sFullRotations(sRlips *K, int rowsToRotate, int numColumns,
					int fRow, int fCol)
{
	
	// Calculate rotation parameters
	// Stages
	int totalStages = rowsToRotate + numColumns - 1;
	
	// Loop through stages
	int stage;
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		// in this stage
		int nn;
		if (rowsToRotate > numColumns)
		{
			nn = numColumns;
		}
		else
		{
			nn = rowsToRotate;
		}
		int numRotations;
		numRotations = min(stage, nn);
		numRotations = min(numRotations,
						   totalStages - stage + 1);
		
		// Find first row and column for rotations
		int firstRow, firstCol;
		firstRow = min(stage, rowsToRotate) - 1 + fRow;
		firstCol = max(1, stage - (rowsToRotate - 1))
				   - 1 + fCol;
		
		// number of rotations/work-groups
		// Every single rotation is made in its own
		// work-group
		size_t localSize = K->sizeWorkgroup;
		size_t globalSize = localSize * numRotations;
		
		int err1;
		
		// Set up kernel arguments
		int n = 0;
		// R matrix array
		err1 =  clSetKernelArg (*K->fullRotKernel, 
								n++, 
								sizeof(cl_mem), 
								&K->dRmat);
		// Buffer array
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(cl_mem), 
								 &K->dBufferMat);
		// First row
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(int), 
								 &firstRow);
		// First column
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(int), 
								 &firstCol);
		// Number of columns in R matrix
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(int), 
								 &K->numRmatCols);	
								 
		// Handle possible errors				
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel arguments! Error code %d.\n", (int) err1);
		}
	
		// Launch full rotation kernel
		err1 = clEnqueueNDRangeKernel(*K->commandqueue, 
									  *K->fullRotKernel, 
									  1, 
									  NULL, 
									  &globalSize, 
									  &localSize, 
									  0, 
									  NULL, 
									  NULL);
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel execution! Error code %d.\n", (int) err1);
		}
	}
	
	// Finish	
	return;
}
///////////////////////////////////////////////////////////////
// End of sFullRotations
///////////////////////////////////////////////////////////////


/*
cFullRotations
=============

Performs full rotations for between target
and buffer matrices

Complex version

Arguments:

	K				Complex RLIPS structure
	rowsToRotate	Number of rows to be rotated
	numColumns		Number of columns to be rotated
	fRow			First row to be rotated
	fCol			First column to be rotated
*/
void cFullRotations(cRlips *K, int rowsToRotate, int numColumns, 
					int fRow, int fCol)
{
	// Calculate rotation parameters
	// Stages
	int totalStages = rowsToRotate + numColumns - 1;

	// Loop through stages
	int stage;
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		// in this stage
		int nn;
		if (rowsToRotate > numColumns)
		{
			nn = numColumns;
		}
		else
		{
			nn = rowsToRotate;
		}
		
		int numRotations;
		numRotations = min(stage, nn);
		numRotations = min(numRotations, 
						   totalStages - stage + 1);
		
		// Find first row and column for rotations
		int firstRow, firstCol;
		firstRow = min(stage, rowsToRotate) - 1 + fRow;
		firstCol = max(1, stage - (rowsToRotate - 1)) 
				   - 1 + fCol;
		
		// number of rotations/work-groups
		// Every single rotation is made in its own
		// work-group
		size_t localSize = K->sizeWorkgroup;
		size_t globalSize = localSize * numRotations;
		
		int err1;
		
		// Set up kernel arguments
		int n = 0;
		// R matrix real part
		err1 =  clSetKernelArg (*K->fullRotKernel, 
								n++, 
								sizeof(cl_mem), 
								&K->dRmat_r);
		// R matrix imaginary part
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(cl_mem), 
								 &K->dRmat_i);
		// Rotation buffer real part
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(cl_mem), 
								 &K->dBufferMat_r);
		// Rotation buffer imaginary part
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(cl_mem), 
								 &K->dBufferMat_i);
		// First row
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(int), 
								 &firstRow);
		// First column						 
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(int), 
								 &firstCol);
		// Number of columns in R matrix						 
		err1 |=  clSetKernelArg (*K->fullRotKernel, 
								 n++, 
								 sizeof(int), 
								 &K->numRmatCols);	
		
		//	Handle errors		
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel arguments! Error code %d.\n", err1);
		}

		// Launch full rotation kernel
		err1 = clEnqueueNDRangeKernel(*K->commandqueue, 
									  *K->fullRotKernel, 
									  1, 
									  NULL, 
									  &globalSize, 
									  &localSize, 
									  0, 
									  NULL, 
									  NULL);
									  
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel execution! Error code %d.\n", err1);
		}
	}

	// Finish
	return;
}
///////////////////////////////////////////////////////////////
// End of cFullRotations
///////////////////////////////////////////////////////////////


/*
sPartialRotations
=============

Performs partial rotations for between target
and buffer matrices

Arguments:

	K				Real RLIPS structure
	rowsToRotate	Number of rows to be rotated
	numColumns		Number of columns to be rotated
	fRow			First row to be rotated
	fCol			First column to be rotated
*/

void sPartialRotations(sRlips  *K, 
					   int rowsToRotate, 
					   int numColumns, 
					   int fRow, 
					   int fCol)
{

	int err1, n;
	// Calculate rotation parameters
	// Stages
	int totalStages = 2 * rowsToRotate - 3;

	// Loop through stages
	int stage;
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		// for this stage
		int numRotations = 
			min((stage+1) / 2, 
				(2 * rowsToRotate - 1 - stage) / 2 );
				
		// Find the first row and column for rotations		
		int firstRow = 
			min(stage + 1, rowsToRotate)
			- 1 + fRow;
		int firstCol = 
			max(stage - rowsToRotate + 2 , 1)
			- 1;

		// number of rotations/work-groups
		// Every single rotation is made in its own
		// work-group
		size_t localSize = K->sizeWorkgroup;
		size_t globalSize = localSize * numRotations;
		
		// Set up kernel arguments
		n = 0;
		// Buffer matrix
		err1 =  clSetKernelArg (*K->partRotKernel, 
								 n++, 
								 sizeof(cl_mem), 
								 &K->dBufferMat);
		// First rotation row						 
		err1 |=  clSetKernelArg (*K->partRotKernel, 
								 n++, 
								 sizeof(int), 
								 &firstRow);
		// First rotation column						 
		err1 |=  clSetKernelArg (*K->partRotKernel, 
								 n++, 
								 sizeof(int), 
								 &firstCol);
		// Number of R matrix columns						 
		err1 |=  clSetKernelArg (*K->partRotKernel, 
								 n++, 
								 sizeof(int), 
								 &K->numRmatCols);
		// Column offset                                    						 
		err1 |=  clSetKernelArg (*K->partRotKernel, 
								 n++,
								 sizeof(int), 
								 &fCol);	
				
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel arguments! Error code %d.\n", err1);
		}
		
		// Launch rotation kernel
		err1 = clEnqueueNDRangeKernel(*K->commandqueue, 
									  *K->partRotKernel, 
									  1, 
									  NULL, 
									  &globalSize, 
									  &localSize, 
									  0, 
									  NULL, 
									  NULL);
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in partial rotation kernel execution! Error code %d.\n", err1);
		}
	}
	
	// Rotations done. Move Rotated dBuffer in dRmat
	// Calculate offset for copy
	int dRmatOffset = K->numRmatRows * K->numRmatCols;
	
	// Copy from one OpenCL buffer to another
	err1 = clEnqueueCopyBuffer(*K->commandqueue,
							   K->dBufferMat,
							   K->dRmat,
							   0,
							   dRmatOffset * sizeof(float),
							   sizeof(float) * rowsToRotate 
							   		* K->numRmatCols,
							   0,
							   NULL,
							   NULL);
	if ( err1 != CL_SUCCESS)
		{
			printf("Error in buffer copy! Error code %d.\n", err1);
			exit(1);
		}
	
	return;
}
///////////////////////////////////////////////////////////////
// End of sPartialRotations
///////////////////////////////////////////////////////////////


/*
cPartialRotations
=============

Performs partial rotations for between target
and buffer matrices

Complex version

Arguments:

	K				Complex RLIPS structure
	rowsToRotate	Number of rows to be rotated
	numColumns		Number of columns to be rotated
	fRow			First row to be rotated
	fCol			First column to be rotated
*/

void cPartialRotations(cRlips  *K, 
					   int rowsToRotate, 
					   int numColumns, 
					   int fRow, 
					   int fCol)
{
	int err1,n;
	// Calculate rotation parameters
	// Stages
	int totalStages = 2 * rowsToRotate - 3;

	// Loop through stages
	int stage;
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		// for this stage
		int numRotations = 
			min((stage+1) / 2, 
				(2 * rowsToRotate - 1 - stage) / 2 );
				
		// Find the first row and column for rotations	 		
		int firstRow = 
			min(stage + 1, 
				rowsToRotate) 
				- 1 + fRow;
		int firstCol = 
			max(stage - rowsToRotate + 2 , 1) 
			- 1;

		// number of rotations/work-groups
		// Every single rotation is made in its own
		// work-group
		size_t localSize = K->sizeWorkgroup;
		size_t globalSize = localSize * numRotations;
		
		// Set up kernel arguments
		n = 0;
		// Real paert of buffer matrix
		err1 =  clSetKernelArg (*K->partRotKernel, 
								n++, 
								sizeof(cl_mem), 
								&K->dBufferMat_r);
		// Imaginary part of buffer matrix						
		err1 =  clSetKernelArg (*K->partRotKernel, 
								n++, 
								sizeof(cl_mem), 
								&K->dBufferMat_i);
		// First rotation row						
		err1 |=  clSetKernelArg (*K->partRotKernel, 
								 n++, 
								 sizeof(int), 
								 &firstRow);
		// First rotation column						 
		err1 |=  clSetKernelArg (*K->partRotKernel, 
								 n++, 
								 sizeof(int), 
								 &firstCol);
		// Number of columns in R matrix						 
		err1 |=  clSetKernelArg (*K->partRotKernel, 
								 n++, 
								 sizeof(int), 
								 &K->numRmatCols);
		// Column offset						 
		err1 |=  clSetKernelArg (*K->partRotKernel, 
								 n++, 
								 sizeof(int), 
								 &fCol);	
					
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel arguments! Error code %d.\n", err1);
		}

		// Launch rotation kernel
		err1 = clEnqueueNDRangeKernel(*K->commandqueue, 
									  *K->partRotKernel, 
									  1, 
									  NULL, 
									  &globalSize, 
									  &localSize, 
									  0, 
									  NULL, 
									  NULL);
									  
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in partial rotation kernel execution! Error code %d.\n", err1);
		}
	}
	
	// Rotations done. Move Rotated dBuffer in dRmat
	// Calculate offset for copy
	int dRmatOffset = K->numRmatRows * K->numRmatCols;

	// Copy both real and imaginary parts of the buffer 
	// into R matrix buffers
	err1  = clEnqueueCopyBuffer(*K->commandqueue,
								K->dBufferMat_r,
								K->dRmat_r,
								0,
								dRmatOffset * sizeof(float),
								sizeof(float) * rowsToRotate 
									* K->numRmatCols,
								0,
								NULL,
								NULL);
	err1 |= clEnqueueCopyBuffer(*K->commandqueue,
								K->dBufferMat_i,
								K->dRmat_i,
								0,
								dRmatOffset * sizeof(float),
								sizeof(float) * rowsToRotate 
									* K->numRmatCols,
								0,
								NULL,
								NULL);
	if ( err1 != CL_SUCCESS)
		{
			printf("Error in buffer copy! Error code %d.\n", err1);
		}
	return;
}
///////////////////////////////////////////////////////////////
// End of cPartialRotations
///////////////////////////////////////////////////////////////