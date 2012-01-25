//file: rot_full.c

// ocllips internal functions for full rotations


// (c) 2011- University of Oulu, Finland
// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
// Licensed under FreeBSD license. See file LICENSE for details.

#include "rotations.h"
#include<stdio.h>



#define ridx(A,B,C) ((B)+(A)*(2*(C)-(A)-1)/2)
#define yidx(A,B,C) ((A)*(C)+(B))

#define min(A,B) (((A)<=(B))?(A):(B))
#define max(A,B) (((A)>(B))?(A):(B))




void sFullRotations(sOcllips *K, int rowsToRotate, int numColumns, int fRow, int fCol)
{
	
	// Calculate rotation parameters
	// Stages
	int totalStages = rowsToRotate + numColumns - 1;
	
	//printf("\tFull rotations, Total Stages: %d\n",totalStages);
	

	// Loop through stages
	int stage;
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		int numRotations;
		numRotations = min(stage, rowsToRotate);
		numRotations = min(numRotations, totalStages - stage + 1);
		
		int firstRow, firstCol;
		firstRow = min(stage, rowsToRotate) - 1 + fRow;
		firstCol = max(1, stage - (rowsToRotate - 1)) - 1 + fCol;
		
		//printf("\t\tStage: %d numRot: %d firstRow: %d firstCol: %d\n",stage,numRotations,firstRow,firstCol);
		
		// number of rotations/work-groups
		size_t localSize = K->sizeWorkgroup;
		size_t globalSize = localSize * numRotations;
		
		int err1;
		
		// Set up kernel arguments
		int n = 0;
		err1 =  clSetKernelArg (*K->fullRotKernel, n++, sizeof(cl_mem), &K->dRmat);
		err1 =  clSetKernelArg (*K->fullRotKernel, n++, sizeof(cl_mem), &K->dBufferMat);
		err1 |=  clSetKernelArg (*K->fullRotKernel, n++, sizeof(int), &firstRow);
		err1 |=  clSetKernelArg (*K->fullRotKernel, n++, sizeof(int), &firstCol);
		err1 |=  clSetKernelArg (*K->fullRotKernel, n++, sizeof(int), &K->numRmatCols);	
		//err1 |=  clSetKernelArg (partialRotKernel, n++, sizeof(cl_float) * localSize, NULL);	
		//err1 |=  clSetKernelArg (partialRotKernel, n++, sizeof(cl_float) * localSize, NULL);					
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel arguments! Error code %d.\n", (int) err1);
		}
		
		
		//printf("GlobalSize: %d\n",globalSize);
	
		// Launch rotation kernel
		err1 = clEnqueueNDRangeKernel(*K->commandqueue, *K->fullRotKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel execution! Error code %d.\n", (int) err1);
		}
		//clFinish(A.commandqueue);
	

	}

	return;
}


void cFullRotations(cOcllips *K, int rowsToRotate, int numColumns, int fRow, int fCol)
{
	// Calculate rotation parameters
	// Stages
	int totalStages = rowsToRotate + numColumns - 1;
	
	printf("\tFull rotations, Total Stages: %d\n",totalStages);
	

	// Loop through stages
	int stage;
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		int numRotations;
		numRotations = min(stage, rowsToRotate);
		numRotations = min(numRotations, totalStages - stage + 1);
		
		int firstRow, firstCol;
		firstRow = min(stage, rowsToRotate) - 1 + fRow;
		firstCol = max(1, stage - (rowsToRotate - 1)) - 1 + fCol;
		
		printf("\t\tStage: %d numRot: %d firstRow: %d firstCol: %d\n",stage,numRotations,firstRow,firstCol);
		
		// number of rotations/work-groups
		size_t localSize = K->sizeWorkgroup;
		size_t globalSize = localSize * numRotations;
		
		int err1;
		
		// Set up kernel arguments
		int n = 0;
		err1 =  clSetKernelArg (*K->fullRotKernel, n++, sizeof(cl_mem), &K->dRmat_r);
		err1 =  clSetKernelArg (*K->fullRotKernel, n++, sizeof(cl_mem), &K->dRmat_i);
		err1 =  clSetKernelArg (*K->fullRotKernel, n++, sizeof(cl_mem), &K->dBufferMat_r);
		err1 =  clSetKernelArg (*K->fullRotKernel, n++, sizeof(cl_mem), &K->dBufferMat_i);
		err1 |=  clSetKernelArg (*K->fullRotKernel, n++, sizeof(int), &firstRow);
		err1 |=  clSetKernelArg (*K->fullRotKernel, n++, sizeof(int), &firstCol);
		err1 |=  clSetKernelArg (*K->fullRotKernel, n++, sizeof(int), &K->numRmatCols);	
		//err1 |=  clSetKernelArg (partialRotKernel, n++, sizeof(cl_float) * localSize, NULL);	
		//err1 |=  clSetKernelArg (partialRotKernel, n++, sizeof(cl_float) * localSize, NULL);					
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel arguments! Error code %d.\n", err1);
		}
		
		
		//printf("GlobalSize: %d\n",globalSize);
	
		// Launch rotation kernel
		err1 = clEnqueueNDRangeKernel(*K->commandqueue, *K->fullRotKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel execution! Error code %d.\n", err1);
		}
		//clEnqueueBarrier(*K->commandqueue);
		clFinish(*K->commandqueue);

	}

	return;
}



void sPartialRotations(sOcllips  *K, int rowsToRotate, int numColumns, int fRow, int fCol)
{

	int err1, n;
	// Calculate rotation parameters
	// Stages
	int totalStages = 2 * rowsToRotate - 3;
	
	//printf("\tPartial rotations, Total Stages: %d\n",totalStages);
	

	// Loop through stages
	int stage;
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		int numRotations = min( (stage+1) / 2, (2 * rowsToRotate - 1 - stage) / 2 );
		int firstRow = min( stage + 1 , rowsToRotate ) - 1 + fRow;
		int firstCol = max( stage - rowsToRotate + 2 , 1) - 1;
		
		//printf("\t\tStage: %d numRot: %d firstRow: %d firstCol: %d fCol: %d\n",stage,numRotations,firstRow,firstCol,fCol);
		
		// number of rotations/work-groups
		size_t localSize = K->sizeWorkgroup;
		size_t globalSize = localSize * numRotations;
		
		// Set up kernel arguments
		n = 0;
		err1 =  clSetKernelArg (*K->partRotKernel, n++, sizeof(cl_mem), &K->dBufferMat);
		err1 |=  clSetKernelArg (*K->partRotKernel, n++, sizeof(int), &firstRow);
		err1 |=  clSetKernelArg (*K->partRotKernel, n++, sizeof(int), &firstCol);
		err1 |=  clSetKernelArg (*K->partRotKernel, n++, sizeof(int), &K->numRmatCols);
		err1 |=  clSetKernelArg (*K->partRotKernel, n++, sizeof(int), &fCol);	
		//err1 |=  clSetKernelArg (partialRotKernel, n++, sizeof(cl_float) * localSize, NULL);	
		//err1 |=  clSetKernelArg (partialRotKernel, n++, sizeof(cl_float) * localSize, NULL);					
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel arguments! Error code %d.\n", err1);
		}
		
		
		//printf("GlobalSize: %d\n",globalSize);
	
		// Launch rotation kernel
		err1 = clEnqueueNDRangeKernel(*K->commandqueue, *K->partRotKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in partial rotation kernel execution! Error code %d.\n", err1);
		}
		//clFinish(A.commandqueue);
	

	}
	//clFinish(*K->commandqueue);

	// Rotations done. Move Rotated dBuffer in dRmat
	//printf("\trowsInR before bufferCopy: %d\n",rowsInR);
	int dRmatOffset = K->numRmatRows * K->numRmatCols;
	//printf("RmatOffset: %d\n",dRmatOffset);
	//printf("\tElementsCopied: %d\n",rowsToRotate * nCols16);
	//printf("rowsToRotate: %d\n",rowsToRotate);
	err1 = clEnqueueCopyBuffer(*K->commandqueue,K->dBufferMat,K->dRmat,0,dRmatOffset * sizeof(float),sizeof(float) * rowsToRotate * K->numRmatCols,0,NULL,NULL);
	if ( err1 != CL_SUCCESS)
		{
			printf("Error in buffer copy! Error code %d.\n", err1);
			exit(1);
		}
	//clFinish(*K->commandqueue);
	
	// Update variables
	//rowsInR += rowsToRotate;
	//rowsIndBuffer = 0;
	
				
	
	return;
}



void cPartialRotations(cOcllips  *K, int rowsToRotate, int numColumns, int fRow, int fCol)
{
	int err1,n;
	// Calculate rotation parameters
	// Stages
	int totalStages = 2 * rowsToRotate - 3;
	
	printf("\tPartial rotations, Total Stages: %d\n",totalStages);
	

	// Loop through stages
	int stage;
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		int numRotations = min( (stage+1) / 2, (2 * rowsToRotate - 1 - stage) / 2 );
		int firstRow = min( stage + 1 , rowsToRotate ) - 1 + fRow;
		int firstCol = max( stage - rowsToRotate + 2 , 1) - 1;
		
		printf("\t\tStage: %d numRot: %d firstRow: %d firstCol: %d fCol: %d\n",stage,numRotations,firstRow,firstCol,fCol);
		
		// number of rotations/work-groups
		size_t localSize = K->sizeWorkgroup;
		size_t globalSize = localSize * numRotations;
		
		// Set up kernel arguments
		n = 0;
		err1 =  clSetKernelArg (*K->partRotKernel, n++, sizeof(cl_mem), &K->dBufferMat_r);
		err1 =  clSetKernelArg (*K->partRotKernel, n++, sizeof(cl_mem), &K->dBufferMat_i);
		err1 |=  clSetKernelArg (*K->partRotKernel, n++, sizeof(int), &firstRow);
		err1 |=  clSetKernelArg (*K->partRotKernel, n++, sizeof(int), &firstCol);
		err1 |=  clSetKernelArg (*K->partRotKernel, n++, sizeof(int), &K->numRmatCols);
		err1 |=  clSetKernelArg (*K->partRotKernel, n++, sizeof(int), &fCol);	
		//err1 |=  clSetKernelArg (partialRotKernel, n++, sizeof(cl_float) * localSize, NULL);	
		//err1 |=  clSetKernelArg (partialRotKernel, n++, sizeof(cl_float) * localSize, NULL);					
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in kernel arguments! Error code %d.\n", err1);
		}
		
		
		//printf("GlobalSize: %d\n",globalSize);
	
		// Launch rotation kernel
		err1 = clEnqueueNDRangeKernel(*K->commandqueue, *K->partRotKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if ( err1 != CL_SUCCESS)
		{
			printf("Error in partial rotation kernel execution! Error code %d.\n", err1);
		}
		//clFinish(*K->commandqueue);
	

	}
	//clFinish(*K->commandqueue);
	clEnqueueBarrier(*K->commandqueue);

	// Rotations done. Move Rotated dBuffer in dRmat
	//printf("\trowsInR before bufferCopy: %d\n",rowsInR);
	int dRmatOffset = K->numRmatRows * K->numRmatCols;
	printf("\tRmatOffset: %d\n",dRmatOffset);
	printf("\tElementsCopied: %d\n",rowsToRotate * K->numRmatCols);
	err1  = clEnqueueCopyBuffer(*K->commandqueue,K->dBufferMat_r,K->dRmat_r,0,dRmatOffset * sizeof(float),sizeof(float) * rowsToRotate * K->numRmatCols,0,NULL,NULL);
	//clEnqueueBarrier(*K->commandqueue);
	err1 |= clEnqueueCopyBuffer(*K->commandqueue,K->dBufferMat_i,K->dRmat_i,0,dRmatOffset * sizeof(float),sizeof(float) * rowsToRotate * K->numRmatCols,0,NULL,NULL);
	if ( err1 != CL_SUCCESS)
		{
			printf("Error in buffer copy! Error code %d.\n", err1);
		}
	//clFinish(*K->commandqueue);
	//clEnqueueBarrier(*K->commandqueue);
	// Update variables
	//rowsInR += rowsToRotate;
	//rowsIndBuffer = 0;
	
	
	return;
}