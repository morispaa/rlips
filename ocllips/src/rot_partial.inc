	// Calculate rotation parameters
	// Stages
	totalStages = 2 * rowsToRotate - 3;
	
	//printf("\tPartial rotations, Total Stages: %d\n",totalStages);
	

	// Loop through stages
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		numRotations = min( (stage+1) / 2, (2 * rowsToRotate - 1 - stage) / 2 );
		firstRow = min( stage + 1 , rowsToRotate ) - 1 + fRow;
		firstCol = max( stage - rowsToRotate + 2 , 1) - 1;
		
		//printf("\t\tStage: %d numRot: %d firstRow: %d firstCol: %d fCol: %d\n",stage,numRotations,firstRow,firstCol,fCol);
		
		// number of rotations/work-groups
		localSize = K->sizeWorkgroup;
		globalSize = localSize * numRotations;
		
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
	clFinish(*K->commandqueue);

	// Rotations done. Move Rotated dBuffer in dRmat
	//printf("\trowsInR before bufferCopy: %d\n",rowsInR);
	dRmatOffset = K->numRmatRows * K->numRmatCols;
	//printf("\tRmatOffset: %d\n",dRmatOffset);
	//printf("\tElementsCopied: %d\n",rowsToRotate * nCols16);
	err1 = clEnqueueCopyBuffer(*K->commandqueue,K->dBufferMat,K->dRmat,0,dRmatOffset * sizeof(float),sizeof(float) * rowsToRotate * K->numRmatCols,0,NULL,NULL);
	if ( err1 != CL_SUCCESS)
		{
			printf("Error in buffer copy! Error code %d.\n", err1);
		}
	clFinish(*K->commandqueue);
	
	// Update variables
	//rowsInR += rowsToRotate;
	//rowsIndBuffer = 0;
	
				