	// Calculate rotation parameters
	// Stages
	totalStages = rowsToRotate + numColumns - 1;
	
	//printf("\tFull rotations, Total Stages: %d\n",totalStages);
	

	// Loop through stages
	for (stage = 1; stage <= totalStages; stage++)
	{
		// Calculate number of rotations
		numRotations = min(stage, rowsToRotate);
		numRotations = min(numRotations, totalStages - stage + 1);
		firstRow = min(stage, rowsToRotate) - 1 + fRow;
		firstCol = max(1, stage - (rowsToRotate - 1)) - 1 + fCol;
		
		//printf("\t\tStage: %d numRot: %d firstRow: %d firstCol: %d\n",stage,numRotations,firstRow,firstCol);
		
		// number of rotations/work-groups
		localSize = K->sizeWorkgroup;
		globalSize = localSize * numRotations;
		
		// Set up kernel arguments
		n = 0;
		err1 =  clSetKernelArg (*K->fullRotKernel, n++, sizeof(cl_mem), &K->dRmat);
		err1 =  clSetKernelArg (*K->fullRotKernel, n++, sizeof(cl_mem), &K->dBufferMat);
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
		//clFinish(A.commandqueue);
	

	}
	
				