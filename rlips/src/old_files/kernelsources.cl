// file: partialrot.cl
//
// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
// (c) 2011 University of Oulu, Finland
//
// Licensed under freeBSD license. See file LICENSE for details.

 
 __kernel void full_rotations( __global float* Rmat,__global float* BufferMat, int firstRow, int firstCol, int nCols)
{
	int currentRotation = get_group_id(0); 
	int lid = get_local_id(0);
	int locSize = get_local_size(0);
	
	local float rotCos;
	local float rotSin;
	float locR1;
	float locR2;
	local int skip;
	local int swap;
	local int rotRow, rotCol, firstBlock, numBlocks; 
	
	skip = 0;
	swap = 0;
	
	// Calculate rotation coefficients (This is a very simple preliminary version. Must be fixed soon!) 
	if (lid == 0)
	{
		rotRow = firstRow - currentRotation;
		rotCol = firstCol + currentRotation;
		firstBlock = rotCol / locSize;
		numBlocks = nCols / locSize - firstBlock;
		
	
		float a, b, sqab;
		a = Rmat[rotCol * nCols + rotCol];
		b = BufferMat[rotRow * nCols + rotCol];
		sqab = sqrt(a*a + b*b);
		if (sqab < 0.00001f)
		{
			rotCos = 1.0f;
			rotSin = 0.0f;
			skip = 1;
		}
		else if (fabs(a) < 0.00001f)
		{
			rotCos = 0.0f;
			rotSin = b/sqab;
			swap = 1;
		}
		else
		{
			rotCos = a/sqab;
			rotSin = b/sqab;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Check for common special cases
	if (skip)
	{
		return;
	}
	
	int curBlock;
	for(curBlock = 0; curBlock < numBlocks; curBlock++)
	{
		// Load data
		locR1 = Rmat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];
		locR2 = BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];
		
		//barrier(CLK_LOCAL_MEM_FENCE);
		
		if (swap)
		{
			Rmat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = rotSin * locR2;
		
			BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = -rotSin * locR1;
		}
		else
		{
			Rmat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = rotCos * locR1 + rotSin * locR2;
		
			BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = rotCos * locR2 - rotSin * locR1;
		}
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	

	
}
 
 



__kernel void partial_rotations( __global float* BufferMat, int firstRow, int firstCol, int nCols,int colOffset)
{
	int currentRotation = get_group_id(0); 
	int lid = get_local_id(0);
	int locSize = get_local_size(0);
	
	local float rotCos;
	local float rotSin;
	//local float locR1[BLOCK_SIZE];
	//local float locR2[BLOCK_SIZE];
	float locR1;
	float locR2;
	local int skip;
	local int swap;
	local int rotRow, rotCol, firstBlock, numBlocks; 
	
	skip = 0;
	swap = 0;
	
	// Calculate rotation coefficients (This is a very simple preliminary version. Must be fixed soon!) 
	if (lid == 0)
	{
		rotRow = firstRow - currentRotation;
		rotCol = firstCol + currentRotation;
		firstBlock = (rotCol + colOffset) / locSize;
		numBlocks = nCols / locSize - firstBlock;
		
	
		float a, b, sqab;
		a = BufferMat[rotCol * nCols + rotCol + colOffset];
		b = BufferMat[rotRow * nCols + rotCol + colOffset];
		sqab = sqrt(a*a + b*b);
		if (sqab < 0.00001f || fabs(b) < 0.00001f)
		{
			rotCos = 1.0f;
			rotSin = 0.0f;
			skip = 1;
		}
		else if (fabs(a) < 0.00001f)
		{
			rotCos = 0.0f;
			rotSin = b/sqab;
			swap = 1;
		}
		else
		{
			rotCos = a/sqab;
			rotSin = b/sqab;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Check for common special cases
	if (skip)
	{
		return;
	}
	
	int curBlock;
	for(curBlock = 0; curBlock < numBlocks; curBlock++)
	{
		// Load data
		locR1 = BufferMat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];
		locR2 = BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];
				
		if (swap)
		{
			BufferMat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = rotSin * locR2;
		
			BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = -rotSin * locR1;
		}
		else
		{
			BufferMat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = rotCos * locR1 + rotSin * locR2;
		
			BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = rotCos * locR2 - rotSin * locR1;
		}
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	

	
}