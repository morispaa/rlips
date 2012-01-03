// file: partialrot.cl
//
// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
// (c) 2011 University of Oulu, Finland
//
// Licensed under freeBSD license. See file LICENSE for details.

 
 #define cmult_r(a,b,x,y) ((a)*(x) - (b)*(y))
 #define cmult_i(a,b,x,y) ((a)*(y) + (b)*(x))
 
 
 __kernel void full_rotations( __global float* Rmat_r,__global float* Rmat_i,
 								__global float* BufferMat_r, __global float* BufferMat_i,
 								int firstRow, int firstCol, int nCols)
{
	int currentRotation = get_group_id(0); 
	int lid = get_local_id(0);
	int locSize = get_local_size(0);
	
	local float rotCos_r;
	local float rotCos_i;
	local float rotSin_r;
	local float rotSin_i;
	
	float locR1_r;
	float locR1_i;
	float locR2_r;
	float locR2_i;
	
	local int skip;
	local int swap;
	local int skiprot;
	local int rotRow, rotCol, firstBlock, numBlocks; 
	
	skip = 0;
	swap = 0;
	skiprot = 0;
	
	// Calculate rotation coefficients (This is a very simple preliminary version. Must be fixed soon!) 
	if (lid == 0)
	{
		rotRow = firstRow - currentRotation;
		rotCol = firstCol + currentRotation;
		firstBlock = rotCol / locSize;
		numBlocks = nCols / locSize - firstBlock;
		
	
		float a_r, a_i, b_r, b_i, sqab, modA, modB, tmpA, tmpB;
		a_r = Rmat_r[rotCol * nCols + rotCol];
		a_i = Rmat_i[rotCol * nCols + rotCol];
		b_r = BufferMat_r[rotRow * nCols + rotCol];
		b_i = BufferMat_i[rotRow * nCols + rotCol];

		tmpA = a_r*a_r + a_i*a_i;
		tmpB = b_r*b_r + b_i*b_i;
		sqab = sqrt(tmpA + tmpB);
		modA = sqrt(tmpA);
		modB = sqrt(tmpB);
		
		if (sqab < 0.00001f)
		{			
			//rotCos_r = a_r/modA;
			//rotCos_i = -a_i/modA;
			//rotSin_r = 0.0f;
			//rotSin_i = 0.0f;
			skip = 1;
		}
		else if (modB < 0.00001f)
		{
			rotCos_r = a_r/modA;
			rotCos_i = -a_i/modA;
			rotSin_r = 0.0f;
			rotSin_i = 0.0f;
			skiprot = 1;
		}
		else if (modA < 0.00001f)
		{
			rotCos_r = 0.0f;
			rotCos_i = 0.0f;
						
			rotSin_r =  b_r/modB;
			rotSin_i = -b_i/modB;
			swap = 1;
		}
		else
		{
			rotCos_r =  a_r/sqab;
			rotSin_r =  b_r/sqab;
			rotCos_i = -a_i/sqab;
			rotSin_i = -b_i/sqab;
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
		locR1_r = Rmat_r[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];
		locR2_r = BufferMat_r[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];
		locR1_i = Rmat_i[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];
		locR2_i = BufferMat_i[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];
		
		//barrier(CLK_LOCAL_MEM_FENCE);
		
		if (swap)
		{
			Rmat_r[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotSin_r,rotSin_i,locR2_r,locR2_i);
			Rmat_i[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotSin_r,rotSin_i,locR2_r,locR2_i);
			
			BufferMat_r[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(-rotSin_r,rotSin_i,locRi_r,locR1_i);
			BufferMat_i[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(-rotSin_r,rotSin_i,locRi_r,locR1_i);
		}
		else if (skiprot)
		{
			Rmat_r[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotCos_r,rotCos_i,locR1_r,locR1_i);
			Rmat_i[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotCos_r,rotCos_i,locR1_r,locR1_i);
			
			BufferMat_r[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotCos_r,-rotCos_i,locR2_r,locR2_i);
			BufferMat_i[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotCos_r,-rotCos_i,locR2_r,locR2_i);
		}
		else
		{
			Rmat_r[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotCos_r,rotCos_i,locR1_r,locR1_i) + cmult_r(rotSin_r,rotSin_i,locR2_r,locR2_i);
			Rmat_i[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotCos_r,rotCos_i,locR1_r,locR1_i) + cmult_i(rotSin_r,rotSin_i,locR2_r,locR2_i);
			
			BufferMat_r[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotCos_r,-rotCos_i,locR2_r,locR2_i) + cmult_r(-rotSin_r,rotSin_i,locR1_r,locR1_i);
			BufferMat_i[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotCos_r,-rotCos_i,locR2_r,locR2_i) + cmult_i(-rotSin_r,rotSin_i,locR1_r,locR1_i);
		}
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	

	
}
 
 



__kernel void partial_rotations( __global float* BufferMat_r, __global float* BufferMat_i, int firstRow, int firstCol, int nCols,int colOffset)
{
	int currentRotation = get_group_id(0); 
	int lid = get_local_id(0);
	int locSize = get_local_size(0);
	
	local float rotCos_r;
	local float rotCos_i;
	local float rotSin_r;
	local float rotSin_i;
	
	float locR1_r;
	float locR1_i;
	float locR2_r;
	float locR2_i;
	
	local int skip;
	local int swap;
	local int skiprot;
	local int rotRow, rotCol, firstBlock, numBlocks; 
	
	skip = 0;
	swap = 0;
	skiprot = 0;
	
	// Calculate rotation coefficients (This is a very simple preliminary version. Must be fixed soon!) 
	if (lid == 0)
	{
		rotRow = firstRow - currentRotation;
		rotCol = firstCol + currentRotation;
		firstBlock = (rotCol + colOffset) / locSize;
		numBlocks = nCols / locSize - firstBlock;
		
		float a_r, a_i, b_r, b_i, sqab, modA, modB, tmpA, tmpB;
		a_r = BufferMat_r[rotCol * nCols + rotCol];
		a_i = BufferMat _i[rotCol * nCols + rotCol];
		b_r = BufferMat_r[rotRow * nCols + rotCol];
		b_i = BufferMat_i[rotRow * nCols + rotCol];

		tmpA = a_r*a_r + a_i*a_i;
		tmpB = b_r*b_r + b_i*b_i;
		sqab = sqrt(tmpA + tmpB);
		modA = sqrt(tmpA);
		modB = sqrt(tmpB);
		
		if (sqab < 0.00001f)
		{			
			//rotCos_r = a_r/modA;
			//rotCos_i = -a_i/modA;
			//rotSin_r = 0.0f;
			//rotSin_i = 0.0f;
			skip = 1;
		}
		else if (modB < 0.00001f)
		{
			rotCos_r = a_r/modA;
			rotCos_i = -a_i/modA;
			rotSin_r = 0.0f;
			rotSin_i = 0.0f;
			skiprot = 1;
		}
		else if (modA < 0.00001f)
		{
			rotCos_r = 0.0f;
			rotCos_i = 0.0f;
						
			rotSin_r =  b_r/modB;
			rotSin_i = -b_i/modB;
			swap = 1;
		}
		else
		{
			rotCos_r =  a_r/sqab;
			rotSin_r =  b_r/sqab;
			rotCos_i = -a_i/sqab;
			rotSin_i = -b_i/sqab;
		}
		/*
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
		*/
		
		
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
		locR1_r = BufferMat_r[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];
		locR2_r = BufferMat_r[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];
		locR1_i = BufferMat_i[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];
		locR2_i = BufferMat_i[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];
		//locR1 = BufferMat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];
		//locR2 = BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];
		
		if (swap)
		{
			BufferMat_r[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotSin_r,rotSin_i,locR2_r,locR2_i);
			BufferMat_i[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotSin_r,rotSin_i,locR2_r,locR2_i);
			
			BufferMat_r[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(-rotSin_r,rotSin_i,locRi_r,locR1_i);
			BufferMat_i[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(-rotSin_r,rotSin_i,locRi_r,locR1_i);
		}
		else if (skiprot)
		{
			BufferMat_r[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotCos_r,rotCos_i,locR1_r,locR1_i);
			BufferMat_i[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotCos_r,rotCos_i,locR1_r,locR1_i);
			
			BufferMat_r[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotCos_r,-rotCos_i,locR2_r,locR2_i);
			BufferMat_i[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotCos_r,-rotCos_i,locR2_r,locR2_i);
		}
		else
		{
			BufferMat_r[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotCos_r,rotCos_i,locR1_r,locR1_i) + cmult_r(rotSin_r,rotSin_i,locR2_r,locR2_i);
			BufferMat_i[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotCos_r,rotCos_i,locR1_r,locR1_i) + cmult_i(rotSin_r,rotSin_i,locR2_r,locR2_i);
			
			BufferMat_r[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_r(rotCos_r,-rotCos_i,locR2_r,locR2_i) + cmult_r(-rotSin_r,rotSin_i,locR1_r,locR1_i);
			BufferMat_i[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = cmult_i(rotCos_r,-rotCos_i,locR2_r,locR2_i) + cmult_i(-rotSin_r,rotSin_i,locR1_r,locR1_i);
		}
		/*		
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
		*/
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	

	
}