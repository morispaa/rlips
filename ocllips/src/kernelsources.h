//file: kernelsources.h

// OpenCL-LIPS compute kernel source codes

// (c) 2011- University of Oulu, Finland
// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
// Licensed under FreeBSD license. See file LICENSE for details.


#ifndef __KERNELSOURCES_H
#define __KERNELSOURCES_H



// Single precision real
const char *sKernelSource = "\n" \
" __kernel void full_rotations( __global float* Rmat,__global float* BufferMat, int firstRow, int firstCol, int nCols)\n" \
"{\n" \
"	int currentRotation = get_group_id(0); \n" \
"	int lid = get_local_id(0);\n" \
"	int locSize = get_local_size(0);\n" \
"	\n" \
"	local float rotCos;\n" \
"	local float rotSin;\n" \
"\n" \
"	float locR1;\n" \
"	float locR2;\n" \
"	local int skip;\n" \
"	local int swap;\n" \
"	local int rotRow, rotCol, firstBlock, numBlocks; \n" \
"	\n" \
"	skip = 0;\n" \
"	swap = 0;\n" \
"	\n" \
"	if (lid == 0)\n" \
"	{\n" \
"		rotRow = firstRow - currentRotation;\n" \
"		rotCol = firstCol + currentRotation;\n" \
"		firstBlock = rotCol / locSize;\n" \
"		numBlocks = nCols / locSize - firstBlock;\n" \
"		\n" \
"	\n" \
"		float a, b, sqab;\n" \
"		a = Rmat[rotCol * nCols + rotCol];\n" \
"		b = BufferMat[rotRow * nCols + rotCol];\n" \
"		sqab = sqrt(a*a + b*b);\n" \
"		if (sqab < 0.00001f)\n" \
"		{\n" \
"			rotCos = 1.0f;\n" \
"			rotSin = 0.0f;\n" \
"			skip = 1;\n" \
"		}\n" \
"		else if (fabs(a) < 0.00001f)\n" \
"		{\n" \
"			rotCos = 0.0f;\n" \
"			rotSin = b/sqab;\n" \
"			swap = 1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			rotCos = a/sqab;\n" \
"			rotSin = b/sqab;\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	\n" \
"	if (skip)\n" \
"	{\n" \
"		return;\n" \
"	}\n" \
"	\n" \
"	int curBlock;\n" \
"	for(curBlock = 0; curBlock < numBlocks; curBlock++)\n" \
"	{\n" \
"		locR1 = Rmat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];\n" \
"		locR2 = BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];\n" \
"				\n" \
"		if (swap)\n" \
"		{\n" \
"			Rmat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = -rotSin * locR1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			Rmat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = rotCos * locR1 + rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = rotCos * locR2 - rotSin * locR1;\n" \
"		}\n" \
"	}	\n" \
"	//barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"}\n" \
"\n" \
"\n" \
"__kernel void partial_rotations( __global float* BufferMat, int firstRow, int firstCol, int nCols,int colOffset)\n" \
"{\n" \
"	int currentRotation = get_group_id(0); \n" \
"	int lid = get_local_id(0);\n" \
"	int locSize = get_local_size(0);\n" \
"	\n" \
"	local float rotCos;\n" \
"	local float rotSin;\n" \
"	float locR1;\n" \
"	float locR2;\n" \
"	local int skip;\n" \
"	local int swap;\n" \
"	local int rotRow, rotCol, firstBlock, numBlocks; \n" \
"	\n" \
"	skip = 0;\n" \
"	swap = 0;\n" \
"	\n" \
"	if (lid == 0)\n" \
"	{\n" \
"		rotRow = firstRow - currentRotation;\n" \
"		rotCol = firstCol + currentRotation;\n" \
"		firstBlock = (rotCol + colOffset) / locSize;\n" \
"		numBlocks = nCols / locSize - firstBlock;\n" \
"		\n" \
"	\n" \
"		float a, b, sqab;\n" \
"		a = BufferMat[rotCol * nCols + rotCol + colOffset];\n" \
"		b = BufferMat[rotRow * nCols + rotCol + colOffset];\n" \
"		sqab = sqrt(a*a + b*b);\n" \
"		if (sqab < 0.00001f || fabs(b) < 0.00001f)\n" \
"		{\n" \
"			rotCos = 1.0f;\n" \
"			rotSin = 0.0f;\n" \
"			skip = 1;\n" \
"		}\n" \
"		else if (fabs(a) < 0.00001f)\n" \
"		{\n" \
"			rotCos = 0.0f;\n" \
"			rotSin = b/sqab;\n" \
"			swap = 1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			rotCos = a/sqab;\n" \
"			rotSin = b/sqab;\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	\n" \
"	if (skip)\n" \
"	{\n" \
"		return;\n" \
"	}\n" \
"	\n" \
"	int curBlock;\n" \
"	for(curBlock = 0; curBlock < numBlocks; curBlock++)\n" \
"	{\n" \
"		locR1 = BufferMat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize];\n" \
"		locR2 = BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize];\n" \
"\n" \
"		if (swap)\n" \
"		{\n" \
"			BufferMat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = -rotSin * locR1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			BufferMat[rotCol * nCols + lid + (firstBlock + curBlock) * locSize] = rotCos * locR1 + rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + lid + (firstBlock + curBlock) * locSize] = rotCos * locR2 - rotSin * locR1;\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	//barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	\n" \
"\n" \
"	\n" \
"}\n";







#endif