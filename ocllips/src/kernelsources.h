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
"		//firstBlock = rotCol / locSize;\n" \
"		firstBlock = rotCol / 16 * 16;\n" \
"		//numBlocks = nCols / locSize - firstBlock;\n" \
"		numBlocks = nCols - firstBlock;\n" \
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
"	//for(curBlock = 0; curBlock < numBlocks; curBlock++)\n" \
"	for(curBlock = rotCol / 32 * 32 + lid; curBlock < nCols; curBlock += locSize)\n" \
"	{\n" \
"		locR1 = Rmat[rotCol * nCols + (curBlock)];\n" \
"		locR2 = BufferMat[rotRow * nCols + (curBlock)];\n" \
"				\n" \
"		if (swap)\n" \
"		{\n" \
"			Rmat[rotCol * nCols + (curBlock) ] = rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + (curBlock)] = -rotSin * locR1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			Rmat[rotCol * nCols + (curBlock)] = rotCos * locR1 + rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + (curBlock) ] = rotCos * locR2 - rotSin * locR1;\n" \
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
"		firstBlock = (rotCol + colOffset) / 16 * 16;\n" \
"		numBlocks = nCols - firstBlock;\n" \
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
"	for(curBlock = (rotCol + colOffset) / 32 * 32 + lid; curBlock < nCols; curBlock+=locSize)\n" \
"	{\n" \
"		locR1 = BufferMat[rotCol * nCols + (curBlock)];\n" \
"		locR2 = BufferMat[rotRow * nCols + (curBlock)];\n" \
"\n" \
"		if (swap)\n" \
"		{\n" \
"			BufferMat[rotCol * nCols + (curBlock)] = rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + (curBlock)] = -rotSin * locR1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			BufferMat[rotCol * nCols + (curBlock)] = rotCos * locR1 + rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + (curBlock)] = rotCos * locR2 - rotSin * locR1;\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	//barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	\n" \
"\n" \
"	\n" \
"}\n";


const char *cKernelSource = "// file: partialrot.cl\n" \
"//\n" \
"// Written by Mikko Orispaa <mikko.orispaa@oulu.fi>\n" \
"// (c) 2011 University of Oulu, Finland\n" \
"//\n" \
"// Licensed under freeBSD license. See file LICENSE for details.\n" \
"\n" \
" \n" \
" #define cmult_r(a,b,x,y) ((a)*(x) - (b)*(y))\n" \
" #define cmult_i(a,b,x,y) ((a)*(y) + (b)*(x))\n" \
" \n" \
" \n" \
" __kernel void full_rotations( __global float* Rmat_r,__global float* Rmat_i,\n" \
" 								__global float* BufferMat_r, __global float* BufferMat_i,\n" \
" 								int firstRow, int firstCol, int nCols)\n" \
"{\n" \
"	int currentRotation = get_group_id(0); \n" \
"	int lid = get_local_id(0);\n" \
"	int locSize = get_local_size(0);\n" \
"	\n" \
"	local float rotCos_r;\n" \
"	local float rotCos_i;\n" \
"	local float rotSin_r;\n" \
"	local float rotSin_i;\n" \
"	\n" \
"	float locR1_r;\n" \
"	float locR1_i;\n" \
"	float locR2_r;\n" \
"	float locR2_i;\n" \
"	\n" \
"	local int skip;\n" \
"	local int swap;\n" \
"	local int skiprot;\n" \
"	local int rotRow, rotCol, firstBlock, numBlocks; \n" \
"	\n" \
"	skip = 0;\n" \
"	swap = 0;\n" \
"	skiprot = 0;\n" \
"	\n" \
"	// Calculate rotation coefficients (This is a very simple preliminary version. Must be fixed soon!) \n" \
"	if (lid == 0)\n" \
"	{\n" \
"		rotRow = firstRow - currentRotation;\n" \
"		rotCol = firstCol + currentRotation;\n" \
"		firstBlock = rotCol / locSize;\n" \
"		numBlocks = nCols / locSize - firstBlock;\n" \
"		\n" \
"	\n" \
"		float a_r, a_i, b_r, b_i, sqab, modA, modB, tmpA, tmpB;\n" \
"		a_r = Rmat_r[rotCol * nCols + rotCol];\n" \
"		a_i = Rmat_i[rotCol * nCols + rotCol];\n" \
"		b_r = BufferMat_r[rotRow * nCols + rotCol];\n" \
"		b_i = BufferMat_i[rotRow * nCols + rotCol];\n" \
"\n" \
"		tmpA = a_r*a_r + a_i*a_i;\n" \
"		tmpB = b_r*b_r + b_i*b_i;\n" \
"		sqab = sqrt(tmpA + tmpB);\n" \
"		modA = sqrt(tmpA);\n" \
"		modB = sqrt(tmpB);\n" \
"		\n" \
"		if (sqab < 0.00001f)\n" \
"		{			\n" \
"			//rotCos_r = a_r/modA;\n" \
"			//rotCos_i = -a_i/modA;\n" \
"			//rotSin_r = 0.0f;\n" \
"			//rotSin_i = 0.0f;\n" \
"			skip = 1;\n" \
"		}\n" \
"		else if (modB < 0.00001f)\n" \
"		{\n" \
"			rotCos_r = a_r/modA;\n" \
"			rotCos_i = -a_i/modA;\n" \
"			rotSin_r = 0.0f;\n" \
"			rotSin_i = 0.0f;\n" \
"			skiprot = 1;\n" \
"		}\n" \
"		else if (modA < 0.00001f)\n" \
"		{\n" \
"			rotCos_r = 0.0f;\n" \
"			rotCos_i = 0.0f;\n" \
"						\n" \
"			rotSin_r =  b_r/modB;\n" \
"			rotSin_i = -b_i/modB;\n" \
"			swap = 1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			rotCos_r =  a_r/sqab;\n" \
"			rotSin_r =  b_r/sqab;\n" \
"			rotCos_i = -a_i/sqab;\n" \
"			rotSin_i = -b_i/sqab;\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	\n" \
"	// Check for common special cases\n" \
"	if (skip)\n" \
"	{\n" \
"		return;\n" \
"	}\n" \
"	\n" \
"	int curBlock;\n" \
"	for(curBlock = rotCol / 32 * 32 + lid; curBlock < nCols; curBlock += locSize)\n" \
"	{\n" \
"		// Load data\n" \
"		locR1_r = Rmat_r[rotCol * nCols + curBlock];\n" \
"		locR1_i = Rmat_i[rotCol * nCols + curBlock];\n" \
"		locR2_r = BufferMat_r[rotRow * nCols + curBlock];\n" \
"		locR2_i = BufferMat_i[rotRow * nCols + curBlock];\n" \
"		\n" \
"		//barrier(CLK_LOCAL_MEM_FENCE);\n" \
"		\n" \
"		if (swap)\n" \
"		{\n" \
"			Rmat_r[rotCol * nCols + curBlock] = cmult_r(rotSin_r,rotSin_i,locR2_r,locR2_i);\n" \
"			Rmat_i[rotCol * nCols + curBlock] = cmult_i(rotSin_r,rotSin_i,locR2_r,locR2_i);\n" \
"			\n" \
"			BufferMat_r[rotRow * nCols + curBlock] = cmult_r(-rotSin_r,rotSin_i,locR1_r,locR1_i);\n" \
"			BufferMat_i[rotRow * nCols + curBlock] = cmult_i(-rotSin_r,rotSin_i,locR1_r,locR1_i);\n" \
"		}\n" \
"		else if (skiprot)\n" \
"		{\n" \
"			Rmat_r[rotCol * nCols + curBlock] = cmult_r(rotCos_r,rotCos_i,locR1_r,locR1_i);\n" \
"			Rmat_i[rotCol * nCols + curBlock] = cmult_i(rotCos_r,rotCos_i,locR1_r,locR1_i);\n" \
"			\n" \
"			BufferMat_r[rotRow * nCols + curBlock] = cmult_r(rotCos_r,-rotCos_i,locR2_r,locR2_i);\n" \
"			BufferMat_i[rotRow * nCols + curBlock] = cmult_i(rotCos_r,-rotCos_i,locR2_r,locR2_i);\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			Rmat_r[rotCol * nCols + curBlock] = cmult_r(rotCos_r,rotCos_i,locR1_r,locR1_i) + cmult_r(rotSin_r,rotSin_i,locR2_r,locR2_i);\n" \
"			Rmat_i[rotCol * nCols + curBlock] = cmult_i(rotCos_r,rotCos_i,locR1_r,locR1_i) + cmult_i(rotSin_r,rotSin_i,locR2_r,locR2_i);\n" \
"			\n" \
"			BufferMat_r[rotRow * nCols + curBlock] = cmult_r(rotCos_r,-rotCos_i,locR2_r,locR2_i) + cmult_r(-rotSin_r,rotSin_i,locR1_r,locR1_i);\n" \
"			BufferMat_i[rotRow * nCols + curBlock] = cmult_i(rotCos_r,-rotCos_i,locR2_r,locR2_i) + cmult_i(-rotSin_r,rotSin_i,locR1_r,locR1_i);\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	//barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	\n" \
"\n" \
"	\n" \
"}\n" \
" \n" \
" \n" \
"\n" \
"\n" \
"\n" \
"__kernel void partial_rotations( __global float* BufferMat_r, __global float* BufferMat_i, int firstRow, int firstCol, int nCols,int colOffset)\n" \
"{\n" \
"	int currentRotation = get_group_id(0); \n" \
"	int lid = get_local_id(0);\n" \
"	int locSize = get_local_size(0);\n" \
"	\n" \
"	local float rotCos_r;\n" \
"	local float rotCos_i;\n" \
"	local float rotSin_r;\n" \
"	local float rotSin_i;\n" \
"	\n" \
"	float locR1_r;\n" \
"	float locR1_i;\n" \
"	float locR2_r;\n" \
"	float locR2_i;\n" \
"	\n" \
"	local int skip;\n" \
"	local int swap;\n" \
"	local int skiprot;\n" \
"	local int rotRow, rotCol, firstBlock, numBlocks; \n" \
"	\n" \
"	skip = 0;\n" \
"	swap = 0;\n" \
"	skiprot = 0;\n" \
"	\n" \
"	// Calculate rotation coefficients (This is a very simple preliminary version. Must be fixed soon!) \n" \
"	if (lid == 0)\n" \
"	{\n" \
"		rotRow = firstRow - currentRotation;\n" \
"		rotCol = firstCol + currentRotation;\n" \
"		firstBlock = (rotCol + colOffset) / locSize;\n" \
"		numBlocks = nCols / locSize - firstBlock;\n" \
"		\n" \
"		float a_r, a_i, b_r, b_i, sqab, modA, modB, tmpA, tmpB;\n" \
"		a_r = BufferMat_r[rotCol * nCols + rotCol + colOffset];\n" \
"		a_i = BufferMat_i[rotCol * nCols + rotCol + colOffset];\n" \
"		b_r = BufferMat_r[rotRow * nCols + rotCol + colOffset];\n" \
"		b_i = BufferMat_i[rotRow * nCols + rotCol + colOffset];\n" \
"\n" \
"		tmpA = a_r*a_r + a_i*a_i;\n" \
"		tmpB = b_r*b_r + b_i*b_i;\n" \
"		sqab = sqrt(tmpA + tmpB);\n" \
"		modA = sqrt(tmpA);\n" \
"		modB = sqrt(tmpB);\n" \
"		\n" \
"		if (sqab < 0.00001f)\n" \
"		{			\n" \
"			//rotCos_r = a_r/modA;\n" \
"			//rotCos_i = -a_i/modA;\n" \
"			//rotSin_r = 0.0f;\n" \
"			//rotSin_i = 0.0f;\n" \
"			skip = 1;\n" \
"		}\n" \
"		else if (modB < 0.00001f)\n" \
"		{\n" \
"			rotCos_r = a_r/modA;\n" \
"			rotCos_i = -a_i/modA;\n" \
"			rotSin_r = 0.0f;\n" \
"			rotSin_i = 0.0f;\n" \
"			skiprot = 1;\n" \
"		}\n" \
"		else if (modA < 0.00001f)\n" \
"		{\n" \
"			rotCos_r = 0.0f;\n" \
"			rotCos_i = 0.0f;\n" \
"						\n" \
"			rotSin_r =  b_r/modB;\n" \
"			rotSin_i = -b_i/modB;\n" \
"			swap = 1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			rotCos_r =  a_r/sqab;\n" \
"			rotCos_i = -a_i/sqab;\n" \
"			rotSin_r =  b_r/sqab;\n" \
"			rotSin_i = -b_i/sqab;\n" \
"		}\n" \
"		/*\n" \
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
"		*/\n" \
"		\n" \
"		\n" \
"	}\n" \
"	\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	\n" \
"	// Check for common special cases\n" \
"	if (skip)\n" \
"	{\n" \
"		return;\n" \
"	}\n" \
"	\n" \
"	int curBlock;\n" \
"	for(curBlock = (rotCol + colOffset) / 32 * 32 + lid; curBlock < nCols; curBlock += locSize)\n" \
"	{\n" \
"		// Load data\n" \
"		locR1_r = BufferMat_r[rotCol * nCols + curBlock];\n" \
"		locR1_i = BufferMat_i[rotCol * nCols + curBlock];\n" \
"		locR2_r = BufferMat_r[rotRow * nCols + curBlock];\n" \
"		locR2_i = BufferMat_i[rotRow * nCols + curBlock];\n" \
"		//locR1 = BufferMat[rotCol * nCols + curBlock];\n" \
"		//locR2 = BufferMat[rotRow * nCols + curBlock];\n" \
"		\n" \
"		if (swap)\n" \
"		{\n" \
"			BufferMat_r[rotCol * nCols + curBlock] = cmult_r(rotSin_r,rotSin_i,locR2_r,locR2_i);\n" \
"			BufferMat_i[rotCol * nCols + curBlock] = cmult_i(rotSin_r,rotSin_i,locR2_r,locR2_i);\n" \
"			\n" \
"			BufferMat_r[rotRow * nCols + curBlock] = cmult_r(-rotSin_r,rotSin_i,locR1_r,locR1_i);\n" \
"			BufferMat_i[rotRow * nCols + curBlock] = cmult_i(-rotSin_r,rotSin_i,locR1_r,locR1_i);\n" \
"		}\n" \
"		else if (skiprot)\n" \
"		{\n" \
"			BufferMat_r[rotCol * nCols + curBlock] = cmult_r(rotCos_r,rotCos_i,locR1_r,locR1_i);\n" \
"			BufferMat_i[rotCol * nCols + curBlock] = cmult_i(rotCos_r,rotCos_i,locR1_r,locR1_i);\n" \
"			\n" \
"			BufferMat_r[rotRow * nCols + curBlock] = cmult_r(rotCos_r,-rotCos_i,locR2_r,locR2_i);\n" \
"			BufferMat_i[rotRow * nCols + curBlock] = cmult_i(rotCos_r,-rotCos_i,locR2_r,locR2_i);\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			BufferMat_r[rotCol * nCols + curBlock] = cmult_r(rotCos_r,rotCos_i,locR1_r,locR1_i) + cmult_r(rotSin_r,rotSin_i,locR2_r,locR2_i);\n" \
"			BufferMat_i[rotCol * nCols + curBlock] = cmult_i(rotCos_r,rotCos_i,locR1_r,locR1_i) + cmult_i(rotSin_r,rotSin_i,locR2_r,locR2_i);\n" \
"			\n" \
"			BufferMat_r[rotRow * nCols + curBlock] = cmult_r(rotCos_r,-rotCos_i,locR2_r,locR2_i) + cmult_r(-rotSin_r,rotSin_i,locR1_r,locR1_i);\n" \
"			BufferMat_i[rotRow * nCols + curBlock] = cmult_i(rotCos_r,-rotCos_i,locR2_r,locR2_i) + cmult_i(-rotSin_r,rotSin_i,locR1_r,locR1_i);\n" \
"		}\n" \
"		/*		\n" \
"		if (swap)\n" \
"		{\n" \
"			BufferMat[rotCol * nCols + curBlock] = rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + curBlock] = -rotSin * locR1;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			BufferMat[rotCol * nCols + curBlock] = rotCos * locR1 + rotSin * locR2;\n" \
"		\n" \
"			BufferMat[rotRow * nCols + curBlock] = rotCos * locR2 - rotSin * locR1;\n" \
"		}\n" \
"		*/\n" \
"	}\n" \
"	\n" \
"	//barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	\n" \
"\n" \
"	\n" \
"}\n";



#endif