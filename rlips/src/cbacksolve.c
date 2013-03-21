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

// Complex back substitution algorithm, which is missing
// from R for some reason

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<R.h>
#include<Rinternals.h>

#include "rlips.h"

// Define macros for complex multiplication
#define CPLX_MULT_R(ar,ai,br,bi) ((ar)*(br) - (ai)*(bi))
#define CPLX_MULT_I(ar,ai,br,bi) ((ar)*(bi) + (ai)*(br))


/*
cbacksolve
==========

Backsubstitution solver for complex data

Arguments

R		Complex upper triangular matrix
Y		Complex matrix

Returns

res		Complex matrix

*/
SEXP cbacksolve(SEXP R, SEXP Y)
{
	int i,j,k;
	
	SEXP res;

	// Allocate R data in C side
	Rcomplex *rr = COMPLEX(R);
	Rcomplex *yy = COMPLEX(Y);
	
	// get the dimensions of R and Y
	SEXP Rdim = getAttrib(R,R_DimSymbol);
	SEXP Ydim = getAttrib(Y,R_DimSymbol);
	
	// I 	number of rows in Y
	// J 	number of cols in Y
	int I = INTEGER(Ydim)[0];
	int J = INTEGER(Ydim)[1];
	
	// Allocate and protect the result matrix
	PROTECT(res = allocMatrix(CPLXSXP,I,J));
	
	// Copy matrix Y to array res
	for (i = 0; i < I * J ; i++)
	{
		
		COMPLEX(res)[i].r = yy[i].r;
		COMPLEX(res)[i].i = yy[i].i;
	}
	
	// Auxiliary variables	
	double Ur,Ui,tmp;
	
	// This is standard (complex) 
	// backsubstitution algorithm
	// Starting from the last row, go up to
	// the second row
	for (i = I - 1 ; i >= 1; i--)
	{
		Ur = rr[i + i * I].r;
		Ui = rr[i + i * I].i;
		
		// Handle every column of Y separately
		for (k = 0 ; k < J ; k++)
		{
			tmp = COMPLEX(res)[i + k * I].r;
			COMPLEX(res)[i + k * I].r = (CPLX_MULT_R(
								COMPLEX(res)[i + k * I].r,
								COMPLEX(res)[i + k * I].i,
								Ur,
								-Ui
								))/(Ur*Ur + Ui*Ui);
								
			COMPLEX(res)[i + k * I].i =	(CPLX_MULT_I(
								tmp,
								COMPLEX(res)[i + k * I].i,
								Ur,
								-Ui
								))/(Ur*Ur + Ui*Ui);	
			
			for (j = 0 ; j < i ; j++)
			{
				COMPLEX(res)[j + k * I].r 
					= COMPLEX(res)[j + k * I].r
					- (CPLX_MULT_R(
							COMPLEX(res)[i + k * I].r,
							COMPLEX(res)[i + k * I].i,
							rr[j + i * I].r,
							rr[j + i * I].i
							));
				
				
				COMPLEX(res)[j + k * I].i 
					= COMPLEX(res)[j + k * I].i
					- (CPLX_MULT_I(
							COMPLEX(res)[i + k * I].r,
							COMPLEX(res)[i + k * I].i,
							rr[j + i * I].r,
							rr[j + i * I].i
							));
			}
		}
	}
	
	// Handle the first row separately
	for (k = 0; k < J ; k++)
	{
		tmp = COMPLEX(res)[k * I].r;
		COMPLEX(res)[k * I].r = (CPLX_MULT_R(
							COMPLEX(res)[k * I].r,
							COMPLEX(res)[k * I].i,
							rr[0].r,
							-rr[0].i
							))/(rr[0].r * rr[0].r 
							    + rr[0].i * rr[0].i);
		
		COMPLEX(res)[k * I].i = (CPLX_MULT_I(
							tmp,
							COMPLEX(res)[k * I].i,
							rr[0].r,
							-rr[0].i
							))/(rr[0].r * rr[0].r 
							    + rr[0].i * rr[0].i);
	}
	
	// Unprotect the result and return it
	UNPROTECT(1);	
	return res;
}