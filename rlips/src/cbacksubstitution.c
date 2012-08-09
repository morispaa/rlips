// Complex back substitution algorithm, which is missing
// from R for some reason

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<R.h>
#include<Rinternals.h>

#include "rlips.h"

#define CPLX_MULT_R(ar,ai,br,bi) ((ar)*(br) - (ai)*(bi))
#define CPLX_MULT_I(ar,ai,br,bi) ((ar)*(bi) + (ai)*(br))


SEXP cbacksolve(SEXP R, SEXP Y)
{
	int i,j,k;
	SEXP res;
	//R = coerceVector(R,CPLXSXP);
	//Y = coerceVector(Y,CPLXSXP);
	
	Rcomplex *rr = COMPLEX(R);
	Rcomplex *yy = COMPLEX(Y);
	
	// get the dimension of R and Y
	SEXP Rdim = getAttrib(R,R_DimSymbol);
	SEXP Ydim = getAttrib(Y,R_DimSymbol);
	int I = INTEGER(Ydim)[0];
	int J = INTEGER(Ydim)[1];
	
	
	// Allocate the result matrix
	PROTECT(res = allocMatrix(CPLXSXP,I,J));
	
	// Copy Y to res
	for (i = 0; i < I * J ; i++)
	{
		
		COMPLEX(res)[i].r = yy[i].r;
		COMPLEX(res)[i].i = yy[i].i;
	}
	
	//COMPLEX(res)[0].r = yy[0].r;
	//COMPLEX(res)[0].i = yy[0].i;
	
	//double *tmp = malloc(J * sizeof(double));
	
	double Ur,Ui;
	
	for (i = I - 1 ; i >= 1; i--)
	{
		Ur = rr[i + i * I].r;
		Ui = rr[i + i * I].i;
		
		for (k = 0 ; k < J ; k++)
		{
			COMPLEX(res)[i + k * I].r = (CPLX_MULT_R(
								COMPLEX(res)[i + k * I].r,
								COMPLEX(res)[i + k * I].i,
								Ur,
								-Ui
								))/(Ur*Ur + Ui*Ui);
			COMPLEX(res)[i + k * I].i =	(CPLX_MULT_I(
								COMPLEX(res)[i + k * I].r,
								COMPLEX(res)[i + k * I].i,
								Ur,
								-Ui
								))/(Ur*Ur + Ui*Ui);	
			
			for (j = 0 ; j < i ; j++)
			{
				COMPLEX(res)[j + k * I].r = COMPLEX(res)[j + k * I].r
						- (CPLX_MULT_R(
								COMPLEX(res)[i + k * I].r,
								COMPLEX(res)[i + k * I].i,
								rr[j + i * I].r,
								rr[j + i * I].i
								));
				
				
				COMPLEX(res)[j + k * I].i = COMPLEX(res)[j + k * I].i
						- (CPLX_MULT_I(
								COMPLEX(res)[i + k * I].r,
								COMPLEX(res)[i + k * I].i,
								rr[j + i * I].r,
								rr[j + i * I].i
								));
				
			}
				
		}
	}
	
	for (k = 0; k < J ; k++)
	{
		COMPLEX(res)[k * I].r = (CPLX_MULT_R(
							COMPLEX(res)[k * I].r,
							COMPLEX(res)[k * I].i,
							rr[0].r,
							-rr[0].i
							))/(rr[0].r * rr[0].r + rr[0].i * rr[0].i);
		
		COMPLEX(res)[k * I].i = (CPLX_MULT_I(
							COMPLEX(res)[k * I].r,
							COMPLEX(res)[k * I].i,
							rr[0].r,
							-rr[0].i
							))/(rr[0].r * rr[0].r + rr[0].i * rr[0].i);
	}
	
	UNPROTECT(1);
	
	return res;
	
	
	//return R_NilValue;
}