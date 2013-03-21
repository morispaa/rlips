## file:rlips.solve.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



## Solve system
## Arguments:
##
##	e		Initialized rlips environment
##  calculate.covariance
##			Flag for covariance matrix calculation,
##			default = FALSE
## full.covariance
##			Flag for full covariance matrix
## 			calculation, if FALSE, only
##			diagonal elements (variances) 
##			are calculated. 
##			If calculate.covariance=FALSE, this
##			flag has no effect.
rlips.solve <- function(e,calculate.covariance=FALSE,
						full.covariance=FALSE)
{
	# Make sure that we are using an active environment
	if (!e$active)
	{
		stop("Not an active rlips environment! Nothing done!")
	}
	
	# If buffer has non-rotated rows, rotate them now
	if (e$brows > 1) rlips.rotate(e)
	
	# Get the problem data from the GPU
	# i.e. the target matrix R and the target vector Y
	rlips.get.data(e)
    
    # Use backsubstitution algorithm to get the solution
    # For complex backsubstitution we use our own routine
    # since it is missing in R
    if (e$type == 's')
    {
    	e$solution <- backsolve(e$R.mat,e$Y.mat)
    }
    else if (e$type == 'c')
    {
    	e$solution <- .Call("cbacksolve",
    						e$R.mat,
    						e$Y.mat,
    						PACKAGE="rlips")
    }
    
    # A posteriori covariance calculation
    if (calculate.covariance)
    {
    	# Calculate full covariance matrix
    	if (full.covariance)
    	{
    		# Real problem
    		if (e$type == 's')
    		{
    			# Calculate the inverse of target matrix R
      			e$covariance <- backsolve(
      								e$R.mat,
      								diag(rep(1,e$ncols)))
      			# Covariance = (R^(-1)) * (R^(-1))^T
      			e$covariance <- e$covariance %*% 
      							t(e$covariance)
      		}
      		# Complex problem
      		else if (e$type == 'c')
      		{
      			# Here we use solve instead of complex backsolve
      			# becouse it is not optimised for such a large 
      			# measurement vectors
      			e$covariance <- solve(
      							e$R.mat,
      							diag(rep(1,e$ncols)))
      			# Covariance = (R^(-1)) * (R^(-1))^H
      			e$covariance <- e$covariance %*% 
      							Conj(t(e$covariance))
      		}
      	}
      	# Calculate only the diagonal of the covariance matrix
      	else
      	{
      		# Real problem
      		if (e$type == 's')
    		{
      			e$iR.mat <- backsolve(e$R.mat,
      								  diag(rep(1,e$ncols)))
      			e$covariance<- rep(0,e$ncols)
      			# Calculate only diagonal terms 
      			# of the covariance matrix
      			for (i in 1:e$ncols)
      			{
      				e$covariance[i] <-
      					sum(e$iR.mat[i,i:e$ncols]**2)
      			}
      		}
      		# Complex problem
      		else if (e$type == 'c')
      		{
      			e$iR.mat <- solve(e$R.mat,diag(rep(1,e$ncols)))
      			e$covariance<- rep(0,e$ncols)
      			# Calculate only diagonal terms 
      			# of the covariance matrix
      			for (i in 1:e$ncols)
      			{
      				e$covariance[i] <- 
      					sum(e$iR.mat[i,i:e$ncols] *
      					Conj(e$iR.mat[i,i:e$ncols]))
      			}
      		}
      	}
    }
}
