## file:rlips.solve.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



## Solve system

rlips.solve <- function(e,calculate.covariance=FALSE,full.covariance=FALSE)
{
	# Make sure that we are using an active environment
	if (!e$active)
	{
		stop("Not an active rlips environment! Nothing done!")
	}
	
	if (e$brows > 1) rlips.rotate(e)
	
	if (e$active)
	{
		rlips.get.data(e)
    
    if (e$type == 's')
    {
    	e$solution <- backsolve(e$R.mat,e$Y.mat)
    }
    else if (e$type == 'c')
    {
    	
    	#e$solution <- solve(e$R.mat,e$Y.mat)
    	e$solution <- .Call("cbacksolve",
    						e$R.mat,
    						e$Y.mat,
    						PACKAGE="rlips")
    }
    
    if (calculate.covariance)
    {
    	if (full.covariance)
    	{
    		if (e$type == 's')
    		{
      			e$covariance <- backsolve(e$R.mat,diag(rep(1,e$ncols)))
      			e$covariance <- e$covariance %*% t(e$covariance)
      		}
      		else if (e$type == 'c')
      		{
      			# Here we use solve instead of complex backsolve
      			# becouse it is not optimised for such a large measurement
      			# vectors
      			e$covariance <- solve(e$R.mat,diag(rep(1,e$ncols)))
      			e$covariance <- e$covariance %*% Conj(t(e$covariance))
      		}
      	}
      	else
      	{
      		if (e$type == 's')
    		{
      			e$iR.mat <- backsolve(e$R.mat,diag(rep(1,e$ncols)))
      			e$covariance<- rep(0,e$ncols)
      			for (i in 1:e$ncols)
      			{
      				e$covariance[i] <- sum(e$iR.mat[i,i:e$ncols]**2)
      			}
      		}
      		else if (e$type == 'c')
      		{
      			e$iR.mat <- solve(e$R.mat,diag(rep(1,e$ncols)))
      			e$covariance<- rep(0,e$ncols)
      			for (i in 1:e$ncols)
      			{
      				e$covariance[i] <- sum(e$iR.mat[i,i:e$ncols]*Conj(e$iR.mat[i,i:e$ncols]))
      			}
      		}
      	}
    }
    
	}
}
