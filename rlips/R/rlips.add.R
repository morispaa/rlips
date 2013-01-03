## file:rlips.add.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



## Add data to RLIPS problem
## Arguments:
##
##	e		Initialized rlips environment
##	A.data	Theory matrix data. Given as a matrix OR as a vector (row-major format)
##	M.data	Measurement matrix data. Given as a matrix OR as a row-major vector
##	E.data	Measurement error covariance matrix, OR vector giving error variances OR
##			single number giving constant error variance for all measurements.
##			Default is E.data==1, i.e. errors do not have any effect to the result. 
rlips.add <- function(e,A.data,M.data,E.data=1)
{ 
	# Make sure that we are using an active environment
	if (!e$active)
	{
		stop("Not an active rlips environment! Nothing done!")
	}
  
	## Check that the arguments are in right form
	
	## Is error given as a matrix
	if (is.matrix(E.data))
	{
		Emat=TRUE
	}
	else
	{
		Emat=FALSE
	}
	
	## Check given theory matrix data

	if (is.vector(A.data))
	## If given as a vector check that its length is a multiple of e$ncols 	
	{
		if (length(A.data)/e$ncols != as.integer(length(A.data)/e$ncols))
		{
			stop('rlips.add: theory matrix has wrong size!')
		}
		else
		## Calculate the number of matrix rows fed in
		{
			num.rows <- length(A.data)/e$ncols
		}
		
		## Reshape vector into a matrix
		A.data <- matrix(A.data,e$ncols,num.rows,byrow=T)
	}	
	else
	## Theory matrix data given as a matrix. Check number of columns.
	{
		if (ncol(A.data) != e$ncols)
		{
			stop('rlips.add: theory matrix has wrong number of columns!')
		}
		else
		{
			num.rows <- nrow(A.data)
		}
	}
	
	## Check given measurement matrix

	if (is.vector(M.data))
	## If given as a vector, check that its length is right (given the rows in A.data)
	{
		if (length(M.data) != e$nrhs * num.rows)
		{
			stop('rlips.add: measurement vector has wrong size!')
		}
		else
		{
			## reshape as matrix
			M.data <- matrix(M.data,e$nrhs,num.rows,by.row=T)
		}
	}
	else
	## If given as a matrix, check its dimensions
	{
		if (!all(c(num.rows,e$nrhs)==dim(M.data)))
		{
			stop('rlips.add: measurement matrix has wrong shape!')
		}
	}
	

	## Check given error data
	
	if (Emat)
	## If errors are given as a covariance matrix, check its dimensions.
	## NB: Positive definitiveness or symmetricity is not checked!
	{
		if (!all(c(num.rows,num.rows)==dim(E.data)))
		{
			stop('rlips.add: error covariance matrix has wrong size!')
		}

		## Multiply theory matrix and measurement data with the inverse of
		## the Cholesky factor of the error covariance matrix
		## (Whitening of the noise)
		C <- chol(E.data)
		A.data <- backsolve(C,A.data,transpose=T)
		M.data <- backsolve(C,M.data,transpose=T)
		
	}
	else
	## Errors given as a vector or a scalar value
	{
		if (length(E.data)==1)
		## If error variance is given as a single number, expand it to a vector.
		{
			E.data <- rep(E.data,num.rows)
		}
		else
		## if given as a vector, check its length.
		{
			if ( length(E.data) != num.rows)
			{
				stop('rlips.add: error vector has wrong length!')
			}
		}
		
		## Divide both theory matrix rows and measurements by the square root of
		## the inverse of the variance (whitening the noise)
		A.data <- diag(1/sqrt(E.data)) %*% A.data
		M.data <- diag(1/sqrt(E.data)) %*% M.data
	}
  
	## Move data to buffer row by row
	for (i in seq(num.rows))
	{
		e$buffer[e$brows+1,1:e$ncols] <- A.data[i,]
		e$buffer[e$brows+1,(e$ncols+1):(e$ncols+e$nrhs)] <- M.data[i,]
		
		e$brows <- e$brows + 1
		
		# If buffer becomes full, rotate it
		if (e$brows >= e$nbuf)
		{
			rlips.rotate(e)
		}
	}



}
