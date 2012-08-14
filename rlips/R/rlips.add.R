## file:rlips.add.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



## Add data 
rlips.add <- function(e,A.data,M.data,E.data=1)
{
  
  #ttt <- proc.time()
  # Make sure that we are using an active environment
  if (!e$active)
  {
    stop("Not an active rlips environment! Nothing done!")
  }
  
  ## Is error given as a matrix
	if (is.matrix(E.data))
	{
		Emat=TRUE
	}
	else
	{
		Emat=FALSE
	}
	
	## theory matrix
	if (is.vector(A.data))
	{
		if (length(A.data)/e$ncols != as.integer(length(A.data)/e$ncols))
		{
			stop('rlips.add: theory matrix has wrong size!')
		}
		else
		{
			num.rows <- length(A.data)/e$ncols
		}
		
		## Reshape into a matrix
		A.data <- t(matrix(A.data,e$ncols,num.rows))
	}	
	else
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
	
	## measurement

	
	if (is.vector(M.data))
	{
		if (length(M.data) != e$nrhs * num.rows)
		{
			stop('rlips.add: measurement vector has wrong size!')
		}
		else
		{
			## reshape as matrix
			M.data <- t(matrix(M.data,e$nrhs,num.rows))
		}
	}
	else
	{
		if (!all(c(num.rows,e$nrhs)==dim(M.data)))
		{
			stop('rlips.add: measurement matrix has wrong shape!')
		}
	}
	

	
	## error
	if (Emat)
	{
		if (!all(c(num.rows,num.rows)==dim(E.data)))
		{
			stop('rlips.add: error covariance matrix has wrong size!')
		}
		else
		{
			C <- chol(E.data)
			A.data <- backsolve(C,A.data,transpose=T)
			M.data <- backsolve(C,M.data,transpose=T)
			E.data <- rep(1.0,num.rows)	
		}	
	}
	else
	{
		if (length(E.data)==1)
		{
			E.data <- rep(E.data,num.rows)
		}
		else
		{
			if ( length(E.data) != num.rows)
			{
				stop('rlips.add: error vector has wrong length!')
			}
		}
	}
  
  
  
#	# Make sure all data are matrices
#	A.data <-as.matrix(A.data)
#	M.data <-as.matrix(M.data)
#	E.data <-as.matrix(E.data)
#	
#	# Data dimensions & check
#	A.dim <- dim(A.data)
#	M.dim <- dim(M.data)
#	E.dim <- dim(E.data)
#	
#	data.rows <- A.dim[1]
#	
#	# Check dims of A
#	if (A.dim[2] != e$ncols)
#	{
#		stop('Error in rlips.add: A.data has wrong number of columns!')
#	}
#	
#	# Check dims of M
#	if (all(M.dim != c(data.rows,e$nrhs)))
#	{
#		stop('Error in rlips.add: M.data has wrong size!')
#	}
#	
#	## cat(all(E.dim==c(1,1)),'\n')
#	
#	# Check dims of E and get rid of it.
#	# It can be either single scalar, vector or matrix
#	if (all(E.dim == c(1,1)))
#	{
#		# Single scalar error variance
#		if (E.data != 1)
#		{
#			err <- 1/sqrt(as.vector(E.data))
#			A.data <- as.vector(err) * A.data
#			M.data <- as.vector(err) * M.data
#		}
#	}
#	else if (all(E.dim == c(data.rows,1)))
#	{
#		# Error variances given as vector, i.e., diagonal of
#		# error covariance matrix
#		err <- 1/sqrt(err)
#		A.data <- diag(as.vector(err)) %*% A.data
#		M.data <- as.vector(err) * M.data
#	}
#	else if (all(E.dim == c(data.rows,data.rows)))
#	{
#		# Full error covariance matrix
#		C <- chol(E.data)
#		A.data <- backsolve(C,A.data,transpose=T)
#		M.data <- backsolve(C,M.data,transpose=T)
#	}
#	else
#	{
#		stop('Error in rlips.add: E.data has wrong size!')
#	}
	
	#cat("rotation init: ",proc.time()-ttt,"\n")	
    


	# Move data to buffer
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
