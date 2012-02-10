## file: ocllips.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



## Init RLIPS object
rlips.init <- function(ncols,nrhs,type='s',nbuf=ncols,workgroup.size=128) 
{
	e <- new.env()
	
	e$ref <- c(0,0)
	e$ncols <- ncols
	e$nrhs <- nrhs
  #if (nbuf > ncols)
  #{
  #  cat("nbuf too large! Setting nbuf to ",ncols,"\n")
  #  e$nbuf <- ncols
  #}
  #else
  #{
    e$nbuf <- nbuf  
  #}

  if (workgroup.size%%16 != 0)
  {
    cat("Warning! Workgroup size REALLY should be multiple of 16, but whatever...\n")
  }
  e$wg.size <- workgroup.size
	
  e$type <- type ## Other types not yet implemented
	
  if (type != 's' && type != 'c')
  {
    stop('Only single precission real and complex types implemented! Exiting!')
  }
	
	e$nrows <- 0 # Number of total rows fed into system
	e$brows <- 0 # Number of rows in the buffers
	e$rrows <- 0 # Number of rows in R matrix
    e$buffer.cols <- floor((ncols + nrhs + 32 - 1)/32) * 32 # Number of columns in the buffer matrix. 
                                                                                              # Holds both data and measurements and is a multiple of 
                                                                                              # workgroup.size.
    #e$buffer.cols <- ncols + nrhs                                                                                          
    e$buffer <- matrix(0,e$nbuf,e$buffer.cols)
	
  # At this point, only single precision real is implemented.
	if (e$type == 's')
	{
		e$ref = .Call("sInitOcllips",
			ncols,
			nrhs,
			e$nbuf,
          		e$wg.size,PACKAGE="rlips")
	}
	else if (type == 'c')
	{
		.Call("cInitOcllips",
			e$ref,
			ncols,
			nrhs,
			e$nbuf,
          		e$wg.size,PACKAGE="rlips")
	}
 	else
 	{
 		stop('rlips.init: type not recognized!')
 	}


	# We need this to ensure that we do not reallocate allocated objects
	# or deallocate already deallocated objects. Doing so would cause R to crash.
	e$active <-TRUE
	
	return(e)
}


## dispose RLIPS object
rlips.dispose <- function(e)
{
	if (e$active)
	{
		if (e$type == 's')
		{
			.C("sKillOcllips",
				as.integer(e$ref),PACKAGE="rlips")
		}
		else if (e$type == 'c')
		{
			.C("cKillOcllips",
				as.integer(e$ref),PACKAGE="rlips")
		}
		else
		{
			stop('rlips.dispose: type not recognized!')
		}
	}
	else
	{
		cat('Warning rlips.dispose: not an active RLIPS environment!\n','Nothing done!\n',sep='')
	}
	
	e$active <- FALSE
	
	
	
}



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

## Do the rotations
rlips.rotate <- function(e)
{
	#cat("Going to rotate ",e$brows," buffer rows\n",sep="")

  #if (e$brows > e$nbuf)
  #{
  #  stop("Something fishy going on? Environment should not have this many buffer rows at this point! Doin' nuthin'!")
  #}
	if (e$brows > 0)
	{
		#rlips.data <- matrix(t(e$buffer[1:e$brows,]),e$brows*e$buffer.cols)
  	
    	## rotate first nbuf rows
    	if (e$type == 's')
    	{
    		.Call("sRotateOcllips",
    			e$ref,
    			t(e$buffer[1:e$brows,]),
    			e$brows,
    			PACKAGE="rlips")
    	}
    	else if (e$type == 'c')
    	{
    		.C("cRotateOcllips",
    			as.integer(e$ref),
    			as.double(Re(rlips.data)),
    			as.double(Im(rlips.data)),		
    			as.integer(e$brows),PACKAGE="rlips")
    	}
    	
    	e$buffer <- matrix(0,e$nbuf,e$buffer.cols)
    	e$nrows <- e$nrows + e$browse.pkgs
    	e$rrows <- min(e$ncols,e$nrows)
    	e$brows <- 0
    	
	}


}

## Solve system

rlips.solve <- function(e,calculate.covariance=FALSE,full.covariance=FALSE)
{
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
    	e$solution <- solve(e$R.mat,e$Y.mat)
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


rlips.get.data <- function(e)
{
	if (e$brows > 0) rlips.rotate(e)
	
	if (e$type == 's')
	{
		res <- .C("sGetDataOcllips",
				as.integer(e$ref),
				data = double(e$ncols * e$buffer.cols),
				data.rows = integer(1),PACKAGE="rlips")
		#data <- res$data	
		data.mat <- matrix(res$data,e$ncols,e$buffer.cols,byrow=TRUE)	
	}
	else if (e$type == 'c')
	{
		res <- .C("cGetDataOcllips",
				as.integer(e$ref),
				data = double(e$ncols * e$buffer.cols),
				data.i = double(e$ncols * e$buffer.cols),
				data.rows = integer(1),PACKAGE="rlips")
		data.mat <- matrix(res$data + 1i*res$data.i,e$ncols,e$buffer.cols,byrow=TRUE)			
	}
	
	#tt<-proc.time()
	#data.mat <- matrix(data,e$ncols,e$buffer.cols,byrow=TRUE)
			
	e$R.mat <- data.mat[,1:e$ncols]
	e$Y.mat <- data.mat[,(e$ncols+1):(e$ncols+e$nrhs)]
	e$ddd.rows <- res$data.rows
	#cat("Matrix manipulation: ",proc.time()-tt,"\n")
	


}
## 


rlips.test <- function(type,size,buffersize = size[2],loop=1,wg.size=128,return.data=FALSE,averaging.fun=mean)
{
  ncols <- size[2]
	rows <- size[1]
	A<-matrix(rnorm(ncols*rows),rows,ncols)
	sol<-rnorm(ncols)
	if (type == 'c')
	{
		A <- A + 1i*matrix(rnorm(ncols*rows),rows,ncols)
		sol <- sol + 1i*rnorm(ncols)
	}
	#if (type=='c' || type=='z')
	#{
	#	A <- A + 1i*matrix(rnorm(ncols*rows),rows,ncols)
	#}
	
	m<-A%*%sol
	
	#n<-length(buffersize)
	acc <- rep(0,loop)
	times <- rep(0,loop)
  	flops <- 2 * ncols**3 + 3 * ncols**2 - 5 * ncols + 6 * (rows - ncols) * ncols + 3 * (rows - ncols) * ncols * (ncols + 1)
	
	for(k in 1:loop)
	{
	  ss<-rlips.problem(type,A,m,buffersize,wg.size,return.data)
	  times[k] <- ss$time[3]
	  acc[k] <- max(abs(sol - ss$sol))	
	}
  
	a.time <- averaging.fun(times)
	a.acc <- averaging.fun(acc)
    Gflops <- flops/1.0E9 / a.time
	
	if (return.data)
	{
		return(list(times=a.time,accuracy=a.acc,Gflops=Gflops,R=ss$R,Y=ss$Y))
	}
	else
	{
		return(list(times=a.time,accuracy=a.acc,Gflops=Gflops))
	}
}

rlips.problem <- function(type,A,m,bsize,wg.size,return.data)
{
	ncols <- ncol(A)
	h<-rlips.init(ncols,1,type,bsize,wg.size)
	tt <- proc.time()
	rlips.add(h,A,m,1)
	rlips.solve(h)
	tt2<-proc.time()
	#cat("LIPS time:",tt2-tt,"\n")
	aa <- h$solution
	Rmat<-0
	Ymat<-0
	if(return.data)
	{
		rlips.get.data(h)
		Rmat<-h$R.mat
		Ymat<-h$Y.mat
	}
	rlips.dispose(h)
	#cat(' init:',t1,"\n",sep=" ")
	#cat('  add:',t2,"\n",sep=" ")
	#cat('solve:',t3,"\n",sep=" ")
	return(list(sol=aa,time=tt2-tt,R=Rmat,Y=Ymat))	
}

