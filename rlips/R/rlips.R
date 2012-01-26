## file: ocllips.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



## Init RLIPS object
rlips.init <- function(ncols,nrhs,type='s',nbuf=100,workgroup.size=64) 
{
	e <- new.env()
	
	e$ref <- c(0,0)
	e$ncols <- ncols
	e$nrhs <- nrhs
  if (nbuf > ncols)
  {
    cat("nbuf too large! Setting nbuf to ",ncols,"\n")
    e$nbuf <- ncols
  }
  else
  {
    e$nbuf <- nbuf  
  }

  if (workgroup.size%%16 != 0)
  {
    cat("Warning! Workgroup size REALLY should be multiple of 16, but whatever...\n")
  }
  e$wg.size <- workgroup.size
	
  e$type <- type ## Other types not yet implemented
	
  if (type != 's' && type != 'c')
  {
    stop('Only single precission real type implemented! Exiting!')
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
		e$ref <- .C("sInitOcllips",
					aa=integer(2),
					as.integer(ncols),
					as.integer(nrhs),
					as.integer(e$nbuf),
          as.integer(e$wg.size))$aa
	}
	else if (type == 'c')
	{
		e$ref <- .C("cInitOcllips",
					aa=integer(2),
					as.integer(ncols),
					as.integer(nrhs),
					as.integer(e$nbuf),
          as.integer(e$wg.size))$aa
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
				as.integer(e$ref))
		}
		else if (e$type == 'c')
		{
			.C("cKillOcllips",
				as.integer(e$ref))
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
  
	# Make sure all data are matrices
	A.data <-as.matrix(A.data)
	M.data <-as.matrix(M.data)
	E.data <-as.matrix(E.data)
	
	# Data dimensions & check
	A.dim <- dim(A.data)
	M.dim <- dim(M.data)
	E.dim <- dim(E.data)
	
	data.rows <- A.dim[1]
	
	# Check dims of A
	if (A.dim[2] != e$ncols)
	{
		stop('Error in rlips.add: A.data has wrong number of columns!')
	}
	
	# Check dims of M
	if (all(M.dim != c(data.rows,e$nrhs)))
	{
		stop('Error in rlips.add: M.data has wrong size!')
	}
	
	## cat(all(E.dim==c(1,1)),'\n')
	
	# Check dims of E and get rid of it.
	# It can be either single scalar, vector or matrix
	if (all(E.dim == c(1,1)))
	{
		# Single scalar error variance
		if (E.data != 1)
		{
			err <- 1/sqrt(as.vector(E.data))
			A.data <- as.vector(err) * A.data
			M.data <- as.vector(err) * M.data
		}
	}
	else if (all(E.dim == c(data.rows,1)))
	{
		# Error variances given as vector, i.e., diagonal of
		# error covariance matrix
		err <- 1/sqrt(err)
		A.data <- diag(as.vector(err)) %*% A.data
		M.data <- as.vector(err) * M.data
	}
	else if (all(E.dim == c(data.rows,data.rows)))
	{
		# Full error covariance matrix
		C <- chol(E.data)
		A.data <- backsolve(C,A.data,transpose=T)
		M.data <- backsolve(C,M.data,transpose=T)
	}
	else
	{
		stop('Error in rlips.add: E.data has wrong size!')
	}
	
	#cat("rotation init: ",proc.time()-ttt,"\n")	
    

 #  #ttt<-proc.time()
#   #Join A and M and insert in buffer
#   buffer <- matrix(0,data.rows,e$buffer.cols)
#   buffer[,1:e$ncols] <- A.data
#   buffer[,(e$ncols+1):(e$ncols+e$nrhs)] <- M.data
#    
#   # Add new buffer rows to buffer
#   if (e$brows > 0)
#   {
#   e$buffer <- rbind(e$buffer,buffer)
#   }
#   else
#   {
#   	e$buffer <- buffer
#   }
#   
#   e$brows <- e$brows + data.rows
#   
#   #cat("rotation joining: ",proc.time()-ttt,"\n")

	# Move data to buffer
	for (i in seq(data.rows))
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


  # #ttt<-proc.time()
#   # Make rotations if necessary
#   loops <- floor(e$brows/e$nbuf)
#   #cat("loops: ",loops,"\n")
#   #if (loops > 0)
#   #{
#   for (q in 1:loops)
#   #while (e$brows >= e$nbuf)
#   {
#   	#q<-1
#   	start.row <- (q-1) * e$nbuf + 1
#   	end.row <- q * e$nbuf
#   	#qqq<-proc.time()
#   	data <- matrix(t(e$buffer[start.row:end.row,]),e$nbuf*e$buffer.cols)
#   	#cat("buffer manipulation: ",proc.time()-qqq,"\n")
#   	
#     ## rotate first nbuf rows
#     if (e$type == 's')
#     {
#     	.C("sRotateOcllips",
#     		as.integer(e$ref),
#     		as.double(data),
#     		as.integer(e$nbuf))
#     }
#     else if (e$type == 'c')
#     {
#     	.C("cRotateOcllips",
#     		as.integer(e$ref),
#     		as.double(Re(data)),
#     		as.double(Im(data)),
#     		as.integer(e$nbuf))
#     }	
#     #cat("Rotation:\n",tt,"\n")
#     
#     #qqq<-proc.time()
#     #e$buffer <- e$buffer[-(1:e$nbuf),]
#     #cat("buffer manipulation: ",proc.time()-qqq,"\n")
#     e$brows <- e$brows - e$nbuf
#   #}
#   }
#   #cat("rotation loop: ",proc.time()-ttt,"\n")
#   
#   e$buffer <- e$buffer[-(1:(e$nbuf*loops)),]
#   
#   # Update internal parameters
#   e$nrows <- e$nrows + data.rows
#   e$rrows <- min(e$ncols,e$rrows + data.rows)
#   

}

## Do the rotations
rlips.rotate <- function(e)
{
	#cat("Going to rotate ",e$brows," buffer rows\n",sep="")

  if (e$brows > e$nbuf)
  {
    stop("Something fishy going on? Environment should not have this many buffer rows at this point! Doin' nuthin'!")
  }
	if (e$brows > 0)
	{
		data <- matrix(t(e$buffer[1:e$brows,]),e$brows*e$buffer.cols)
  	
    	## rotate first nbuf rows
    	if (e$type == 's')
    	{
    		.C("sRotateOcllips",
    			as.integer(e$ref),
    			as.single(data),
    			as.integer(e$brows))
    	}
    	else if (e$type == 'c')
    	{
    		.C("cRotateOcllips",
    			as.integer(e$ref),
    			as.double(Re(data)),
    			as.double(Im(data)),		
    			as.integer(e$brows))
    	}
    	
    	e$buffer <- matrix(0,e$nbuf,e$buffer.cols)
    	e$nrows <- e$nrows + e$browse.pkgs
    	e$rrows <- min(e$ncols,e$nrows)
    	e$brows <- 0
    	
	}


}

## Solve system

rlips.solve <- function(e,calculate.covariance=FALSE)
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
    	if (e$type == 's')
    	{
      		e$covariance <- backsolve(e$R.mat,diag(rep(1,e$ncols)))
      		e$covariance <- e$covariance %*% t(e$covariance)
      	}
      	else if (e$type == 'c')
      	{
      		e$covariance <- solve(e$R.mat,diag(rep(1,e$ncols)))
      		e$covariance <- e$covariance %*% t(e$covariance)
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
				data = single(e$ncols * e$buffer.cols),
				data.rows = integer(1))
		#data <- res$data	
		data.mat <- matrix(as.double(res$data),e$ncols,e$buffer.cols,byrow=TRUE)	
	}
	else if (e$type == 'c')
	{
		res <- .C("cGetDataOcllips",
				as.integer(e$ref),
				data = double(e$ncols * e$buffer.cols),
				data.i = double(e$ncols * e$buffer.cols),
				data.rows = integer(1))
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


rlips.test <- function(type,size,buffersizes,loop=1,wg.size=128,return.data=FALSE)
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
	
	n<-length(buffersizes)
	acc <- rep(0,n)
	times <- rep(0,n)
  	flops <- 2 * ncols**3 + 3 * ncols**2 - 5 * ncols + 6 * (rows - ncols) * ncols + 3 * (rows - ncols) * ncols * (ncols + 1)
	
	for(k in 1:loop)
	{
	  for (i in 1:n)
	  {
	  	ss<-rlips.problem(type,A,m,buffersizes[i],wg.size,return.data)
	  	times[i] <- times[i] + ss$time[3]
	  	acc[i] <- acc[i] + max(abs(sol - ss$sol))

			
	  }
	}
  
	times <- times/loop
	acc <- acc/loop
    Gflops <- flops/1.0E9 / times
	
	if (return.data)
	{
		return(list(times=times,accuracy=acc,Gflops=Gflops,R=ss$R,Y=ss$Y))
	}
	else
	{
		return(list(times=times,accuracy=acc,Gflops=Gflops))
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

