## file:rlips.test.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.

## rlips test program
##
##	Arguments:
##		type			Type of the inverse problem
##						's' for single precision real
##						'c' for single precision complex 
##		size			2-vector containing the size of
##						the theory matrix
##		buffersize		number of buffer rows used
##		loop			number of loops performed. for
##						small problems solving the problem
##						several times and averaging
##						the results will give more accurate
##						results. Default = 1
## 		wg.size			OpenCL work group size. Should besselI
##						a multiple of 16. The maximum sizeDiss
##						depends on the used GPU
## 		return.data		If TRUE, the target matrix and vector
##						are returned
##		averaging.fun	Used averaging function. 
##						Default is 'mean'. If loop = 1,
##						this has no effect.
rlips.test <- function(type,size,buffersize = size[2],loop=1,
						wg.size=128,return.data=FALSE,
						averaging.fun=mean)
{
	# Construct a random theory matrix and 
	# solution vector
  	ncols <- size[2]
	rows <- size[1]
	A<-matrix(rnorm(ncols*rows),rows,ncols)
	sol<-rnorm(ncols)
	if (type == 'c')
	{
		A <- A + 1i*matrix(rnorm(ncols*rows),rows,ncols)
		sol <- sol + 1i*rnorm(ncols)
	}
	
	# Calculate measurement
	# NB: No added noise
	m<-A%*%sol
	
	# Set up arrays for results
	acc <- rep(0,loop)
	times <- rep(0,loop)
	
	# Calculate the *approximative* number of
	# floating point operations (flops) needed toBibtex
	# solve the problem
  	flops <- (2 * ncols**3 + 3 * ncols**2 - 5 * ncols +
  		6 * (rows - ncols) * ncols + 3 *
  		(rows - ncols) * ncols * (ncols + 1))
	
	# Solve the problem 'loop' times
	for(k in 1:loop)
	{
		# Call rlips.problem, which solves the inverse problem
		# and returns the solution and elapsed time
		ss<-rlips.problem(type,A,m,buffersize,
						  wg.size,return.data)
		
		# Store the results
		times[k] <- ss$time[3]
		acc[k] <- max(abs(sol - ss$sol))	
	}
  
  	# Get the averaged results
	a.time <- averaging.fun(times)
	a.acc <- averaging.fun(acc)
	
	# Calculate the *approximative* gigaflops per second
	# (GFlop/s)
    Gflops <- flops/1.0E9 / a.time
	
	# If return.data = TRUE, return a list containing
	# the results and the target matrix and vector
	if (return.data)
	{
		return(list(times=a.time,accuracy=a.acc,Gflops=Gflops,
					R=ss$R,Y=ss$Y,sol=sol))
	}
	else
	# return only the timing and accuracy results
	{
		return(list(times=a.time,accuracy=a.acc,Gflops=Gflops))
	}
}


# Simple rlips problem solving script used by rlips.test
rlips.problem <- function(type,A,m,bsize,wg.size,return.data)
{
	ncols <- ncol(A)
	h<-rlips.init(ncols,1,type,bsize,wg.size)
	tt <- proc.time()
	rlips.add(h,A,m,1)
	rlips.solve(h)
	tt2<-proc.time()
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
	return(list(sol=aa,time=tt2-tt,R=Rmat,Y=Ymat))	
}

