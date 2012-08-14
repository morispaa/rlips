## file:rlips.test.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



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
		return(list(times=a.time,accuracy=a.acc,Gflops=Gflops,R=ss$R,Y=ss$Y,sol=sol))
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

