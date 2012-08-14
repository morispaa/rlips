## file:rlips.get.data.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.




rlips.get.data <- function(e)
{
	if (e$brows > 0) rlips.rotate(e)
	
	if (e$type == 's')
	{
		data <- .Call("sGetDataOcllips",
				e$ref,
				PACKAGE="rlips")
		#data <- res$data	
		data.mat <- matrix(data,e$ncols,e$buffer.cols,byrow=TRUE)	
	}
	else if (e$type == 'c')
	{
		data <- .Call("cGetDataOcllips",
				e$ref,
				PACKAGE="rlips")
				#cat(length(data),'\n')
		data <- data[seq(1,2 * e$ncols * e$buffer.cols,by=2)] + 1i*data[seq(2,2 * e$ncols * e$buffer.cols,by=2)]
		data.mat <- matrix(data,e$ncols,e$buffer.cols,byrow=TRUE)			
	}
	
	#tt<-proc.time()
	#data.mat <- matrix(data,e$ncols,e$buffer.cols,byrow=TRUE)
			
	e$R.mat <- data.mat[,1:e$ncols]
	dim(e$R.mat)<-c(e$ncols,e$ncols)
	e$Y.mat <- data.mat[,(e$ncols+1):(e$ncols+e$nrhs)]
	dim(e$Y.mat)<-c(e$ncols,e$nrhs)
	#e$ddd.rows <- res$data.rows
	#cat("Matrix manipulation: ",proc.time()-tt,"\n")
	


}
## 