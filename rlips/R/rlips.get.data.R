## file:rlips.get.data.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.

## Fetch problem data from GPU
## Arguments:
##  e		rlips environment
rlips.get.data <- function(e)
{
	if (e$brows > 0) rlips.rotate(e)
	
	# According to the problem type, call the appropriate
	# C routine which fetches the data from GPU to 
	# CPU memory
	if (e$type == 's')
	{
		data <- .Call("sGetDataRlips",
					  e$ref,
					  PACKAGE="rlips")
				
		# Reshape the received vector as a matrix
		data.mat <- matrix(data,e$ncols,e$buffer.cols,byrow=TRUE)	
	}
	else if (e$type == 'c')
	{
		data <- .Call("cGetDataRlips",
					  e$ref,
					  PACKAGE="rlips")
				
		# Transform the received real vector into a complex
		# vector
		data <- data[seq(1,2 * e$ncols * e$buffer.cols,by=2)] +
			1i*data[seq(2,2 * e$ncols * e$buffer.cols,by=2)]
			
		# Reshape as a matrix
		data.mat <- matrix(data,e$ncols,e$buffer.cols,byrow=TRUE)			
	}
	
	# Get target matrix and set its dimensions
	e$R.mat <- data.mat[,1:e$ncols]
	dim(e$R.mat)<-c(e$ncols,e$ncols)
	
	# Get target vector and set its dimensions
	e$Y.mat <- data.mat[,(e$ncols+1):(e$ncols+e$nrhs)]
	dim(e$Y.mat)<-c(e$ncols,e$nrhs)



}
## 