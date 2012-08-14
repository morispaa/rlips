## file:rlips.init.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.




## Initialize RLIPS object
## Arguments:
##	ncols		number of unknowns/columns in the theory matrix
##	nrhs		number of measurements/columns in the measurement vector/matrix
##	type		problem type; 's' for single precision real, 
##				'c' for single precision complex
##	nbuf		data buffer rows
##	workgroup.size
##				OpenCL workgroup size, best choice depends on the used GPU
rlips.init <- function(ncols,nrhs,type='s',nbuf=ncols,workgroup.size=128) 
{
	# Create new environment for the problem data
	e <- new.env()
	
	# Set problem parameters. 
	# e$ref holds two integers that are used to form the C pointer for this
	# problem.
	e$ref <- c(0,0)
	e$ncols <- ncols
	e$nrhs <- nrhs
    e$nbuf <- nbuf  

	# Check workgroup size. Usually, it should be a multiple of 16. The optimal
	# size depends on the used GPU.
	if (workgroup.size%%16 != 0)
	{
		warning("Workgroup size REALLY should be multiple of 16!")
	}
	e$wg.size <- workgroup.size

	e$type <- type
	
	# Check type parameter. At this point, only single precision real and
	# complex problems are implemented. Also, only high-end GPU's are capable
	# to use double precision.	
	if (type != 's' && type != 'c')
	{
		stop('Only single precission real and complex types implemented! Exiting!')
	}
	
	# Initialize some problem parameters
	e$nrows <- 0 # Number of total rows fed into system
	e$brows <- 0 # Number of rows in the buffers
	e$rrows <- 0 # Number of rows in R matrix
	
	# Number of columns in the buffer matrix.
	# Holds both data and measurements and is a multiple of
	# workgroup size.
	#e$buffer.cols <- floor((ncols + nrhs + e$wg.size - 1)/e$wg.size) * e$wg.size  
	e$buffer.cols <- floor((ncols + nrhs + 32 - 1)/32) * 32  
                                                                                               
                                                                                              
	#e$buffer.cols <- ncols + nrhs                                                                                          
	e$buffer <- matrix(0,e$nbuf,e$buffer.cols)

	# At this point, only single precision real is implemented.
	# Call OpenCL initialization routine using .Call
	# See file rlips.c
	if (e$type == 's')
	{
		e$ref = .Call("sInitOcllips",
			ncols,
			nrhs,
			e$nbuf,
			e$wg.size,
			PACKAGE="rlips")
	}
	else if (type == 'c')
	{
		e$ref = .Call("cInitOcllips",
			ncols,
			nrhs,
			e$nbuf,
			e$wg.size,
			PACKAGE="rlips")
	}
 	else
 	{
 		stop('rlips.init: type not recognized!')
 	}

	# Flag current RLIPS environment as active.
	# We need this to ensure that we do not reallocate allocated objects
	# or deallocate already deallocated objects. Doing so would cause R to crash.
	e$active <-TRUE
	
	return(e)
}

