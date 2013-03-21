## file:rlips.rotate.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



## Do the rotations
##
##	Arguments:
##		e		Active RLIPS environmet
rlips.rotate <- function(e)
{
	# Make sure that we are using an active environment
	if (!e$active)
	{
		stop("Not an active rlips environment! Nothing done!")
	}
	
	# Rotate only, if there is something to rotate
	if (e$brows > 0)
	{  	
		# Depending on e$type, use the right C routine 
		# <s|c>RotateRlips
		# Notice that the buffer is transposed, because 
		# R stores matrices in column-major order and 
		# C in row-major order.
		if (e$type == 's')
		{
			.Call("sRotateRlips",
				  e$ref,
				  t(e$buffer[1:e$brows,]),
				  e$brows,
				  PACKAGE="rlips")
		}
		else if (e$type == 'c')
		{
			.Call("cRotateRlips",
				  e$ref,
				  # Complex data is separated into real and
				  # imaginary parts
				  Re(t(e$buffer[1:e$brows,])),
				  Im(t(e$buffer[1:e$brows,])),
				  e$brows,
				  PACKAGE="rlips")
		}
		
		# After rotations, update internal variables and 
		# empty the buffer matrix.
		e$buffer <- matrix(0,e$nbuf,e$buffer.cols)
		e$nrows <- e$nrows + e$brows
		e$rrows <- min(e$ncols,e$nrows)
		e$brows <- 0	
	}
}


