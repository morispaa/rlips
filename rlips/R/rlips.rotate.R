## file:rlips.rotate.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.



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
    		.Call("cRotateOcllips",
    			e$ref,
    			#Re(rlips.data),
    			#Im(rlips.data),
				Re(t(e$buffer[1:e$brows,])),
				Im(t(e$buffer[1:e$brows,])),
    			e$brows,
				PACKAGE="rlips")
    	}
    	
    	e$buffer <- matrix(0,e$nbuf,e$buffer.cols)
    	e$nrows <- e$nrows + e$browse.pkgs
    	e$rrows <- min(e$ncols,e$nrows)
    	e$brows <- 0
    	
	}


}