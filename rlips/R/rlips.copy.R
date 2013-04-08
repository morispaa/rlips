## file:rlips.add.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.

## Copy RLIPS environment
## Arguments:
##  oenv	rlips environment to be copied
##
## Returns:
##  nenv	new rlips environment that is identical
##			with oenv 
##
## This is a quick hack! A better one should be 
## written ASAP!

rlips.copy <- function(oenv)
{
	# Init a new problem using same parameters as the old one
	nenv <- rlips.init(oenv$ncols,oenv$nrhs,oenv$type,
					   oenv$nbuf,oenv$wg.size)
	
	# Rotate the old problem and get its data
	rlips.rotate(oenv)
	rlips.get.data(oenv)
	
	# Add data from old problem to the new oneway
	rlips.add(nenv,oenv$R.mat,oenv$Y.mat)
	
	# Return the new problem
	return(nenv)
}
