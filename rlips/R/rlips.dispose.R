## file:rlips.dispose.R
## (c) 2011- University of Oulu, Finland
## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>
## Licensed under FreeBSD license. See file LICENSE for details.


## dispose RLIPS object
## Arguments:
##  e		rlips environment
rlips.dispose <- function(e)
{
	# Check that e actually is an active rlips environment
	if (e$active)
	{
		# Depending on the type of the environment call the proper
		# C routine which deallocates memory and tidies things up.
		if (e$type == 's')
		{
			.Call("sKillRlips",
				e$ref,PACKAGE="rlips")
		}
		else if (e$type == 'c')
		{
			.Call("cKillRlips",
				e$ref,PACKAGE="rlips")
		}
		else
		{
			stop('rlips.dispose: type not recognized! Nothing done!')
		}
		
		# Flag environment as inactive.
		e$active <- FALSE	
	}
	else
	{
		stop('rlips.dispose: Not an active RLIPS environment! Nothing done!')
	}
}
