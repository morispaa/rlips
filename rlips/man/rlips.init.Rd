\name{rlips.init}
\alias{rlips.init}

\title{Initialize rlips environment}

\description{
	Initializes a new rlips environment.}
	
\usage{
	rlips.init(ncols,nrhs,type='s',nbuf=ncols,workgroup.size=128) 
	}
	
\arguments{
	
	\item{ncols}{Integer giving the number of unknowns, i.e. the number of
	columns in the theory matrix.}
	
	\item{nrhs}{Integer giving the number of columns in the measurement matrix.}
	
	\item{type}{Problem numerical type. One of the following character strings:
		\describe{
			\item{'s'}{Single precision real}
			%\item{'d'}{Double precision real}
			\item{'c'}{Single precision complex}
			%\item{'z'}{Double precision complex}
		}}
	
	\item{nbuf}{Size (number of rows) of the rotation buffer. Default is the number of unknowns, i.e. 
	default rotation buffer is of the same size as the problem after rotations. Size of the buffer can
	have a significant impact on performance. The optimal size of the buffer depends on the problem and
	also on computer hardware. Small buffers conserve computer memory but tend to slow rlips down.}
	
	\item{workgroup.size}{Size of the OpenCL workgroup. Optimal size depends on GPU hardware. Usually, This
	should be of form \code{2**n}, where \code{n = 2,3,...,10}. Typical optimal value for NVIDIA GPU's is 128 and
	for AMD GPU's 256. Your mileage may vary.}

}


\value{
R environment. This environment contains a number of internal variables needed by
\code{rrlips}. After the rlips problem is solved (see \code{\link{rlips.solve}})
it also contains the solution and (if calculated) posteriori covariance matrix.
}



\seealso{\code{\link{rlips.dispose}}, \code{\link{rlips.add}}, \code{\link{rlips.solve}}.}

\author{Mikko Orispaa (University of Oulu, Finland) \email{mikko.orispaa@oulu.fi}}

\keyword{algebra}

