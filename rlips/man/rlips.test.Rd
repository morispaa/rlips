\name{rlips.test}
\alias{rlips.test}

\title{RLIPS test}

\description{Simple test to check the speed and accuracy of RLIPS.}

\usage{
rlips.test(type,size,buffersize=size[2],loop=1,wg.size=128,return.data=FALSE,averaging.fun=mean)
}

\arguments{
\item{type}{Type of the problem. \code{'s'} for single precision real, or \code{'c'} for
single precision complex.}
\item{size}{2-vector holding the size (rows,columns) of the test problem.}
\item{buffersize}{RLIPS rotation buffersize. The default is the number of columns/unknowns of the problem.}
\item{loop}{Number of tests performed. Default is 1. If \code{loop} is larger than 1, the results are averaged using
the function \code{averaging.fun}.}
\item{wg.size}{OpenCL workgroup size.}
\item{return.data}{If \code{TRUE}, the rotated system matrices are returned.}
\item{averaging.fun}{Function used in averaging the results, if \code{loop} is greater than 1. Default is arithmetic mean.}
}

\value{Returns a list with components:
\describe{
\item{time}{Elapsed time for solving the problem using RLIPS.}
\item{accuracy}{Maximum absolute error.}
\item{Gflops}{Approximate Gigaflops (floating point operations per second).}
\item{R}{If \code{return.data=TRUE}, the rotated upper triangular target matrix.}
\item{Y}{If \code{return.data=TRUE}, the rotated target vector.}
}
}

\details{This test routine can (and should) be used to trim the rotation buffer and OpenCL
workgroup sizes for the used hardware. }



\author{Mikko Orispaa (University of Oulu, Finland) \email{mikko.orispaa@oulu.fi}}

%\examples{}

\keyword{algebra}
