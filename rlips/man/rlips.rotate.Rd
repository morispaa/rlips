\name{rlips.rotate}
\alias{rlips.rotate}

\title{Force Givens rotation}
\description{Rotates all unrotated data of a rlips environment}

\usage{rlips.rotate(e)}
\arguments{
	\item{e}{Existing rlips environment with unrotated data}
	}

\value{
	None. The unrotated data is rotated into rlips system.
}

\details{\code{rlips.rotate} forces the Givens rotations on unrotated rlips data. Especially for large problems this can reduce the memory consumption and solution time remarkably. On the other hand, overusing this command will decrease the performance.

The size of the rotation buffer can be controlled by \code{buffersize}. The default \code{ncols} should be good for most situations.
}

%\note{}

\author{Mikko Orispaa (University of Oulu, Finland) \email{mikko.orispaa@oulu.fi}}

%\seealso{}

%\references{}


%\examples{}


\keyword{algebra}