\name{rlips.solve}
\alias{rlips.solve}

\title{Solve rlips problem}
\description{Solve the problem fed into rlips environment.}

\usage{rlips.solve(e,calculate.covariance=FALSE,full.covariance=FALSE)}
\arguments{
	\item{e}{An existing rlips environment with enough data feeded in.}
	\item{calculate.covariance}{Flag for calculating the posteriori covariance matrix.}
	\item{full.covariance}{Flag for calculating the full posteriori covariance matrix..}
	}

\value{None. The solution is written into rlips environment variable \code{e$solution}. The  posteriori covariance matrix (if calculated) is written into environment variable \code{e$covariance}.}

\details{The calculation of the posteriori covariance matrix is controlled by the two flags \code{calculate.covariance} and \code{full.covariance}. 

If \code{calculate.covariance=FALSE} the posteriori covariance matrix is not calculated at all.

If \code{calculate.covariance=TRUE}, then if \code{full.covariance=TRUE} the full posteriori covariance matrix is calculated. If \code{full.covariance=FALSE}, only the posteriori variances (i.e. the diagonal of the 
posteriori covariance matrix) is calculated. This is much faster than calculating the full matrix!



    }


\note{Rlips does not check that there exists enough data in the problem, nor that the problem is solvable 
(i.e. there exists enough linearly independent data rows in the theory matrix).

This routine calls first \link{rlips.get.data}, which fetches the rotated system from the GPU. The rotated upper triangular
target matrix R is put into \code{e$R.mat} and the corresponding rotated target vector into \code{e$Y.mat}.


}

\author{Mikko Orispaa (University of Oulu, Finland) \email{mikko.orispaa@oulu.fi}}

%\seealso{}

%\references{}


%\examples{}


\keyword{algebra}
