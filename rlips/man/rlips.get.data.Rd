\name{rlips.get.data}
\alias{rlips.get.data}

\title{Fetch the rotated data from the GPU}

\description{Fetches the target matrix R and the target vector Y from the GPU. }

\usage{
rlips.get.data(e)
}

\arguments{
\item{e}{Initialized RLIPS environment}
}

\value{None.}

\details{
The rotated system matrices are fetched from the GPU. The upper triangular target matrix R is
put into \code{e$R.mat} and the corresponding target vector Y is put into \code{e$Y.mat}.

Note that this routine is called every time the command \code{rlips.solve} is used, so there is no
reason to use this in normal operation.
}



\author{Mikko Orispaa (University of Oulu, Finland) \email{mikko.orispaa@oulu.fi}}

%\examples{}

\keyword{algebra}
