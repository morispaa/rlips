\name{RLIPS}
\alias{rlips}
\alias{Rlips}

\title{RLIPS}

\description{R Linear Inverse Problem Solver}

\details{RLIPS is a linear solver for large (overdetermined) linear systems.
It utilizes GPU's to speed up calculations. The critical routines of RLIPS are 
written in C and OpenCL.

RLIPS transforms the original linear problem into a upper triangular one using 
Givens rotations. Upper triangular system is then solved using backsubstitution. 

\bold{RLIPS functions:}

\code{\link{rlips.init}} RLIPS initialization

\code{\link{rlips.dispose}} RLIPS object deletion

\code{\link{rlips.add}} Data feeding

\code{\link{rlips.solve}} Problem solving
 
%\code{\link{rlips.copy}} Prbolem cloning

%\code{\link{rlips.resize}} Unknown marginalization/addition

\code{\link{rlips.rotate}} Force Givens rotations on added data

%\code{\link{rlips.put}} Insert target matrix or target vector into a rlips problem manually

%\code{\link{rlips.get}} Fetch the target matrix or target vector manually

%\code{\link{rlips.demo}} Simple rlips demonstration

\code{\link{rlips.test}} 

}

\note{Copyright (c) 2011--2012, University of Oulu, Finland.

RLIPS is licensed under freeBSD License.}

\author{Mikko Orispaa (University of Oulu, Finland) \email{mikko.orispaa@oulu.fi}}

%\seealso{}

%\references{}


%\examples{}


\keyword{algebra}
