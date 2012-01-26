.onLoad <- function(libname,pkgname)
{
	cat('##\n',
		'## R Linear Inverse Problem Solver (RLIPS)\n',
		'## \n',
		'## Copyright (c) 2011 University of Oulu, Finland\n',
		'## Written by Mikko Orispaa <mikko.orispaa@oulu.fi>\n',
		'## Licensed under FreeBSD license\n',sep="")
	
	library.dynam(pkgname,pkgname,lib.loc=libname)
}