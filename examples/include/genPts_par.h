/*
 *
 * @author: Milinda Fernando
 * School of Computing, University of Utah
 * @date: 11/10/2015
 *
 *
 * Contains the Normal(Gaussian) and Logarithmic normal random number (octants) generator based on new c+11 rand engines.
 *
 *
 * */


#ifndef GEN_GAUSS_H
#define GEN_GAUSS_H


#include <iostream>
#include "mpi.h"
#include <iostream>
#include <random>
#include <dendro.h>
#include "TreeNode.h"
//#include <chrono>


void genGauss(const double& sd, const long int numPts, int dim, char * filePrefix,MPI_Comm comm);
void genGauss(const double& sd, const DendroIntL numPts, int dim, std::vector<double> & xyz);
void genLogarithmicGauss(const double& sd, const int numPts, int dim, char * filePrefix,MPI_Comm comm);
void pts2Octants(std::vector<ot::TreeNode> & pNodes,double * pts, DendroIntL totPts, unsigned int dim ,unsigned int maxDepth);

#endif