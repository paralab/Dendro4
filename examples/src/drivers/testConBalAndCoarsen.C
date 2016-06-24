#include "mpi.h"
#include "petsc.h"
#include "sys.h"
#include "parUtils.h"
#include "octUtils.h"
#include "TreeNode.h"
#include "externVars.h"
#include "dendro.h"
#include "treenode2vtk.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "hcurvedata.h"


#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

struct Particle
{
  int charge = 0;
  int intX = 0;
  int intY = 0; 
  int intZ = 0;
  double radius = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

double gaussian(double mean, double std_deviation);
inline bool ParticleInOctant(ot::TreeNode &octant, Particle &particle);
inline void CenterOfCharge(std::vector<Particle> &inOctant, Particle &newOctant, unsigned int &maxRange);

int main(int argc, char ** argv ) {	
  int size, rank;
  double startTime, endTime;
  double localTime, globalTime;
  double gSize[3];
  unsigned int local_num_pts = 5000;
  unsigned int dim = 3;
  unsigned int maxDepth = 8;
  unsigned int maxRange = 1<<maxDepth;
  unsigned int maxNumPts = 1;
  int obtainedDepth = 0;
  int numFilePoints = 0;
  const int MAX_OCT = 100;
  std::string filename = "N3000_Phi0.1.xyz";
  
  _InitializeHcurve(dim);

  bool incCorner = 1;  
  std::vector<ot::TreeNode> linOct;
  std::vector<ot::TreeNode> balOct;
  std::vector<ot::TreeNode> coarseOct;
  std::vector<ot::TreeNode> coarseOct2;  
  std::vector<ot::TreeNode> tmpNodes;
  std::vector<Particle> particles;
  std::vector<Particle> allParticles;
  std::vector<Particle> balOct_Particles;
  std::vector<double> pts;
  std::vector<double> allPts;
  std::vector<ot::TreeNode> coarsen1;
  std::vector<ot::TreeNode> coarsen2;
  std::vector<Particle> particle1;
  std::vector<Particle> particle2;
  int level = 0;
  std::vector<ot::TreeNode>::iterator balItr;
  std::vector<Particle>::iterator parItr;
  std::vector<ot::TreeNode>::iterator coarseItr;
  std::vector<std::vector<ot::TreeNode> > octants;
  std::vector<std::vector<Particle> > octParticles;
  unsigned int ptsLen;
  DendroIntL localSz, totalSz;

  PetscInitialize(&argc, &argv, "options", NULL);
  ot::RegisterEvents();

#ifdef PETSC_USE_LOG
  int stages[4];
  PetscLogStageRegister( "Main",&stages[0]);
  PetscLogStageRegister( "P2O",&stages[1]);
  PetscLogStageRegister( "Bal",&stages[2]);
  PetscLogStageRegister( "Coarsen",&stages[3]);
#endif

#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[0]);
#endif

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  
  // Reads in the file
  if(rank == 0)
    {
      std::ifstream pointsFile;
      pointsFile.open(filename.c_str());
      pointsFile >> numFilePoints;
      allParticles.resize(numFilePoints);
      std::string temp = "";
      double x_scale = 0;
      double y_scale = 0;
      double z_scale = 0;
      pointsFile >> temp;
      pointsFile >> temp;
      pointsFile >> x_scale >> y_scale >> z_scale;
      for(int i = 0; i < numFilePoints; ++i)
	{
	  char letter;
	  double x, y, z;
	  pointsFile >> letter >> x >> y >> z;
	  double x_s = x/x_scale;
	  double y_s = y/y_scale;
	  double z_s = z/z_scale;
	  allPts.push_back(x_s);
	  allPts.push_back(y_s);
	  allPts.push_back(z_s);
	  allParticles[i].charge = 1;
	  allParticles[i].x = x_s;
	  allParticles[i].intX = (int)(x_s * maxRange);
	  allParticles[i].y = y_s;
	  allParticles[i].intY = (int)(y_s * maxRange);
	  allParticles[i].z = z_s;
	  allParticles[i].intZ = (int)(z_s * maxRange);
	}
      pointsFile.close();
      local_num_pts = allPts.size() / size;
    }
  
  // Sends numFilePoints and local_num_points to the others
  MPI_Bcast(&local_num_pts, 1, MPI::INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&numFilePoints, 1, MPI::INT, 0, MPI_COMM_WORLD);
  
  // Keeping this until I get a fix for "overflow"
  if(numFilePoints % size != 0)
    {
      if(rank == 0)
	std::cerr << "Number of points isn't divisible by the rank. Exiting." 
		  << std::endl;
      return -1;
    }

  // Resizes the number of points for each node
  pts.resize(local_num_pts/size);
  allParticles.resize(numFilePoints);

  std::cout << "local num pts: " << local_num_pts << std::endl;
  
  // Scatter the data
  MPI_Scatter(&allPts.front(), local_num_pts/size, MPI::DOUBLE, &pts.front(), local_num_pts/size, MPI::DOUBLE, 0, MPI_COMM_WORLD);

  // Broadcast the points
  MPI_Bcast(&allParticles.front(), sizeof(Particle) * numFilePoints, MPI::BYTE, 0, MPI_COMM_WORLD);

  particles = allParticles;

  if(argc > 1) {
    //local_num_pts = atoi(argv[1]);
  }

  if(argc > 2) {
    dim = atoi(argv[2]);
  }

  if(argc > 3) {
    maxDepth = atoi(argv[3]);
  }

  if(argc > 4) {
    maxNumPts = atoi(argv[4]);
  }

  if(argc > 5) {
    incCorner = (bool)(atoi(argv[5]));
  }

  MPI_Barrier(MPI_COMM_WORLD);

  ptsLen = pts.size();
  std::cout << "ptsLen: " << ptsLen << std::endl;

  // Scaling issue
  // visualize octree
  for(int i = 0; i < ptsLen; i+=3) {
    if( (pts[i] > 0.0) &&
        (pts[i+1] > 0.0)
        && (pts[i+2] > 0.0) &&
        ( ((unsigned int)(pts[i]*((double)(1u << maxDepth)))) < (1u << maxDepth))  &&
        ( ((unsigned int)(pts[i+1]*((double)(1u << maxDepth)))) < (1u << maxDepth))  &&
        ( ((unsigned int)(pts[i+2]*((double)(1u << maxDepth)))) < (1u << maxDepth)) ) {
      tmpNodes.push_back( ot::TreeNode((unsigned int)(pts[i]*(double)(1u << maxDepth)),
            (unsigned int)(pts[i+1]*(double)(1u << maxDepth)),
            (unsigned int)(pts[i+2]*(double)(1u << maxDepth)),
            maxDepth,dim,maxDepth) );
    }
  }
  pts.clear();

  par::removeDuplicates<ot::TreeNode>(tmpNodes,false,MPI_COMM_WORLD);
  linOct = tmpNodes;
  tmpNodes.clear();
  par::partitionW<ot::TreeNode>(linOct, NULL,MPI_COMM_WORLD);
  // reduce and only print the total ...
  localSz = linOct.size();
  par::Mpi_Reduce<DendroIntL>(&localSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank==0) {
    std::cout<<"# pts= " << totalSz<<std::endl;
  }

  pts.resize(3*(linOct.size()));
  ptsLen = (3*(linOct.size()));
  for(int i = 0; i < linOct.size(); i++) {
    pts[3*i] = (((double)(linOct[i].getX())) + 0.5)/((double)(1u << maxDepth));
    pts[(3*i)+1] = (((double)(linOct[i].getY())) +0.5)/((double)(1u << maxDepth));
    pts[(3*i)+2] = (((double)(linOct[i].getZ())) +0.5)/((double)(1u << maxDepth));
  }//end for i
  linOct.clear();

  gSize[0] = 1.0;
  gSize[1] = 1.0;
  gSize[2] = 1.0;

#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif

  MPI_Barrier(MPI_COMM_WORLD);	

#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[1]);
#endif
  startTime = MPI_Wtime();
  // Creates the octTree, returned into linOct
  ot::points2Octree(pts, gSize, linOct, dim, maxDepth, maxNumPts, MPI_COMM_WORLD);
  endTime = MPI_Wtime();
  localTime = endTime - startTime;
#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif
  par::Mpi_Reduce<double>(&localTime, &globalTime, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  if(!rank){
    std::cout <<"P2n Time: "<<globalTime << "secs " << std::endl;
  }
  pts.clear();

  localSz = linOct.size();
  par::Mpi_Reduce<DendroIntL>(&localSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank==0) {
    std::cout<<"linOct.size = " << totalSz<<std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);	

#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[2]);
#endif
  treeNodesTovtk(linOct, rank, "N3000_lin-1");

  startTime = MPI_Wtime();
  // Balances the octTree, returns it in balOct
  ot::balanceOctree (linOct, balOct, dim, maxDepth, incCorner, MPI_COMM_WORLD, NULL, NULL);
  endTime = MPI_Wtime();
  localTime = endTime - startTime;
#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif

  

  //treeNodesTovtk(balOct, rank, "N3000_bal-1");

  par::Mpi_Reduce<double>(&localTime, &globalTime, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  if(!rank){
    std::cout <<"Bal Time: "<<globalTime << "secs " << std::endl;
  }
  linOct.clear();

  localSz = balOct.size();
  par::Mpi_Reduce<DendroIntL>(&localSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank==0) {
    std::cout<<"balOct.size = " << totalSz<<std::endl;
  }


#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[3]);
#endif
  startTime = MPI_Wtime();
  // Builds the x,y,z,charge for the initial balanced octants
  balOct_Particles.resize(balOct.size());
  int parCount = 0;
  int parTotal = 0;
  int index = 0;
  std::vector<Particle> inOctant;

  for(balItr = balOct.begin(); balItr != balOct.end(); ++balItr)
    {
      for(parItr = particles.begin(); parItr != particles.end(); ++parItr)
	{
	  if(ParticleInOctant(*balItr, *parItr))
	    {
	      inOctant.push_back(*parItr);
	      particles.erase(parItr);
	      ++parCount;
	      --parItr;
	    }
	}
      index = std::distance(balOct.begin(), balItr);
      CenterOfCharge(inOctant, balOct_Particles[index], maxRange);
      inOctant.clear();
    }

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
    std::cout << "rank 0: " << parCount << '\t' << balOct.size() << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 1)
    std::cout << "rank 1: " << parCount << '\t' << balOct.size() << std::endl;

  MPI_Reduce(&parCount, &parTotal, 1, MPI::INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  parItr = balOct_Particles.begin();
  octants.resize(octants.size() + 1);
  octParticles.resize(octParticles.size() + 1);
  for(balItr = balOct.begin(); balItr != balOct.end(); ++balItr)
    {
      octants[level].push_back(*balItr);
      octParticles[level].push_back(*parItr);
      ++parItr;
    }

  ++level;

  if(rank == 0)
    {
      std::cout << "Time for making balOct particles: " << MPI_Wtime() - startTime << " sec" << std::endl;
      std::cout << "Balanced Matches: " << parTotal << std::endl;
    }

  coarsen1 = balOct;

  do
    {
      std::cout << "level: " << level << std::endl;
      ot::coarsenOctree(coarsen1, coarsen2, dim, maxDepth, MPI_COMM_WORLD, false, NULL, NULL);
      bool done = true;
      std::vector<ot::TreeNode>::iterator coarse1Itr;
      std::vector<ot::TreeNode>::iterator coarse2Itr = coarsen2.begin();
      octants.resize(octants.size() + 1);
      octParticles.resize(octParticles.size() + 1);
      octants[level].resize(coarsen2.size());
      octParticles[level].resize(coarsen2.size());
      std::vector<Particle> inCoarse;
      for(coarse1Itr = coarsen1.begin(); coarse1Itr != coarsen1.end(); ++coarse1Itr)
	{
	  done = true;
	  if((*coarse2Itr).isAncestor(*coarse1Itr) || (*coarse2Itr) == (*coarse1Itr))
	    {
	      int index = std::distance(coarsen1.begin(), coarse1Itr);
	      inCoarse.push_back(octParticles[level-1][index]);
	      done = false;
	    }
	  if(done == true)
	    {
	      int coarseIndex = std::distance(coarsen2.begin(), coarse2Itr);
	      CenterOfCharge(inCoarse, octParticles[level][coarseIndex], maxRange);
	      ++coarse2Itr;
	      if(inCoarse.size() > 0)
		--coarse1Itr;
	      inCoarse.clear();
	    }
	}

      coarsen1 = coarsen2;
      coarsen2.clear();
      ++level;
      
    }while(coarsen1.size() > MAX_OCT);

  std::cout << "rank " << rank << std::endl;
  

   /*

  // Coarsens octTree by 1 level
  startTime = MPI_Wtime();
  ot::coarsenOctree(balOct, coarseOct, dim, maxDepth, MPI_COMM_WORLD, false, NULL, NULL);
  endTime = MPI_Wtime();
  localTime = endTime - startTime;
#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif
  par::Mpi_Reduce<double>(&localTime, &globalTime, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  if(!rank){
    std::cout <<"coarse Time: "<<globalTime << " secs " << std::endl;
  }

  treeNodesTovtk(coarseOct, rank, "N3000_Coars1-1");

  // Build the coarsened particles
  startTime = MPI_Wtime();
  int matches = 0;
  int totalMatches = 0;
  index = 0;
  int coarseIndex = 0;
  bool done = true;
  std::vector<Particle> inCoarse;
  std::vector<Particle> coarse_Particles(coarseOct.size());
  std::cout << "coarse size: " << coarseOct.size() << std::endl;
  std::cout << "bal size: " << balOct.size() << std::endl;
  coarseItr = coarseOct.begin();
  for(balItr = balOct.begin(); balItr != balOct.end(); ++balItr)
    {
      done = true;
      // Have to use or since the coarsened octant can be the same 
      if((*coarseItr).isAncestor(*balItr) || (*coarseItr) == (*balItr))
	{
	  index = std::distance(balOct.begin(), balItr);
	  inCoarse.push_back(balOct_Particles[index]);
	  ++matches;
	  done = false;
	}
      if(done == true)
	{
	  coarseIndex = std::distance(coarseOct.begin(), coarseItr);
	  CenterOfCharge(inCoarse, coarse_Particles[coarseIndex], maxRange);
	  ++coarseItr;
	  if(inCoarse.size() > 0)
	    --balItr;
	  inCoarse.clear();
	}
    }
  if(rank == 0)
    std::cout << "Time for coarsening particles: " << MPI_Wtime() - startTime << " sec" << std::endl;

  MPI_Reduce(&matches, &totalMatches, 1, MPI::INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(rank == 0)
    std::cout << "Coarsened Matches: " << totalMatches << std::endl;

  balOct.clear();

  localSz = coarseOct.size();
  par::Mpi_Reduce<DendroIntL>(&localSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank==0) {
    std::cout<<"coarseOct.size = " << totalSz<<std::endl;
  }

  ot::coarsenOctree(coarseOct, coarseOct2, dim, maxDepth, MPI_COMM_WORLD, false, NULL, NULL);

  localSz = coarseOct2.size();
  par::Mpi_Reduce<DendroIntL>(&localSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank == 0)
    std::cout << "coarseOct2.size = " << totalSz << std::endl;

  treeNodesTovtk(coarseOct2, rank, "N3000_Coars2-1");

  coarseOct.clear();
  coarseOct2.clear();
  */
  PetscFinalize();
}//end main

double gaussian(double mean, double std_deviation) {
  static double t1 = 0, t2=0;
  double x1, x2, x3, r;

  using namespace std;

  // reuse previous calculations
  if(t1) {
    const double tmp = t1;
    t1 = 0;
    return mean + std_deviation * tmp;
  }
  if(t2) {
    const double tmp = t2;
    t2 = 0;
    return mean + std_deviation * tmp;
  }

  // pick randomly a point inside the unit disk
  do {
    x1 = 2 * drand48() - 1;
    x2 = 2 * drand48() - 1;
    x3 = 2 * drand48() - 1;
    r = x1 * x1 + x2 * x2 + x3*x3;
  } while(r >= 1);

  // Box-Muller transform
  r = sqrt(-2.0 * log(r) / r);

  // save for next call
  t1 = (r * x2);
  t2 = (r * x3);

  return mean + (std_deviation * r * x1);
}//end gaussian

void CoarsenParticles(std::vector<ot::TreeNode> &fine, std::vector<ot::TreeNode> &coarse, std::vector<Particle> &fineP, std::vector<Particle> &coarseP)
{
  
}

inline bool ParticleInOctant(ot::TreeNode &octant, Particle &particle)
{
  bool particle_x = false;
  bool particle_y = false;
  bool particle_z = false;

  if(octant.minX() <= particle.intX && particle.intX <= octant.maxX())
    particle_x = true;
  if(octant.minY() <= particle.intY && particle.intY <= octant.maxY())
    particle_y = true;
  if(octant.minZ() <= particle.intZ && particle.intZ <= octant.maxZ())
    particle_z = true;
  return (particle_x && particle_y && particle_z);
}

inline void CenterOfCharge(std::vector<Particle> &inOctant, Particle &newOctant, unsigned int &maxRange)
{
  if(inOctant.size() > 0)
    {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      for(int i = 0; i < inOctant.size(); ++i)
	{
	  x += inOctant[i].x;
	  y += inOctant[i].y;
	  z += inOctant[i].z;
	  newOctant.charge += inOctant[i].charge;
	}
      newOctant.x = x/inOctant.size();
      newOctant.y = y/inOctant.size();
      newOctant.z = z/inOctant.size();
      newOctant.intX = newOctant.x * maxRange;
      newOctant.intY = newOctant.y * maxRange;
      newOctant.intZ = newOctant.z * maxRange;
    }
  return;
}
