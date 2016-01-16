
#include "mpi.h"
#include "petsc.h"
#include "sys.h"
#include <vector>
#include "TreeNode.h"
#include "parUtils.h"
#include "oda.h"
#include "handleStencils.h"
#include "odaJac.h"
#include "colors.h"
#include <cstdlib>
#include <cstring>
#include "externVars.h"
#include "dendro.h"
#include "genPts_par.h"

//Don't want time to be synchronized. Need to check load imbalance.
#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

#ifdef PETSC_USE_LOG
//user-defined variables
int Jac1DiagEvent;
int Jac1MultEvent;
int Jac1FinestDiagEvent;
int Jac1FinestMultEvent;
#endif

double**** LaplacianType2Stencil; 
double**** MassType2Stencil; 

int main(int argc, char ** argv ) {	
  int size, rank;
  bool incCorner = 1;  
  unsigned int numPts;
  unsigned int solveU = 0;
  unsigned int writeB = 0;
  unsigned int numLoops = 100;
  char Kstr[20];
  char pFile[50],bFile[50],uFile[50];
  double gSize[3];
  unsigned int ptsLen;
  unsigned int maxNumPts= 1;
  unsigned int dim=3;
  unsigned int maxDepth=30;
  bool compressLut=true;
  bool genPts=true;
  double localTime, totalTime;
  double startTime, endTime;
  DendroIntL locSz, totalSz;
  std::vector<ot::TreeNode> linOct, balOct;
  std::vector<double> pts;
  DendroIntL TotalPts=10000;

  PetscInitialize(&argc,&argv,"options",NULL);
  ot::RegisterEvents();
  ot::DA_Initialize(MPI_COMM_WORLD);

#ifdef PETSC_USE_LOG
  PetscClassId classid;
  PetscClassIdRegister("Dendro",&classid);

  PetscLogEventRegister("ODAmatDiag",classid, &Jac1DiagEvent);
  PetscLogEventRegister("ODAmatMult",classid, &Jac1MultEvent);
  PetscLogEventRegister("ODAmatDiagFinest",classid, &Jac1FinestDiagEvent);
  PetscLogEventRegister("ODAmatMultFinest",classid, &Jac1FinestMultEvent);
  int stages[4];
  PetscLogStageRegister("P2O.",&stages[0]);
  PetscLogStageRegister("Bal",&stages[1]);
  PetscLogStageRegister("ODACreate",&stages[2]);
  PetscLogStageRegister("MatVec",&stages[3]);
#endif

  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << "inpfile  maxDepth[30] solveU[0]\
      writeB[0] dim[3] maxNumPtsPerOctant[1] incCorner[1] numLoops[100] compressLut[1] genPts[1] totalPts[10000]" << std::endl;
    return -1;
  }
  if(argc > 2) {
    maxDepth = atoi(argv[2]);
  }
  if(argc > 3) {
    solveU = atoi(argv[3]);
  }
  if(argc > 4) {
    writeB = atoi(argv[4]);
  }
  if(argc > 5) {
    dim = atoi(argv[5]);
  }
  if(argc > 6) {
    maxNumPts = atoi(argv[6]);
  }

  if(argc > 7) { incCorner = (bool)(atoi(argv[7]));}
  if(argc > 8) { numLoops = atoi(argv[8]); }
  if(argc > 9) { compressLut = (bool)(atoi(argv[9]));}
  if(argc > 10) { genPts = (bool)(atoi(argv[10]));}
  if(argc >11 ) {TotalPts=atol(argv[11]);}

  if(genPts)
  {

    long local_numPts=TotalPts/size;
    long lc_size=0;
    lc_size=((rank+1)*TotalPts)/size-(rank*TotalPts)/size;
    genGauss(0.15,lc_size,dim,argv[1],MPI_COMM_WORLD);

  }

#ifdef HILBERT_ORDERING
  G_MAX_DEPTH = maxDepth;
  G_dim = dim;
  _InitializeHcurve();
#endif

  if (!rank) {
    std::cout << BLU << "===============================================" << NRM << std::endl;
    std::cout << " Input Parameters"  << std::endl;
    std::cout << " Input File Prefix:"<<argv[1]  << std::endl;
    std::cout << " Gen Pts files:: "<< genPts  << std::endl;
    std::cout << " Total Number of Points:: "<<TotalPts<<std::endl;
    std::cout << " Max Depth:"<<maxDepth<<std::endl;
    //std::cout << " Number of psuedo Processors:: "<<num_pseudo_proc<<std::endl;
    std::cout << BLU << "===============================================" << NRM << std::endl;
  }


  strcpy(bFile,argv[1]);
  ot::int2str(rank,Kstr);
  strcat(bFile,Kstr);
  strcat(bFile,"_\0");
  ot::int2str(size,Kstr);
  strcat(bFile,Kstr);
  strcpy(pFile,bFile);
  strcpy(uFile,bFile);
  strcat(bFile,"_Bal.ot\0");
  strcat(pFile,".pts\0");
  strcat(uFile,".sol\0");

  //Points2Octree....
  if(!rank){
    std::cout << " reading  "<<pFile<<std::endl; // Point size
  }
  ot::readPtsFromFile(pFile, pts);
  if(!rank){
    std::cout << " finished reading  "<<pFile<<std::endl; // Point size
  }
  ptsLen = pts.size();
  std::vector<ot::TreeNode> tmpNodes;
  for(int i=0;i<ptsLen;i+=3) {
    if( (pts[i] > 0.0) &&
        (pts[i+1] > 0.0)  
        && (pts[i+2] > 0.0) &&
        ( ((unsigned int)(pts[i]*((double)(1u << maxDepth)))) < (1u << maxDepth))  &&
        ( ((unsigned int)(pts[i+1]*((double)(1u << maxDepth)))) < (1u << maxDepth))  &&
        ( ((unsigned int)(pts[i+2]*((double)(1u << maxDepth)))) < (1u << maxDepth)) ) {
#ifdef __DEBUG__
      assert((i+2) < ptsLen);
#endif
      tmpNodes.push_back( ot::TreeNode((unsigned int)(pts[i]*(double)(1u << maxDepth)),
            (unsigned int)(pts[i+1]*(double)(1u << maxDepth)),
            (unsigned int)(pts[i+2]*(double)(1u << maxDepth)),
            maxDepth,dim,maxDepth) );
    }
  }
  pts.clear();
//  if(!rank) {
//    std::cout << "Number of Nodes Read:" << tmpNodes.size() << std::endl;
//    for(int i=0;i<tmpNodes.size();i++)
//      std::cout<<"Node:"<<tmpNodes[i]<<std::endl;
//  }
  par::removeDuplicates<ot::TreeNode>(tmpNodes,false,MPI_COMM_WORLD);	
  linOct = tmpNodes;
  tmpNodes.clear();
  par::partitionW<ot::TreeNode>(linOct, NULL,MPI_COMM_WORLD);
  // reduce and only print the total ...
  locSz = linOct.size();
  par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank==0) {
    std::cout<<"# pts= " << totalSz<<std::endl;
  }

  pts.resize(3*(linOct.size()));
  ptsLen = (3*(linOct.size()));
  for(int i=0;i<linOct.size();i++) {
    pts[3*i] = (((double)(linOct[i].getX())) + 0.5)/((double)(1u << maxDepth));
    pts[(3*i)+1] = (((double)(linOct[i].getY())) +0.5)/((double)(1u << maxDepth));
    pts[(3*i)+2] = (((double)(linOct[i].getZ())) +0.5)/((double)(1u << maxDepth));
  }//end for i
  linOct.clear();
  gSize[0] = 1.;
  gSize[1] = 1.;
  gSize[2] = 1.;

  MPI_Barrier(MPI_COMM_WORLD);	
#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[0]);
#endif
  startTime = MPI_Wtime();
  ot::points2Octree(pts, gSize, linOct, dim, maxDepth, maxNumPts, MPI_COMM_WORLD);
  endTime = MPI_Wtime();
#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif
  localTime = endTime - startTime;
  par::Mpi_Reduce<double>(&localTime, &totalTime, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  if(!rank){
    std::cout <<"P2n Time: "<<totalTime << std::endl;
  }
  // reduce and only print the total ...
  locSz = linOct.size();
  par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank==0) {
    std::cout<<"# of Unbalanced Octants: " << totalSz<<std::endl;
  }
  pts.clear();

  //Balancing...
  MPI_Barrier(MPI_COMM_WORLD);	
#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[1]);
#endif
  startTime = MPI_Wtime();
  ot::balanceOctree (linOct, balOct, dim, maxDepth, incCorner, MPI_COMM_WORLD, NULL, NULL);
  endTime = MPI_Wtime();
#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif
  linOct.clear();
  if(writeB) { 
    ot::writeNodesToFile(bFile,balOct);
  }
  // compute total inp size and output size
  locSz = balOct.size();
  localTime = endTime - startTime;
  par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<double>(&localTime, &totalTime, 1, MPI_MAX, 0, MPI_COMM_WORLD);

  if(!rank) {
    std::cout << "# of Balanced Octants: "<< totalSz << std::endl;       
    std::cout << "bal Time: "<<totalTime << std::endl;
  }

  //ODA ...
  MPI_Barrier(MPI_COMM_WORLD);	
#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[2]);
#endif
  assert(!(balOct.empty()));
  startTime = MPI_Wtime();
  ot::DA da(balOct, MPI_COMM_WORLD, MPI_COMM_WORLD, compressLut);
  endTime = MPI_Wtime();
#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif
  balOct.clear();
  // compute total inp size and output size
  locSz = da.getNodeSize();
  localTime = endTime - startTime;
  par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<double>(&localTime, &totalTime, 1, MPI_MAX, 0, MPI_COMM_WORLD);

  if(!rank) {
    std::cout << "Total # Vertices: "<< totalSz << std::endl;       
    std::cout << "Time to build ODA: "<<totalTime << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);	
#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[3]);
#endif

  Mat J;
  Vec in, out, diag;
  PetscScalar zero = 0.0;

  //Nodal, Non-Ghosted
  da.createVector(in,false,false,1);
  da.createVector(out,false,false,1);
  da.createVector(diag,false,false,1);

  createLmatType2(LaplacianType2Stencil);
  createMmatType2(MassType2Stencil);
  if(!rank) {
    std::cout << "Created stencils."<< std::endl;
  }

  if(!rank) {
    std::cout<<rank << " Creating Jacobian" << std::endl;
  }

  iC(CreateJacobian1(&da,&J));

  if(!rank) {
    std::cout<<rank << " Computing Jacobian" << std::endl;
  }

  iC(ComputeJacobian1(&da,J));

  if(!rank) {
    std::cout<<rank << " Finished computing Jacobian" << std::endl;
  }

  VecSet(in, zero);

  for(unsigned int i=0;i<numLoops;i++) {
    iC(Jacobian1MatGetDiagonal(J, diag));
    iC(Jacobian1MatMult(J, in, out));
  }

  VecDestroy(&in);
  VecDestroy(&out);
  VecDestroy(&diag);

  iC(Jacobian1MatDestroy(J));

  destroyLmatType2(LaplacianType2Stencil);
  destroyMmatType2(MassType2Stencil);

  if(!rank) {
    std::cout << "Destroyed stencils."<< std::endl;
  }


  //! Quality of the partition ...


  DendroIntL maxNodeSize, minNodeSize,
          maxBdyNode, minBdyNode,
          maxIndepSize, minIndepSize,
          maxElementSize, minElementSize;

  DendroIntL localSz;

  localSz = da.getNodeSize();
  par::Mpi_Reduce<DendroIntL>(&localSz, &maxNodeSize, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &minNodeSize, 1, MPI_MIN, 0, MPI_COMM_WORLD);

  localSz = da.getBoundaryNodeSize();
  par::Mpi_Reduce<DendroIntL>(&localSz, &maxBdyNode, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &minBdyNode, 1, MPI_MIN, 0, MPI_COMM_WORLD);

  localSz = da.getElementSize();
  par::Mpi_Reduce<DendroIntL>(&localSz, &maxElementSize, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &minElementSize, 1, MPI_MIN, 0, MPI_COMM_WORLD);

  localSz = da.getIndependentSize();
  par::Mpi_Reduce<DendroIntL>(&localSz, &maxIndepSize, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &minIndepSize, 1, MPI_MIN, 0, MPI_COMM_WORLD);


  unsigned long diff,diff_min,diff_max,diff_mean;
  unsigned int min,max;
  std::ofstream myfile1;
  std::ofstream myfile2;
  char ptsFileName1[256];
  char ptsFileName2[256];
  sprintf(ptsFileName1, "%s_%d_%d_%d_%d", "nodeListComplete", maxDepth,TotalPts,rank, size);
  sprintf(ptsFileName2, "%s_%d_%d_%d_%d", "nodeListNonHangin", maxDepth,TotalPts,rank, size);
  myfile1.open(ptsFileName1);
  myfile2.open(ptsFileName2);
  myfile1<<"ODA_NODE_LIST COMPLETE"<<std::endl;
  myfile2<<"ODA NODE LIST NON HANGIN"<<std::endl;
  for(da.init<ot::DA_FLAGS::ALL>();da.curr()<da.end<ot::DA_FLAGS::ALL>();da.next<ot::DA_FLAGS::ALL>())
  {
    unsigned int  nodeList[8];
    da.getNodeIndices(nodeList);
    myfile1<<"Node List for element:\t"<<da.curr()<<":";
    min=nodeList[0];
    max=nodeList[0];
    myfile1<<nodeList[0]<<",";

    if(da.isGhost(nodeList[0]))
      myfile2<<nodeList[0]<<",";


    for(int i=1;i<8;i++)
    {
      if(i<8) {
        myfile1 << nodeList[i] << ",";
      }
      else {
        myfile1 << nodeList[i];
      }

      if(!da.isGhost(nodeList[i]))
      {

        if(i<8) {
          myfile2 << nodeList[i] << ",";
        }
        else {
          myfile2 << nodeList[i];
        }

      }

      if(max<nodeList[i] & !da.isGhost(nodeList[i]))
        max=nodeList[i];

      if(min>nodeList[i] & !da.isGhost(nodeList[i]))
        min=nodeList[i];


    }

    diff=max-min;
    myfile1<<"\t Diff:"<<diff<<std::endl;
    myfile2<<"\t Diff:"<<diff<<std::endl;

  }

  myfile1.close();


//  par::Mpi_Reduce(&min,&min_g,1,MPI_MIN,0,MPI_COMM_WORLD);
//  par::Mpi_Reduce(&max,&max_g,1,MPI_MAX,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff,&diff_min,1,MPI_MIN,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff,&diff_max,1,MPI_MAX,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff,&diff_mean,1,MPI_SUM,0,MPI_COMM_WORLD);

  diff_mean=diff_mean/size;

  if (!rank) {

    std::cout << RED<<"=====================QUALITY OF ODA========================================"<<NRM<<std::endl;
    std::cout << "Nodes          \t(" << minNodeSize << ", " << maxNodeSize << ")" << std::endl;
    std::cout << "Boundary Node  \t(" << minBdyNode << ", " << maxBdyNode << ")" << std::endl;
    std::cout << "Element Size   \t(" << minElementSize << ", " << maxElementSize << ")" << std::endl;
    std::cout << "Independent    \t(" << minIndepSize << ", " << maxIndepSize << ")" << std::endl;
    std::cout << RED<<"=====================NODELIST STATISTICS========================================"<<NRM<<std::endl;
    std::cout << RED<<"Diff Global Min:"<<diff_min<<NRM<<std::endl;
    std::cout << RED<<"Diff Global Max:"<<diff_max<<NRM<<std::endl;
    std::cout << RED<<"Diff Global Mean:"<<diff_mean<<NRM<<std::endl;
    std::cout << RED<<"==========================================================================="<<NRM<<std::endl;

  }



#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif
  if (!rank) {
    std::cout << GRN << "Finalizing PETSC" << NRM << std::endl;
  }
  ot::DA_Finalize();
  PetscFinalize();



}//end function

