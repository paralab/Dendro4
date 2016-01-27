
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

//#include <cxxabi.h>
//#include <execinfo.h>

#include "genPts_par.h"
#include <climits>

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
  char pFile[256],bFile[256],uFile[256];
  double gSize[3];
  unsigned int ptsLen;
  unsigned int maxNumPts= 1;
  unsigned int dim=3;
  unsigned int maxDepth=30;
  bool compressLut=true;
  bool genPts=true;
  bool genRegGrid=false;
  double localTime, totalTime;
  double startTime, endTime;
  DendroIntL locSz, totalSz;
  std::vector<ot::TreeNode> linOct, balOct;
  std::vector<double> pts;
  DendroIntL grainSize =10000;
  char nlistFName[256];

  PetscInitialize(&argc,&argv,"options",NULL);
  ot::RegisterEvents();
  ot::DA_Initialize(MPI_COMM_WORLD);
  PetscErrorPrintf = PetscErrorPrintfNone;
  unsigned int regLev;


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
      writeB[0] dim[3] maxNumPtsPerOctant[1] incCorner[1] numLoops[100] compressLut[1] genPts[1] totalPts[10000] nlistFileName genRegGrid[false] regLev" << std::endl;
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
  if(argc >11 ) { grainSize =atol(argv[11]);}
  sprintf(nlistFName, "%s_%d_%d_%d_%d.%s", argv[12], maxDepth, grainSize, rank, size, "bin");

  if(argc >13) {genRegGrid=(bool)(atoi(argv[13])); regLev=atoi(argv[14]);}

#ifdef HILBERT_ORDERING
  G_MAX_DEPTH = maxDepth;
  G_dim = dim;
  _InitializeHcurve();
#endif


  if (!rank) {
    std::cout << BLU << "===============================================" << NRM << std::endl;
    std::cout << " Input Parameters" << std::endl;
    std::cout << " Input File Prefix:" << argv[1] << std::endl;
    std::cout << " Gen Pts files:: " << genPts << std::endl;
    std::cout << " Number of Points per process:: " << grainSize << std::endl;
    std::cout << " Max Depth:" << maxDepth << std::endl;
    std::cout << " Gen Regular Grid:"<<genRegGrid<<std::endl;
    std::cout << " Regular grid Level:"<<regLev<<std::endl;
    std::cout << BLU << "===============================================" << NRM << std::endl;
  }



if(!genRegGrid) {

  if (genPts) {
    genGauss(0.15, grainSize, dim, argv[1], MPI_COMM_WORLD);

  }




  strcpy(bFile, argv[1]);
  ot::int2str(rank, Kstr);
  strcat(bFile, Kstr);
  strcat(bFile, "_\0");
  ot::int2str(size, Kstr);
  strcat(bFile, Kstr);
  strcpy(pFile, bFile);
  strcpy(uFile, bFile);
  strcat(bFile, "_Bal.ot\0");
  strcat(pFile, ".pts\0");
  strcat(uFile, ".sol\0");

  //Points2Octree....
  if (!rank) {
    std::cout << " reading  " << pFile << std::endl; // Point size
  }
  ot::readPtsFromFile(pFile, pts);
  if (!rank) {
    std::cout << " finished reading  " << pFile << std::endl; // Point size
  }
  ptsLen = pts.size();
  MPI_Barrier(MPI_COMM_WORLD);
  std::vector<ot::TreeNode> tmpNodes;
  for (int i = 0; i < ptsLen; i += 3) {
    if ((pts[i] > 0.0) &&
        (pts[i + 1] > 0.0)
        && (pts[i + 2] > 0.0) &&
        (((unsigned int) (pts[i] * ((double) (1u << maxDepth)))) < (1u << maxDepth)) &&
        (((unsigned int) (pts[i + 1] * ((double) (1u << maxDepth)))) < (1u << maxDepth)) &&
        (((unsigned int) (pts[i + 2] * ((double) (1u << maxDepth)))) < (1u << maxDepth))) {
#ifdef __DEBUG__
      assert((i+2) < ptsLen);
#endif
      tmpNodes.push_back(ot::TreeNode((unsigned int) (pts[i] * (double) (1u << maxDepth)),
                                      (unsigned int) (pts[i + 1] * (double) (1u << maxDepth)),
                                      (unsigned int) (pts[i + 2] * (double) (1u << maxDepth)),
                                      maxDepth, dim, maxDepth));
    }
  }
  pts.clear();
//  if(!rank) {
//    std::cout << "Number of Nodes Read:" << tmpNodes.size() << std::endl;
//    for(int i=0;i<tmpNodes.size();i++)
//      std::cout<<"Node:"<<tmpNodes[i]<<std::endl;
//  }
  par::removeDuplicates<ot::TreeNode>(tmpNodes, false, MPI_COMM_WORLD);
  linOct = tmpNodes;
  tmpNodes.clear();
  par::partitionW<ot::TreeNode>(linOct, NULL, MPI_COMM_WORLD);
  // reduce and only print the total ...
  locSz = linOct.size();
  par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "# pts= " << totalSz << std::endl;
  }
  //std::cout<<"linOct:"<<linOct.size()<<std::endl;
  pts.resize(3 * (linOct.size()));
  ptsLen = (3 * (linOct.size()));
  for (int i = 0; i < linOct.size(); i++) {
    pts[3 * i] = (((double) (linOct[i].getX())) + 0.5) / ((double) (1u << maxDepth));
    pts[(3 * i) + 1] = (((double) (linOct[i].getY())) + 0.5) / ((double) (1u << maxDepth));
    pts[(3 * i) + 2] = (((double) (linOct[i].getZ())) + 0.5) / ((double) (1u << maxDepth));
  }//end for i
  linOct.clear();
  gSize[0] = 1.;
  gSize[1] = 1.;
  gSize[2] = 1.;

  //std::cout << rank << " : reached barrier" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  //std::cout << rank << " : reached barrier" << std::endl;
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
  if (!rank) {
    std::cout << "P2n Time: " << totalTime << std::endl;
  }
  // reduce and only print the total ...
  locSz = linOct.size();
  par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "# of Unbalanced Octants: " << totalSz << std::endl;
  }
  pts.clear();

  //Balancing...
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef PETSC_USE_LOG
  PetscLogStagePush(stages[1]);
#endif
  startTime = MPI_Wtime();
  ot::balanceOctree(linOct, balOct, dim, maxDepth, incCorner, MPI_COMM_WORLD, NULL, NULL);
  endTime = MPI_Wtime();
#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif
  linOct.clear();
  if (writeB) {
    ot::writeNodesToFile(bFile, balOct);
  }
  // compute total inp size and output size
  locSz = balOct.size();
  localTime = endTime - startTime;
  par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<double>(&localTime, &totalTime, 1, MPI_MAX, 0, MPI_COMM_WORLD);

  if (!rank) {
    std::cout << "# of Balanced Octants: " << totalSz << std::endl;
    std::cout << "bal Time: " << totalTime << std::endl;
  }
}else
{
  if(!rank)
    std::cout<<"Generating Regular Grid"<<std::endl;

  assert(regLev<=maxDepth);
  ot::createRegularOctree(balOct,regLev,dim,maxDepth,MPI_COMM_WORLD);


  DendroIntL localSz=balOct.size();
  DendroIntL sizeG=0;

  par::Mpi_Reduce(&localSz,&sizeG,1,MPI_SUM,0,MPI_COMM_WORLD);


  if(!rank)
    std::cout<<"Bal Oct Size:"<<sizeG<<std::endl;


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

#ifdef HILBERT_ORDERING
  da.computeHilbertRotations();
  if(!rank)
    std::cout<<"ODA Rotation pattern computation completed for Hilbert. "<<std::endl;
#endif


 /*
  * This Code is to Visually check the build node list correctness by writing the nodes to vtk file.
  *
  * unsigned int nodeList[8];
  unsigned int nIndex=0.4*(da.end<ot::DA_FLAGS::ALL>());// (unsigned int )0.3*da.end<ot::DA_FLAGS::ALL>();
  std::vector<ot::TreeNode> node;
  ot::TreeNode tmp;

  bool state=false;

  for(da.init<ot::DA_FLAGS::ALL>();da.curr()< da.end<ot::DA_FLAGS::ALL>();da.next<ot::DA_FLAGS::ALL>())
  {
    da.getNodeIndices(nodeList);
    if(da.curr()<da.getIdxElementBegin())
    {
      state=true;
      Point p=da.getCurrentOffset();
      tmp=ot::TreeNode(1,p.xint(),p.yint(),p.zint(),da.getLevel(da.curr()),3,da.getMaxDepth());
      node.push_back(tmp);
      da.getNodeIndices(nodeList);
      treeNodesTovtk(node,rank,"node_pre");
      break;
    }

  }

  std::vector<ot::TreeNode> keys;
  unsigned int curentIndex;
 if(state) {
   for (da.init<ot::DA_FLAGS::ALL>(); da.curr() < da.end<ot::DA_FLAGS::ALL>(); da.next<ot::DA_FLAGS::ALL>()) {
     curentIndex = da.curr();

     for (int w = 0; w < 8; w++) {
       if (curentIndex == nodeList[w]) {
         Point p = da.getCurrentOffset();
         tmp = ot::TreeNode(1, p.xint(), p.yint(), p.zint(), da.getLevel(curentIndex), 3, da.getMaxDepth());
         keys.push_back(tmp);
       }
     }
   }
   treeNodesTovtk(keys, rank, "keys_pre");
 }

  node.clear();
  keys.clear();
*/
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
          maxBdyNode, meanBdyNode,minBdyNode,
          maxIndepSize, minIndepSize,
          maxElementSize, minElementSize;

  DendroIntL localSz;

  DendroIntL preGhost[3];//,postGhost[3];


  localSz = da.getNodeSize();
  par::Mpi_Reduce<DendroIntL>(&localSz, &maxNodeSize, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &minNodeSize, 1, MPI_MIN, 0, MPI_COMM_WORLD);

  localSz =da.getBoundaryNodeSize();//da.getPreGhostElementSize();//da.getPrePostBoundaryNodesSize();//
  par::Mpi_Reduce<DendroIntL>(&localSz, &maxBdyNode, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &minBdyNode, 1, MPI_MIN, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &meanBdyNode,1,MPI_SUM,0,MPI_COMM_WORLD);
  meanBdyNode=meanBdyNode/size;

  localSz=da.getPreGhostElementSize();
  par::Mpi_Reduce<DendroIntL>(&localSz,preGhost,1,MPI_MIN,0,MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz,(preGhost+1),1,MPI_SUM,0,MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz,(preGhost+2),1,MPI_MAX,0,MPI_COMM_WORLD);

  preGhost[1]=preGhost[1]/size;




  localSz = da.getElementSize();
  par::Mpi_Reduce<DendroIntL>(&localSz, &maxElementSize, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &minElementSize, 1, MPI_MIN, 0, MPI_COMM_WORLD);

  localSz = da.getIndependentSize();
  par::Mpi_Reduce<DendroIntL>(&localSz, &maxIndepSize, 1, MPI_MAX, 0, MPI_COMM_WORLD);
  par::Mpi_Reduce<DendroIntL>(&localSz, &minIndepSize, 1, MPI_MIN, 0, MPI_COMM_WORLD);


  unsigned long diff_pre=0,diff_mine=0,diff_post=0;
  unsigned long diff_min_pre=LONG_MAX,diff_max_pre=0,diff_sum_pre;
  unsigned long diff_min_mine=LONG_MAX,diff_max_mine=0,diff_sum_mine;
  unsigned long diff_min_post=LONG_MAX,diff_max_post=0,diff_sum_post;

  unsigned long diff_min_pre_g, diff_max_pre_g;
  unsigned long diff_min_mine_g, diff_max_mine_g;
  unsigned long diff_min_post_g, diff_max_post_g;


  double diff_mean_pre, diff_mean_mine, diff_mean_post;



  int actCnt=0;
  int actCnt_g=0;

  if(da.iAmActive()) {
    actCnt=1;
    FILE* outfile = fopen(nlistFName, "wb");

    for (da.init<ot::DA_FLAGS::ALL>(); da.curr() < da.end<ot::DA_FLAGS::ALL>(); da.next<ot::DA_FLAGS::ALL>()) {
      unsigned int nodeList[8];
      da.getNodeIndices(nodeList);
       unsigned int index;
       unsigned long min_pre=LONG_MAX,max_pre=0, min_mine=LONG_MAX,max_mine=0, min_post=LONG_MAX,max_post=0;
         for (int i = 0; i < 8; i++) {
            index=nodeList[i];
            if(index<da.getIdxElementBegin()) // pre ghost element
            {
              if(min_pre>index)
                min_pre=index;

              if(max_pre<index)
                max_pre=index;

            }else if(index<da.getIdxElementEnd()) // this is a my element.
            {

              if(min_mine>index)
                min_mine=index;

              if(max_mine<index)
                max_mine=index;

            }else // this is a post ghost element.
            {

              if(min_post>index)
                min_post=index;

              if(max_post<index)
                max_post=index;
            }

          }
      index=da.curr();
      if(max_pre!=0 && min_pre!=LONG_MAX)
        diff_pre=max_pre-min_pre;
      else
        diff_pre=0;

      if(max_mine!=0 && min_mine!=LONG_MAX)
        diff_mine=max_mine-min_mine;
      else
        diff_mine=0;

      if(max_post!=0 && min_post!=LONG_MAX)
        diff_post=max_post-min_post;
      else
        diff_post=0;

      //std::cout<<"Current Node:"<<da.curr()<<"diff_pre:"<<diff_pre<<" diff_mine:"<<diff_mine<<" diff_post:"<<diff_post<<std::endl;


      fwrite(&index,sizeof(unsigned int),1,outfile);
      fwrite(nodeList,sizeof(unsigned int),8,outfile);
      fwrite(&diff_pre,sizeof(unsigned long),1,outfile);
      fwrite(&diff_mine,sizeof(unsigned long),1,outfile);
      fwrite(&diff_post,sizeof(unsigned long),1,outfile);



      if(diff_min_pre>diff_pre)
         diff_min_pre=diff_pre;

      if(diff_min_mine>diff_mine)
        diff_min_mine=diff_mine;

      if(diff_min_post>diff_post)
        diff_min_post=diff_post;


      if(diff_max_pre<diff_pre)
        diff_max_pre=diff_pre;

      if(diff_max_mine<diff_mine)
        diff_max_mine=diff_mine;

      if(diff_max_post<diff_post)
        diff_max_post=diff_post;






    }

    fclose(outfile);

/*
    FILE * instream=fopen(nlistFName,"rb");
    unsigned int nlist[8];
    unsigned long diff[3];
    unsigned int index;
    do {
      fread(&index,sizeof(unsigned int), 1,instream);
      fread(nlist, sizeof(unsigned int), 8, instream);
      fread(diff, sizeof(unsigned long), 3, instream);
      std::cout<<"index:"<<index<<"Nlist:"<<nlist[0]<<","<<nlist[1]<<","<<nlist[2]<<","<<nlist[3]<<","<<nlist[4]<<","<<nlist[5]<<","<<nlist[6]<<","<<nlist[7]<<", Diff:"<<diff[0]<<","<<diff[1]<<","<<diff[2]<<std::endl;
    }while(nlist!=NULL);

    fclose(instream);
*/



//    std::cout<<"diff_pre (min,max):"<<diff_min_pre<<","<<diff_max_pre<<std::endl;
//    std::cout<<"diff_mine (min,max):"<<diff_min_mine<<","<<diff_max_mine<<std::endl;
//    std::cout<<"diff_post (min,max):"<<diff_min_post<<","<<diff_max_post<<std::endl;

   }



  par::Mpi_Reduce(&actCnt,&actCnt_g,1,MPI_SUM,0,MPI_COMM_WORLD);

  par::Mpi_Reduce(&diff_max_pre,&diff_min_pre_g,1,MPI_MIN,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff_max_pre,&diff_max_pre_g,1,MPI_MAX,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff_max_pre,&diff_sum_pre,1,MPI_SUM,0,MPI_COMM_WORLD);



  par::Mpi_Reduce(&diff_max_mine,&diff_min_mine_g,1,MPI_MIN,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff_max_mine,&diff_max_mine_g,1,MPI_MAX,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff_max_mine,&diff_sum_mine,1,MPI_SUM,0,MPI_COMM_WORLD);



  par::Mpi_Reduce(&diff_max_post,&diff_min_post_g,1,MPI_MIN,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff_max_post,&diff_max_post_g,1,MPI_MAX,0,MPI_COMM_WORLD);
  par::Mpi_Reduce(&diff_max_post,&diff_sum_post,1,MPI_SUM,0,MPI_COMM_WORLD);






  if (!rank) {

    diff_mean_pre=diff_sum_pre/(double)(actCnt_g);
    diff_mean_mine=diff_sum_mine/(double)(actCnt_g);
    diff_mean_post=diff_sum_post/(double)(actCnt_g);



    std::cout << RED<<"=====================QUALITY OF ODA========================================"<<NRM<<std::endl;
    std::cout << "Nodes          \t(" << minNodeSize << ", " << maxNodeSize << ")" << std::endl;
    std::cout << "Boundary Node (Overall)  \t(" << minBdyNode << ", " <<meanBdyNode<<" ," << maxBdyNode << ")" << std::endl;
    std::cout << "Boundary Node (Inter Process)\t"<<"("<<preGhost[0]<<", "<<preGhost[1]<<", "<<preGhost[2]<<" )"<<std::endl;
    std::cout << "Element Size   \t(" << minElementSize << ", " << maxElementSize << ")" << std::endl;
    std::cout << "Independent    \t(" << minIndepSize << ", " << maxIndepSize << ")" << std::endl;
    std::cout << RED<<"=====================NODELIST STATISTICS========================================"<<NRM<<std::endl;
    std::cout << RED<< "Number of active ODA's:"<<actCnt_g<<NRM<<std::endl;
    std::cout << RED<<"Diff Pre Ghost:"<<"("<<diff_min_pre_g<<", "<<diff_mean_pre<<", "<<diff_max_pre_g<<")"<<NRM<<std::endl;
    std::cout << RED<<"Diff My elements:"<<"("<<diff_min_mine_g<<", "<<diff_mean_mine<<", "<<diff_max_mine_g<<")"<<NRM<<std::endl;
    std::cout << RED<<"Diff Phost Ghost:"<<"("<<diff_min_post_g<<", "<<diff_mean_post<<", "<<diff_max_post_g<<")"<<NRM<<std::endl;
    std::cout << RED<<"==========================================================================="<<NRM<<std::endl;

  }



#ifdef PETSC_USE_LOG
  PetscLogStagePop();
#endif
  if (!rank) {
    std::cout << GRN << "Finalizing PETSC" << NRM << std::endl;
  }


#ifdef HILBERT_ORDERING
  delete [] rotations;
  rotations=NULL;
  delete [] HILBERT_TABLE;
  HILBERT_TABLE=NULL;
#endif


  ot::DA_Finalize();
  PetscFinalize();



}//end function

