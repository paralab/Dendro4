
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


//void handler (int sig) {
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//  // char fname[256];
//  // sprintf(fname, "trace%.2d", rank);
//  // FILE *out = fopen(fname, "w");
//  unsigned int max_frames = 63;
//
//  // if (!rank) {
//  printf("%s---------------------------------%s\n", RED, NRM);
//  printf("%sError:%s signal %d:\n", RED, NRM, sig);
//  printf("%s---------------------------------%s\n", RED, NRM);
//  printf("\n%s======= stack trace =======%s\n", GRN, NRM);
//  // }
//
//  // fprintf(out, "======= stack trace =======\n");
//
//  // storage array for stack trace address data
//  void *addrlist[max_frames + 1];
//
//  // retrieve current stack addresses
//  int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void *));
//
//  if (addrlen == 0) {
//    // if (!rank)
//    fprintf(stderr, "%s  <empty, possibly corrupt>%s\n",RED, NRM);
//
//    // fprintf(out, "    <empty, possibly corrupt>\n");
//    return;
//  }
//
//  // resolve addresses into strings containing "filename(function+address)",
//  // this array must be free()-ed
//  char **symbollist = backtrace_symbols(addrlist, addrlen);
//
//  // allocate string which will be filled with the demangled function name
//  size_t funcnamesize = 256;
//  char *funcname = (char *) malloc(funcnamesize);
//
//  // iterate over the returned symbol lines. skip the first, it is the
//  // address of this function.
//  for (int i = 1; i < addrlen; i++) {
//    char *begin_name = 0, *begin_offset = 0, *end_offset = 0;
//
//    // find parentheses and +address offset surrounding the mangled name:
//    // ./module(function+0x15c) [0x8048a6d]
//    for (char *p = symbollist[i]; *p; ++p) {
//      if (*p == '(')
//        begin_name = p;
//      else if (*p == '+')
//        begin_offset = p;
//      else if (*p == ')' && begin_offset) {
//        end_offset = p;
//        break;
//      }
//    }
//
//    if (begin_name && begin_offset && end_offset
//        && begin_name < begin_offset) {
//      *begin_name++ = '\0';
//      *begin_offset++ = '\0';
//      *end_offset = '\0';
//
//      // mangled name is now in [begin_name, begin_offset) and caller
//      // offset in [begin_offset, end_offset). now apply
//      // __cxa_demangle():
//
//      int status;
//      char *ret = abi::__cxa_demangle(begin_name,
//                                      funcname, &funcnamesize, &status);
//      if (status == 0) {
//        funcname = ret; // use possibly realloc()-ed string
//        // if (!rank)
//        printf("%s[%.2d]%s%s : %s%s%s : \n",RED,rank,YLW, symbollist[i], MAG, funcname,NRM);
//
//        // fprintf(out, "%s : %s : ", symbollist[i], funcname);
//      }
//      else {
//        // demangling failed. Output function name as a C function with
//        // no arguments.
//        // if (!rank)
//        printf("%s[%.2d]%s%s : %s%s()%s : \n", RED,rank, YLW, symbollist[i], GRN,begin_name, NRM);
//
//        // fprintf(out, "%s : %s() : ", symbollist[i], begin_name);
//      }
//      size_t p = 0;
//      char syscom[256];
//      while(symbollist[i][p] != '(' && symbollist[i][p] != ' ' && symbollist[i][p] != 0)
//        ++p;
//
//      sprintf(syscom,"addr2line %p -e %.*s", addrlist[i], p, symbollist[i]);
//      //last parameter is the file name of the symbol
//      system(syscom);
//    }
//    else {
//      // couldn't parse the line? print the whole line.
//      // if (!rank)
//      printf("%sCouldn't Parse:%s  %s\n", RED, NRM, symbollist[i]);
//
//      // fprintf(out, "  %s\n", symbollist[i]);
//    }
//  }
//
//  free(funcname);
//  free(symbollist);
//  // fclose(out);
//
//  exit(1);
//}




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
  DendroIntL grainSize =10000;

  PetscInitialize(&argc,&argv,"options",NULL);
  ot::RegisterEvents();
  ot::DA_Initialize(MPI_COMM_WORLD);
  PetscErrorPrintf = PetscErrorPrintfNone;



//  signal(SIGSEGV, handler);   // install our handler
//  signal(SIGTERM, handler);   // install our handler


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
  if(argc >11 ) { grainSize =atol(argv[11]);}

  if(genPts)
  {
    genGauss(0.15,grainSize,dim,argv[1],MPI_COMM_WORLD);

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
    std::cout << "Number of Points per process:: " << grainSize << std::endl;
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
  MPI_Barrier(MPI_COMM_WORLD);
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
  //std::cout<<"linOct:"<<linOct.size()<<std::endl;
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

#ifdef HILBERT_ORDERING
  da.computeHilbertRotations();
  if(!rank)
    std::cout<<"ODA Rotation pattern computation completed for Hilbert. "<<std::endl;
#endif




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


  unsigned long diff_pre=0,diff_mine=0,diff_post=0;
  unsigned long diff_min_pre=LONG_MAX,diff_max_pre=0,diff_sum_pre;
  unsigned long diff_min_mine=LONG_MAX,diff_max_mine=0,diff_sum_mine;
  unsigned long diff_min_post=LONG_MAX,diff_max_post=0,diff_sum_post;

  unsigned long diff_min_pre_g, diff_max_pre_g;
  unsigned long diff_min_mine_g, diff_max_mine_g;
  unsigned long diff_min_post_g, diff_max_post_g;


  double diff_mean_pre, diff_mean_mine, diff_mean_post;

  unsigned long min_pre=LONG_MAX,max_pre=0, min_mine=LONG_MAX,max_mine=0, min_post=LONG_MAX,max_post=0;
   char fileName[256];
  sprintf(fileName, "%s_%d_%d_%d_%d.%s", "nlist", maxDepth, grainSize, rank, size,"bin");
  int actCnt=0;
  int actCnt_g=0;

  if(da.iAmActive()) {
    actCnt=1;
    FILE* outfile = fopen(fileName,"wb");

    for (da.init<ot::DA_FLAGS::ALL>(); da.curr() < da.end<ot::DA_FLAGS::ALL>(); da.next<ot::DA_FLAGS::ALL>()) {
      unsigned int nodeList[8];
      da.getNodeIndices(nodeList);
       unsigned int index;
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

    /*FILE * instream=fopen(fileName,"rb");
    unsigned int nlist[8];
    unsigned long diff[3];
    unsigned int index;
    do {
      fread(&index,sizeof(unsigned int), 1,instream);
      fread(nlist, sizeof(unsigned int), 8, instream);
      fread(diff, sizeof(unsigned long), 3, instream);
      std::cout<<"index:"<<index<<"Nlist:"<<nlist[0]<<","<<nlist[1]<<","<<nlist[2]<<","<<nlist[3]<<","<<nlist[4]<<","<<nlist[5]<<","<<nlist[6]<<","<<nlist[7]<<", Diff:"<<diff[0]<<","<<diff[1]<<","<<diff[2]<<std::endl;
    }while(nlist!=NULL);

    fclose(instream);*/




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
    std::cout << "Boundary Node  \t(" << minBdyNode << ", " << maxBdyNode << ")" << std::endl;
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

