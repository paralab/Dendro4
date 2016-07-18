//
// Created by milinda on 6/23/16.
// @author: Milinda Shayamal Fernando
// School of Computing, University of Utah
//
// Example (similar version of the tstMatvec but using the treeSort functionality implemented in sfcSort.h)
//

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
#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>
#include <time.h>

#include "genPts_par.h"
#include <climits>
#include <chrono>
#include <thread>
#include "sfcSort.h"
#include "testUtils.h"


double**** LaplacianType2Stencil;
double**** MassType2Stencil;

#ifdef PETSC_USE_LOG
//user-defined variables
int Jac1DiagEvent;
int Jac1MultEvent;
int Jac1FinestDiagEvent;
int Jac1FinestMultEvent;
#endif

char sendComMapFileName[256];
char recvComMapFileName[256];


const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}




int main(int argc, char ** argv )
{



    int npes, rank;
    bool incCorner = 1;

    // Default values.
    unsigned int solveU = 0;
    unsigned int writeB = 0;
    unsigned int numLoops = 100;
    double tol=0.1;

    MPI_Comm globalComm=MPI_COMM_WORLD;

    char Kstr[20];
    char pFile[256],bFile[256],uFile[256];

    double gSize[3];
    unsigned int ptsLen;
    DendroIntL grainSize =10000;
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

    PetscInitialize(&argc,&argv,"options",NULL);

    ot::RegisterEvents();
    ot::DA_Initialize(globalComm);
    PetscErrorPrintf = PetscErrorPrintfNone;

    MPI_Comm_size(globalComm,&npes);
    MPI_Comm_rank(globalComm,&rank);


    if(argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << "inpfile grainSz[10000] dim[3] maxDepth[30] tol[0-1] genPts[1] solveU[0] writeB[0] maxNumPtsPerOctant[1] incCorner[1] numLoops[100] compressLut[1]" << std::endl;
        return -1;
    }

    if(argc > 2) {
        grainSize = atoi(argv[2]);
    }
    if(argc > 3) {
        dim = atoi(argv[3]);
    }
    if(argc > 4) {
        maxDepth = atoi(argv[4]);
    }
    if(argc > 5) {
        tol = atof(argv[5]);
    }
    if(argc > 6) {
        genPts = atoi(argv[6]);
    }
    if(argc > 7) {
        solveU = atoi(argv[7]);
    }
    if(argc > 8) {
        writeB = atoi(argv[8]);
    }
    if(argc > 9) {
        maxNumPts = atoi(argv[9]);
    }if(argc > 10) {
        incCorner = atoi(argv[10]);
    }if(argc > 11) {
        numLoops = atoi(argv[11]);
    }if(argc > 12) {
        compressLut = atoi(argv[12]);
    }


#ifdef HILBERT_ORDERING
    G_MAX_DEPTH = maxDepth;
    G_dim = dim;
#endif
    _InitializeHcurve(dim);

#ifdef POWER_MEASUREMENT_TIMESTEP
    time_t rawtime;
    struct tm * ptm;
#endif


    if (!rank) {
        std::cout << BLU << "===============================================" << NRM << std::endl;
        std::cout << " Input Parameters" << std::endl;
        std::cout << " Input File Prefix:" << argv[1] << std::endl;
        std::cout << " Gen Pts files: " << genPts << std::endl;
        std::cout << " Number of Points per process: " << grainSize << std::endl;
        std::cout << " Dim: "<<dim<<std::endl;
        std::cout << " Max Depth: " << maxDepth << std::endl;
        std::cout << " Tol: "<<tol<<std::endl;
        std::cout << " MatVec number of iterations: "<<numLoops<<std::endl;
        std::cout << BLU << "===============================================" << NRM << std::endl;


    }


#ifdef HILBERT_ORDERING
    sprintf(sendComMapFileName,"sendCommMap_H_tol_%f_npes_%d_pts_%d_ps%d_%d.csv",tol,npes,grainSize,rank,npes);
    sprintf(recvComMapFileName,"recvCommMap_H_tol_%f_npes_%d_pts_%d_ps%d_%d.csv",tol,npes,grainSize,rank,npes);

#else
    sprintf(sendComMapFileName,"sendCommMap_M_tol_%f_npes_%d_pts_%d_ps%d_%d.csv",tol,npes,grainSize,rank,npes);
        sprintf(recvComMapFileName,"recvCommMap_M_tol_%f_npes_%d_pts_%d_ps%d_%d.csv",tol,npes,grainSize,rank,npes);

#endif



    if(genPts==1)
    {
    genGauss(0.15,grainSize,dim,argv[1],globalComm);
    }

    //genGauss(0.15,grainSize,dim,pts);

    sprintf(pFile, "%s%d_%d.pts", argv[1], rank, npes);
    //std::cout<<"Attempt to Read "<<ptsFileName<<std::endl;

    //Read pts from files
    if (!rank) {
        std::cout << RED " Reading  " << argv[1] << NRM << std::endl; // Point size
    }
    ot::readPtsFromFile(pFile, pts);

    if (!rank) {
        std::cout << GRN " Finished reading  " << argv[1] << NRM << std::endl; // Point size
    }


    ptsLen=pts.size();
    std::vector<ot::TreeNode> tmpNodes;
    DendroIntL totPts=grainSize*dim;

#ifdef DIM_2

    for (DendroIntL i = 0; i < totPts; i += 2) {
        if ((pts[i] > 0.0) &&
            (pts[i + 1] > 0.0) &&
            (((unsigned int) (pts[i] * ((double) (1u << maxDepth)))) < (1u << maxDepth)) &&
            (((unsigned int) (pts[i + 1] * ((double) (1u << maxDepth)))) < (1u << maxDepth)) ) {

            tmpNodes.push_back(ot::TreeNode((unsigned int) (pts[i] * (double) (1u << maxDepth)),
                                            (unsigned int) (pts[i + 1] * (double) (1u << maxDepth)),
                                            0,maxDepth, dim, maxDepth));
        }
    }

#else
    for (DendroIntL i = 0; i < totPts; i += 3) {
        if ((pts[i] > 0.0) &&
            (pts[i + 1] > 0.0)
            && (pts[i + 2] > 0.0) &&
            (((unsigned int) (pts[i] * ((double) (1u << maxDepth)))) < (1u << maxDepth)) &&
            (((unsigned int) (pts[i + 1] * ((double) (1u << maxDepth)))) < (1u << maxDepth)) &&
            (((unsigned int) (pts[i + 2] * ((double) (1u << maxDepth)))) < (1u << maxDepth))) {

            tmpNodes.push_back(ot::TreeNode((unsigned int) (pts[i] * (double) (1u << maxDepth)),
                                            (unsigned int) (pts[i + 1] * (double) (1u << maxDepth)),
                                            (unsigned int) (pts[i + 2] * (double) (1u << maxDepth)),
                                            maxDepth, dim, maxDepth));
        }
    }
#endif

    pts.clear();

    SFC::parSort::SFC_Sort_RemoveDuplicates(tmpNodes,tol,maxDepth,false,globalComm);
    std::swap(linOct,tmpNodes);

    //assert( par::test::isUniqueAndSorted(linOct,globalComm));


    //par::partitionW(linOct,NULL,globalComm);

    //SFC::parSort::SFC_3D_Sort(linOct,tol,maxDepth,globalComm);


    locSz = linOct.size();
    par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << " # pts= " << totalSz << std::endl;
    }

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
#ifdef POWER_MEASUREMENT_TIMESTEP
    time ( &rawtime );
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    ptm = gmtime ( &rawtime );
    if(!rank) std::cout<<" points2Octree Begin: "<<(ptm->tm_year+1900)<<"-"<<(ptm->tm_mon+1)<<"-"<<ptm->tm_mday<<" "<<(ptm->tm_hour%24)<<":"<<ptm->tm_min<<":"<<ptm->tm_sec<<std::endl;
#endif

    startTime = MPI_Wtime();
    ot::points2Octree(pts, gSize, linOct, dim, maxDepth, maxNumPts, MPI_COMM_WORLD);
    endTime = MPI_Wtime();
#ifdef POWER_MEASUREMENT_TIMESTEP
    time ( &rawtime );
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    ptm = gmtime ( &rawtime );
    if(!rank) std::cout<<" points2Octree End: "<<(ptm->tm_year+1900)<<"-"<<(ptm->tm_mon+1)<<"-"<<ptm->tm_mday<<" "<<(ptm->tm_hour%24)<<":"<<ptm->tm_min<<":"<<ptm->tm_sec<<std::endl;
#endif

    //SFC::parSort::SFC_3D_Sort(linOct,tol,maxDepth,globalComm);


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

#ifdef POWER_MEASUREMENT_TIMESTEP
    time ( &rawtime );
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    ptm = gmtime ( &rawtime );
    if(!rank) std::cout<<" balOCt Begin: "<<(ptm->tm_year+1900)<<"-"<<(ptm->tm_mon+1)<<"-"<<ptm->tm_mday<<" "<<(ptm->tm_hour%24)<<":"<<ptm->tm_min<<":"<<ptm->tm_sec<<std::endl;
#endif

    startTime = MPI_Wtime();
    ot::balanceOctree(linOct, balOct, dim, maxDepth, incCorner, MPI_COMM_WORLD, NULL, NULL);
    endTime = MPI_Wtime();
#ifdef POWER_MEASUREMENT_TIMESTEP
    time ( &rawtime );
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    ptm = gmtime ( &rawtime );
    if(!rank) std::cout<<" balOCt End: "<<(ptm->tm_year+1900)<<"-"<<(ptm->tm_mon+1)<<"-"<<ptm->tm_mday<<" "<<(ptm->tm_hour%24)<<":"<<ptm->tm_min<<":"<<ptm->tm_sec<<std::endl;
#endif

    //SFC::parSort::SFC_3D_Sort(balOct,tol,maxDepth,globalComm);


    locSz = balOct.size();
    localTime = endTime - startTime;
    par::Mpi_Reduce<DendroIntL>(&locSz, &totalSz, 1, MPI_SUM, 0, MPI_COMM_WORLD);
    par::Mpi_Reduce<double>(&localTime, &totalTime, 1, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!rank) {
        std::cout << "# of Balanced Octants: " << totalSz << std::endl;
        std::cout << "bal Time: " << totalTime << std::endl;
    }

    //treeNodesTovtk(balOct,rank,"balOCt_dendro");

#ifdef POWER_MEASUREMENT_TIMESTEP
    time ( &rawtime );
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    ptm = gmtime ( &rawtime );
    if(!rank) std::cout<<" oda Begin: "<<(ptm->tm_year+1900)<<"-"<<(ptm->tm_mon+1)<<"-"<<ptm->tm_mday<<" "<<(ptm->tm_hour%24)<<":"<<ptm->tm_min<<":"<<ptm->tm_sec<<std::endl;
#endif

    startTime = MPI_Wtime();
    ot::DA da(balOct, MPI_COMM_WORLD, MPI_COMM_WORLD,tol,compressLut);
    endTime = MPI_Wtime();

#ifdef POWER_MEASUREMENT_TIMESTEP
    time ( &rawtime );
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    ptm = gmtime ( &rawtime );
    if(!rank) std::cout<<" oda End: "<<(ptm->tm_year+1900)<<"-"<<(ptm->tm_mon+1)<<"-"<<ptm->tm_mday<<" "<<(ptm->tm_hour%24)<<":"<<ptm->tm_min<<":"<<ptm->tm_sec<<std::endl;
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


#ifdef POWER_MEASUREMENT_TIMESTEP

    time ( &rawtime );
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    ptm = gmtime ( &rawtime );
    if(!rank) std::cout<<" MatVec Begin: "<<(ptm->tm_year+1900)<<"-"<<(ptm->tm_mon+1)<<"-"<<ptm->tm_mday<<" "<<(ptm->tm_hour%24)<<":"<<ptm->tm_min<<":"<<ptm->tm_sec<<std::endl;
#endif

    for(unsigned int i=0;i<numLoops;i++) {
        iC(Jacobian1MatGetDiagonal(J, diag));
        iC(Jacobian1MatMult(J, in, out));
    }

#ifdef POWER_MEASUREMENT_TIMESTEP
  time ( &rawtime );
  ptm = gmtime ( &rawtime );
  if(!rank) std::cout<<" MatVec End: "<<(ptm->tm_year+1900)<<"-"<<(ptm->tm_mon+1)<<"-"<<ptm->tm_mday<<" "<<(ptm->tm_hour%24)<<":"<<ptm->tm_min<<":"<<ptm->tm_sec<<std::endl;
#endif

    VecDestroy(&in);
    VecDestroy(&out);
    VecDestroy(&diag);

    iC(Jacobian1MatDestroy(J));

    destroyLmatType2(LaplacianType2Stencil);
    destroyMmatType2(MassType2Stencil);

    if(!rank) {
        std::cout << "Destroyed stencils."<< std::endl;
    }

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


}