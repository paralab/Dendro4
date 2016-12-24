static char help[] = "Driver for heat";

#include "externVars.h"
#include "dendro.h"
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <vector>

#include "petscksp.h"

#include "timeInfo.h"

#include "massMatrix.h"
#include "parabolic.h"
#include "stiffnessMatrix.h"

// #include "VecIO.h"
#include "rhs.h"

int main(int argc, char **argv)
{       
  PetscInitialize(&argc, &argv, "heat.opt", help);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  int Ns = 32;
  unsigned int dof = 1;

  char problemName[PETSC_MAX_PATH_LEN];
  char filename[PETSC_MAX_PATH_LEN];

  double t0 = 0.0;
  double dt = 0.001;
  double t1 = 0.01;

  // double dtratio = 1.0;
  DM  da;         // Underlying scalar DA - for scalar properties

  Vec rho;        // density - elemental scalar

  // Initial conditions
  Vec initialTemperature; 

  timeInfo ti;

  PetscBool mf = PETSC_FALSE;
  bool mfree = false;
  PetscOptionsGetBool(0, "-mfree", &mf, 0);
  
  if (mf == PETSC_TRUE) {
    mfree = true;
  } else 
    mfree = false;

  // get Ns
  CHKERRQ ( PetscOptionsGetInt(0,"-Ns",&Ns,0) );
  CHKERRQ ( PetscOptionsGetScalar(0,"-t0",&t0,0) );
  CHKERRQ ( PetscOptionsGetScalar(0,"-t1",&t1,0) );
  CHKERRQ ( PetscOptionsGetScalar(0,"-dt",&dt,0) );
  CHKERRQ ( PetscOptionsGetString(PETSC_NULL, "-pn",problemName,PETSC_MAX_PATH_LEN-1,PETSC_NULL));

  // Time info for timestepping
  ti.start = t0;
  ti.stop  = t1;
  ti.step  = dt;

  if (!rank) {
    std::cout << "Grid size is " << Ns+1 << " and NT is " << (int)ceil(1.0/dt) << std::endl;
  }

  // create DA
  CHKERRQ ( DMDACreate3d ( PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, 
                    Ns+1, Ns+1, Ns+1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                    1, 1, 0, 0, 0, &da) );
  massMatrix* Mass = new massMatrix(feMat::PETSC); // Mass Matrix
  stiffnessMatrix* Stiffness = new stiffnessMatrix(feMat::PETSC); // Stiffness matrix
  forceVector* Force = new forceVector(feVec::PETSC);  // force term

  // create vectors 
  CHKERRQ( DMCreateGlobalVector(da, &rho) );

  CHKERRQ( DMCreateGlobalVector(da, &initialTemperature) );

  // Set initial conditions
  CHKERRQ( VecSet ( initialTemperature, 0.0) ); 

  VecZeroEntries( rho );

  int x, y, z, m, n, p;
  int mx,my,mz, xne, yne, zne;

  CHKERRQ( DMDAGetCorners(da, &x, &y, &z, &m, &n, &p) ); 
  CHKERRQ( DMDAGetInfo(da,0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0) ); 

  if (x+m == mx) {
    xne=m-1;
  } else {
    xne=m;
  }
  if (y+n == my) {
    yne=n-1;
  } else {
    yne=n;
  }
  if (z+p == mz) {
    zne=p-1;
  } else {
    zne=p;
  }

  double acx,acy,acz;
  double hx = 1.0/((double)Ns);

  // SET MATERIAL PROPERTIES ...
  unsigned int elemSize = Ns*Ns*Ns;  // number of elements
  std::cout << "Elem size is " << elemSize << std::endl;
  unsigned int nodeSize = (Ns+1)*(Ns+1)*(Ns+1);  // number of nodes

  // Set Elemental material properties
  PetscScalar ***initialTemperatureArray, ***rhoArray;

  CHKERRQ(DMDAVecGetArray(da, initialTemperature, &initialTemperatureArray));
  CHKERRQ(DMDAVecGetArray(da, rho, &rhoArray));

  std::cout << "Setting initial guess." << std::endl;

  // loop through all nodes ...
  for (int k=z; k < z+p; k++) {
    for (int j=y; j < y+n; j++) {
      for (int i=x; i < x+m; i++) {
        double coords[3] = { hx*i, hx*j, hx*k };
        double ic = sin(M_PI * coords[0]) * sin(M_PI * coords[1]) * sin(M_PI * coords[2]);
        //std::cout << "ic at " << coords[0] << ", " << coords[1] << ", " << coords[2] << ": " << ic << "\n";

        initialTemperatureArray[k][j][i] = ic;
        rhoArray[k][j][i] = 1.0;
      } // end i
    } // end j
  } // end k

  std::cout << "Finished initial conditions loop." << std::endl;

  CHKERRQ( DMDAVecRestoreArray ( da, initialTemperature, &initialTemperatureArray ) );
  CHKERRQ( DMDAVecRestoreArray ( da, rho, &rhoArray ) );

/*{
    std::stringstream ss;
    ss << "ic.m";

    PetscViewer view;
    PetscViewerCreate(PETSC_COMM_WORLD, &view);
    PetscViewerPushFormat(view, PETSC_VIEWER_ASCII_MATLAB); 
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, ss.str().c_str(), &view);
    VecView(initialTemperature, view);
    PetscViewerDestroy(&view);
}*/
  // write_vector("ic.plt", initialTemperature, da);
 
  // DONE - SET MATERIAL PROPERTIES ...
  unsigned int numSteps = (unsigned int)(ceil(( ti.stop - ti.start)/ti.step));
  std::cout << "Numsteps is " << numSteps << std::endl;

  // Setup Matrices and Force Vector ...
  Mass->setProblemDimensions(1.0, 1.0, 1.0);
  Mass->setDA(da);
  Mass->setDof(dof);

  Stiffness->setProblemDimensions(1.0, 1.0, 1.0);
  Stiffness->setDA(da);
  Stiffness->setDof(dof);
  Stiffness->setNuVec(rho);

  Force->setProblemDimensions(1.0, 1.0, 1.0);
  Force->setDA(da);
  Force->setDof(dof);

  // Newmark time stepper ...
  parabolic *ts = new parabolic; 

  ts->setMassMatrix(Mass);
  ts->setStiffnessMatrix(Stiffness);
  ts->setForceVector(Force);
  ts->setTimeFrames(1);

  ts->setInitialTemperature(initialTemperature);

  ts->setTimeInfo(&ti);
  ts->setAdjoint(false); // set if adjoint or forward
  ts->setDAForMonitor(da);
  //ts->useMatrixFree(mfree);

  if (!rank)
    std::cout <<"Initializing parabolic"<< std::endl;

  double itime = MPI_Wtime();
	ts->init(); // initialize IMPORTANT 
  if (!rank)
    std::cout <<"Starting parabolic Solve"<< std::endl;
  double stime = MPI_Wtime();
  ts->solve();// solve 
  double etime = MPI_Wtime();
  if (!rank)
    std::cout <<"Done parabolic"<< std::endl;
  if (!rank) {
		std::cout << "Total time for init is " << stime - itime << std::endl;
    std::cout << "Total time for solve is " << etime - stime << std::endl;
  }

  PetscFinalize();
}

