// example to get nodes and interpolate between two DAs.

#include "mpi.h"
#include "petsc.h"
#include "sys.h"
#include "parUtils.h"
#include "octUtils.h"
#include "odaUtils.h"
#include "TreeNode.h"
#include <cstdlib>
#include "externVars.h"
#include "dendro.h"
#include <TreeNode.h>
#include <cmath>


#include <sub_oda.h>

#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

#ifndef iC
#define iC(fun) {CHKERRQ(fun);}
#endif

#define NODE_0 1u
#define NODE_1 2u
#define NODE_2 4u
#define NODE_3 8u
#define NODE_4 16u
#define NODE_5 32u
#define NODE_6 64u
#define NODE_7 128u

#define DOF 1


// move to DA utils.
void saveNodalVecAsVTK(ot::DA* da, Vec vec, double* gsz, const char *fname);
void saveNodalVecAsVTK(ot::subDA* da, Vec vec, double* gsz, const char *fname);
void setScalarByFunction(ot::subDA* da, Vec vec, std::function<double(double,double,double)> f, double *gSize);

void interp_global_to_local(PetscScalar* glo, PetscScalar* __restrict loc, ot::DA* m_octDA);
void interp_local_to_global(PetscScalar* __restrict loc, PetscScalar* glo, ot::DA* m_octDA);



int main(int argc, char ** argv ) {
  int size, rank;
  unsigned int dim = 3;
  unsigned maxDepth = 30;
  std::vector<ot::TreeNode> nodes, nodes_bal;

  PetscInitialize(&argc, &argv, "options", NULL);
  ot::RegisterEvents();
  ot::DA_Initialize(MPI_COMM_WORLD);
  _InitializeHcurve(3);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double gsz[3] = {8.0, 8.0, 8.0};
 
  double sub_max[3] = {5.5875, 5.5875, 8.0};
  double ctr[3] = {5.5, 5.5, 5.5};
  double yr[2] = {4.4375, 5.5875};         //  .25 .375 .4375 .5 .5875 .625 .75

  //  double gsz[3] = {10.66, 10.66, 10.66};

  // double sub_max[3] = {8.66, 8.66, 10.66};
  // double ctr[3] = {4.33, 4.33, 5.33};
  // double yr[2] = {0.0, 8.66};

  double r = 0.25;
  
  auto fx_refine = [ctr, r, yr](double x, double y, double z) -> double { 
    if ( (y < yr[0]) || (y > yr[1]) || (x < yr[0]) || (x > yr[1])  ) return -1.0;
    return sqrt((x-ctr[0])*(x-ctr[0]) + (z-ctr[2])*(z-ctr[2])) - r; 
  };

  auto fx_retain = [ctr, r, yr](double x, double y, double z) -> double { 
    if ( (y < yr[0]) || (y > yr[1]) || (x < yr[0]) || (x > yr[1])  ) return -1.0;
    return 1.0; 
  };

  auto fx_u = [yr](double x, double y, double z) -> double { 
    return 1.0 + sin(0.25*M_PI*x)*sin(0.25*M_PI*y)*sin(0.25*M_PI*z)  ; //cos(M_PI*(1.0 - (x-0.5)*(x-0.5)*2.0))*cos(M_PI*(1.0 - (y-0.5)*(y-0.5)*2.0));
  };
    
  ot::DA *main_da =  ot::function_to_DA(fx_refine, fx_retain, 3, 5, 100, gsz, MPI_COMM_WORLD);
  ot::subDA *da =  new ot::subDA(main_da, fx_retain, gsz);


  PetscScalar zero = 1.0, nrm;
  Vec v;
  // if(!rank) std::cout << "Creating vector" << std::endl;
  da->createVector(v, false, false, DOF);

  VecSet(v, zero);
  
  setScalarByFunction(da, v, fx_u, gsz);

  // if (!rank) std::cout << "Saving vtk" << std::endl;
  saveNodalVecAsVTK(da, v, gsz, "orig_function" );

  if (!rank) std::cout << "======================================" << std::endl;
  // === === === === === ===   
  // ctr[0] = 6.6;
  // ctr[1] = 6.6;
  // ctr[2] = 6.6;

  // yr[0] = 5.5375;
  // yr[1] = 6.6875;

  // r = 0.4;

  ot::DA *main_da_new =  ot::function_to_DA(fx_refine, fx_retain, 4, 6, 100, gsz, MPI_COMM_WORLD);  
  ot::subDA *da_new =  new ot::subDA(main_da_new, fx_retain, gsz);

  // MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) 
    std::cout << "Constructed new DA" << std::endl;

  std::vector<double> pts;
  getNodeCoordinates(da_new, pts, gsz, sub_max);

  // std::cout << "num pts for interpolate: " << pts.size() << std::endl;

  Vec v_n;
  // std::cout << rank << ": creating new vector" << std::endl;
  da_new->createVector(v_n, false, false, DOF);

  PetscScalar minus_two = -2.0;
  VecSet(v_n, minus_two);
  
  // now interpolate 
  interpolateData(da, v, v_n, NULL, 1, pts, gsz, sub_max);

  // write out ...
  if (!rank) std::cout << "Saving interpolated vtk" << std::endl;
  saveNodalVecAsVTK(da_new, v_n, gsz, "interpolated_function" );

  // clean up
  VecDestroy(&v);
  VecDestroy(&v_n);

  // wrap up
  ot::DA_Finalize();
  PetscFinalize();
}//end main


void saveNodalVecAsVTK(ot::subDA* da, Vec vec, double* gSize, const char *file_prefix) {
  int rank, size;
  char fname[256];


	MPI_Comm_rank(da->getComm(), &rank);
	MPI_Comm_size(da->getComm(), &size);

  sprintf(fname, "%s_%05d.vtk", file_prefix, rank);

  if ( !rank ) std::cout << "Writing to VTK file: " << fname << std::endl;

  std::ofstream out;
  out.open( fname );


  out << "# vtk DataFile Version 2.0" << std::endl;
  out << "DENDRO OCTREES" << std::endl;
  out << "ASCII" << std::endl;
  out << "DATASET UNSTRUCTURED_GRID" << std::endl;

  int dim = 3;

  int unit_points = 1 << dim;
  int num_cells = 0; // da->getElementSize();
    
  for ( da->init<ot::DA_FLAGS::WRITABLE>(); 
         da->curr() < da->end<ot::DA_FLAGS::WRITABLE>(); 
        da->next<ot::DA_FLAGS::WRITABLE>() ) {

    num_cells++;
  }
  int num_vertices = num_cells * (unit_points);

  out << "POINTS " << num_vertices << " float" << std::endl;

  { // dim = 3

    unsigned int len; //!  ??
    unsigned int xl, yl, zl;  //! ??

    int num_data_field = 2; // rank and data
    

    int dof=DOF;
    PetscScalar *_vec=NULL;

    da->vecGetBuffer(vec,   _vec, false, false, true,  dof);

    da->ReadFromGhostsBegin<PetscScalar>(_vec, dof);
    da->ReadFromGhostsEnd<PetscScalar>(_vec);

    unsigned int maxD = da->getMaxDepth();
    unsigned int lev;
    double hx, hy, hz;
    Point pt;
    
    // double gSize[3] = {1.0, 1.0, 1.0};

    double xFac = gSize[0]/((double)(1<<(maxD-1)));
    double yFac = gSize[1]/((double)(1<<(maxD-1)));
    double zFac = gSize[2]/((double)(1<<(maxD-1)));
    double xx, yy, zz;
    unsigned int idx[8];

    DendroIntL* l2g = da->getLocalToGlobalMap();
         
         /*
    std::cout << "=== Printing initials ===" << std::endl;
    da->init<ot::DA_FLAGS::ALL>();
    std::cout << rank << ": init<All> " << da->curr() << std::endl; // ", " << l2g[da->curr()] << 
    da->init<ot::DA_FLAGS::INDEPENDENT>();
    std::cout << rank << ": init<Indep> " << da->curr() << std::endl;
    da->init<ot::DA_FLAGS::W_DEPENDENT>();
    std::cout << rank << ": init<W_Dep> " << da->curr() << std::endl;
    da->init<ot::DA_FLAGS::WRITABLE>();
    std::cout << rank << ": init<Writ> " << da->curr() << std::endl;
    */

    for ( da->init<ot::DA_FLAGS::WRITABLE>(); 
         da->curr() < da->end<ot::DA_FLAGS::WRITABLE>(); 
        da->next<ot::DA_FLAGS::WRITABLE>() ) {
      // set the value
      lev = da->getLevel(da->curr());
      hx = xFac*(1<<(maxD - lev));
      hy = yFac*(1<<(maxD - lev));
      hz = zFac*(1<<(maxD - lev));

      pt = da->getCurrentOffset();

      xx = pt.x()*xFac; yy = pt.y()*yFac; zz = pt.z()*zFac;

      out << pt.x()*xFac << " " <<  pt.y()*yFac << " " << pt.z()*zFac << std::endl;
      out << pt.x()*xFac + hx << " " <<  pt.y()*yFac << " " << pt.z()*zFac << std::endl;
      out << pt.x()*xFac + hx << " " <<  pt.y()*yFac + hy << " " << pt.z()*zFac << std::endl;
      out << pt.x()*xFac << " " <<  pt.y()*yFac + hy << " " << pt.z()*zFac << std::endl;

      out << pt.x()*xFac << " " <<  pt.y()*yFac << " " << pt.z()*zFac + hz<< std::endl;
      out << pt.x()*xFac + hx << " " <<  pt.y()*yFac << " " << pt.z()*zFac + hz << std::endl;
      out << pt.x()*xFac + hx << " " <<  pt.y()*yFac + hy << " " << pt.z()*zFac + hz << std::endl;
      out << pt.x()*xFac << " " <<  pt.y()*yFac + hy << " " << pt.z()*zFac + hz << std::endl;
      
      // num_cells++;
    }

    int num_cells_elements = num_cells * unit_points + num_cells;

    out << "CELLS " << num_cells << " " << num_cells_elements << std::endl;

    for (int i = 0; i < num_cells; i++) {
      out << unit_points << " ";
      for (int j = 0; j < unit_points; j++) {
        out << (i * unit_points + j) << " ";
      }
      out << std::endl;
    }

    out << "CELL_TYPES " << num_cells << std::endl;
    for (int i = 0; i < num_cells; i++) {
      out << VTK_HEXAHEDRON << std::endl;
    }

    //myfile<<"CELL_DATA "<<num_cells<<std::endl;

    out << std::endl;
    out << "POINT_DATA " << num_vertices  << std::endl;
    out << "SCALARS foo float 1" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;

    PetscScalar* local = new PetscScalar[8];

    for ( da->init<ot::DA_FLAGS::WRITABLE>(); 
         da->curr() < da->end<ot::DA_FLAGS::WRITABLE>(); 
        da->next<ot::DA_FLAGS::WRITABLE>() ) {

          // std::cout << da->curr() << ", "; // std::endl;

      da->getNodeIndices(idx);
  
      // @hari - fix this. temporarily bug in 
      // interp_global_to_local(_vec, local, da->global_domain());
      for (int q=0; q<8; ++q)
        local[q] = _vec[idx[q]];

        out << local[0] << " ";
        out << local[1] << " ";
        out << local[3] << " ";
        out << local[2] << " ";
        out << local[4] << " ";
        out << local[5] << " ";
        out << local[7] << " ";
        out << local[6] << " ";
        /*
        out << _vec[idx[0]] << " ";
        out << _vec[idx[1]] << " ";
        out << _vec[idx[3]] << " ";
        out << _vec[idx[2]] << " ";
        out << _vec[idx[4]] << " ";
        out << _vec[idx[5]] << " ";
        out << _vec[idx[7]] << " ";
        out << _vec[idx[6]] << " "; */

    }

    out << std::endl;

    delete [] local;

    da->vecRestoreBuffer(vec,  _vec, false, false, true,  dof);
  }

  out.close();
}

void saveNodalVecAsVTK(ot::DA* da, Vec vec, double* gSize, const char *file_prefix) {
  int rank, size;
  char fname[256];


	MPI_Comm_rank(da->getComm(), &rank);
	MPI_Comm_size(da->getComm(), &size);

  sprintf(fname, "%s_%05d.vtk", file_prefix, rank);

  if ( !rank ) std::cout << "Writing to VTK file: " << fname << std::endl;

  std::ofstream out;
  out.open( fname );


  out << "# vtk DataFile Version 2.0" << std::endl;
  out << "DENDRO OCTREES" << std::endl;
  out << "ASCII" << std::endl;
  out << "DATASET UNSTRUCTURED_GRID" << std::endl;

  int dim = 3;

  int unit_points = 1 << dim;
  int num_cells = 0; // da->getElementSize();
    
  for ( da->init<ot::DA_FLAGS::WRITABLE>(); 
         da->curr() < da->end<ot::DA_FLAGS::WRITABLE>(); 
        da->next<ot::DA_FLAGS::WRITABLE>() ) {

    num_cells++;
  }
  int num_vertices = num_cells * (unit_points);

  out << "POINTS " << num_vertices << " float" << std::endl;

  { // dim = 3

    unsigned int len; //!  ??
    unsigned int xl, yl, zl;  //! ??

    int num_data_field = 2; // rank and data
    

    int dof=DOF;
    PetscScalar *_vec=NULL;

    da->vecGetBuffer(vec,   _vec, false, false, true,  dof);

    da->ReadFromGhostsBegin<PetscScalar>(_vec, dof);
    da->ReadFromGhostsEnd<PetscScalar>(_vec);

    unsigned int maxD = da->getMaxDepth();
    unsigned int lev;
    double hx, hy, hz;
    Point pt;
    
    // double gSize[3] = {1.0, 1.0, 1.0};

    double xFac = gSize[0]/((double)(1<<(maxD-1)));
    double yFac = gSize[1]/((double)(1<<(maxD-1)));
    double zFac = gSize[2]/((double)(1<<(maxD-1)));
    double xx, yy, zz;
    unsigned int idx[8];

    for ( da->init<ot::DA_FLAGS::WRITABLE>(); 
         da->curr() < da->end<ot::DA_FLAGS::WRITABLE>(); 
        da->next<ot::DA_FLAGS::WRITABLE>() ) {
      // set the value
      lev = da->getLevel(da->curr());
      hx = xFac*(1<<(maxD - lev));
      hy = yFac*(1<<(maxD - lev));
      hz = zFac*(1<<(maxD - lev));

      pt = da->getCurrentOffset();

      xx = pt.x()*xFac; yy = pt.y()*yFac; zz = pt.z()*zFac;

      out << pt.x()*xFac << " " <<  pt.y()*yFac << " " << pt.z()*zFac << std::endl;
      out << pt.x()*xFac + hx << " " <<  pt.y()*yFac << " " << pt.z()*zFac << std::endl;
      out << pt.x()*xFac + hx << " " <<  pt.y()*yFac + hy << " " << pt.z()*zFac << std::endl;
      out << pt.x()*xFac << " " <<  pt.y()*yFac + hy << " " << pt.z()*zFac << std::endl;

      out << pt.x()*xFac << " " <<  pt.y()*yFac << " " << pt.z()*zFac + hz<< std::endl;
      out << pt.x()*xFac + hx << " " <<  pt.y()*yFac << " " << pt.z()*zFac + hz << std::endl;
      out << pt.x()*xFac + hx << " " <<  pt.y()*yFac + hy << " " << pt.z()*zFac + hz << std::endl;
      out << pt.x()*xFac << " " <<  pt.y()*yFac + hy << " " << pt.z()*zFac + hz << std::endl;
      
      // num_cells++;
    }

    int num_cells_elements = num_cells * unit_points + num_cells;

    out << "CELLS " << num_cells << " " << num_cells_elements << std::endl;

    for (int i = 0; i < num_cells; i++) {
      out << unit_points << " ";
      for (int j = 0; j < unit_points; j++) {
        out << (i * unit_points + j) << " ";
      }
      out << std::endl;
    }

    out << "CELL_TYPES " << num_cells << std::endl;
    for (int i = 0; i < num_cells; i++) {
      out << VTK_HEXAHEDRON << std::endl;
    }

    //myfile<<"CELL_DATA "<<num_cells<<std::endl;

    out << std::endl;
    out << "POINT_DATA " << num_vertices  << std::endl;
    out << "SCALARS foo float 1" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;

    PetscScalar* local = new PetscScalar[8];

    for ( da->init<ot::DA_FLAGS::WRITABLE>(); 
         da->curr() < da->end<ot::DA_FLAGS::WRITABLE>(); 
        da->next<ot::DA_FLAGS::WRITABLE>() ) {
      da->getNodeIndices(idx);
      interp_global_to_local(_vec, local, da);

        out << local[0] << " ";
        out << local[1] << " ";
        out << local[3] << " ";
        out << local[2] << " ";
        out << local[4] << " ";
        out << local[5] << " ";
        out << local[7] << " ";
        out << local[6] << " ";
        /*
        out << _vec[idx[0]] << " ";
        out << _vec[idx[1]] << " ";
        out << _vec[idx[3]] << " ";
        out << _vec[idx[2]] << " ";
        out << _vec[idx[4]] << " ";
        out << _vec[idx[5]] << " ";
        out << _vec[idx[7]] << " ";
        out << _vec[idx[6]] << " "; */

    }

    out << std::endl;

/*
    out << "FIELD OCTREE_DATA " << num_data_field << std::endl;

    out << "cell_level 1 " << num_cells << " int" << std::endl;

    for ( da->init<ot::DA_FLAGS::ALL>(); da->curr() < da->end<ot::DA_FLAGS::ALL>(); da->next<ot::DA_FLAGS::ALL>() ) {
      int l = da->getLevel(da->curr());
      out << l << " ";
    }

    out << std::endl;

    out << "mpi_rank 1 " << num_cells << " int" << std::endl;
    for (int i = 0; i < num_cells; i++)
      out << rank << " ";

    out << std::endl;
*/
    da->vecRestoreBuffer(vec,  _vec, false, false, true,  dof);
  }

  out.close();
}

void setScalarByFunction(ot::subDA* da, Vec vec, std::function<double(double,double,double)> f, double *gSize) {
	int dof=1;
  PetscScalar *_vec=NULL;

  da->vecGetBuffer(vec,   _vec, false, false, false,  dof);

  // da->ReadFromGhostsBegin<PetscScalar>(_vec, dof);
	// da->ReadFromGhostsEnd<PetscScalar>(_vec);

  unsigned int maxD = da->getMaxDepth();
	unsigned int lev;
	double hx, hy, hz;
	Point pt, parPt;

	double xFac = gSize[0]/((double)(1<<(maxD-1)));
	double yFac = gSize[1]/((double)(1<<(maxD-1)));
	double zFac = gSize[2]/((double)(1<<(maxD-1)));
	double xx[8], yy[8], zz[8];
	unsigned int idx[8];

  for ( da->init<ot::DA_FLAGS::ALL>(); da->curr() < da->end<ot::DA_FLAGS::ALL>(); da->next<ot::DA_FLAGS::ALL>() ) {
    // set the value
    lev = da->getLevel(da->curr());
    hx = xFac*(1<<(maxD - lev));
    hy = yFac*(1<<(maxD - lev));
    hz = zFac*(1<<(maxD - lev));

    pt = da->getCurrentOffset();

    //! get the correct coordinates of the nodes ...
    // unsigned int chNum = da->getChildNumber();
    unsigned char hangingMask = da->getHangingNodeIndex(da->curr());

    // if hanging, use parents, else mine.

    xx[0] = pt.x()*xFac; yy[0] = pt.y()*yFac; zz[0] = pt.z()*zFac;
    xx[1] = pt.x()*xFac+hx; yy[1] = pt.y()*yFac; zz[1] = pt.z()*zFac;
    xx[2] = pt.x()*xFac; yy[2] = pt.y()*yFac+hy; zz[2] = pt.z()*zFac;
    xx[3] = pt.x()*xFac+hx; yy[3] = pt.y()*yFac+hy; zz[3] = pt.z()*zFac;

    xx[4] = pt.x()*xFac; yy[4] = pt.y()*yFac; zz[4] = pt.z()*zFac+hz;
    xx[5] = pt.x()*xFac+hx; yy[5] = pt.y()*yFac; zz[5] = pt.z()*zFac+hz;
    xx[6] = pt.x()*xFac; yy[6] = pt.y()*yFac+hy; zz[6] = pt.z()*zFac+hz;
    xx[7] = pt.x()*xFac+hx; yy[7] = pt.y()*yFac+hy; zz[7] = pt.z()*zFac +hz;

    da->getNodeIndices(idx);
    for (int i=0; i<8; ++i) {
      if ( ! ( hangingMask & ( 1u << i ) ) ) {
        _vec[idx[i]] =  f(xx[i], yy[i], zz[i]); //! use correct coordinate
      }
    }
    //  std::cout << "Setting value: " << idx[0] << " to " << f(xx,yy,zz) << std::endl;
  }

  da->vecRestoreBuffer(vec,  _vec, false, false, false,  dof);

}

void interp_global_to_local(PetscScalar* glo, PetscScalar* __restrict loc, ot::DA* da) {
	unsigned int idx[8];
	unsigned char hangingMask = da->getHangingNodeIndex(da->curr());
	unsigned int chNum = da->getChildNumber();
	da->getNodeIndices(idx);

  unsigned int m_uiDof = 1;

  // std::cout << chNum << std::endl;

  switch (chNum) {

    case 0:
      // 0,7 are not hanging
		  for (size_t i = 0; i < m_uiDof; i++) {
				loc[i] = glo[m_uiDof*idx[0]+i];

        if ( hangingMask & NODE_1 )
          loc[m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] );
        else
          loc[m_uiDof + i] = glo[m_uiDof*idx[1]+i];

        if ( hangingMask & NODE_2 )
          loc[2*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[2]+i] );
        else
          loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];

        if ( hangingMask & NODE_3 )
          loc[3*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i]);
        else
          loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];

        if ( hangingMask & NODE_4 )
          loc[4*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[4]+i] );
        else
          loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];

        if ( hangingMask & NODE_5 )
          loc[5*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i]);
        else
          loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];

        if ( hangingMask & NODE_6 )
          loc[6*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[6]+i]);
        else
          loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];

        loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];
      }
      break;
    case 1:
      // 1,6 are not hanging
		  for (size_t i = 0; i < m_uiDof; i++) {

        if ( hangingMask & NODE_0 )
          loc[i] = 0.5 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] );
        else
          loc[i] = glo[m_uiDof*idx[0]+i] ;

        loc[m_uiDof + i] = glo[m_uiDof*idx[1]+i];

        if ( hangingMask & NODE_2 )
          loc[2*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i]);
        else
          loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];

        if ( hangingMask & NODE_3 )
          loc[3*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[3]+i] );
        else
          loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];

        if ( hangingMask & NODE_4 )
          loc[4*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i]);
        else
          loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];

        if ( hangingMask & NODE_5 )
          loc[5*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[5]+i] );
        else
          loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];

        loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];

        if ( hangingMask & NODE_7 )
          loc[7*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];
      }
      break;
    case 2:
      // 2,5 are not hanging
		  for (size_t i = 0; i < m_uiDof; i++) {

        if ( hangingMask & NODE_0 )
          loc[i] = 0.5 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[2]+i] );
        else
          loc[i] = glo[m_uiDof*idx[0]+i] ;

        if ( hangingMask & NODE_1 )
          loc[1*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i]);
        else
          loc[1*m_uiDof + i] = glo[m_uiDof*idx[1]+i];

        loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];

        if ( hangingMask & NODE_3 )
          loc[3*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i] );
        else
          loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];

        if ( hangingMask & NODE_4 )
          loc[4*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[6]+i]);
        else
          loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];

        loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];

        if ( hangingMask & NODE_6 )
          loc[6*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[6]+i] );
        else
          loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];

        if ( hangingMask & NODE_7 )
          loc[7*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];
      }
      break;
    case 3:
      // 3,4 are not hanging
		  for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 )
          loc[i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i]);
        else
          loc[i] = glo[m_uiDof*idx[0]+i];

        if ( hangingMask & NODE_1 )
          loc[m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[3]+i] );
        else
          loc[m_uiDof + i] = glo[m_uiDof*idx[1]+i];

        if ( hangingMask & NODE_2 )
          loc[2*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i] );
        else
          loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];

        loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];

        loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];

        if ( hangingMask & NODE_5 )
          loc[5*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];

        if ( hangingMask & NODE_6 )
          loc[6*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];

        if ( hangingMask & NODE_7 )
          loc[7*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[7]+i] );
        else
          loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];
      }
      break;
    case 4:
		  // 4,3 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 )
          loc[i] = 0.5 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[4]+i] );
        else
          loc[i] = glo[m_uiDof*idx[0]+i];

        if ( hangingMask & NODE_1 )
          loc[1*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i]);
        else
          loc[1*m_uiDof + i] = glo[m_uiDof*idx[1]+i];

        if ( hangingMask & NODE_2 )
          loc[2*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[6]+i]);
        else
          loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];

        loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];

        loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];

        if ( hangingMask & NODE_5 )
          loc[5*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i] );
        else
          loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];

        if ( hangingMask & NODE_6 )
          loc[6*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[6]+i] );
        else
          loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];

        if ( hangingMask & NODE_7 )
          loc[7*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];
      }
      break;
    case 5:
      // 5,2 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 )
          loc[i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i]);
        else
          loc[i] = glo[m_uiDof*idx[0]+i];

        if ( hangingMask & NODE_1 )
          loc[m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[5]+i] );
        else
          loc[m_uiDof + i] = glo[m_uiDof*idx[1]+i];

        loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];

        if ( hangingMask & NODE_3 )
          loc[3*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];

        if ( hangingMask & NODE_4 )
          loc[4*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i] );
        else
          loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];

        loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];

        if ( hangingMask & NODE_6 )
          loc[6*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];

        if ( hangingMask & NODE_7 )
          loc[7*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[7]+i] );
        else
          loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];
      }
      break;
    case 6:
      // 6,1 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 )
          loc[i] = 0.25 * ( glo[m_uiDof*idx[0]+i] + glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i]);
        else
          loc[i] = glo[m_uiDof*idx[0]+i];

        loc[m_uiDof + i] = glo[m_uiDof*idx[1]+i];

        if ( hangingMask & NODE_2 )
          loc[2*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[6]+i] );
        else
          loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];

        if ( hangingMask & NODE_3 )
          loc[3*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];

        if ( hangingMask & NODE_4 )
          loc[4*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[6]+i] );
        else
          loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];

        if ( hangingMask & NODE_5 )
          loc[5*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];

        loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];

        if ( hangingMask & NODE_7 )
          loc[7*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i] );
        else
          loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];

      }
      break;
    case 7:
      // 7,0 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        loc[i] = glo[m_uiDof*idx[0]+i];

        if ( hangingMask & NODE_1 )
          loc[1*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[1]+i] + glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[1*m_uiDof + i] = glo[m_uiDof*idx[1]+i];

        if ( hangingMask & NODE_2 )
          loc[2*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[2]+i] + glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];

        if ( hangingMask & NODE_3 )
          loc[3*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[3]+i] + glo[m_uiDof*idx[7]+i] );
        else
          loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];

        if ( hangingMask & NODE_4 )
          loc[4*m_uiDof + i] = 0.25 * ( glo[m_uiDof*idx[4]+i] + glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i]);
        else
          loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];

        if ( hangingMask & NODE_5 )
          loc[5*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[5]+i] + glo[m_uiDof*idx[7]+i] );
        else
          loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];

        if ( hangingMask & NODE_6 )
          loc[6*m_uiDof + i] = 0.5 * ( glo[m_uiDof*idx[6]+i] + glo[m_uiDof*idx[7]+i] );
        else
          loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];

        loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];
      }
      break;
    default:
			std::cout<<"in glo_to_loc: incorrect child num = " << chNum << std::endl;
			assert(false);
      break;
  } // switch

} // glo_to_loc

void interp_local_to_global(PetscScalar* __restrict loc, PetscScalar* glo, ot::DA* da) {
  unsigned int idx[8];
	unsigned char hangingMask = da->getHangingNodeIndex(da->curr());
	unsigned int chNum = da->getChildNumber();
	da->getNodeIndices(idx);

  unsigned int m_uiDof = 1;

  switch (chNum) {
    case 0:
      // 0,7 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        glo[m_uiDof*idx[0]+i] += loc[i];
        if ( hangingMask & NODE_1 ) {
          glo[m_uiDof*idx[1]+i] += 0.5*loc[m_uiDof + i];
          glo[m_uiDof*idx[0]+i] += 0.5*loc[m_uiDof + i];
        } else {
          glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];
        }
        if ( hangingMask & NODE_2 ) {
          glo[m_uiDof*idx[2]+i] += 0.5*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[0]+i] += 0.5*loc[2*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];
        }
        if ( hangingMask & NODE_3 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[2]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[3*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
        }
        if ( hangingMask & NODE_4 ) {
          glo[m_uiDof*idx[4]+i] += 0.5*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[0]+i] += 0.5*loc[4*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];
        }
        if ( hangingMask & NODE_5 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[4]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[5*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
        }
        if ( hangingMask & NODE_6 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[2]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[4]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[6*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
        }
        glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
      }
      break;
    case 1:
		  for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 ) {
          glo[m_uiDof*idx[0]+i] += 0.5*loc[i];
          glo[m_uiDof*idx[1]+i] += 0.5*loc[i];
        } else {
          glo[m_uiDof*idx[0]+i] += loc[i];
        }

        glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];

        if ( hangingMask & NODE_2 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[2]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[2*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];
        }

        if ( hangingMask & NODE_3 ) {
          glo[m_uiDof*idx[1]+i] += 0.5*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.5*loc[3*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
        }

        if ( hangingMask & NODE_4 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[4]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[4*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];
        }

        if ( hangingMask & NODE_5 ) {
          glo[m_uiDof*idx[1]+i] += 0.5*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.5*loc[5*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
        }
          
        glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
        
        if ( hangingMask & NODE_7 ) {
          glo[m_uiDof*idx[1]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[7*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
        }
      }
      break;
    case 2:
      // 2,5 are not hanging
		  for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 ) {
          glo[m_uiDof*idx[0]+i] += 0.5*loc[i];
          glo[m_uiDof*idx[2]+i] += 0.5*loc[i];
        } else {
          glo[m_uiDof*idx[0]+i] += loc[i];
        }

        if ( hangingMask & NODE_1 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[2]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[m_uiDof + i];
        } else {
          glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];
        }

        glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];

        if ( hangingMask & NODE_3 ) {
          glo[m_uiDof*idx[2]+i] += 0.5*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.5*loc[3*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
        }

        if ( hangingMask & NODE_4 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[2]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[4]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[4*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];
        }

        glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
          
        if ( hangingMask & NODE_6 ) {
          glo[m_uiDof*idx[2]+i] += 0.5*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.5*loc[6*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
        }
        
        if ( hangingMask & NODE_7 ) {
          glo[m_uiDof*idx[2]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[7*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
        }
      }
      break;
    case 3:
      // 3,4 are not hanging
		  for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[2]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[i];
        } else {
          glo[m_uiDof*idx[0]+i] += loc[i];
        }

        if ( hangingMask & NODE_1 ) {
          glo[m_uiDof*idx[1]+i] += 0.5*loc[m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.5*loc[m_uiDof + i];
        } else {
          glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];
        }

        if ( hangingMask & NODE_2 ) {
          glo[m_uiDof*idx[2]+i] += 0.5*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.5*loc[2*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];
        }

        glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
        glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];

        if ( hangingMask & NODE_5 ) {
          glo[m_uiDof*idx[1]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[5*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
        }
        if ( hangingMask & NODE_6 ) {
          glo[m_uiDof*idx[2]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[6*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
        }
        if ( hangingMask & NODE_7 ) {
          glo[m_uiDof*idx[3]+i] += 0.5*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.5*loc[7*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
        }
      }
      break;
    case 4:
		  // 4,3 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 ) {
          glo[m_uiDof*idx[0]+i] += 0.5*loc[i];
          glo[m_uiDof*idx[4]+i] += 0.5*loc[i];
        } else {
          glo[m_uiDof*idx[0]+i] += loc[i];
        }
        if ( hangingMask & NODE_1 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[4]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[m_uiDof + i];
        } else {
          glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];
        }

        if ( hangingMask & NODE_2 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[2]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[4]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[2*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];
        }

        glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
        glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];
        
        if ( hangingMask & NODE_5 ) {
          glo[m_uiDof*idx[4]+i] += 0.5*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.5*loc[5*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
        }
        if ( hangingMask & NODE_6 ) {
          glo[m_uiDof*idx[4]+i] += 0.5*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.5*loc[6*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
        }
        if ( hangingMask & NODE_7 ) {
          glo[m_uiDof*idx[4]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[7*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
        }
      }
      break;
    case 5:
      // 5,2 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[4]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[i];
        } else {
          glo[m_uiDof*idx[0]+i] += loc[i];
        }
        if ( hangingMask & NODE_1 ) {
          glo[m_uiDof*idx[1]+i] += 0.5*loc[m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.5*loc[m_uiDof + i];
        } else {
          glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];
        }
        glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];
        if ( hangingMask & NODE_3 ) {
          glo[m_uiDof*idx[1]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[3*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
        }
        if ( hangingMask & NODE_4 ) {
          glo[m_uiDof*idx[4]+i] += 0.5*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.5*loc[4*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];
        }
        glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
        if ( hangingMask & NODE_6 ) {
          glo[m_uiDof*idx[4]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[6*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
        }
        if ( hangingMask & NODE_7 ) {
          glo[m_uiDof*idx[5]+i] += 0.5*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.5*loc[7*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
        }
      }
      break;
    case 6:
      // 6,1 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        if ( hangingMask & NODE_0 ) {
          glo[m_uiDof*idx[0]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[1]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[4]+i] += 0.25*loc[i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[i];
        } else {
          glo[m_uiDof*idx[0]+i] += loc[i];
        }
        glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];
        if ( hangingMask & NODE_2 ) {
          glo[m_uiDof*idx[2]+i] += 0.5*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.5*loc[2*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];
        }
        if ( hangingMask & NODE_3 ) {
          glo[m_uiDof*idx[2]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[3*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
        }
        if ( hangingMask & NODE_4 ) {
          glo[m_uiDof*idx[4]+i] += 0.5*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.5*loc[4*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];
        }
        if ( hangingMask & NODE_5 ) {
          glo[m_uiDof*idx[4]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[5*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
        }
        glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
        if ( hangingMask & NODE_7 ) {
          glo[m_uiDof*idx[6]+i] += 0.5*loc[7*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.5*loc[7*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
        }
      }
      break;
    case 7:
      // 7,0 are not hanging
      for (size_t i = 0; i < m_uiDof; i++) {
        glo[m_uiDof*idx[0]+i] += loc[i];
        if ( hangingMask & NODE_1 ) {
          glo[m_uiDof*idx[1]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[m_uiDof + i];
        } else {
          glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];
        }
        if ( hangingMask & NODE_2 ) {
          glo[m_uiDof*idx[2]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[3]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[2*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[2*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];
        }
        if ( hangingMask & NODE_3 ) {
          glo[m_uiDof*idx[3]+i] += 0.5*loc[3*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.5*loc[3*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
        }

        if ( hangingMask & NODE_4 ) {
          glo[m_uiDof*idx[4]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[5]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[6]+i] += 0.25*loc[4*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.25*loc[4*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];
        }
        if ( hangingMask & NODE_5 ) {
          glo[m_uiDof*idx[5]+i] += 0.5*loc[5*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.5*loc[5*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
        }
        if ( hangingMask & NODE_6 ) {
          glo[m_uiDof*idx[6]+i] += 0.5*loc[6*m_uiDof + i];
          glo[m_uiDof*idx[7]+i] += 0.5*loc[6*m_uiDof + i];
        } else {
          glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
        }
        glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
      }
    break;
    default:
			std::cout<<"in loc_to_glo: incorrect child num = " << chNum << std::endl;
			assert(false);
      break;
  } // switch chNum
} // loc_to_glo

