
#include "mpi.h"
#include "petsc.h"
#include "sys.h"
#include "parUtils.h"
#include "octUtils.h"
#include "TreeNode.h"
#include <cstdlib>
#include "externVars.h"
#include "dendro.h"
#include <TreeNode.h>

#include <treenode2vtk.h>

#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

#ifndef iC
#define iC(fun) {CHKERRQ(fun);}
#endif

double gSize[3];

#define SQRT_3 1.7320508075688772

void saveTreeAsVTK(std::vector<ot::TreeNode>& tree, const char *file_prefix, MPI_Comm comm);

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

  double ctr[3] = {0.5, 0.5, 0.5};
  double r = 0.2;
  
  auto fx = [ctr, r](double x, double y, double z) { return sqrt((x-ctr[0])*(x-ctr[0]) + (y-ctr[1])*(y-ctr[1])) - r; };
  
  function2Octree(fx, nodes, 8, false, MPI_COMM_WORLD);
    
  // write out to vtk file ...
  saveTreeAsVTK(nodes, "cylinder_hole", MPI_COMM_WORLD);
    
  // wrap up
  ot::DA_Finalize();
  PetscFinalize();
}//end main

void saveTreeAsVTK(std::vector<ot::TreeNode>& tree, const char *file_prefix, MPI_Comm comm) {
  int rank=0, size;
  char fname[256];


	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

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
  int num_vertices = tree.size() * (unit_points);
  int num_cells = tree.size();

  out << "POINTS " << num_vertices << " float" << std::endl;

  { // dim = 3

    unsigned int len; //!  ??
    unsigned int xl, yl, zl;  //! ??

    int num_data_field = 2; // rank and data
    int num_cells_elements = num_cells * unit_points + num_cells;

    int dof=1;
    PetscScalar *_vec=NULL;

    unsigned int maxD = 30;
    unsigned int lev;
    double h1;
    Point pt;

    double h = 1.0/((double)(1<<(maxD-1)));
    
    double xx, yy, zz;
    unsigned int idx[8];

    for (auto elem: tree ) {
        // if ( elem.getLevel() != depth ) continue;
        
        h1 = h * ( 1 << (maxD - elem.getLevel())); 
        
        // check and split
        pt = elem.getAnchor();
        pt *= h ;

      out << pt.x() << " " <<  pt.y() << " " << pt.z() << std::endl;
      out << pt.x() + h1 << " " <<  pt.y() << " " << pt.z() << std::endl;
      out << pt.x() + h1 << " " <<  pt.y() + h1 << " " << pt.z() << std::endl;
      out << pt.x() << " " <<  pt.y() + h1 << " " << pt.z() << std::endl;

      out << pt.x() << " " <<  pt.y() << " " << pt.z() + h1<< std::endl;
      out << pt.x() + h1 << " " <<  pt.y() << " " << pt.z() + h1 << std::endl;
      out << pt.x() + h1 << " " <<  pt.y() + h1 << " " << pt.z() + h1 << std::endl;
      out << pt.x() << " " <<  pt.y() + h1 << " " << pt.z() + h1 << std::endl;
    }

    out << "CELLS " << tree.size() << " " << num_cells_elements << std::endl;

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

    out << std::endl;

  }

  out.close();
}


