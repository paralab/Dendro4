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
#include <string>
#include <Point.h>
#include <colors.h>

#include <iomanip>

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


#define SQRT_3 1.7320508075688772

void saveNodalVecAsVTK(ot::DA* da, Vec vec, double* gsz, const char *fname);
void saveNodalVecAsVTK(ot::subDA* da, Vec vec, double* gsz, const char *fname);

void interp_global_to_local(PetscScalar* glo, PetscScalar* __restrict loc, ot::DA* m_octDA);
void interp_local_to_global(PetscScalar* __restrict loc, PetscScalar* glo, ot::DA* m_octDA);

void build_taly_coordinates(double *coords, const Point &pt, const Point &h) {
  const double &hx = h.x();
  const double &hy = h.y();
  const double &hz = h.z();

  coords[0] = pt.x();
  coords[1] = pt.y();
  coords[2] = pt.z();
  coords[3] = coords[0] + hx;
  coords[4] = coords[1];
  coords[5] = coords[2];
  coords[6] = coords[0] + hx;
  coords[7] = coords[1] + hy;
  coords[8] = coords[2];
  coords[9] = coords[0];
  coords[10] = coords[1] + hy;
  coords[11] = coords[2];
  coords[12] = coords[0];
  coords[13] = coords[1];
  coords[14] = coords[2] + hz;
  coords[15] = coords[0] + hx;
  coords[16] = coords[1];
  coords[17] = coords[2] + hz;
  coords[18] = coords[0] + hx;
  coords[19] = coords[1] + hy;
  coords[20] = coords[2] + hz;
  coords[21] = coords[0];
  coords[22] = coords[1] + hy;
  coords[23] = coords[2] + hz;
}

double ValueAtInit(const std::array<double,3> &pt) {
  /// Calculate the diffuse interface phi
  double phiDiffuse = 0.0;


  double amplitude = 0.2;

  /// This is interface
  double delta = 0.025;

  double locationInt = 2.0;

  std::array<double,3> centerofGaussian = {0.5, 0.0, 0.5};

  double stdDevGaussian = 0.2;

  /// Gaussian
  double gaussian =
    amplitude * exp(-((((pow((pt[0] - centerofGaussian[0]), 2)) / stdDevGaussian)) +
          ((pow((pt[2] - centerofGaussian[2]), 2)) / (stdDevGaussian))));

  phiDiffuse =
    tanh(((locationInt - pt[2]) - gaussian) / (sqrt(2) *delta));


  return phiDiffuse;
}

std::vector<unsigned int> calc_refine_func_userProvidedDim(
    ot::DA *da, 
  const std::function<double(double, double, double)> &f_phi,
  const double* channel_min, 
  const double* channel_max, 
  const unsigned int refine_lvl_base, 
  const unsigned int refine_lvl_interface, 
  const unsigned int refine_lvl_channel_wall,
  bool enable_subda,
  double* problemSize,
	const unsigned int initialRefinementLevel) {
	
  double octToPhysScale[3];
	const unsigned int maxD = da->getMaxDepth();
	octToPhysScale[0] = problemSize[0] / ((PetscScalar)(1 << (maxD - 1)));
	octToPhysScale[1] = problemSize[1] / ((PetscScalar)(1 << (maxD - 1)));
	octToPhysScale[2] = problemSize[2] / ((PetscScalar)(1 << (maxD - 1)));

	std::vector<unsigned int> refinement;
	da->createVector(refinement, true, true, 1);
	//  for (unsigned int i = 0; i < refinement.size(); i++) {
	//    refinement[i] = 8;
	//  }
	// loop over owned elements
	for (da->init<ot::DA_FLAGS::WRITABLE>();
	     da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
	     da->next<ot::DA_FLAGS::WRITABLE>()) {
		const unsigned int curr = da->curr();
		const int lev = da->getLevel(da->curr());
		Point h(octToPhysScale[0] * (1u << (maxD - lev)),
		        octToPhysScale[1] * (1u << (maxD - lev)),
		        octToPhysScale[2] * (1u << (maxD - lev)));
		Point pt = da->getCurrentOffset();
		pt.x() *= octToPhysScale[0];
		pt.y() *= octToPhysScale[1];
		pt.z() *= octToPhysScale[2];
		double coords[8 * 3];

    // @check makrand
    double refine_interface_tol = 0.85;
    
    // @check makrand
    build_taly_coordinates(coords, pt, h);
		
    // bool for at channel wall
		std::array<std::vector<double>, 8> nodeCoords;
		for (unsigned int n = 0; n < 8; n++) {
			nodeCoords[n] = {coords[n * 3], coords[n * 3 + 1], coords[n * 3 + 2]};
		}
		//find if coords are within subDA bounds
		bool withinSubDA;
		const double epsInOut = 1 * (problemSize[0] / pow(2, initialRefinementLevel));
		withinSubDA =std::any_of(nodeCoords.begin(), nodeCoords.end(), [&](std::vector<double>  d){
				bool in_channel =
						(((channel_min[0]) <= d[0] && d[0] < (channel_max[0] + epsInOut)) &&
						 ((channel_min[1]) <= d[1] && d[1] < (channel_max[1] + epsInOut)) &&
						 ((channel_min[2]) <= d[2] && d[2] < (channel_max[2] + epsInOut)));
				return in_channel;
		});

		std::array<double, 8> phi;
		// get the phi values at each node (if subda this needs to be within the subda bounds
		if (enable_subda){
			if (withinSubDA){
				for (unsigned int n = 0; n < 8; n++) {
					phi[n] = f_phi(coords[n * 3], coords[n * 3 + 1], coords[n * 3 + 2]);
				}
			} else{
				for (unsigned int n = 0; n < 8; n++) {
					phi[n] = 1.0;
				}
			}
		}else{
			// case for the fullDA
			for (unsigned int n = 0; n < 8; n++) {
				phi[n] = f_phi(coords[n * 3], coords[n * 3 + 1], coords[n * 3 + 2]);
			}
		}
		// get the phi values at each node
		//std::array<double, 8> phi;
		//for (unsigned int n = 0; n < 8; n++) {
		//	phi[n] = f_phi(coords[n * 3], coords[n * 3 + 1], coords[n * 3 + 2]);
		//}
		// Any node is on the channel boundary make refine_wall true
		/// We use the initial refinement level to find an optimum distance from
		/// the
		/// analytical function any node of the
		/// element can have and use that to refine the elements
		///(baselvl + 1)
		bool refineWall = false;
		// This is satisfied when the half of the boundaries conform with the
		// original DA out of which subDA was carved
		// out. In this case it is guaranteed to have a node very very close to
		// the analytical boundary, for the sides
		// which correspond to the sides of the original DA.
			double eps_originalDAconformingBoundaries = 1e-16;
			double eps_boundariesDuetosubDAcarve =
					0.55 * (problemSize[0] / pow(2, initialRefinementLevel));
			refineWall =
					std::any_of(nodeCoords.begin(), nodeCoords.end(), [&](std::vector<double> d) {
							bool onChannelWall = false;
							bool onChannelInlet = (std::fabs(d[0] - channel_min[0]) <
							                       eps_originalDAconformingBoundaries);
							bool onChannelOutlet = (std::fabs(d[0] - channel_max[0]) <
							                        eps_boundariesDuetosubDAcarve);
							bool bottom_wall = (std::fabs(d[1] - channel_min[1]) <
							                    eps_originalDAconformingBoundaries);
							bool top_wall = (std::fabs(d[1] - channel_max[1]) <
							                 eps_boundariesDuetosubDAcarve);
							bool front_wall = (std::fabs(d[2] - channel_min[2]) <
							                   eps_originalDAconformingBoundaries);
							bool back_wall = (std::fabs(d[2] - channel_max[2]) <
							                  eps_boundariesDuetosubDAcarve);
							// refine walls except inlet and outlet
							bool noInletOutlet = false;
							if (noInletOutlet) {
								onChannelWall =
										bottom_wall || top_wall || front_wall || back_wall;
							} else {
								onChannelWall = bottom_wall || top_wall || front_wall ||
								                back_wall || onChannelInlet || onChannelOutlet;
							}
							return onChannelWall;
					});
		// }
		// refine if any nodes are below the phi interface tolerance (abs(phi) <
		// interface_tol) or if the sign changes
		bool refine_phi =
				std::any_of(phi.begin(), phi.end(),
				            [&](double d) {
						            return (std::fabs(d) < refine_interface_tol);
				            }) ||
				((*std::min_element(phi.begin(), phi.end()) > 0) !=
				 (*std::max_element(phi.begin(), phi.end()) > 0));
		if (refine_phi) {
			refinement[curr] = refine_lvl_interface;
		} else if (refineWall) {
			refinement[curr] = refine_lvl_channel_wall;
	 } else if (!refine_phi && !refineWall) {
			refinement[curr] = refine_lvl_base;
		} else if (refine_phi && refineWall) {
			refinement[curr] = std::min(refine_lvl_interface, refine_lvl_channel_wall);
		}
	}
	return refinement;
}

std::vector<unsigned int> calc_refine_func_estimated_domain_size(
		ot::DA *da,
		const std::function<double(double, double, double)> &f_phi,
		double *problemSize,
		const double* channel_min,
		const double* channel_max,
		const unsigned int refine_lvl_base,
		const unsigned int refine_lvl_interface,
		const unsigned int refine_lvl_channel_wall,
		bool enable_subda,
		const double *estimatedDomainMin,
		const double *estimatedDomainMax,
		const unsigned int initialRefinementLevel) {


	double octToPhysScale[3];
	const unsigned int maxD = da->getMaxDepth();
	octToPhysScale[0] = problemSize[0] / ((PetscScalar)(1 << (maxD - 1)));
	octToPhysScale[1] = problemSize[1] / ((PetscScalar)(1 << (maxD - 1)));
	octToPhysScale[2] = problemSize[2] / ((PetscScalar)(1 << (maxD - 1)));

	std::vector<unsigned int> refinement;
	da->createVector(refinement, true, true, 1);
	//  for (unsigned int i = 0; i < refinement.size(); i++) {
	//    refinement[i] = 8;
	//  }

	// loop over owned elements
	for (da->init<ot::DA_FLAGS::WRITABLE>();
	     da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
	     da->next<ot::DA_FLAGS::WRITABLE>()) {
		const unsigned int curr = da->curr();
		const int lev = da->getLevel(da->curr());

		Point h(octToPhysScale[0] * (1u << (maxD - lev)),
		        octToPhysScale[1] * (1u << (maxD - lev)),
		        octToPhysScale[2] * (1u << (maxD - lev)));

		Point pt = da->getCurrentOffset();
		pt.x() *= octToPhysScale[0];
		pt.y() *= octToPhysScale[1];
		pt.z() *= octToPhysScale[2];

		double coords[8 * 3];
    // @check makrand
		build_taly_coordinates(coords, pt, h);

		// bool for at channel wall
		std::array<std::vector<double>, 8> nodeCoords;
		for (unsigned int n = 0; n < 8; n++) {
			nodeCoords[n] = {coords[n * 3], coords[n * 3 + 1], coords[n * 3 + 2]};
		}

		//find if coords are within subDA bounds
		bool withinSubDA;
		const double epsInOut = 1 * (problemSize[0] / pow(2, initialRefinementLevel));
		withinSubDA =std::any_of(nodeCoords.begin(), nodeCoords.end(), [&](std::vector<double> d){
				bool in_channel =
						(((channel_min[0]) <= d[0] && d[0] < (channel_max[0] + epsInOut)) &&
						 ((channel_min[1]) <= d[1] && d[1] < (channel_max[1] + epsInOut)) &&
						 ((channel_min[2]) <= d[2] && d[2] < (channel_max[2] + epsInOut)));
				return in_channel;
		});

		std::array<double, 8> phi;
		// get the phi values at each node (if subda this needs to be within the subda bounds
		

		if (withinSubDA){
				for (unsigned int n = 0; n < 8; n++) {
					phi[n] = f_phi(coords[n * 3], coords[n * 3 + 1], coords[n * 3 + 2]);
				}
		} else{
				for (unsigned int n = 0; n < 8; n++) {
					phi[n] = 1.0;
				}
		}


		// get the phi values at each node
		//std::array<double, 8> phi;
		//for (unsigned int n = 0; n < 8; n++) {
		//	phi[n] = f_phi(coords[n * 3], coords[n * 3 + 1], coords[n * 3 + 2]);
		//}

		// Any node is on the channel boundary make refine_wall true
		/// We use the initial refinement level to find an optimum distance from
		/// the
		/// analytical function any node of the
		/// element can have and use that to refine the elements
		///(baselvl + 2)
		bool refineWall = false;

		// This is satisfied when the half of the boundaries conform with the
		// original DA out of which subDA was carved
		// out. In this case it is guaranteed to have a node very very close to
		// the analytical boundary, for the sides
		// which correspond to the sides of the original DA.
		double eps = 1e-16;
		refineWall =
				std::any_of(nodeCoords.begin(), nodeCoords.end(), [&](std::vector<double> d) {
						bool onChannelWall = false;
						bool onChannelInlet =
								(std::fabs(d[0] - estimatedDomainMin[0]) < eps);
						bool onChannelOutlet =
								(std::fabs(d[0] - estimatedDomainMax[0]) < eps);

						bool bottom_wall = (std::fabs(d[1] - estimatedDomainMin[1]) < eps);
						bool top_wall = (std::fabs(d[1] - estimatedDomainMax[1]) < eps);

						bool front_wall = (std::fabs(d[2] - estimatedDomainMin[2]) < eps);
						bool back_wall = (std::fabs(d[2] - estimatedDomainMax[2]) < eps);

						// refine walls except inlet and outlet
						bool noInletOutlet = false;
						if (noInletOutlet) {
							onChannelWall = bottom_wall || top_wall || front_wall || back_wall;
						} else {
							onChannelWall = bottom_wall || top_wall || front_wall ||
							                back_wall || onChannelInlet || onChannelOutlet;
						}
						return onChannelWall;
				});


		// refine if any nodes are below the phi interface tolerance (abs(phi) <
		// interface_tol) or if the sign changes
    double refine_interface_tol=0.85; // @check makrand
		bool refine_phi =
				std::any_of(phi.begin(), phi.end(),
				            [&](double d) {
						            return (std::fabs(d) < refine_interface_tol);
				            }) ||
				((*std::min_element(phi.begin(), phi.end()) > 0) !=
				 (*std::max_element(phi.begin(), phi.end()) > 0));

		if (refine_phi) {
			refinement[curr] = refine_lvl_interface;
		} else if (refineWall) {
			refinement[curr] = refine_lvl_channel_wall;
		} else if (!refine_phi && !refineWall) {
			refinement[curr] = refine_lvl_base;
		} else if (refine_phi && refineWall) {
			refinement[curr] = std::min(refine_lvl_interface, refine_lvl_channel_wall);
		}
	}
	return refinement;
}


void printApproximateWork(ot::subDA* octDA) {

	MPI_Comm comm = octDA->getComm();

	int rank, nProcs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nProcs);

	const int N_NODES = 0;
	const int N_GHOSTED_NODES = 1;
	const int N_ELEMENTS = 2;
	const int N_GHOSTED_ELEMENTS = 3;
	unsigned int counts[4];
	counts[N_NODES] = octDA->getNodeSize();
	counts[N_GHOSTED_NODES] = octDA->getGhostedNodeSize();
	counts[N_ELEMENTS] = octDA->getElementSize();
	counts[N_GHOSTED_ELEMENTS] = octDA->getGhostedElementSize();


  std::cout << "\t" << MAG << rank << NRM << " : " << GRN << counts[2] << NRM << " e " << GRN << counts[0] << NRM << " n" << std::endl; 

	unsigned int mins[4], maxs[4], avgs[4];
	for (int i = 0; i < 4; i++) {
		MPI_Allreduce(&counts[i], &mins[i], 1, MPI_UNSIGNED, MPI_MIN, comm);
		MPI_Allreduce(&counts[i], &maxs[i], 1, MPI_UNSIGNED, MPI_MAX, comm);
		MPI_Allreduce(&counts[i], &avgs[i], 1, MPI_UNSIGNED, MPI_SUM, comm);
		avgs[i] /= nProcs;
	}
	if (!rank) {
		std::cout << std::endl;
    std::cout << "N_NODES - min: " << mins[N_NODES] << ", max: " << maxs[N_NODES] << ", avg: " <<  avgs[N_NODES] <<
		std::endl;
		std::cout << "N_GHOSTED_NODES - min: " << mins[N_GHOSTED_NODES] << ", max: " << maxs[N_GHOSTED_NODES] << ", avg: "
		<< avgs[N_GHOSTED_NODES] << std::endl;
		std::cout << "N_ELEMENTS - min: " << mins[N_ELEMENTS] << ", max: " << maxs[N_ELEMENTS] << ", avg: " <<
		avgs[N_ELEMENTS] << std::endl;
		std::cout << "N_GHOSTED_ELEMENTS - min: " << mins[N_GHOSTED_ELEMENTS] << ", max: " << maxs[N_GHOSTED_ELEMENTS]<<
		", avg: " << avgs[N_GHOSTED_ELEMENTS] << std::endl;
	}
}


int main(int argc, char ** argv ) {
	int size, rank;
	unsigned int dim = 3;
	unsigned maxDepth = 30;
 
  /* TODO
    1. Adjust weights for retained, non-retained and boundary nodes.
    2. Try to make DA construction more balanced.
  */


	std::vector<ot::TreeNode> nodes, nodes_bal;

	//PetscInitialize(&argc, &argv, "options", NULL);
	PetscInitialize (&argc,&argv,(char*)0,NULL);

	ot::RegisterEvents();
	ot::DA_Initialize(MPI_COMM_WORLD);

	_InitializeHcurve(3);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ot::subDA* octDA = NULL;

	/// Define the channel geometry
	const double channelMin[3] = {0.0, 0.0, 0.0};
	const double channelMax[3] = {1.0, 4.0, 1.0};
	double daScalingFactor[3] = {4.0, 4.0, 4.0};
	const unsigned int refine_lvl_base = 7;
	const unsigned int refine_lvl_interface = 10;
	const unsigned int refine_lvl_channel_wall = 7;
	bool enable_subda = true;
	const unsigned int initialRefinementLevel = 5;
	const unsigned int max_depth = 30;
	const bool refine_walls = false;

	/// Setup function for carving out a subDA from the DA
	/// less than 0 = remove, greater than 0 = keep
	std::function<double(double, double, double)> fx_retain;
	fx_retain = [&](double x, double y, double z) {
			bool in_channel = ((channelMin[0] < x && x < channelMax[0]) &&
			                   (channelMin[1] < y && y < channelMax[1]) &&
			                   (channelMin[2] < z && z < channelMax[2]));
			if (in_channel) {
				return 1.0;
			}
			return -1.0;
	};

	/// setup function for wall refinement
	auto fx_refine_walls = [&](double x, double y, double z) {
			if (!refine_walls) {
				return 1.0;
			}

			const double eps = 1e-12;
			bool in_channel =
					(((channelMin[0] + eps) < x && x < (channelMax[0] - eps)) &&
					 ((channelMin[1] + eps) < y && y < (channelMax[1] - eps)) &&
					 ((channelMin[2] + eps) < z && z < (channelMax[2] - eps)));
			return in_channel ? 1.0 : -1.0;
	};


	/// fullDA
	ot::DA *mainDA = NULL; ///for the refine level formulation
	// if (!rank) printf("start by createRegularOctree\n");
	createRegularOctree(nodes, initialRefinementLevel, 3, max_depth, MPI_COMM_WORLD);
	///Create regular octree by invoking Da constructor
	ot::DA *base_da =
			new ot::DA(nodes, PETSC_COMM_WORLD, PETSC_COMM_WORLD, 0.3);
	nodes.clear();
	// if(!rank) std::cout << YLW << "Before calc_refine_func 1" NRM << std::endl;
	// Use channel refined DA in the phi based refinement function
	bool refineOjects = true;
	std::vector<unsigned int> levels = calc_refine_func_userProvidedDim(
			base_da,
			[&](double x, double y, double z) -> double {
					std::array<double, 3> pp = {x, y, z};
					return ValueAtInit(pp);
			},
			channelMin,
			channelMax,
			refine_lvl_base,
			refine_lvl_interface,
			refine_lvl_channel_wall,
			enable_subda,
			daScalingFactor,
			initialRefinementLevel);

	ot::DA *mid_da =
			ot::remesh_DA(base_da, levels, daScalingFactor, fx_refine_walls, fx_retain, 1000, PETSC_COMM_WORLD);

  levels.clear();

  // if(!rank) std::cout << GRN << "Finished remesh_DA" << NRM << std::endl;

	delete base_da;
	/// Estimate the size of the subDA for the given refinement
	double subDA_boundingBoxEstimate_Min[3];
	double subDA_boundingBoxEstimate_Max[3];

	ot::subDA* subDAforBoundingBoxEstimate =
				new ot::subDA(mid_da, fx_retain, daScalingFactor);
	subDAforBoundingBoxEstimate->getBoundingBox(subDA_boundingBoxEstimate_Min, subDA_boundingBoxEstimate_Max);
	delete subDAforBoundingBoxEstimate;

	levels.clear();
	// Coarsening step
	refineOjects = true;
	levels = calc_refine_func_estimated_domain_size(
			mid_da,
			[&](double x, double y, double z) -> double {
					std::array<double, 3> pp = {x, y, z};
					return ValueAtInit(pp);
			},
			daScalingFactor,
			channelMin,
			channelMax,
			refine_lvl_base,
			refine_lvl_interface,
			refine_lvl_channel_wall,
			enable_subda,
			subDA_boundingBoxEstimate_Min,
			subDA_boundingBoxEstimate_Max,
			initialRefinementLevel);

	///Makrand-debug: save fullda with weights:
	//std::vector<ot::TreeNode> mainDATreeNodes;
	//mainDATreeNodes = ot::remesh_DA_Treenode (mid_da, levels, daScalingFactor, fx_refine_walls, fx_retain, 1000,
	//                                          PETSC_COMM_WORLD, 2);
  // if(!rank) std::cout << MAG << "!==!==!==!==!==!==!==" << NRM << std::endl;
  // if(!rank) std::cout << YLW << "Calling second remesh_DA" << NRM << std::endl;
	mainDA = ot::remesh_DA(mid_da, levels, daScalingFactor, fx_refine_walls, fx_retain, 1000, PETSC_COMM_WORLD);
  levels.clear();

  // if(!rank) std::cout << GRN << "After second remesh_DA" << NRM << std::endl;
  // if(!rank) std::cout << MAG << "!==!==!==!==!==!==!==" << NRM << std::endl;
	delete mid_da;
	/// Subda dimensions
	double subDAmin[3];
	double subDAmax[3];

	octDA = new ot::subDA(mainDA, fx_retain, daScalingFactor);
	
  
  octDA->getBoundingBox(subDAmin, subDAmax);

	printApproximateWork(octDA);

	MPI_Barrier(PETSC_COMM_WORLD);
	// clean up
	delete mainDA;
	delete octDA;
	// wrap up
	ot::DA_Finalize();
	// std::cout << rank << GRN " === OT finalize ===" <<  NRM << std::endl;
	PetscFinalize();
	if (!rank) std::cout << BLU << "<== All done ==>" << NRM << std::endl;
}//end main


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

    // std::cout << "Printing initials." << std::endl;
    // da->init<ot::DA_FLAGS::ALL>();
    // std::cout << rank << ": init<All> " << da->curr() << std::endl;
    // da->init<ot::DA_FLAGS::INDEPENDENT>();
    // std::cout << rank << ": init<Indep> " << da->curr() << std::endl;
    // da->init<ot::DA_FLAGS::W_DEPENDENT>();
    // std::cout << rank << ": init<W_Dep> " << da->curr() << std::endl;

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
    delete [] local;

    da->vecRestoreBuffer(vec,  _vec, false, false, true,  dof);
  }

  out.close();
}

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

/*
void correct_elemental_matrix(PetscScalar *Ke, DA* da, int m_uiDof) {
  /// not the most efficient. Will make efficient later.
  //  
  //  - pass in pre-allocated arrays for Q, Qt and temp 
  //  - build Qt while building Q.

  unsigned int idx[8];
	unsigned char hangingMask = da->getHangingNodeIndex(da->curr());
	unsigned int chNum = da->getChildNumber();
	da->getNodeIndices(idx);

  //=== Step 1
  // build interp matrix - keep it 8x8
  double *Q  = new double[64];
  double *Qt = new double[64];

  for (unsigned int i=0; i<64; ++i) {
    Q[i]  = 0.0;
    Qt[i] = 0.0;
  }

  switch (chNum) {
    case 0:
      // 0,7 are not hanging
      Q[0] = 1.0;
      if ( hangingMask & NODE_1 ) {
        Q[8] = 0.5; Q[9] = 0.5;
      } else {
        Q[9] = 1.0;
      }  
      if ( hangingMask & NODE_2 ) {
        Q[16] = 0.5; Q[18] = 0.5;
      } else {
        Q[18] = 1.0;
      }
      if ( hangingMask & NODE_3 ) {
        Q[24] = 0.25; Q[25] = 0.25;
        Q[26] = 0.25; Q[27] = 0.25;
      } else {
        Q[27] = 1.0;
      }
      if ( hangingMask & NODE_4 ) {
        Q[32] = 0.5;  Q[36] = 0.5;
      } else {
        Q[36] = 1.0;
      }
      if ( hangingMask & NODE_5 ) {
        Q[40] = 0.25; Q[41] = 0.25;
        Q[44] = 0.25; Q[45] = 0.25;
      } else {
        Q[45] = 1.0;
      }
      if ( hangingMask & NODE_6 ) {
        Q[48] = 0.25; Q[50] = 0.25;
        Q[52] = 0.25; Q[54] = 0.25;
      } else {
        Q[54] = 1.0;
      }
      Q[63] = 1.0;
      break;
    case 1:
      // 1,6 are not hanging
      if ( hangingMask & NODE_0 ) {
        Q[0] = 0.5; Q[1] = 0.5;
      } else {
        Q[0] = 1.0;
      }
      Q[9] = 1.0;  
      if ( hangingMask & NODE_2 ) {
        Q[16] = 0.25; Q[17] = 0.25;
        Q[18] = 0.25; Q[19] = 0.25;
      } else {
        Q[18] = 1.0;
      }
      if ( hangingMask & NODE_3 ) {
        Q[25] = 0.5; Q[27] = 0.5;
      } else {
        Q[27] = 1.0;
      }
      if ( hangingMask & NODE_4 ) {
        Q[32] = 0.25; Q[33] = 0.25;
        Q[36] = 0.25; Q[37] = 0.25;
      } else {
        Q[36] = 1.0;
      }
      if ( hangingMask & NODE_5 ) {
        Q[41] = 0.5; Q[45] = 0.5;
      } else {
        Q[45] = 1.0;
      }
      Q[54] = 1.0;
      if ( hangingMask & NODE_7 ) {
        Q[57] = 0.25; Q[59] = 0.25;
        Q[61] = 0.25; Q[63] = 0.25;
      } else {
        Q[63] = 1.0;
      }
      break;
    case 2:
      // 2,5 are not hanging
      if ( hangingMask & NODE_0 ) {
        Q[0] = 0.5; Q[2] = 0.5;
      } else {
        Q[0] = 1.0;
      }
      if ( hangingMask & NODE_1 ) {
        Q[8] = 0.25; Q[9] = 0.25; 
        Q[10] = 0.25; Q[11] = 0.25;
      } else {
        Q[9] = 1.0;
      }
      Q[18] = 1.0;
      if ( hangingMask & NODE_3 ) {
        Q[26] = 0.5; Q[27] = 0.5;
      } else {
        Q[27] = 1.0;
      }
      if ( hangingMask & NODE_4 ) {
        Q[32] = 0.25; Q[34] = 0.25;
        Q[36] = 0.25; Q[38] = 0.25;
      } else {
        Q[36] = 1.0;
      }
      Q[45] = 1.0;
      if ( hangingMask & NODE_6 ) {
        Q[50] = 0.5; Q[54] = 0.5;
      } else {
        Q[54] = 1.0;
      }
      if ( hangingMask & NODE_7 ) {
        Q[58] = 0.25; Q[59] = 0.25;
        Q[62] = 0.25; Q[63] = 0.25;
      } else {
        Q[63] = 1.0;
      }
      break;
    case 3:
      // 3,4 are not hanging
      if ( hangingMask & NODE_0 ) {
        Q[0] = 0.25; Q[1] = 0.25;
        Q[2] = 0.25; Q[3] = 0.25;
      } else {
        Q[0] = 1.0;
      }
      if ( hangingMask & NODE_1 ) {
        Q[9] = 0.5; Q[11] = 0.5;
      } else {
        Q[9] = 1.0;
      }
      if ( hangingMask & NODE_2 ) {
        Q[18] = 0.5; Q[19] = 0.5;
      } else {
        Q[18] = 1.0;
      }
      Q[27] = 1.0;
      Q[36] = 1.0;
      if ( hangingMask & NODE_5 ) {
        Q[41] = 0.25; Q[43] = 0.25;
        Q[45] = 0.25; Q[47] = 0.25;
      } else {
        Q[45] = 1.0;
      }
      if ( hangingMask & NODE_6 ) {
        Q[50] = 0.25; Q[51] = 0.25;
        Q[54] = 0.25; Q[55] = 0.25;
      } else {
        Q[54] = 1.0;
      }
      if ( hangingMask & NODE_7 ) {
        Q[59] = 0.5; Q[63] = 0.5;
      } else {
        Q[63] = 1.0;
      }
      break;
    case 4:
		  // 4,3 are not hanging
      if ( hangingMask & NODE_0 ) {
        Q[0] = 0.5; Q[4] = 0.5;
      } else {
        Q[0] = 1.0;
      }
      if ( hangingMask & NODE_1 ) {
        Q[8] = 0.25; Q[9] = 0.25;
        Q[12] = 0.25; Q[13] = 0.25;
      } else {
        Q[9] = 1.0;
      }
      if ( hangingMask & NODE_2 ) {
        Q[16] = 0.25; Q[18] = 0.25;
        Q[20] = 0.25; Q[22] = 0.25;
      } else {
        Q[18] = 1.0;
      }
      Q[27] = 1.0;
      Q[36] = 1.0;
      if ( hangingMask & NODE_5 ) {
        Q[44] = 0.5; Q[45] = 0.5;
      } else {
        Q[45] = 1.0;
      }
      if ( hangingMask & NODE_6 ) {
        Q[52] = 0.5; Q[54] = 0.5;
      } else {
        Q[54] = 1.0;
      }
      if ( hangingMask & NODE_7 ) {
        Q[60] = 0.25; Q[61] = 0.25;
        Q[62] = 0.25; Q[63] = 0.25;
      } else {
        Q[63] = 1.0;
      }
      break;
    case 5:
      // 5,2 are not hanging
      if ( hangingMask & NODE_0 ) {
        Q[0] = 0.25; Q[1] = 0.25;
        Q[4] = 0.25; Q[5] = 0.25;
      } else {
        Q[0] = 1.0;
      }
      if ( hangingMask & NODE_1 ) {
        Q[9] = 0.5; Q[13] = 0.5;
      } else {
        Q[9] = 1.0;
      }
      Q[18] = 1.0;
      if ( hangingMask & NODE_3 ) {
        Q[25] = 0.25; Q[27] = 0.25;
        Q[29] = 0.25; Q[31] = 0.25;
      } else {
        Q[27] = 1.0;
      }
      if ( hangingMask & NODE_4 ) {
        Q[36] = 0.5;  Q[37] = 0.5;
      } else {
        Q[36] = 1.0;
      }
      Q[45] = 1.0;
      if ( hangingMask & NODE_6 ) {
        Q[52] = 0.25; Q[53] = 0.25;
        Q[54] = 0.25; Q[55] = 0.25;
      } else {
        Q[54] = 1.0;
      }
      if ( hangingMask & NODE_7 ) {
        Q[61] = 0.5; Q[63] = 0.5;
      } else {
        Q[63] = 1.0;
      }
      break;
    case 6:
      // 6,1 are not hanging
       if ( hangingMask & NODE_0 ) {
        Q[0] = 0.25; Q[1] = 0.25;
        Q[4] = 0.25; Q[5] = 0.25;
      } else {
        Q[0] = 1.0;
      }
      Q[9] = 1.0;
      if ( hangingMask & NODE_2 ) {
        Q[18] = 0.5; Q[22] = 0.5;
      } else {
        Q[18] = 1.0;
      }
      if ( hangingMask & NODE_3 ) {
        Q[26] = 0.25; Q[27] = 0.25;
        Q[30] = 0.25; Q[31] = 0.25;
      } else {
        Q[27] = 1.0;
      }
      if ( hangingMask & NODE_4 ) {
        Q[36] = 0.5;  Q[38] = 0.5;
      } else {
        Q[36] = 1.0;
      }
      if ( hangingMask & NODE_5 ) {
        Q[44] = 0.25; Q[45] = 0.25;
        Q[46] = 0.25; Q[47] = 0.25;
      } else {
        Q[45] = 1.0;
      }
      Q[54] = 1.0;
      if ( hangingMask & NODE_7 ) {
        Q[62] = 0.5; Q[63] = 0.5;
      } else {
        Q[63] = 1.0;
      }
      break;
    case 7:
      // 7,0 are not hanging
      Q[0] = 1.0;
      if ( hangingMask & NODE_1 ) {
        Q[9] = 0.25; Q[11] = 0.25; 
        Q[13] = 0.25; Q[15] = 0.25;
      } else {
        Q[9] = 1.0;
      }
      if ( hangingMask & NODE_2 ) {
        Q[18] = 0.25; Q[19] = 0.25;
        Q[22] = 0.25; Q[23] = 0.25;
      } else {
        Q[18] = 1.0;
      }
      if ( hangingMask & NODE_3 ) {
        Q[27] = 0.5; Q[31] = 0.5;
      } else {
        Q[27] = 1.0;
      }
      if ( hangingMask & NODE_4 ) {
        Q[36] = 0.25; Q[37] = 0.25;
        Q[38] = 0.25; Q[39] = 0.25;
      } else {
        Q[36] = 1.0;
      }
      if ( hangingMask & NODE_5 ) {
        Q[45] = 0.5; Q[47] = 0.5;
      } else {
        Q[45] = 1.0;
      }
      if ( hangingMask & NODE_6 ) {
        Q[54] = 0.5; Q[55] = 0.5;
      } else {
        Q[54] = 1.0;
      }
      Q[63] = 1.0;
      break;
    default:
			std::cout<<"in correct elemental matrix: incorrect child num = " << chNum << std::endl;
			assert(false);
      break;
  } // switch

  // === Allocate 
  unsigned int n = 8*m_uiDof;
  double* T = new double [n*n];

  //=== Step 2
  // right multiply by Q
  // T = Ke * Q
  for (unsigned int j=0; j<8; ++j) {
    for (unsigned int i=0; i<8; ++i) {
      for (unsigned int d=0; d<m_uiDof; ++d) {
        T[ m_uiDof*(8*j + i) + d]=0;
        for(unsigned int k=0; k<8; ++k) {
          // T [j, i] = Ke [j,k] * Q[k,i]
          T[ m_uiDof*(8*j + i) + d ] += Ke[ m_uiDof*(8*j + k) + d ] * Q[8*k + i];
        } // k
      } // dof
    } // j
  } // i

  //=== Step 3
  // left multiply by Qt
  // Ke = Qt * T
  for (unsigned int j=0; j<8; ++j) {
    for (unsigned int i=0; i<8; ++i) {
      for (unsigned int d=0; d<m_uiDof; ++d) {
        Ke[ m_uiDof*(8*j + i) + d] = 0;
        for(unsigned int k=0; k<8; ++k) {
          // Ke [j, i] = Q [j,k] * T[k,i]
          Ke[ m_uiDof*(8*j + i) + d ] += T[ m_uiDof*(8*k + i) + d ] * Q[8*j + k];
        } // k
      } // dof
    } // j
  } // i


  // clean up
  delete [] Q;
  delete [] Qt;
  delete [] T;

}
*/

