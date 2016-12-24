#include "Point.h"

template <typename T>
feMatrix<T>::feMatrix() {
	m_daType = PETSC;
	m_DA    = NULL;
	m_octDA   = NULL;
	m_stencil = NULL;
	m_uiDof = 1;
	m_ucpLut  = NULL;

	// initialize the stencils ...
	initStencils();
}

template <typename T>
feMatrix<T>::feMatrix(daType da) {
#ifdef __DEBUG__
	assert ( ( da == PETSC ) || ( da == OCT ) );
#endif
	m_daType = da;
	m_DA    = NULL;
	m_octDA   = NULL;
	m_stencil = NULL;
	m_ucpLut  = NULL;

	// initialize the stencils ...
	initStencils();
	if (da == OCT)
		initOctLut();
}

template <typename T>
void feMatrix<T>::initOctLut() {
	//Note: It is not symmetric.
	unsigned char tmp[8][8]={
		{0,1,2,3,4,5,6,7},
		{1,3,0,2,5,7,4,6},
		{2,0,3,1,6,4,7,5},
		{3,2,1,0,7,6,5,4},
		{4,5,0,1,6,7,2,3},
		{5,7,1,3,4,6,0,2},
		{6,4,2,0,7,5,3,1},
		{7,6,3,2,5,4,1,0}
	};

	//Is Stored in  ROW_MAJOR Format.
	typedef unsigned char* charPtr;
	m_ucpLut = new charPtr[8];
	for (int i=0;i<8;i++) {
		m_ucpLut[i] = new unsigned char[8];
		for (int j=0;j<8;j++) {
			m_ucpLut[i][j] = tmp[i][j];
		}
	}
}

template <typename T>
feMatrix<T>::~feMatrix() {
}


#undef __FUNCT__
#define __FUNCT__ "feMatrix_MatGetDiagonal"
template <typename T>
bool feMatrix<T>::MatGetDiagonal(Vec _diag, double scale){
	PetscFunctionBegin;
#ifdef __DEBUG__
	assert ( ( m_daType == PETSC ) || ( m_daType == OCT ) );
#endif

	int ierr;

	// PetscScalar zero=0.0;

	if (m_daType == PETSC) {

		PetscInt x,y,z,m,n,p;
		PetscInt mx,my,mz;
		int xne,yne,zne;

		PetscScalar ***diag;
		Vec diagLocal;

		/* Get all corners*/
		if (m_DA == NULL)
			std::cerr << "Da is null" << std::endl;
		ierr = DMDAGetCorners(m_DA, &x, &y, &z, &m, &n, &p); CHKERRQ(ierr);
		/* Get Info*/
		ierr = DMDAGetInfo(m_DA,0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0); CHKERRQ(ierr);

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

		ierr = DMGetLocalVector(m_DA, &diagLocal); CHKERRQ(ierr);
		ierr = VecZeroEntries(diagLocal);

		// ierr = DAGlobalToLocalBegin(m_DA, _diag, INSERT_VALUES, diagLocal); CHKERRQ(ierr);
		// ierr = DAGlobalToLocalEnd(m_DA, _diag, INSERT_VALUES, diagLocal); CHKERRQ(ierr);


		ierr = DMDAVecGetArray(m_DA, diagLocal, &diag);

		// Any derived class initializations ...
		preMatVec();

		// loop through all elements ...
		for (int k=z; k<z+zne; k++) {
			for (int j=y; j<y+yne; j++) {
				for (int i=x; i<x+xne; i++) {
					ElementalMatGetDiagonal(i, j, k, diag, scale);
				} // end i
			} // end j
		} // end k

		postMatVec();

		ierr = DMDAVecRestoreArray(m_DA, diagLocal, &diag); CHKERRQ(ierr);


		ierr = DMLocalToGlobalBegin(m_DA, diagLocal, ADD_VALUES,_diag); CHKERRQ(ierr);
		ierr = DMLocalToGlobalEnd(m_DA, diagLocal, ADD_VALUES, _diag); CHKERRQ(ierr);

		ierr = DMRestoreLocalVector(m_DA, &diagLocal); CHKERRQ(ierr);


	} else {
		// loop for octree DA.
		PetscScalar *diag=NULL;

		// get Buffers ...
		//Nodal,Non-Ghosted,Read,1 dof, Get in array and get ghosts during computation
		m_octDA->vecGetBuffer(_diag, diag, false, false, false, m_uiDof);

		preMatVec();

		// loop through all elements ...
		for ( m_octDA->init<ot::DA_FLAGS::ALL>(); m_octDA->curr() < m_octDA->end<ot::DA_FLAGS::ALL>(); m_octDA->next<ot::DA_FLAGS::ALL>() ) {
			ElementalMatGetDiagonal( m_octDA->curr(), diag, scale);
		}//end

		postMatVec();

		// Restore Vectors ..
		m_octDA->vecRestoreBuffer(_diag, diag, false, false, false, m_uiDof);
	}

	PetscFunctionReturn(0);
}



/**
* 	@brief		The matrix-vector multiplication routine that is used by
* 				matrix-free methods.
* 	@param		_in	PETSc Vec which is the input vector with whom the
* 				product is to be calculated.
* 	@param		_out PETSc Vec, the output of M*_in
* 	@return		bool true if successful, false otherwise.
*
*  The matrix-vector multiplication routine that is used by matrix-free
* 	methods. The product is directly calculated from the elemental matrices,
*  which are computed by the ElementalMatrix() function. Use the Assemble()
*  function for matrix based methods.
**/
#undef __FUNCT__
#define __FUNCT__ "feMatrix_MatVec"
template <typename T>
bool feMatrix<T>::MatVec(Vec _in, Vec _out, double scale){
	PetscFunctionBegin;

#ifdef __DEBUG__
	assert ( ( m_daType == PETSC ) || ( m_daType == OCT ) );
#endif

	int ierr;
	// PetscScalar zero=0.0;

	if (m_daType == PETSC) {

		PetscInt x,y,z,m,n,p;
		PetscInt mx,my,mz;
		int xne,yne,zne;

		PetscScalar ***in, ***out;
		Vec inlocal, outlocal;

		/* Get all corners*/
		if (m_DA == NULL)
			std::cerr << "Da is null" << std::endl;
		ierr = DMDAGetCorners(m_DA, &x, &y, &z, &m, &n, &p); CHKERRQ(ierr);
		/* Get Info*/
		ierr = DMDAGetInfo(m_DA,0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0); CHKERRQ(ierr);

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

		// std::cout << x << "," << y << "," << z << " + " << xne <<","<<yne<<","<<zne<<std::endl;

		// Get the local vector so that the ghost nodes can be accessed
		ierr = DMGetLocalVector(m_DA, &inlocal); CHKERRQ(ierr);
		ierr = DMGetLocalVector(m_DA, &outlocal); CHKERRQ(ierr);
		// ierr = VecDuplicate(inlocal, &outlocal); CHKERRQ(ierr);

		ierr = DMGlobalToLocalBegin(m_DA, _in, INSERT_VALUES, inlocal); CHKERRQ(ierr);
		ierr = DMGlobalToLocalEnd(m_DA, _in, INSERT_VALUES, inlocal); CHKERRQ(ierr);
		// ierr = DAGlobalToLocalBegin(m_DA, _out, INSERT_VALUES, outlocal); CHKERRQ(ierr);
		// ierr = DAGlobalToLocalEnd(m_DA, _out, INSERT_VALUES, outlocal); CHKERRQ(ierr);

		ierr = VecZeroEntries(outlocal);

		ierr = DMDAVecGetArray(m_DA, inlocal, &in);
		ierr = DMDAVecGetArray(m_DA, outlocal, &out);

		// Any derived class initializations ...
		preMatVec();

		// loop through all elements ...
		for (int k=z; k<z+zne; k++) {
			for (int j=y; j<y+yne; j++) {
				for (int i=x; i<x+xne; i++) {
					// std::cout << i <<"," << j << "," << k << std::endl;
					ElementalMatVec(i, j, k, in, out, scale);
				} // end i
			} // end j
		} // end k

		postMatVec();

		ierr = DMDAVecRestoreArray(m_DA, inlocal, &in); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(m_DA, outlocal, &out); CHKERRQ(ierr);

		ierr = DMLocalToGlobalBegin(m_DA, outlocal, ADD_VALUES, _out); CHKERRQ(ierr);
		ierr = DMLocalToGlobalEnd(m_DA, outlocal, ADD_VALUES, _out); CHKERRQ(ierr);

		ierr = DMRestoreLocalVector(m_DA, &inlocal); CHKERRQ(ierr);
		ierr = DMRestoreLocalVector(m_DA, &outlocal); CHKERRQ(ierr);
		// ierr = VecDestroy(outlocal); CHKERRQ(ierr);

	} else {
		// loop for octree DA.


		PetscScalar *out=NULL;
		PetscScalar *in=NULL;

		// get Buffers ...
		//Nodal,Non-Ghosted,Read,1 dof, Get in array and get ghosts during computation
		m_octDA->vecGetBuffer(_in,   in, false, false, true,  m_uiDof);
		m_octDA->vecGetBuffer(_out, out, false, false, false, m_uiDof);

		// start comm for in ...
		//m_octDA->updateGhostsBegin<PetscScalar>(in, false, m_uiDof);
		// m_octDA->ReadFromGhostsBegin<PetscScalar>(in, false, m_uiDof);
		m_octDA->ReadFromGhostsBegin<PetscScalar>(in, m_uiDof);
		preMatVec();

		// Independent loop, loop through the nodes this processor owns..
		for ( m_octDA->init<ot::DA_FLAGS::INDEPENDENT>(), m_octDA->init<ot::DA_FLAGS::WRITABLE>(); m_octDA->curr() < m_octDA->end<ot::DA_FLAGS::INDEPENDENT>(); m_octDA->next<ot::DA_FLAGS::INDEPENDENT>() ) {
			ElementalMatVec( m_octDA->curr(), in, out, scale);
		}//end INDEPENDENT

		// Wait for communication to end.
		//m_octDA->updateGhostsEnd<PetscScalar>(in);
		m_octDA->ReadFromGhostsEnd<PetscScalar>(in);

		// Dependent loop ...
		for ( m_octDA->init<ot::DA_FLAGS::DEPENDENT>(), m_octDA->init<ot::DA_FLAGS::WRITABLE>(); m_octDA->curr() < m_octDA->end<ot::DA_FLAGS::DEPENDENT>(); m_octDA->next<ot::DA_FLAGS::DEPENDENT>() ) {
			ElementalMatVec( m_octDA->curr(), in, out, scale);
		}//end DEPENDENT

		postMatVec();

		// Restore Vectors ...
		m_octDA->vecRestoreBuffer(_in,   in, false, false, true,  m_uiDof);
		m_octDA->vecRestoreBuffer(_out, out, false, false, false, m_uiDof);

	}

	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "feMatrix_MatAssemble"
template <typename T>
bool feMatrix<T>::GetAssembledMatrix(Mat *J, MatType mtype) {
	PetscFunctionBegin;

	int ierr;

#ifdef __DEBUG__
	assert ( ( m_daType == PETSC ) || ( m_daType == OCT ) );
#endif
	if (m_daType == PETSC) {
		Mat K;

		// Petsc Part ..
		unsigned int elemMatSize = m_uiDof*8;

		PetscScalar* Ke = new PetscScalar[elemMatSize*elemMatSize];
		MatStencil *idx = new MatStencil[elemMatSize];

		PetscInt x,y,z,m,n,p;
		PetscInt mx,my,mz;
		int xne,yne,zne;

		/* Get all corners*/
		if (m_DA == NULL)
			std::cerr << "Da is null" << std::endl;
		ierr = DMDAGetCorners(m_DA, &x, &y, &z, &m, &n, &p); CHKERRQ(ierr);
		/* Get Info*/
		ierr = DMDAGetInfo(m_DA,0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0); CHKERRQ(ierr);

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

		// Get the matrix from the DA ...
		// DAGetMatrix(m_DA, mtype, &J);
		// DAGetMatrix(m_DA, MATAIJ, &K);
		DMCreateMatrix(m_DA, &K);

		MatZeroEntries(K);

		preMatVec();
		// loop through all elements ...
		for (int k=z; k<z+zne; k++) {
			for (int j=y; j<y+yne; j++) {
				for (int i=x; i<x+xne; i++) {
					int idxMap[8][3]={
						{k, j, i},
						{k,j,i+1},
						{k,j+1,i},
						{k,j+1,i+1},
						{k+1, j, i},
						{k+1,j,i+1},
						{k+1,j+1,i},
						{k+1,j+1,i+1}
					};
					for (unsigned int q=0; q<8; q++) {
						for (unsigned int dof=0; dof<m_uiDof; dof++) {
							idx[m_uiDof*q + dof].i = idxMap[q][2];
							idx[m_uiDof*q + dof].j = idxMap[q][1];
							idx[m_uiDof*q + dof].k = idxMap[q][0];
							idx[m_uiDof*q + dof].c = dof;
						}
					}
					GetElementalMatrix(i, j, k, Ke);
					// Set Values
					// @check if rows/cols need to be interchanged.
					MatSetValuesStencil(K, elemMatSize, idx, elemMatSize, idx, Ke, ADD_VALUES);
				} // end i
			} // end j
		} // end k
		postMatVec();

		MatAssemblyBegin (K, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd   (K, MAT_FINAL_ASSEMBLY);

		*J = K;

		delete [] Ke;
		delete [] idx;

	} else {
		if(!(m_octDA->computedLocalToGlobal())) {
			m_octDA->computeLocalToGlobalMappings();
		}
		// Octree part ...
		char matType[30];
		PetscBool typeFound;
		PetscOptionsGetString(PETSC_NULL, "-fullJacMatType", matType, 30, &typeFound);
		if(!typeFound) {
			std::cout<<"I need a MatType for the full Jacobian matrix!"<<std::endl;
			MPI_Finalize();
			exit(0);
		}
		m_octDA->createMatrix(*J, matType, 1);
		MatZeroEntries(*J);
		std::vector<ot::MatRecord> records;

		preMatVec();

		for(m_octDA->init<ot::DA_FLAGS::WRITABLE>(); m_octDA->curr() < m_octDA->end<ot::DA_FLAGS::WRITABLE>();	m_octDA->next<ot::DA_FLAGS::WRITABLE>()) {
			GetElementalMatrix(m_octDA->curr(), records);
			if(records.size() > 500) {
				m_octDA->setValuesInMatrix(*J, records, 1, ADD_VALUES);
			}
		}//end writable
		m_octDA->setValuesInMatrix(*J, records, 1, ADD_VALUES);

		postMatVec();

		MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);
	}


	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "alignElementAndVertices"
template <typename T>
PetscErrorCode feMatrix<T>::alignElementAndVertices(ot::DA * da, stdElemType & sType, unsigned int* indices) {
	PetscFunctionBegin;

	sType = ST_0;
	da->getNodeIndices(indices);

	// not required ....
	// int rank;
	// MPI_Comm_rank(da->getComm(), &rank);

	if (da->isHanging(da->curr())) {

		int childNum = da->getChildNumber();
		Point pt = da->getCurrentOffset();

		unsigned char hangingMask = da->getHangingNodeIndex(da->curr());

		//Change HangingMask and indices based on childNum
		mapVtxAndFlagsToOrientation(childNum, indices, hangingMask);

		unsigned char eType = ((126 & hangingMask)>>1);

		reOrderIndices(eType, indices);
	}//end if hangingElem.
	PetscFunctionReturn(0);
}//end function.

#undef __FUNCT__
#define __FUNCT__ "mapVtxAndFlagsToOrientation"
template <typename T>
PetscErrorCode feMatrix<T>::mapVtxAndFlagsToOrientation(int childNum, unsigned int* indices, unsigned char & mask) {
	PetscFunctionBegin;
	unsigned int tmp[8];
	unsigned char tmpFlags = 0;
	for (int i=0;i<8;i++) {
		tmp[i] = indices[m_ucpLut[childNum][i]];
		tmpFlags = ( tmpFlags | ( ( (1<<(m_ucpLut[childNum][i])) & mask ) ? (1<<i) : 0 ) );
	}
	for (int i=0;i<8;i++) {
		indices[i] = tmp[i];
	}
	mask = tmpFlags;
	PetscFunctionReturn(0);
}//end function

#undef __FUNCT__
#define __FUNCT__ "reOrderIndices"
template <typename T>
PetscErrorCode feMatrix<T>::reOrderIndices(unsigned char eType, unsigned int* indices) {
#ifdef __DEBUG_1
	std::cout << "Entering " << __func__ << std::endl;
#endif
	PetscFunctionBegin;
	unsigned int tmp;
	switch (eType) {
	case  ET_N:
		break;
	case  ET_Y:
		break;
	case  ET_X:
		//Swap 1 & 2, Swap 5 & 6
		tmp = indices[1];
		indices[1] = indices[2];
		indices[2] = tmp;
		tmp = indices[5];
		indices[5] = indices[6];
		indices[6] = tmp;
		break;
	case  ET_XY:
		break;
	case  ET_Z:
		//Swap 2 & 4, Swap 3 & 5
		tmp = indices[2];
		indices[2] = indices[4];
		indices[4] = tmp;
		tmp = indices[3];
		indices[3] = indices[5];
		indices[5] = tmp;
		break;
	case  ET_ZY:
		//Swap 1 & 4, Swap 3 & 6
		tmp = indices[1];
		indices[1] = indices[4];
		indices[4] = tmp;
		tmp = indices[3];
		indices[3] = indices[6];
		indices[6] = tmp;
		break;
	case  ET_ZX:
		//Swap 2 & 4, Swap 3 & 5
		tmp = indices[2];
		indices[2] = indices[4];
		indices[4] = tmp;
		tmp = indices[3];
		indices[3] = indices[5];
		indices[5] = tmp;
		break;
	case  ET_ZXY:
		break;
	case  ET_XY_XY:
		break;
	case  ET_XY_ZXY:
		break;
	case  ET_YZ_ZY:
		//Swap 1 & 4, Swap 3 & 6
		tmp = indices[1];
		indices[1] = indices[4];
		indices[4] = tmp;
		tmp = indices[3];
		indices[3] = indices[6];
		indices[6] = tmp;
		break;
	case  ET_YZ_ZXY:
		//Swap 1 & 4, Swap 3 & 6
		tmp = indices[1];
		indices[1] = indices[4];
		indices[4] = tmp;
		tmp = indices[3];
		indices[3] = indices[6];
		indices[6] = tmp;
		break;
	case  ET_YZ_XY_ZXY:
		break;
	case  ET_ZX_ZX:
		//Swap 2 & 4, Swap 3 & 5
		tmp = indices[2];
		indices[2] = indices[4];
		indices[4] = tmp;
		tmp = indices[3];
		indices[3] = indices[5];
		indices[5] = tmp;
		break;
	case  ET_ZX_ZXY:
		//Swap 2 & 4, Swap 3 & 5
		tmp = indices[2];
		indices[2] = indices[4];
		indices[4] = tmp;
		tmp = indices[3];
		indices[3] = indices[5];
		indices[5] = tmp;
		break;
	case  ET_ZX_XY_ZXY:
		//Swap 1 & 2, Swap 5 & 6
		tmp = indices[1];
		indices[1] = indices[2];
		indices[2] = tmp;
		tmp = indices[5];
		indices[5] = indices[6];
		indices[6] = tmp;
		break;
	case  ET_ZX_YZ_ZXY:
		//Swap 2 & 4, Swap 3 & 5
		tmp = indices[2];
		indices[2] = indices[4];
		indices[4] = tmp;
		tmp = indices[3];
		indices[3] = indices[5];
		indices[5] = tmp;
		break;
	case  ET_ZX_YZ_XY_ZXY:
		break;
	default:
		std::cout<<"in reOrder Etype: "<< (int) eType << std::endl;
		assert(false);
	}
#ifdef __DEBUG_1
	std::cout << "Leaving " << __func__ << std::endl;
#endif
	PetscFunctionReturn(0);
}

template <typename T>
bool feMatrix<T>::MatVec_new(Vec _in, Vec _out, double scale){
	PetscFunctionBegin;

#ifdef __DEBUG__
	assert ( ( m_daType == PETSC ) || ( m_daType == OCT ) );
#endif

	int ierr;
	// PetscScalar zero=0.0;

	// can keep as member variables if required.
	PetscScalar* local_in = new PetscScalar[m_uiDof*8];
	PetscScalar* local_out = new PetscScalar[m_uiDof*8];
	PetscScalar* coords = new PetscScalar[m_uiDof*3];

	if (m_daType == PETSC) {
    // m_dLx, m_dLy, m_dLz
		PetscInt x,y,z,m,n,p;
		PetscInt mx,my,mz;
		int xne,yne,zne;

		PetscScalar ***in, ***out;
		Vec inlocal, outlocal;

		/* Get all corners*/
		if (m_DA == NULL)
			std::cerr << "Da is null" << std::endl;
		ierr = DMDAGetCorners(m_DA, &x, &y, &z, &m, &n, &p); CHKERRQ(ierr);
		/* Get Info*/
		ierr = DMDAGetInfo(m_DA,0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0); CHKERRQ(ierr);

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

		// std::cout << x << "," << y << "," << z << " + " << xne <<","<<yne<<","<<zne<<std::endl;

		// Get the local vector so that the ghost nodes can be accessed
		ierr = DMGetLocalVector(m_DA, &inlocal); CHKERRQ(ierr);
		ierr = DMGetLocalVector(m_DA, &outlocal); CHKERRQ(ierr);
		// ierr = VecDuplicate(inlocal, &outlocal); CHKERRQ(ierr);

		ierr = DMGlobalToLocalBegin(m_DA, _in, INSERT_VALUES, inlocal); CHKERRQ(ierr);
		ierr = DMGlobalToLocalEnd(m_DA, _in, INSERT_VALUES, inlocal); CHKERRQ(ierr);
		// ierr = DAGlobalToLocalBegin(m_DA, _out, INSERT_VALUES, outlocal); CHKERRQ(ierr);
		// ierr = DAGlobalToLocalEnd(m_DA, _out, INSERT_VALUES, outlocal); CHKERRQ(ierr);

		ierr = VecZeroEntries(outlocal);

		ierr = DMDAVecGetArray(m_DA, inlocal, &in);
		ierr = DMDAVecGetArray(m_DA, outlocal, &out);

		// Any derived class initializations ...
		preMatVec();

		// initialize coords

		// loop through all elements ...
		double hx = m_dLx/(mx -1);
		double hy = m_dLy/(my -1);
		double hz = m_dLz/(mz -1);
		for (int k=z; k<z+zne; k++) {
			for (int j=y; j<y+yne; j++) {
				for (int i=x; i<x+xne; i++) {

					// copy data to local
					for (int p=k,idx=0; p<k+2; ++p)
						for (int q=j; q<j+2; ++q) {
							for (int r=m_uiDof*i; r<m_uiDof*(i+2); ++r,++idx) {
								local_in[idx] = in[p][q][r];
							}
						}

				  coords[0] = i*hx; coords[1] = j*hy; coords[2] = k*hz;
					coords[3] = (i+1)*hx; coords[4] = j*hy; coords[5] = k*hz;
					coords[6] = i*hx; coords[7] = (j+1)*hy; coords[8] = k*hz;
					coords[9] = (i+1)*hx; coords[10] = (j+1)*hy; coords[11] = k*hz;
					coords[12] = i*hx; coords[13] = j*hy; coords[14] = (k+1)*hz;
					coords[15] = (i+1)*hx; coords[16] = j*hy; coords[17] = (k+1)*hz;
					coords[18] = i*hx; coords[19] = (j+1)*hy; coords[20] = (k+1)*hz;
					coords[21] = (i+1)*hx; coords[22] = (j+1)*hy; coords[23] = (k+1)*hz;

					ElementalMatVec(local_in, local_out, coords, scale);

					// copy data back
					for (int p=k; p<k+2; ++p)
						for (int q=j; q<j+2; ++q)
							for (int r=m_uiDof*i,idx=0; r<m_uiDof*(i+2); ++r,++idx) {
								in[p][q][r] = local_out[idx];
					}
				} // end i
			} // end j
		} // end k

		postMatVec();

		ierr = DMDAVecRestoreArray(m_DA, inlocal, &in); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(m_DA, outlocal, &out); CHKERRQ(ierr);

		ierr = DMLocalToGlobalBegin(m_DA, outlocal, ADD_VALUES, _out); CHKERRQ(ierr);
		ierr = DMLocalToGlobalEnd(m_DA, outlocal, ADD_VALUES, _out); CHKERRQ(ierr);

		ierr = DMRestoreLocalVector(m_DA, &inlocal); CHKERRQ(ierr);
		ierr = DMRestoreLocalVector(m_DA, &outlocal); CHKERRQ(ierr);
		// ierr = VecDestroy(outlocal); CHKERRQ(ierr);

	} else {
		// loop for octree DA.


		PetscScalar *out=NULL;
		PetscScalar *in=NULL;

		// get Buffers ...
		//Nodal,Non-Ghosted,Read,1 dof, Get in array and get ghosts during computation
		m_octDA->vecGetBuffer(_in,   in, false, false, true,  m_uiDof);
		m_octDA->vecGetBuffer(_out, out, false, false, false, m_uiDof);

		// start comm for in ...
		//m_octDA->updateGhostsBegin<PetscScalar>(in, false, m_uiDof);
		// m_octDA->ReadFromGhostsBegin<PetscScalar>(in, false, m_uiDof);
		m_octDA->ReadFromGhostsBegin<PetscScalar>(in, m_uiDof);
		preMatVec();

		unsigned int maxD = m_octDA->getMaxDepth();
		unsigned int lev;
		double hx, hy, hz;
		Point pt;

		double xFac = m_dLx/((double)(1<<(maxD-1)));
		double yFac = m_dLy/((double)(1<<(maxD-1)));
		double zFac = m_dLz/((double)(1<<(maxD-1)));



		// Independent loop, loop through the nodes this processor owns..
		for ( m_octDA->init<ot::DA_FLAGS::INDEPENDENT>(), m_octDA->init<ot::DA_FLAGS::WRITABLE>(); m_octDA->curr() < m_octDA->end<ot::DA_FLAGS::INDEPENDENT>(); m_octDA->next<ot::DA_FLAGS::INDEPENDENT>() ) {
			lev = m_octDA->getLevel(m_octDA->curr());
			hx = xFac*(1<<(maxD - lev));
			hy = yFac*(1<<(maxD - lev));
			hz = zFac*(1<<(maxD - lev));

			pt = m_octDA->getCurrentOffset();

			// coords
			coords[0] = pt.x()*xFac; coords[1] = pt.y()*yFac; coords[2] = pt.z()*zFac;
			coords[3] = coords[0] + hx; coords[4] = coords[1]; coords[5] = coords[2];
			coords[6] = coords[0] ; coords[7] = coords[1]+hy; coords[8] = coords[2];
			coords[9] = coords[0] + hx; coords[10] = coords[1]+hy; coords[11] = coords[2];
			coords[12] = coords[0] ; coords[13] = coords[1]; coords[14] = coords[2]+hz;
			coords[15] = coords[0] + hx; coords[16] = coords[1]; coords[17] = coords[2]+hz;
			coords[18] = coords[0] ; coords[19] = coords[1]+hy; coords[20] = coords[2]+hz;
			coords[21] = coords[0] + hx; coords[22] = coords[1]+hy; coords[23] = coords[2]+hz;

			// copy and interpolate to local
      // interp_global_to_local(glo, local_in, ot::DA* m_octDA);

			// ElementalMatVec( local_in, local_out, coords, scale);

		  // copy to global
      // interp_local_to_global(local_out, glo, ot::DA* m_octDA);


		}//end INDEPENDENT

		// Wait for communication to end.
		//m_octDA->updateGhostsEnd<PetscScalar>(in);
		m_octDA->ReadFromGhostsEnd<PetscScalar>(in);

		// Dependent loop ...
		for ( m_octDA->init<ot::DA_FLAGS::DEPENDENT>(), m_octDA->init<ot::DA_FLAGS::WRITABLE>(); m_octDA->curr() < m_octDA->end<ot::DA_FLAGS::DEPENDENT>(); m_octDA->next<ot::DA_FLAGS::DEPENDENT>() ) {
			ElementalMatVec( m_octDA->curr(), in, out, scale);
		}//end DEPENDENT

		postMatVec();

		// Restore Vectors ...
		m_octDA->vecRestoreBuffer(_in,   in, false, false, true,  m_uiDof);
		m_octDA->vecRestoreBuffer(_out, out, false, false, false, m_uiDof);

	}

	delete [] local_in;
	delete [] local_out;
	delete [] coords;

	PetscFunctionReturn(0);
}


template <typename T>
PetscErrorCode feMatrix<T>::interp_global_to_local(PetscScalar* glo, PetscScalar* __restrict loc, ot::DA* m_octDA) {
	unsigned int idx[8];
	unsigned char hangingMask = m_octDA->getHangingNodeIndex(m_octDA->curr());
	unsigned int chNum = m_octDA->getChildNumber();
	m_octDA->getNodeIndices(idx);

	unsigned char eType = getEtype(hangingMask, chNum);

	// remap to ch 0
	unsigned char* rot = m_ucpLut[chNum];
	switch ( eType ) {
		case  ET_N:
		  // not hanging, simply copy.
		  for (size_t i = 0; i < m_uiDof; i++) {
				loc[i] = glo[m_uiDof*idx[0]+i];
				loc[m_uiDof + i] = glo[m_uiDof*idx[1]+i];
				loc[2*m_uiDof + i] = glo[m_uiDof*idx[2]+i];
				loc[3*m_uiDof + i] = glo[m_uiDof*idx[3]+i];
				loc[4*m_uiDof + i] = glo[m_uiDof*idx[4]+i];
				loc[5*m_uiDof + i] = glo[m_uiDof*idx[5]+i];
				loc[6*m_uiDof + i] = glo[m_uiDof*idx[6]+i];
				loc[7*m_uiDof + i] = glo[m_uiDof*idx[7]+i];
		  }
			break;
		case  ET_Y: // v2
				for (size_t i = 0; i < m_uiDof; i++) {
					loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
					loc[ rot[1]*m_uiDof + i] =   glo[m_uiDof*idx[rot[1]]+i];
					loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
						                           glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
					loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
					loc[ rot[4]*m_uiDof + i] =   glo[m_uiDof*idx[rot[4]]+i];
					loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
					loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
					loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
			  }
			break;
		case  ET_X: // v1
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] =   glo[m_uiDof*idx[rot[2]]+i];
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] =   glo[m_uiDof*idx[rot[4]]+i];
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_XY: // v1, v2
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] =   glo[m_uiDof*idx[rot[4]]+i];
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_Z: // v4
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = 	 glo[m_uiDof*idx[rot[1]]+i];
			loc[ rot[2]*m_uiDof + i] =   glo[m_uiDof*idx[rot[2]]+i];
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_ZY: // v2, v4
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = 	 glo[m_uiDof*idx[rot[1]]+i];
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_ZX: // v1, v4
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] =   glo[m_uiDof*idx[rot[2]]+i];
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_ZXY: // v1, v2, v4
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_XY_XY: // v1, v2, v3
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[3]]+i] ) / 4.0;
			loc[ rot[4]*m_uiDof + i] =   glo[m_uiDof*idx[rot[4]]+i];
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_XY_ZXY: // v1, v2, v3, v4
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[3]]+i] ) / 4.0;
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_YZ_ZY: // v2, v4, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = 	 glo[m_uiDof*idx[rot[1]]+i];
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[6]]+i] ) / 4.0;
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_YZ_ZXY: // v1, v2, v4, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[6]]+i] ) / 4.0;
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_YZ_XY_ZXY: // v1, v2, v3, v4, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[3]]+i] ) / 4.0;
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] =   glo[m_uiDof*idx[rot[5]]+i];
			loc[ rot[6]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[6]]+i] ) / 4.0;
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_ZX_ZX: // v1, v4, v5
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] =   glo[m_uiDof*idx[rot[2]]+i];
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[5]]+i] +
																	 glo[m_uiDof*idx[rot[5]]+i] +
																	 glo[m_uiDof*idx[rot[5]]+i] +
																	 glo[m_uiDof*idx[rot[5]]+i] ) / 4.0;
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_ZX_ZXY: // v1, v2, v4, v5
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[5]]+i] ) / 4.0;
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_ZX_XY_ZXY: // vi, v2, v3, v4, v5
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[3]]+i] ) / 4.0;
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[5]]+i] ) / 4.0;
			loc[ rot[6]*m_uiDof + i] =   glo[m_uiDof*idx[rot[6]]+i];
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_ZX_YZ_ZXY: // v1, v2, v4, v5, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] =   glo[m_uiDof*idx[rot[3]]+i];
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[5]]+i] ) / 4.0;
			loc[ rot[6]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[6]]+i] ) / 4.0;
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		case  ET_ZX_YZ_XY_ZXY:  // v1, v2, v3, v4, v5, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			loc[ rot[0]*m_uiDof + i] =   glo[m_uiDof*idx[rot[0]]+i];
			loc[ rot[1]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] ) / 2.0 ;
			loc[ rot[2]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] ) / 2.0;
			loc[ rot[3]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[3]]+i] ) / 4.0;
			loc[ rot[4]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] ) / 2.0;
			loc[ rot[5]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[1]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[5]]+i] ) / 4.0;
			loc[ rot[6]*m_uiDof + i] = ( glo[m_uiDof*idx[rot[0]]+i] +
																	 glo[m_uiDof*idx[rot[2]]+i] +
																	 glo[m_uiDof*idx[rot[4]]+i] +
																	 glo[m_uiDof*idx[rot[6]]+i] ) / 4.0;
			loc[ rot[7]*m_uiDof + i] =   glo[m_uiDof*idx[rot[7]]+i];
		}
			break;
		default:
			std::cout<<"in glo_to_loc: "<< (int) eType << std::endl;
			assert(false);
	}

}

template <typename T>
PetscErrorCode feMatrix<T>::interp_local_to_global(PetscScalar* __restrict loc, PetscScalar* glo, ot::DA* m_octDA) {
	unsigned int idx[8];
	unsigned char hangingMask = m_octDA->getHangingNodeIndex(m_octDA->curr());
	unsigned int chNum = m_octDA->getChildNumber();
	m_octDA->getNodeIndices(idx);

	unsigned char eType = getEtype(hangingMask, chNum);

	// remap to ch 0
	unsigned char* rot = m_ucpLut[chNum];

	switch ( eType ) {
		case  ET_N:
		  // not hanging, simply copy.
		  for (size_t i = 0; i < m_uiDof; i++) {
				glo[m_uiDof*idx[0]+i] += loc[i];
				glo[m_uiDof*idx[1]+i] += loc[m_uiDof + i];
				glo[m_uiDof*idx[2]+i] += loc[2*m_uiDof + i];
				glo[m_uiDof*idx[3]+i] += loc[3*m_uiDof + i];
				glo[m_uiDof*idx[4]+i] += loc[4*m_uiDof + i];
				glo[m_uiDof*idx[5]+i] += loc[5*m_uiDof + i];
				glo[m_uiDof*idx[6]+i] += loc[6*m_uiDof + i];
				glo[m_uiDof*idx[7]+i] += loc[7*m_uiDof + i];
		  }
			break;
		case  ET_Y: // v2
				for (size_t i = 0; i < m_uiDof; i++) {
					glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i];
					glo[m_uiDof*idx[rot[1]]+i] += loc[ rot[1]*m_uiDof + i];
					glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i];
					glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
					glo[m_uiDof*idx[rot[4]]+i] += loc[ rot[4]*m_uiDof + i];
					glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
					glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
					glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
			  }
			break;
		case  ET_X: // v1
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += loc[ rot[2]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_XY: // v1, v2
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i];
		  glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
		  glo[m_uiDof*idx[rot[4]]+i] += loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_Z: // v4
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += loc[ rot[2]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_ZY: // v2, v4
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_ZX: // v1, v4
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += loc[ rot[2]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_ZXY: // v1, v2, v4
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_XY_XY: // v1, v2, v3
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_XY_ZXY: // v1, v2, v3, v4
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_YZ_ZY: // v2, v4, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_YZ_ZXY: // v1, v2, v4, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_YZ_XY_ZXY: // v1, v2, v3, v4, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_ZX_ZX: // v1, v4, v5
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += loc[ rot[2]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_ZX_ZXY: // v1, v2, v4, v5
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_ZX_XY_ZXY: // v1, v2, v3, v4, v5
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i]; + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i]; + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_ZX_YZ_ZXY: // v1, v2, v4, v5, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		case  ET_ZX_YZ_XY_ZXY:  // v1, v2, v3, v4, v5, v6
		for (size_t i = 0; i < m_uiDof; i++) {
			glo[m_uiDof*idx[rot[0]]+i] += loc[ rot[0]*m_uiDof + i] + 0.5*loc[ rot[1]*m_uiDof + i] + 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i] + 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[1]]+i] += 0.5*loc[ rot[1]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[2]]+i] += 0.5*loc[ rot[2]*m_uiDof + i] + 0.25*loc[ rot[3]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[3]]+i] += 0.25*loc[ rot[3]*m_uiDof + i];
			glo[m_uiDof*idx[rot[4]]+i] += 0.5*loc[ rot[4]*m_uiDof + i] + 0.25*loc[ rot[5]*m_uiDof + i] + 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[5]]+i] += 0.25*loc[ rot[5]*m_uiDof + i];
			glo[m_uiDof*idx[rot[6]]+i] += 0.25*loc[ rot[6]*m_uiDof + i];
			glo[m_uiDof*idx[rot[7]]+i] += loc[ rot[7]*m_uiDof + i];
		}
			break;
		default:
			std::cout<<"in glo_to_loc: "<< (int) eType << std::endl;
			assert(false);
	}

}
