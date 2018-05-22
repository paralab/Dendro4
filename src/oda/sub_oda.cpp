/**
  @file sub_oda.cpp
  @author Hari Sundar, hsundar@gmail.com
 **/

#include "sub_oda.h"

namespace ot {

//***************Constructor*****************//
subDA::subDA(DA* da, std::function<double ( double, double, double ) > fx_retain, double* gSize) {
  std::cout << "subDA::Constructor - Starting " << std::endl;
  m_da = da;
  unsigned int lev;
  
  unsigned maxDepth = m_da->getMaxDepth() - 1;

  m_uiCommTag = 0;
  m_mpiContexts.clear();

  // std::cout << "subDA: mD= " << maxDepth << std::endl;
  auto inside = [](double d){ return d < 0.0; };

  double hx, hy, hz;
  std::array<double, 8> dist;
  Point pt;

  double xFac = gSize[0]/((double)(1<<(maxDepth)));
  double yFac = gSize[1]/((double)(1<<(maxDepth)));
  double zFac = gSize[2]/((double)(1<<(maxDepth)));

  unsigned int indices[8];

  unsigned int localElemSize = m_da->getGhostedElementSize() + m_da->getBoundaryNodeSize();

  // now process the DA to skip interior elements
  m_ucpSkipList.clear();
  m_ucpSkipList.resize(localElemSize, 0);
  m_uip_DA2sub_ElemMap.resize(localElemSize, 0);

  m_ucpSkipNodeList.clear();
  // m_ucpSkipNodeList.resize(m_da->getGhostedNodeSize(), 1);
  m_ucpSkipNodeList.resize(m_da->getLocalBufferSize(), 1);

  m_uip_DA2sub_NodeMap.resize(m_da->getLocalBufferSize(), 0);

  std::cout << "[subDA::DEBUG] buffer size: " << m_ucpSkipNodeList.size() << ", " << m_da->getLocalBufferSize() << std::endl;

  // m_uiNodeSize + m_uiBoundaryNodeSize + m_uiPreGhostNodeSize + m_uiPreGhostBoundaryNodeSize + m_uiPostGhostNodeSize

  // std::cout << "ghosted node Size: " << m_ucpSkipNodeList.size() << std::endl;
      
  for ( m_da->init<ot::DA_FLAGS::ALL>(); 
        m_da->curr() < m_da->end<ot::DA_FLAGS::ALL>(); 
        m_da->next<ot::DA_FLAGS::ALL>() ) {

          lev = m_da->getLevel(m_da->curr());
          hx = xFac*(1<<(maxDepth +1 - lev));
          hy = yFac*(1<<(maxDepth +1 - lev));
          hz = zFac*(1<<(maxDepth +1 - lev));

          pt = m_da->getCurrentOffset();

          m_da->getNodeIndices(indices);


          dist[0] = fx_retain(pt.x()*xFac, pt.y()*yFac, pt.z()*zFac);
          dist[1] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac, pt.z()*zFac);
          dist[2] = fx_retain(pt.x()*xFac, pt.y()*yFac+hy, pt.z()*zFac);
          dist[3] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac+hy, pt.z()*zFac);

          dist[4] = fx_retain(pt.x()*xFac, pt.y()*yFac, pt.z()*zFac+hz);
          dist[5] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac, pt.z()*zFac+hz);
          dist[6] = fx_retain(pt.x()*xFac, pt.y()*yFac+hy, pt.z()*zFac+hz);
          dist[7] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac+hy, pt.z()*zFac +hz);

          /*
          for (auto q: dist)
            std::cout << q << ", ";
          std::cout << std::endl;
          */

          if ( std::all_of( dist.begin(), dist.end(), inside ) ) {
            // element to skip.
            // std::cout << "subDA: skip element" << std::endl;
            // std::cout << "s" << da->curr() << ", ";
            m_ucpSkipList[m_da->curr()] = 1;
          }
          else {
            // touch nodes ....
            for(int k = 0; k < 8; k++) {
              if ( indices[k] < m_ucpSkipNodeList.size() )
                m_ucpSkipNodeList[indices[k]] = 0;
              // else
              //  std::cout << "skipList node index out of bound: " << indices[k] << std::endl;
            }
          }

        } // for 
   // std::cout << std::endl;
  
  // compute the mapping ...
  unsigned int postG_beg = da->getIdxPostGhostBegin();
  unsigned int elem_beg = da->getIdxElementBegin();
  
  unsigned int sum = 0;
  for (unsigned int i=0; i<m_ucpSkipNodeList.size(); ++i) {
    m_uip_DA2sub_NodeMap[i] = sum;
    if (m_ucpSkipNodeList[i] == 0) sum++;
  }

  //! counts 
  // Elemental ...
  m_uiElementSize = 0;
  m_uiPreGhostElementSize = 0;
  m_uiLocalBufferSize = 0;

  unsigned int j=0;
  for (unsigned int i=0; i<elem_beg; ++i) {
    if (m_ucpSkipList[i] == 0) {
      m_uiPreGhostElementSize++;
      m_uip_DA2sub_ElemMap[i] = j++;
    }
  }
  // std::cout << "subDA::subDA " << m_ucpSkipList.size() << ", " << postG_beg << std::endl;
  for (unsigned int i=elem_beg; i<postG_beg; ++i) {
    assert (i < m_ucpSkipList.size());
    if (m_ucpSkipList[i] == 0) {
      m_uiElementSize++;
      m_uip_DA2sub_ElemMap[i] = j++;
    }
  }
  for (unsigned int i=postG_beg; i<m_ucpSkipList.size(); ++i) {
    if (m_ucpSkipList[i] == 0) {
      m_uip_DA2sub_ElemMap[i] = j++;
    }
  }

  for (unsigned int i=0; i<m_ucpSkipNodeList.size(); ++i)  {
    if (m_ucpSkipNodeList[i] == 0) m_uiLocalBufferSize++;
  }

  m_uip_sub2DA_NodeMap.resize(m_uiLocalBufferSize, 0);
  for (unsigned int i=0; i<m_ucpSkipNodeList.size(); ++i)  {
    if (m_ucpSkipNodeList[i] == 0) {
      m_uip_sub2DA_NodeMap[m_uip_DA2sub_NodeMap[i]] = i;
    }
  }

  std::cout << "[subDA::DEBUG] " << da->getRankAll() << ": ElementSize=" << m_uiElementSize << " , PreGhostElemSize=" << m_uiPreGhostElementSize << std::endl;
  
  // Nodal
  m_uiNodeSize = 0;
  m_uiPreGhostNodeSize = 0;
  m_uiPostGhostNodeSize = 0;

  m_uiPreGhostBoundaryNodeSize=0;
  m_uiBoundaryNodeSize=0;

  for (unsigned int i=0; i<elem_beg; ++i) {
    if (m_ucpSkipNodeList[i] == 0) {
      m_uiPreGhostNodeSize++;
      if ( (m_da->getLevel(i) & ot::TreeNode::NODE ) && (m_da->getLevel(i) & ot::TreeNode::BOUNDARY ) )
        m_uiPreGhostBoundaryNodeSize++;
    }
  }
  for (unsigned int i=elem_beg; i<postG_beg; ++i) {
    if (m_ucpSkipNodeList[i] == 0) {
      m_uiNodeSize++;
      if ( (m_da->getLevel(i) & ot::TreeNode::NODE ) && (m_da->getLevel(i) & ot::TreeNode::BOUNDARY ) )
        m_uiBoundaryNodeSize++;
    }

  }
  for (unsigned int i=postG_beg; i<m_ucpSkipNodeList.size(); ++i) {
    if (m_ucpSkipNodeList[i] == 0) m_uiPostGhostNodeSize++;
  }

  std::cout << "[subDA::DEBUG] " << da->getRankAll() << ": NodeSizes  (" << m_uiPreGhostNodeSize << ") " << m_uiNodeSize << " (" << m_uiPostGhostNodeSize << ")" << std::endl;

  std::cout << "[subDA::DEBUG] " << da->getRankAll() << ": BoundaryNodeSizes  (" << m_uiPreGhostBoundaryNodeSize << ") " << m_uiBoundaryNodeSize << std::endl;

  // scatter map 

  m_uipScatterMap.clear();
  m_uipSendCounts.clear();
  m_uipSendProcs.clear();
  m_uipSendOffsets.clear();
  // new
  unsigned int k=0; 
  unsigned int offset=0;
  for (unsigned int p=0; m_da->getSendProcSize(); ++p) {
    unsigned int cnt=0;
    for (unsigned int i=0; i<m_da->getSendCountsEntry(p); ++i) {
      unsigned int idx = m_da->getScatterMapEntry(k);
      if ( m_ucpSkipNodeList[idx] == 0 ) {
        m_uipScatterMap.push_back(m_uip_DA2sub_NodeMap[idx]);
      }
      cnt++;
      k++;
    }
    if (cnt) {
      m_uipSendProcs.push_back(m_da->getSendProcEntry(p));
      m_uipSendCounts.push_back(cnt);
      m_uipSendOffsets.push_back(offset);
      offset += cnt;
    }
  }

  std::cout << "subDA::Constructor - All done." << std::endl; 

  // old one
  // for (unsigned int i=0; i<m_da->getScatterMapSize(); ++i) {
  //   unsigned int idx = m_da->getScatterMapEntry(i);
    
  //   if ( m_ucpSkipNodeList[idx] == 0 ) {
  //     m_uipScatterMap.push_back(m_uip_DA2sub_NodeMap[idx]);
  //   }
  // } 
 
} // subDA constructor.

subDA::~subDA() {

}

int subDA::createVector(Vec &arr, bool isElemental, bool isGhosted, unsigned int dof) {
    // first determine the length of the vector ...
    unsigned int sz = 0;
    
    if (isElemental) {
      sz = m_uiElementSize;
      if (isGhosted) {
        sz += (m_uiPreGhostElementSize);
      }
    } else {
      sz = m_uiNodeSize + m_uiBoundaryNodeSize;
      if (isGhosted) {
        sz += (m_uiPreGhostNodeSize + m_uiPreGhostBoundaryNodeSize + m_uiPostGhostNodeSize);
      }
    }
    // now for dof ...
    sz *= dof;

    std::cout << "subDA::createVector size: " << sz << std::endl;    

    MPI_Comm comm = m_da->getComm();
    int npes = m_da->getNpesAll();

    // now create the PETSc Vector
    VecCreate(comm, &arr);
    VecSetSizes(arr, sz, PETSC_DECIDE);
    if (npes > 1) {
      VecSetType(arr,VECMPI);
    } else {
      VecSetType(arr,VECSEQ);
    }    
    return 0;
  } // createVector

  int subDA::createMatrix(Mat &M, MatType mtype, unsigned int dof) {
  // first determine the size ...
    unsigned int sz = 0;
    
    sz = dof*(m_uiNodeSize + m_uiBoundaryNodeSize);

    MPI_Comm comm = m_da->getComm();
    int npes = m_da->getNpesAll();

    // now create the PETSc Mat
    // The "parallel direct solver" matrix types like MATAIJSPOOLES are ALL gone in petsc-3.0.0
    // Thus, I (Ilya Lashuk) "delete" all such checks for matrix type.  Hope it is reasonable thing to do.
    PetscBool isAij, isAijSeq, isAijPrl, isSuperLU, isSuperLU_Dist;
    PetscStrcmp(mtype,MATAIJ,&isAij);
    PetscStrcmp(mtype,MATSEQAIJ,&isAijSeq);
    PetscStrcmp(mtype,MATMPIAIJ,&isAijPrl);
    isSuperLU = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU,&isSuperLU);
    isSuperLU_Dist = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU_DIST,&isSuperLU_Dist);

    MatCreate(comm, &M);
    MatSetSizes(M, sz,sz, PETSC_DECIDE, PETSC_DECIDE);
    MatSetType(M,mtype);

    if(isAij || isAijSeq || isAijPrl || isSuperLU || isSuperLU_Dist) {
      if(npes > 1) {
        MatMPIAIJSetPreallocation(M, 53*dof , PETSC_NULL, 53*dof , PETSC_NULL);
      }else {
        MatSeqAIJSetPreallocation(M, 53*dof , PETSC_NULL);
      }
    }

    return 0;
} // create Matrix 

int subDA::computeLocalToGlobalMappings() {
  int rank, npes;
  MPI_Comm comm = m_da->getComm();
  rank = m_da->getRankAll();
  npes = m_da->getNpesAll();

  DendroIntL localNodeSize = getNodeSize();
  DendroIntL off1, globalOffset;
  MPI_Request sendRequest;
  MPI_Status status;
  
  par::Mpi_Scan<DendroIntL>(&localNodeSize, &off1, 1, MPI_SUM, comm); 
  if(rank < (npes-1)) {
    par::Mpi_Issend<DendroIntL>(&off1, 1, rank+1, 0, comm, &sendRequest);
  }

  if(rank) {
    par::Mpi_Recv<DendroIntL>(&globalOffset, 1, rank-1, 0, comm, &status);
  }else {
    globalOffset = 0;
  }
  
  std::vector<DendroIntL> gNumNonGhostNodes(localNodeSize); 
  for(DendroIntL i = 0; i < localNodeSize; i++) {
    gNumNonGhostNodes[i] = (i+globalOffset);   
  }

  vecGetBuffer<DendroIntL>(gNumNonGhostNodes, m_dilpLocalToGlobal, false, false, true, 1);

  if(rank < (npes-1)) {
    MPI_Status statusWait;
    MPI_Wait(&sendRequest, &statusWait);
  }

  ReadFromGhostsBegin<DendroIntL>(m_dilpLocalToGlobal,1);
  ReadFromGhostsEnd<DendroIntL>(m_dilpLocalToGlobal);

  gNumNonGhostNodes.clear();
  m_bComputedLocalToGlobal = true;

  // note, no restore ...

  return 0;
}//end function

int subDA::vecGetBuffer(Vec in, PetscScalar* &out, bool isElemental,
                        bool isGhosted, bool isReadOnly, unsigned int dof) {
  // Some error checks ... make sure the size of Vec in matches those implied
  // by the other params ...
  unsigned int sz = 0;

  int rank, npes;
  MPI_Comm comm = m_da->getComm();
  rank = m_da->getRankAll();
  npes = m_da->getNpesAll();
  
  if (isElemental) {
    sz = m_uiElementSize;
    if (isGhosted) {
      sz += m_uiPreGhostElementSize;
    }
  } else {
    sz = m_uiNodeSize + m_uiBoundaryNodeSize;
    if (isGhosted) {
      sz += (m_uiPreGhostNodeSize + m_uiPreGhostBoundaryNodeSize + m_uiPostGhostNodeSize);
    }
  }
  // now for dof ...
  sz *= dof;

  PetscInt vecSz=0;
  VecGetLocalSize(in, &vecSz);

  if ( sz != vecSz) {
    std::cerr  << "[subDA::DEBUG]" << rank << ": In function " << __func__ << " sizes are unequal, sz is  " 
      << sz << " and vecSz is " << vecSz << std::endl;
    std::cerr << "Params are: isElem " << isElemental << " isGhosted " << isGhosted << std::endl;
    assert(false);
    return -1;; 
  };

  // get the local Petsc Arrray,
  PetscScalar *array = NULL;
  // VecGetArray(in, &array);

  if (isReadOnly) {
    VecGetArrayRead(in, (const PetscScalar**) &array);
  } else {  
    VecGetArray(in, &array);
  }
  
  // allocate except for the case of ghosted-elemental vectors...
  if(isGhosted && isElemental) {
    //simply copy the pointer
    //This is the only case where the buffer will not be the size of the
    //fullLocalBufferSize. 
    out = array;
  }else {
    // First let us allocate for the buffer ... the local buffer will be of full
    // length.
    sz = dof*m_uiLocalBufferSize;

    if(sz) {
      out = new PetscScalar[sz];
      assert(out);
    }else {
      out = NULL;
    }

    //Zero Entries first if you plan to modify the buffer 
    if(!isReadOnly) {
      for(unsigned int i = 0; i < sz; i++) {
        out[i] = 0.0;
      }
    }
  }

  std::cout << "subDA:: vecGetBuffer  local size: " << m_uiLocalBufferSize << std::endl;

  unsigned int vecCnt=0;
  // Now we can populate the out buffer ... and that needs a loop through the
  // elements ...
  if (isGhosted) { // ghosted
    if (isElemental) {
      //Nothing to be done here.
    } else {
      // now copy ...
      for (unsigned int i=0; i<m_uiLocalBufferSize; i++) {
        unsigned int di = m_uip_sub2DA_NodeMap[i];
        // skip the ones that are not nodes ...
        if ( ! (m_da->getLevel(di) & ot::TreeNode::NODE ) ) {
          continue;
        }
        for (unsigned int j=0; j<dof; j++) {
          out[dof*i+j] = array[dof*vecCnt + j];
        }
        vecCnt++;
      }//end for i
    }//end if elemental
  } else { // not ghosted
    if (isElemental) {
      // is a simple copy ...
      for (unsigned int i = m_da->getIdxElementBegin(); i < m_da->getIdxElementEnd(); i++) {
        if (m_ucpSkipList[i]) continue;
        for (unsigned int j = 0; j < dof; j++) {
          out[dof*m_uip_DA2sub_ElemMap[i]+j] = array[dof*vecCnt + j];
        }
        vecCnt++;
      }//end for i
    } else {
      std::cout << "subDA:: vecGetBuffer  local size: " << m_uiLocalBufferSize << std::endl;
      for (unsigned int i = m_da->getIdxElementBegin(); i < m_da->getIdxElementEnd(); i++) {
        // unsigned int di = m_uip_sub2DA_NodeMap[i];
        if ( ! (m_da->getLevel(i) & ot::TreeNode::NODE ) || m_ucpSkipNodeList[i] ) {
          continue;
        }
        for (unsigned int j=0; j<dof; j++) {
          out[dof*m_uip_DA2sub_NodeMap[i]+j] = array[dof*vecCnt + j];
        }
        vecCnt++;
      }//end for i
      for (unsigned int i = m_da->getIdxElementEnd(); i < m_da->getIdxPostGhostBegin(); i++) {
        // add the remaining boundary nodes ...
        // unsigned int di = m_uip_sub2DA_NodeMap[i];
        if ( ! ( (m_da->getLevel(i) & ot::TreeNode::NODE ) &&
              (m_da->getLevel(i) & ot::TreeNode::BOUNDARY ) ) || m_ucpSkipNodeList[i] ) {
          continue;
        }
        for (unsigned int j=0; j<dof; j++) {
          out[dof*m_uip_DA2sub_NodeMap[i]+j] = array[dof*vecCnt + j];
        }
        vecCnt++;
      }//end for i
    }
  }

  if(!(isGhosted && isElemental)) {
    if (isReadOnly) {
      VecRestoreArrayRead(in, (const PetscScalar**) &array);
    } else {
      VecRestoreArray(in, &array);
    }
  }
    
  return 0;
} // vecGetBuffer

int subDA::vecRestoreBuffer(Vec in, PetscScalar* out, bool isElemental, bool isGhosted, bool isReadOnly, unsigned int dof) {
    // Some error checks ... make sure the size of Vec in matches those implied
    // by the other params ...
    unsigned int sz = 0;
    
    if (isElemental) {
      sz = m_uiElementSize;
      if (isGhosted) {
        sz += m_uiPreGhostElementSize;
      }
    } else {
      sz = m_uiNodeSize + m_uiBoundaryNodeSize;
      if (isGhosted) {
        sz += (m_uiPreGhostNodeSize + m_uiPreGhostBoundaryNodeSize + m_uiPostGhostNodeSize);
      }
    }
    // now for dof ...
    sz *= dof; 

    PetscInt vecSz=0;
    VecGetLocalSize(in, &vecSz);

    if ( sz != vecSz) {
      std::cerr  << RED<<"In function subDA::Petsc::" << __func__ <<
        NRM<<" sizes are unequal, sz is  " << sz << " and vecSz is " << vecSz << std::endl;
      std::cerr << "Params are: isElem " << isElemental << " isGhosted " << isGhosted << std::endl;
      assert(false);
      return -1;;
    }

    unsigned int vecCnt=0;

    if(isGhosted && isElemental) {
      //If it is ghosted and elemental, simply restore the array.
      //out was not allocated expicitly in this case. It was just a copy of the
      //array's pointer. The readOnly flag is immaterial for this case.
      if (isReadOnly) {
        VecRestoreArrayRead(in, (const PetscScalar**) &out);
      } else {
        VecRestoreArray(in, &out);
      }
      out = NULL;
    }  else if ( isReadOnly ) {
      // no need to write back ... simply clean up and return
      //Since this is not an elemental and ghosted vector, out was allocated
      //explicitly 
      if(out) {
        delete [] out;
        out = NULL;
      }
    } else {
      //ghosted and elemental is already taken care of. So only need to tackle
      //the other 3 cases.
      // need to write back ...
      // get the local Petsc Arrray,
      PetscScalar *array;
      VecGetArray(in, &array);

      if ( isElemental ) { 
        //non-ghosted, elemental
        for (unsigned int i = m_da->getIdxElementBegin(); i < m_da->getIdxElementEnd(); i++) {
          if (m_ucpSkipList[i]) continue;
          for (unsigned int j = 0; j < dof; j++) {
            array[dof*vecCnt + j] = out[dof*m_uip_DA2sub_ElemMap[i]+j];
          }
          vecCnt++;
        }//end for i
      } else if ( isGhosted ) {
        // nodal and ghosted ...
        for (unsigned int i=0; i<m_uiLocalBufferSize; i++) {
          unsigned int di = m_uip_sub2DA_NodeMap[i];
          // skip the ones that are not nodes ...
          if ( ! (m_da->getLevel(di) & ot::TreeNode::NODE ) ) {
            continue;
          }
          for (unsigned int j=0; j<dof; j++) {
            array[dof*vecCnt + j] = out[dof*i+j];
          }
          vecCnt++;
        }//end for i
      } else {
        // nodal non ghosted ...
        for (unsigned int i = m_da->getIdxElementBegin(); i < m_da->getIdxElementEnd(); i++) {
          // unsigned int di = m_uip_sub2DA_NodeMap[i];
          if ( ! (m_da->getLevel(i) & ot::TreeNode::NODE ) || m_ucpSkipNodeList[i] ) {
            continue;
          }
          for (unsigned int j=0; j<dof; j++) {
            array[dof*vecCnt + j] = out[dof*m_uip_DA2sub_NodeMap[i]+j];
          }
          vecCnt++;
        }//end for i
        for (unsigned int i = m_da->getIdxElementEnd(); i < m_da->getIdxPostGhostBegin(); i++) {
          // add the remaining boundary nodes ...
          // unsigned int di = m_uip_sub2DA_NodeMap[i];
          if ( ! ( (m_da->getLevel(i) & ot::TreeNode::NODE ) &&
              (m_da->getLevel(i) & ot::TreeNode::BOUNDARY ) ) || m_ucpSkipNodeList[i] ) {
            continue;
          } 
          for (unsigned int j=0; j<dof; j++) {
            array[dof*vecCnt + j] = out[dof*m_uip_DA2sub_NodeMap[i]+j];
          }
          vecCnt++;
        }//end for i
      } // else-nodal-non-ghosted

      if (isReadOnly) {
        VecRestoreArrayRead(in, (const PetscScalar**) &array);
      } else {
        VecRestoreArray(in, &array);
      }
      //Since this is not an elemental and ghosted vector, out was allocated
      //explicitly 
      if(out) {
        delete [] out;
        out = NULL;
      }
    }
    return 0;
  } // vecRestoreBuffer

    int subDA::setValuesInMatrix(Mat mat, std::vector<ot::MatRecord> &records, unsigned int dof, InsertMode mode)
    {
      PROF_SET_MAT_VALUES_BEGIN

      assert(m_bComputedLocalToGlobal);
      std::vector<PetscScalar> values;
      std::vector<PetscInt> colIndices;

      //Can make it more efficient later.
      if (!records.empty())
      {
        //Sort Order: row first, col next, val last
        std::sort(records.begin(), records.end());

        unsigned int currRecord = 0;

        while (currRecord < (records.size() - 1))
        {
          values.push_back(records[currRecord].val);
          colIndices.push_back(static_cast<PetscInt>(
              (dof * m_dilpLocalToGlobal[records[currRecord].colIdx]) +
              records[currRecord].colDim));
          if ((records[currRecord].rowIdx != records[currRecord + 1].rowIdx) ||
              (records[currRecord].rowDim != records[currRecord + 1].rowDim))
          {
            PetscInt rowId = static_cast<PetscInt>(
                (dof * m_dilpLocalToGlobal[records[currRecord].rowIdx]) +
                records[currRecord].rowDim);
            MatSetValues(mat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())),
                         (&(*values.begin())), mode);
            
            colIndices.clear();
            values.clear();
          }
          currRecord++;
        } //end while

        PetscInt rowId = static_cast<PetscInt>(
            (dof * m_dilpLocalToGlobal[records[currRecord].rowIdx]) +
            records[currRecord].rowDim);
        if (values.empty())
        {
          //Last row is different from the previous row
          PetscInt colId = static_cast<PetscInt>(
              (dof * m_dilpLocalToGlobal[records[currRecord].colIdx]) +
              records[currRecord].colDim);
          PetscScalar value = records[currRecord].val;
          MatSetValues(mat, 1, &rowId, 1, &colId, &value, mode);
        }
        else
        {
          //Last row is same as the previous row
          values.push_back(records[currRecord].val);
          colIndices.push_back(static_cast<PetscInt>(
              (dof * m_dilpLocalToGlobal[records[currRecord].colIdx]) +
              records[currRecord].colDim));
          MatSetValues(mat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())),
                       (&(*values.begin())), mode);
          colIndices.clear();
          values.clear();
        }
        records.clear();
      } // records not empty

      PROF_SET_MAT_VALUES_END
  }//end function

  int subDA::zeroRowsInMatrix(Mat mat, std::vector<unsigned int>& indices, unsigned int dof,
                             double diag, Vec x, Vec b) {
        
        assert(m_bComputedLocalToGlobal);
        int errCode;
        std::vector<PetscInt> rows;

        if ( !indices.empty() ) {
            PetscInt numRows = dof*indices.size();

            for (auto &i: indices) {
                PetscInt glo_idx = static_cast<PetscInt>( dof*m_dilpLocalToGlobal[i] );
                for (auto d=0; d<dof; ++d)
                    rows.push_back(glo_idx++);
            }

            errCode = MatZeroRows(mat, numRows, rows.data(), diag, x, b);

        } // if not empty

        return errCode;
    }  // end function

  int subDA::getNodeIndices(unsigned int* nodes) {
    int rval = m_da->getNodeIndices(nodes);

    unsigned int idx;
    for (unsigned int i=0; i<8; ++i) {
      idx = nodes[i];
      nodes[i] = m_uip_DA2sub_NodeMap[idx];
    }

    return rval;
  }

}; // namespace ot