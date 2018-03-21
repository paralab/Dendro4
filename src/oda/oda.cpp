
/**
  @file oda.C
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  */

#include "oda.h"
#include "parUtils.h"
#include "colors.h"
#include "testUtils.h"
#include "dendro.h"
#include <iomanip>

#ifdef __DEBUG__
#ifndef __DEBUG_DA__
#define __DEBUG_DA__
#endif
#endif

#ifdef __DEBUG_DA__
#ifndef __DEBUG_DA_PUBLIC__
#define __DEBUG_DA_PUBLIC__
#endif
#endif

#ifdef __DEBUG_DA_PUBLIC__
#ifndef __MEASURE_DA__
#define __MEASURE_DA__
#endif
#endif

namespace ot {

  int DA::computeLocalToGlobalElemMappings() {
    DendroIntL localElemSize = getElementSize();
    DendroIntL off1, globalOffset;
    MPI_Request sendRequest;
    MPI_Status status;
    if(m_bIamActive) {
      par::Mpi_Scan<DendroIntL>(&localElemSize, &off1, 1, MPI_SUM, m_mpiCommActive); 
      if(m_iRankActive < (m_iNpesActive - 1)) {
        par::Mpi_Issend<DendroIntL>(&off1, 1, m_iRankActive+1, 0, m_mpiCommActive, &sendRequest);
      }

      if(m_iRankActive) {
        par::Mpi_Recv<DendroIntL>(&globalOffset, 1, m_iRankActive-1, 0, m_mpiCommActive, &status );
      }else {
        globalOffset = 0;
      }
    }

    //Equivalent to createVector: elemental, non-ghosted, 1 dof
    std::vector<DendroIntL> gNumNonGhostElems(localElemSize); 

    for(DendroIntL i = 0; i < localElemSize; i++) {
      gNumNonGhostElems[i] = (i+globalOffset);   
    }

    vecGetBuffer<DendroIntL>(gNumNonGhostElems,
        m_dilpLocalToGlobalElems, true, false, true, 1);

    if( m_bIamActive && (m_iRankActive < (m_iNpesActive-1)) ) {
      MPI_Status statusWait;
      MPI_Wait(&sendRequest, &statusWait);
    }

    ReadFromGhostElemsBegin<DendroIntL>(m_dilpLocalToGlobalElems,1);
    ReadFromGhostElemsEnd<DendroIntL>(m_dilpLocalToGlobalElems);

    gNumNonGhostElems.clear();
    m_bComputedLocalToGlobalElems = true;

    return 0;
  }//end function

  int DA::computeLocalToGlobalMappings() {
    DendroIntL localNodeSize = getNodeSize();
    DendroIntL off1, globalOffset;
    MPI_Request sendRequest;
    MPI_Status status;
    if(m_bIamActive) {
      par::Mpi_Scan<DendroIntL>(&localNodeSize, &off1, 1, MPI_SUM, m_mpiCommActive); 
      if(m_iRankActive < (m_iNpesActive-1)) {
        par::Mpi_Issend<DendroIntL>(&off1, 1, m_iRankActive+1, 0, m_mpiCommActive, &sendRequest);
      }

      if(m_iRankActive) {
        par::Mpi_Recv<DendroIntL>(&globalOffset, 1, m_iRankActive-1, 0, m_mpiCommActive, &status);
      }else {
        globalOffset = 0;
      }
    }

    std::vector<DendroIntL> gNumNonGhostNodes(localNodeSize); 
    for(DendroIntL i = 0; i < localNodeSize; i++) {
      gNumNonGhostNodes[i] = (i+globalOffset);   
    }

    vecGetBuffer<DendroIntL>(gNumNonGhostNodes, m_dilpLocalToGlobal,
        false, false, true, 1);

    if(m_bIamActive && (m_iRankActive < (m_iNpesActive-1))) {
      MPI_Status statusWait;
      MPI_Wait(&sendRequest, &statusWait);
    }

    ReadFromGhostsBegin<DendroIntL>(m_dilpLocalToGlobal,1);
    ReadFromGhostsEnd<DendroIntL>(m_dilpLocalToGlobal);

    gNumNonGhostNodes.clear();
    m_bComputedLocalToGlobal = true;

    return 0;
  }//end function

    int DA::zeroRowsInMatrix(Mat mat, std::vector<unsigned int>& indices, unsigned int dof,
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

    int DA::setValuesInMatrix(Mat mat, std::vector<ot::MatRecord> &records, unsigned int dof, InsertMode mode)
    {

      PROF_SET_MAT_VALUES_BEGIN

      assert(m_bComputedLocalToGlobal);
      std::vector<PetscScalar> values;
      std::vector<PetscInt> colIndices;

      /* >>= @hari - Oct 12 2017 - debug for Taly integration + SNES
      char talyfile[256];
      sprintf(talyfile, "records.%d.%d.txt", m_iRankAll, m_iNpesAll);

      std::ofstream out(talyfile, std::ofstream::app);

      // =<< @hari - Oct 12 2017 - debug for Taly integration + SNES */

      std::cout << "NodeSize: " << m_uiNodeSize << std::endl;
      std::cout << "BdyNodes: " << m_uiBoundaryNodeSize << std::endl;
      std::cout << "other: " << m_uiPreGhostBoundaryNodeSize << ", " << m_uiPreGhostNodeSize << ", " << std::endl;
      std::cout << "elems: " << m_uiElementBegin << ", " << m_uiIndependentElementBegin << std::endl;

      //Can make it more efficient later.
      if (!records.empty())
      {
        //Sort Order: row first, col next, val last
        std::sort(records.begin(), records.end());

        unsigned int currRecord = 0;

        while (currRecord < (records.size() - 1))
        {
          // >>= Hari, subDA debug
          // std::cout << "setValuesInMatrix: " << " indices ... " << m_uiNodeSize << " === ("  << records[currRecord].rowIdx << ", " << records[currRecord].colIdx << ")" << std::endl;
               
          // =<< subDA debug  

          values.push_back(records[currRecord].val);
//          std::cout << "\t >>= pushed values" << std::endl;
//          std::cout << "\t >>= local2global: " << m_dilpLocalToGlobal[10] << std::endl;
          colIndices.push_back(static_cast<PetscInt>(
              (dof * m_dilpLocalToGlobal[records[currRecord].colIdx]) +
              records[currRecord].colDim));
//          std::cout << "\t  >>= pushed colIndices" << std::endl;
          if ((records[currRecord].rowIdx != records[currRecord + 1].rowIdx) ||
              (records[currRecord].rowDim != records[currRecord + 1].rowDim))
          {
            PetscInt rowId = static_cast<PetscInt>(
                (dof * m_dilpLocalToGlobal[records[currRecord].rowIdx]) +
                records[currRecord].rowDim);
            MatSetValues(mat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())),
                         (&(*values.begin())), mode);
            // >>= @hari - Oct 12 2017 - debug for Taly integration + SNES
            // for (int q = 0; q < colIndices.size(); ++q) {
            //   out << std::setfill('0') << std::setw(5) << rowId << " " << std::setfill('0') << std::setw(5) << colIndices[q] << " " << values[q] << std::endl;
            // }
            // =<< @hari - Oct 12 2017 - debug for Taly integration + SNES

            colIndices.clear();
            values.clear();
          }
          currRecord++;
        } //end while

        std::cout << "=== === === === ===" << std::endl;

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
                      // >>= @hari - Oct 12 2017 - debug for Taly integration + SNES
                      //  out << std::setfill('0') << std::setw(5) << rowId << " " << std::setfill('0') << std::setw(5) << colId << " " << value << std::endl;
                      // =<< @hari - Oct 12 2017 - debug for Taly integration + SNES
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
                      // >>= @hari - Oct 12 2017 - debug for Taly integration + SNES
                      // for (int q = 0; q < colIndices.size(); ++q) {
                      //   out << std::setfill('0') << std::setw(5) << rowId << " " << std::setfill('0') << std::setw(5) << colIndices[q] << " " << values[q] << std::endl;
                      // }
                      // =<< @hari - Oct 12 2017 - debug for Taly integration + SNES
          colIndices.clear();
          values.clear();
        }
        records.clear();
      } // records not empty

      // >>= @hari - Oct 12 2017 - debug for Taly integration + SNES
      // out.close();
      // =<< @hari - Oct 12 2017 - debug for Taly integration + SNES

      PROF_SET_MAT_VALUES_END
  }//end function

  //***************Constructor*****************//
  DA::DA(std::vector<ot::TreeNode> &in, MPI_Comm comm, MPI_Comm activeInputComm,double tol,
      bool compressLut, const std::vector<ot::TreeNode>* blocksPtr,bool* iAmActive ) {

#ifdef __PROF_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_BUILD_DA_BEGIN

      //@milinda

      int rank;
      MPI_Comm_rank(comm,&rank);

      m_bSkipOctants = false;

      m_uiTreeSortTol=tol;
      //if(!rank) std::cout <<"oda tolerance "<<tol<<std::endl;
      DA_FactoryPart0(in, comm, activeInputComm, compressLut, iAmActive);
     // if(!rank)
     //     std::cout<<"ODA Stage 0 completed"<<std::endl;

    if(m_bIamActive) {
      DA_FactoryPart1(in);
      //  if(!rank)
      //      std::cout<<"ODA Stage 1 completed"<<std::endl;
      DA_FactoryPart2(in);
      //  if(!rank)
      //      std::cout<<"ODA Stage 2 completed"<<std::endl;

      DA_FactoryPart3(in, comm, compressLut, blocksPtr, iAmActive);

       // if(!rank)
       //    std::cout<<"ODA Stage 3 completed"<<std::endl;

    }

    PROF_BUILD_DA_END

  }//end constructor

  DA::DA(unsigned int dummy, std::vector<ot::TreeNode> &in, MPI_Comm comm,
      MPI_Comm activeInputComm, bool compressLut, 
      const std::vector<ot::TreeNode> * blocksPtr, bool* iAmActive ) {

#ifdef __PROF_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_BUILD_DA_BEGIN 

      DA_FactoryPart0(in, comm, activeInputComm, compressLut, iAmActive);

    if(m_bIamActive) {
      DA_FactoryPart3(in, comm, compressLut, blocksPtr, iAmActive);
    }

    PROF_BUILD_DA_END
  }//end constructor

  DA::~DA() {
    if (m_ucpOctLevels != NULL) {
      delete [] m_ucpOctLevels;
      m_ucpOctLevels = NULL;
    }

    if(m_dilpLocalToGlobal != NULL) {
      delete [] m_dilpLocalToGlobal;
      m_dilpLocalToGlobal = NULL;
    }

    if(m_dilpLocalToGlobalElems != NULL) {
      delete [] m_dilpLocalToGlobalElems;
      m_dilpLocalToGlobalElems = NULL;
    }

#ifdef HILBERT_ORDERING

    delete[] m_uiParRotID;
      m_uiParRotID=NULL;
    delete[] m_uiParRotIDLev;
      m_uiParRotIDLev=NULL;


#endif
    m_ucpLutRemainders.clear();
    m_uspLutQuotients.clear();
    m_ucpLutMasks.clear();
    m_ucpSortOrders.clear();
    m_uiNlist.clear();
  }

  /************** Domain Access ****************/

Point DA::getNextOffsetByRotation(Point p, unsigned char d)
{

#ifdef HILBERT_ORDERING
    Point m ;
    m=p;
    Point parent;
    unsigned int next_limit = (1u << m_uiDimension) - 1;
    unsigned int rotation_id;
    char next_index;
    unsigned int child_index;
    unsigned int par_x, par_y, par_z, par_level;
    unsigned int len;
    unsigned int num_children = 1u << m_uiDimension; // This is basically the hilbert table offset
    unsigned int rot_offset = num_children << 1;
    unsigned int parRotID;
    unsigned int nextLev = getLevel(m_uiCurrent + 1);
    unsigned int childRotID;
    unsigned int mid_bit;

//    if (!m_uiRotIDComputed) {

      for (int k = d; k >= 0; --k) {
        // special case that next of the root node.We consider it as the first child of the root.
        if (k == 0) {
          assert(m.xint() == 0 && m.yint() == 0 && m.zint() == 0);
          int len = (1u << (m_uiMaxDepth)) - 1;
          m = Point(0, 0, len);//TreeNode(1,0,0,len,1,m_uiDim,m_uiMaxDepth+1);
          break;
        }
        par_level = k - 1;
        GET_HILBERT_CHILDNUMBER(m, k, child_index);
        GET_PARENT(m, par_level, parent);
        rotation_id = parRotID; // calculated from get ChildNumber
        par_x = parent.xint();
        par_y = parent.yint();
        par_z = parent.zint();
        len = m_uiMaxDepth - par_level - 1;//(m_uiMaxDepth-parent.getLevel()-1);
        if (child_index < next_limit) {
          assert(child_index<next_limit);
          m_uiParRotID[m_uiCurrent] = ((child_index << 5) | (parRotID & ROT_ID_MASK));
          m_uiParRotIDLev[m_uiCurrent]=par_level;

          // next octant is in the same level;
          next_index = rotations[rot_offset * rotation_id + child_index + 1] - '0';
          // Note: Just calculation of x,y,x of a child octant for a given octant based on the child index. This is done to eliminate the branching.
          par_x = par_x + (((int)((bool)(next_index& 1u)))<<len);
          par_y = par_y + (((int)((bool)(next_index& 2u)))<<len);
          par_z = par_z + (((int)((bool)(next_index& 4u)))<<len);

          m = Point(par_x, par_y, par_z);
          childRotID = parRotID;
          for (int q = k; q < nextLev; q++) {
            childRotID = HILBERT_TABLE[childRotID * num_children + next_index];
            GET_FIRST_CHILD(m, childRotID, q, m);
            mid_bit = m_uiMaxDepth - q - 1;
            par_z = m.zint();
            par_x = m.xint();
            par_y = m.yint();
            next_index = ((((par_z & (1u << mid_bit)) >> mid_bit) << 2u) |(((par_y & (1u << mid_bit)) >> mid_bit) << 1u) | ((par_x & (1u << mid_bit)) >> mid_bit));
          }
          break;
        } else {
          m = parent;
        }
      }
      return m;
//    }
#endif


}


inline Point DA::getNextOffset(Point p, unsigned char d) {

#ifdef __DEBUG_DA_PUBLIC__
    assert(m_bIamActive);
#endif

#ifdef HILBERT_ORDERING

      unsigned char parRotID;
      unsigned char child_index;
      unsigned char par_level;
      unsigned char len;
      Point m;
      unsigned char nextLev=(m_ucpOctLevels[m_uiCurrent+1]& ot::TreeNode::MAX_LEVEL);
      unsigned char mid_bit;

      parRotID=(m_uiParRotID[m_uiCurrent] & ROT_ID_MASK);
      child_index=(m_uiParRotID[m_uiCurrent] >>5);
      par_level=m_uiParRotIDLev[m_uiCurrent];
      GET_PARENT(p, par_level, m);
      child_index = (rotations[16 * parRotID + child_index + 1] - '0');
      len = m_uiMaxDepth - par_level - 1;

      m=Point((m.xint() + (((int)((bool)(child_index& 1u)))<<len)),(m.yint() + (((int)((bool)(child_index& 2u)))<<len)),(m.zint() + (((int)((bool)(child_index& 4u)))<<len)));
      //childRotID = parRotID;
      for (int q = par_level+1; q < nextLev; q++) {
        parRotID = HILBERT_TABLE[parRotID * 8 + child_index];
        GET_FIRST_CHILD(m, parRotID, q, m);
        mid_bit = m_uiMaxDepth - q - 1;
        child_index = ((((m.zint() & (1u << mid_bit)) >> mid_bit) << 2u) |(((m.yint() & (1u << mid_bit)) >> mid_bit) << 1u) | ((m.xint() & (1u << mid_bit)) >> mid_bit));
      }
  return m;

#else

    unsigned int len = (unsigned int)(1u<<( m_uiMaxDepth - d ) );
    unsigned int len_par = (unsigned int)(1u<<( m_uiMaxDepth - d +1 ) );

    unsigned int i,j,k;

    i = p.xint(); i %= len_par;
    j = p.yint(); j %= len_par;
    k = p.zint(); k %= len_par;
    i /= len;
    j /= len;
    k /= len;

    unsigned int childNum = 4*k + 2*j + i;

    Point p2;
    switch (childNum) {
      case 7:
        p2.x() = p.x() -len; p2.y() = p.y() - len; p2.z() = p.z() -len;
        return getNextOffset(p2, d-1);
      case 0:
        p2.x() = p.x() +len; p2.y() = p.y(); p2.z() = p.z();
        break;
      case 1:
        p2.x() = p.x() -len; p2.y() = p.y() +len; p2.z() = p.z();
        break;
      case 2:
        p2.x() = p.x() +len; p2.y() = p.y(); p2.z() = p.z();
        break;
      case 3:
        p2.x() = p.x() -len; p2.y() = p.y() - len; p2.z() = p.z() +len;
        break;
      case 4:
        p2.x() = p.x() +len; p2.y() = p.y(); p2.z() = p.z();
        break;
      case 5:
        p2.x() = p.x() -len; p2.y() = p.y()+len; p2.z() = p.z();
        break;
      case 6:
        p2.x() = p.x() +len; p2.y() = p.y(); p2.z() = p.z();
        break;
      default:
        std::cerr << "Wrong child number in " << __func__ << std::endl;
        assert(false);
        break;
    } // switch (childNum)

    return p2;
#endif
  }

  void DA::incrementCurrentOffset() {
#ifdef __DEBUG_DA_PUBLIC__
    assert(m_bIamActive);
#endif

    // if it is the first element, simply return the stored offset ...
    if ( m_uiCurrent == (m_uiElementBegin-1)) {
      m_ptCurrentOffset = m_ptOffset;
      return;
    }

#ifdef __DEBUG_DA_PUBLIC__
    if ( m_ucpOctLevels[m_uiCurrent] & ot::TreeNode::BOUNDARY ) {
      std::cerr << RED "ERROR, Boundary eleme in incre Curr offset" NRM << std::endl;
      assert(false);
    }
#endif

#ifdef HILBERT_ORDERING

    unsigned int d = (m_ucpOctLevels[m_uiCurrent] & ot::TreeNode::MAX_LEVEL );
    Point p =m_ptCurrentOffset;
    //Point p2;
    if(!m_uiRotIDComputed) {
      p = getNextOffsetByRotation(p, d);
    }else
    {
      p = getNextOffset(p,d);
    }
    m_ptCurrentOffset=p;

#else
    unsigned char d = (m_ucpOctLevels[m_uiCurrent] & ot::TreeNode::MAX_LEVEL );
    unsigned int len = (unsigned int)(1u<<( m_uiMaxDepth - d ) );
    unsigned int len_par = (unsigned int)(1u<<( m_uiMaxDepth - d +1 ) );

    unsigned int i,j,k;

    i = m_ptCurrentOffset.xint(); 
    j = m_ptCurrentOffset.yint(); 
    k = m_ptCurrentOffset.zint(); 

    i %= len_par;
    j %= len_par;
    k %= len_par;

    i /= len;
    j /= len;
    k /= len;

    unsigned int childNum = 4*k + 2*j + i;

    Point p = m_ptCurrentOffset;
    Point p2;
    switch (childNum) {
      case 7:
        p2.x() = p.x() -len; p2.y() = p.y() - len; p2.z() = p.z() -len;
        p2 = getNextOffset(p2, d-1);
        break;
      case 0:
        p2.x() = p.x() +len; p2.y() = p.y(); p2.z() = p.z();
        break;
      case 1:
        p2.x() = p.x() -len; p2.y() = p.y() +len; p2.z() = p.z();
        break;
      case 2:
        p2.x() = p.x() +len; p2.y() = p.y(); p2.z() = p.z();
        break;
      case 3:
        p2.x() = p.x() -len; p2.y() = p.y() - len; p2.z() = p.z() +len;
        break;
      case 4:
        p2.x() = p.x() +len; p2.y() = p.y(); p2.z() = p.z();
        break;
      case 5:
        p2.x() = p.x() -len; p2.y() = p.y()+len; p2.z() = p.z();
        break;
      case 6:
        p2.x() = p.x() +len; p2.y() = p.y(); p2.z() = p.z();
        break;
      default:
        std::cerr << "Wrong child number in " << __func__ << std::endl;
        assert(false);
        break;
    } // switch (childNum)

    m_ptCurrentOffset = p2;


#endif

  }

  //This is for real octants only, pseudo-boundary octants can not be tested
  //using this. 
  bool DA::isBoundaryOctant(unsigned char *flags) {
#ifdef __DEBUG_DA_PUBLIC__
    assert(m_bIamActive);
#endif

    unsigned char _flags = 0;
    Point pt = getCurrentOffset();
    unsigned int x = pt.xint();
    unsigned int y = pt.yint();
    unsigned int z = pt.zint();
    unsigned int d = getLevel(curr())-1;
    unsigned int maxD = getMaxDepth()-1;
    unsigned int len  = (unsigned int)(1u<<(maxD - d) );
    unsigned int blen = (unsigned int)(1u << maxD);

    if (!x) _flags |= ot::TreeNode::X_NEG_BDY;  
    if (!y) _flags |=  ot::TreeNode::Y_NEG_BDY; 
    if (!z) _flags |=   ot::TreeNode::Z_NEG_BDY;

    if ( (x+len) == blen )  _flags |= ot::TreeNode::X_POS_BDY;
    if ( (y+len) == blen )  _flags |= ot::TreeNode::Y_POS_BDY;
    if ( (z+len) == blen )  _flags |= ot::TreeNode::Z_POS_BDY;

    if(flags) {
      *flags = _flags;
    }
    return _flags;
  }//end function

  /***************** Array Access ********************/
  int DA::createMatrix(Mat &M, MatType mtype, unsigned int dof) {
    // first determine the size ...
    unsigned int sz = 0;
    if(m_bIamActive) {
      sz = dof*(m_uiNodeSize + m_uiBoundaryNodeSize);
    }//end if active

    // now create the PETSc Mat
    // The "parallel direct solver" matrix types like MATAIJSPOOLES are ALL gone in petsc-3.0.0
    // Thus, I (Ilya Lashuk) "delete" all such checks for matrix type.  Hope it is reasonable thing to do.
    PetscBool isAij, isAijSeq, isAijPrl, isSuperLU, isSuperLU_Dist;
    PetscStrcmp(mtype,MATAIJ,&isAij);
    PetscStrcmp(mtype,MATSEQAIJ,&isAijSeq);
    PetscStrcmp(mtype,MATMPIAIJ,&isAijPrl);
    isSuperLU = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU,&isSuperLU);
    isSuperLU_Dist = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU_DIST,&isSuperLU_Dist);

    MatCreate(m_mpiCommAll, &M);
    MatSetSizes(M, sz,sz, PETSC_DECIDE, PETSC_DECIDE);
    MatSetType(M,mtype);

    if(isAij || isAijSeq || isAijPrl || isSuperLU || isSuperLU_Dist) {
      if(m_iNpesAll > 1) {
        MatMPIAIJSetPreallocation(M, 53*dof , PETSC_NULL, 53*dof , PETSC_NULL);
      }else {
        MatSeqAIJSetPreallocation(M, 53*dof , PETSC_NULL);
      }
    }

    return 0;
  }//end function

  int DA::createActiveMatrix(Mat &M, MatType mtype, unsigned int dof) {
    // first determine the size ...
    unsigned int sz = 0;
    if(m_bIamActive) {
      sz = dof*(m_uiNodeSize + m_uiBoundaryNodeSize);

      // now create the PETSc Mat
      PetscBool isAij, isAijSeq, isAijPrl, isSuperLU, isSuperLU_Dist;
      PetscStrcmp(mtype,MATAIJ,&isAij);
      PetscStrcmp(mtype,MATSEQAIJ,&isAijSeq);
      PetscStrcmp(mtype,MATMPIAIJ,&isAijPrl);
      isSuperLU = PETSC_FALSE; //PetscStrcmp(mtype,MATSUPERLU,&isSuperLU);
      isSuperLU_Dist = PETSC_FALSE; //PetscStrcmp(mtype,MATSUPERLU_DIST,&isSuperLU_Dist);

      MatCreate(m_mpiCommActive, &M);
      MatSetSizes(M, sz,sz, PETSC_DECIDE, PETSC_DECIDE);
      MatSetType(M,mtype);

      if(isAij || isAijSeq || isAijPrl || isSuperLU || isSuperLU_Dist) {
        if(m_iNpesActive > 1) {
          MatMPIAIJSetPreallocation(M, 53*dof , PETSC_NULL, 53*dof , PETSC_NULL);
        }else {
          MatSeqAIJSetPreallocation(M, 53*dof , PETSC_NULL);
        }
      }
    }//end if active

    return 0;
  }//end function

  int DA::createVector(Vec &arr, bool isElemental, bool isGhosted, unsigned int dof) {
    // first determine the length of the vector ...
    unsigned int sz = 0;
    if(m_bIamActive) {
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
    }//end if active

    // now create the PETSc Vector
    VecCreate(m_mpiCommAll, &arr);
    VecSetSizes(arr, sz, PETSC_DECIDE);
    if (m_iNpesAll > 1) {
      VecSetType(arr,VECMPI);
    } else {
      VecSetType(arr,VECSEQ);
    }    
    return 0;
  }

  int DA::createActiveVector(Vec &arr, bool isElemental, bool isGhosted, unsigned int dof) {
    // first determine the length of the vector ...
    unsigned int sz = 0;
    if(m_bIamActive) {
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

      // now create the PETSc Vector
      VecCreate(m_mpiCommActive, &arr);
      VecSetSizes(arr, sz, PETSC_DECIDE);
      if (m_iNpesActive > 1) {
        VecSetType(arr, VECMPI);
      } else {
        VecSetType(arr, VECSEQ);
      }    
    }//end if active

    return 0;
  }

  // Obtains a ot::index aligned buffer of the Vector
  int DA::vecGetBuffer(Vec in, PetscScalar* &out, bool isElemental, bool isGhosted,
      bool isReadOnly, unsigned int dof) {
    // Some error checks ... make sure the size of Vec in matches those implied
    // by the other params ...
    unsigned int sz = 0;
    if(m_bIamActive) {
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
    }//end if active

    PetscInt vecSz=0;
    VecGetLocalSize(in, &vecSz);

    if ( sz != vecSz) {
      std::cerr  << m_iRankAll << ": In function " << __func__ << " sizes are unequal, sz is  " 
        << sz << " and vecSz is " << vecSz << std::endl;
      std::cerr << "Params are: isElem " << isElemental << " isGhosted " << isGhosted << std::endl;
      assert(false);
      return -1;; 
    };

    if(!m_bIamActive) {
      assert(m_uiLocalBufferSize == 0);
      assert(m_uiElementBegin == 0);
      assert(m_uiElementEnd == 0);
      assert(m_uiPostGhostBegin == 0);
    }

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

    unsigned int vecCnt=0;
    // Now we can populate the out buffer ... and that needs a loop through the
    // elements ...
    if (isGhosted) {
      if (isElemental) {
        //Nothing to be done here.
      } else {
        // now copy ...
        for (unsigned int i=0; i<m_uiLocalBufferSize; i++) {
          // skip the ones that are not nodes ...
          if ( ! (m_ucpOctLevels[i] & ot::TreeNode::NODE ) ) {
            continue;
          }
          for (unsigned int j=0; j<dof; j++) {
            out[dof*i+j] = array[dof*vecCnt + j];
          }
          vecCnt++;
        }//end for i
      }//end if elemental
    } else {
      if (isElemental) {
        // is a simple copy ...
        for (unsigned int i = m_uiElementBegin; i < m_uiElementEnd; i++) {
          for (unsigned int j = 0; j < dof; j++) {
            out[dof*i+j] = array[dof*vecCnt + j];
          }
          vecCnt++;
        }//end for i
      } else {
        for (unsigned int i = m_uiElementBegin; i < m_uiElementEnd; i++) {
          if ( ! (m_ucpOctLevels[i] & ot::TreeNode::NODE ) ) {
            continue;
          }
          for (unsigned int j=0; j<dof; j++) {
            out[dof*i+j] = array[dof*vecCnt + j];
          }
          vecCnt++;
        }//end for i
        for (unsigned int i = m_uiElementEnd; i < m_uiPostGhostBegin; i++) {
          // add the remaining boundary nodes ...
          if ( ! ( (m_ucpOctLevels[i] & ot::TreeNode::NODE ) &&
                (m_ucpOctLevels[i] & ot::TreeNode::BOUNDARY ) ) ) {
            continue;
          }
          for (unsigned int j=0; j<dof; j++) {
            out[dof*i+j] = array[dof*vecCnt + j];
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
  }

  int DA::vecRestoreBuffer(Vec in, PetscScalar* out, bool isElemental, bool isGhosted, bool isReadOnly, unsigned int dof) {
    // Some error checks ... make sure the size of Vec in matches those implied
    // by the other params ...
    unsigned int sz = 0;
    if(m_bIamActive) {
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
    }//end if active

    PetscInt vecSz=0;
    VecGetLocalSize(in, &vecSz);

    if ( sz != vecSz) {
      std::cerr  << RED<<"In function PETSc::" << __func__ <<
        NRM<<" sizes are unequal, sz is  " << sz << " and vecSz is " << vecSz << std::endl;
      std::cerr << "Params are: isElem " << isElemental << " isGhosted " << isGhosted << std::endl;
      assert(false);
      return -1;;
    }

    if(!m_bIamActive) {
      assert(m_uiLocalBufferSize == 0);
      assert(m_uiElementBegin == 0);
      assert(m_uiElementEnd == 0);
      assert(m_uiPostGhostBegin == 0);
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
        for (unsigned int i = m_uiElementBegin; i < m_uiElementEnd; i++) {
          for (unsigned int j=0; j<dof; j++) {
            array[dof*vecCnt + j] = out[dof*i+j];
          }
          vecCnt++;
        }
      } else if ( isGhosted ) {
        // nodal and ghosted ...
        for (unsigned int i=0; i<sz; i++) {
          // skip the ones that are not nodes ...
          if ( ! (m_ucpOctLevels[i] & ot::TreeNode::NODE ) ) {
            continue;
          }
          for (unsigned int j=0; j<dof; j++) {
            array[dof*vecCnt + j] = out[dof*i+j];
          }
          vecCnt++;
        }
      } else {
        // nodal non ghosted ...
        for (unsigned int i = m_uiElementBegin; i < m_uiElementEnd; i++) {
          if ( ! (m_ucpOctLevels[i] & ot::TreeNode::NODE ) ) {
            continue;
          }
          for (unsigned int j=0; j<dof; j++) {
            array[dof*vecCnt + j] = out[dof*i+j];
          }
          vecCnt++;
        }
        for (unsigned int i = m_uiElementEnd; i < m_uiPostGhostBegin; i++) {
          // add the remaining boundary nodes ...
          if ( ! ( (m_ucpOctLevels[i] & ot::TreeNode::NODE ) &&
                (m_ucpOctLevels[i] & ot::TreeNode::BOUNDARY ) ) ) {
            continue;
          }
          for (unsigned int j=0; j<dof; j++) {
            array[dof*vecCnt + j] = out[dof*i+j];
          }
          vecCnt++;
        }
      }

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
  }

  void DA::updateQuotientCounter() {
#ifdef __DEBUG_DA_PUBLIC__
    assert(m_bIamActive);
#endif

    // m_ucpLutRemainders, m_uspLutQuotients.
    unsigned char _mask = m_ucpLutMasksPtr[2*m_uiCurrent];

    // first let us get the offsets ..
    for (int j=0; j < 8; j++) {
      if ( _mask & (1 << j ) ) {
        m_uiQuotientCounter++;
      }
    }
  }

  unsigned char DA::getHangingNodeIndex(unsigned int i) {
#ifdef __DEBUG_DA_PUBLIC__
    assert(m_bIamActive);
#endif
    return m_ucpLutMasks[2*i + 1];
  }

  void DA::incrementPreGhostOffset() {
#ifdef __DEBUG_DA_PUBLIC__
    assert(m_bIamActive);
#endif

    unsigned char c = m_ucpPreGhostConnectivity[m_uiCurrent];
    if ( c ) {
      // current and next Depths
      unsigned char nd = m_ucpOctLevels[m_uiCurrent+1];
      unsigned char cd = m_ucpOctLevels[m_uiCurrent];
      // current and next octant sizes
      unsigned int ns = (unsigned int)(1u << ( m_uiMaxDepth - nd ) );
      unsigned int cs = (unsigned int)(1u << ( m_uiMaxDepth - cd ) );

      Point curr = m_ptCurrentOffset;

      unsigned int cx = curr.xint();
      unsigned int cy = curr.yint();
      unsigned int cz = curr.zint();
      unsigned int nx = cx;
      unsigned int ny = cy;
      unsigned int nz = cz;

      // @milinda : update next_xyz computation for Hilbert
      //_zzyyxxT
      unsigned char xFlag = ((c & (3<<1) ) >> 1);
      unsigned char yFlag = ((c & (3<<3) ) >> 3);
      unsigned char zFlag = ((c & (3<<5) ) >> 5);

      switch (xFlag) {
        case 0: nx = cx;
                break;
        case 1: nx = (cx - ns);
                break;
        case 2: nx = (cx + cs); 
                break;
        case 3: nx = (cx + cs - ns);
                break;
        default: assert(false);
      }

      switch (yFlag) {
        case 0: ny = cy;
                break;
        case 1: ny = (cy - ns);
                break;
        case 2: ny = (cy + cs); 
                break;
        case 3: ny = (cy + cs - ns);
                break;
        default: assert(false);
      }

      switch (zFlag) {
        case 0: nz = cz;
                break;
        case 1: nz = (cz - ns); 
                break;
        case 2: nz = (cz + cs); 
                break;
        case 3: nz = (cz + cs - ns); 
                break;
        default: assert(false);
      }

      m_ptCurrentOffset = Point(nx,ny,nz);    

    } else {
      m_ptCurrentOffset = m_ptsPreGhostOffsets[m_uiPreGhostQuotientCnt++];     
    }

  }//end function


  void DA::computeHilbertRotations() {

#ifdef HILBERT_ORDERING

      this->init<DA_FLAGS::ALL>();
      m_uiParRotID=new unsigned char[this->end<ot::DA_FLAGS::ALL>()];
      m_uiParRotIDLev=new unsigned char[this->end<ot::DA_FLAGS::ALL>()];

      for(this->init<ot::DA_FLAGS::ALL>();this->curr()<this->end<ot::DA_FLAGS::ALL>();this->next<ot::DA_FLAGS::ALL>());
      m_uiRotIDComputed=true;



#endif

      return;

  }

 void DA::printODAStatistics()
 {

     DendroIntL localSz;

     // Node information

     DendroIntL nodeTlSz[3]; // total available nodes
     DendroIntL glbBndry[3]; // Global boundary
     DendroIntL preGhost[3]; // Pre ghost octants
     DendroIntL indNodes[3];
     DendroIntL postGhost[3];

     // Communication information.
     DendroIntL sendProcCnt[3];
     DendroIntL recvProcCnt[3];
     DendroIntL totalProcCnt[3];

     DendroIntL sendDataCnt[3];
     DendroIntL recvDataCnt[3];
     DendroIntL totalDataCnt[3];

     DendroIntL scatterMp[3];
     DendroIntL elementScatterMp[3];


     int size,rank;
     MPI_Comm_rank(MPI_COMM_WORLD,&rank);
     MPI_Comm_size(MPI_COMM_WORLD,&size);


     if(size==1)
         return;



     // Node Information

     localSz = m_uiNodeSize + m_uiBoundaryNodeSize;
     par::Mpi_Reduce<DendroIntL>(&localSz, nodeTlSz, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (nodeTlSz + 1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (nodeTlSz + 2), 1, MPI_MAX, 0, m_mpiCommActive);
     nodeTlSz[1]=nodeTlSz[1]/size;



     localSz =m_uiBoundaryNodeSize;
     par::Mpi_Reduce<DendroIntL>(&localSz, glbBndry, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (glbBndry+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (glbBndry+2),1,MPI_MAX,0,m_mpiCommActive);
     glbBndry[1]=glbBndry[1]/size;


     localSz=m_uiPreGhostElementSize;
     par::Mpi_Reduce<DendroIntL>(&localSz,preGhost,1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz,(preGhost+1),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz,(preGhost+2),1,MPI_MAX,0,m_mpiCommActive);
     preGhost[1]=preGhost[1]/size;


     localSz = m_uiElementSize;
     par::Mpi_Reduce<DendroIntL>(&localSz, indNodes, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (indNodes+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (indNodes+2), 1, MPI_MAX, 0, m_mpiCommActive);
     indNodes[1]=indNodes[1]/size;

     localSz =m_uiPostGhostNodeSize;
     par::Mpi_Reduce<DendroIntL>(&localSz, postGhost, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (postGhost+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (postGhost+2), 1, MPI_MAX, 0, m_mpiCommActive);
     postGhost[1]=postGhost[1]/size;



     // send proc count

     localSz=m_uipSendProcs.size();
     par::Mpi_Reduce<DendroIntL>(&localSz, sendProcCnt, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (sendProcCnt+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (sendProcCnt+2), 1, MPI_MAX, 0, m_mpiCommActive);
     sendProcCnt[1]=sendProcCnt[1]/size;

     // recv proc count

     localSz=m_uipRecvProcs.size();
     par::Mpi_Reduce<DendroIntL>(&localSz, recvProcCnt, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (recvProcCnt+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (recvProcCnt+2), 1, MPI_MAX, 0, m_mpiCommActive);
     recvProcCnt[1]=recvProcCnt[1]/size;

     // total send

     localSz=m_uipSendOffsets[m_uipSendOffsets.size()-1]+m_uipSendCounts[m_uipSendCounts.size()-1];
     par::Mpi_Reduce<DendroIntL>(&localSz, sendDataCnt, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (sendDataCnt+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (sendDataCnt+2), 1, MPI_MAX, 0, m_mpiCommActive);
     sendDataCnt[1]=sendDataCnt[1]/size;

     DendroIntL minS=0,maxS=0;
     double maxRminS=0;
     for(int i=0;i<m_uipSendCounts.size();i++)
     {
         if(i==0)
         {
             minS=m_uipSendCounts[i];
             maxS=m_uipSendCounts[i];
             continue;
         }

         if(minS>m_uipSendCounts[i])
             minS=m_uipSendCounts[i];

         if(maxS<m_uipSendCounts[i])
             maxS=m_uipSendCounts[i];

     }
     if(minS!=0)
        maxRminS=(double)maxS/(double)minS;
     else
        maxRminS=0;



     DendroIntL minR=0,maxR=0;
     double maxRminR=0;
     for(int i=0;i<m_uipRecvCounts.size();i++)
     {
         if(i==0)
         {
             minR=m_uipRecvCounts[i];
             maxR=m_uipRecvCounts[i];
             continue;
         }

         if(minR>m_uipRecvCounts[i])
             minR=m_uipRecvCounts[i];

         if(maxR<m_uipRecvCounts[i])
             maxR=m_uipRecvCounts[i];

     }
     if(minR!=0)
        maxRminR=(double)maxR/(double)minR;
     else
         maxRminR=0;

     DendroIntL maxComCnt=std::max(maxS,maxR);
     DendroIntL sendCnt[6];
     DendroIntL recvCnt[6];
     DendroIntL comCnt[3];

     double sendR[3];
     double recvR[3];

     par::Mpi_Reduce(&minS,sendCnt,1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce(&minS,(sendCnt+1),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&minS,(sendCnt+2),1,MPI_MAX,0,m_mpiCommActive);
     sendCnt[1]=sendCnt[1]/size;

     par::Mpi_Reduce(&maxS,(sendCnt+3),1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxS,(sendCnt+4),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxS,(sendCnt+5),1,MPI_MAX,0,m_mpiCommActive);
     sendCnt[4]=sendCnt[4]/size;

     par::Mpi_Reduce(&maxRminS,sendR,1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxRminS,(sendR+1),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxRminS,(sendR+2),1,MPI_MAX,0,m_mpiCommActive);
     sendR[1]=sendR[1]/(double)size;


     par::Mpi_Reduce(&minR,recvCnt,1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce(&minR,(recvCnt+1),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&minR,(recvCnt+2),1,MPI_MAX,0,m_mpiCommActive);
     recvCnt[1]=recvCnt[1]/size;

     par::Mpi_Reduce(&maxR,(recvCnt+3),1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxR,(recvCnt+4),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxR,(recvCnt+5),1,MPI_MAX,0,m_mpiCommActive);
     recvCnt[4]=recvCnt[4]/size;

     par::Mpi_Reduce(&maxRminR,recvR,1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxRminR,(recvR+1),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxRminR,(recvR+2),1,MPI_MAX,0,m_mpiCommActive);
     recvR[1]=recvR[1]/(double)size;

     par::Mpi_Reduce(&maxComCnt,comCnt,1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxComCnt,(comCnt+1),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&maxComCnt,(comCnt+2),1,MPI_MAX,0,m_mpiCommActive);

     comCnt[1]=comCnt[1]/size;



     //total recv

     localSz=m_uipRecvOffsets[m_uipRecvOffsets.size()-1]+m_uipRecvCounts[m_uipRecvCounts.size()-1];
     par::Mpi_Reduce<DendroIntL>(&localSz, recvDataCnt, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (recvDataCnt+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (recvDataCnt+2), 1, MPI_MAX, 0, m_mpiCommActive);
     recvDataCnt[1]=recvDataCnt[1]/size;

     // total proc (send + recv)

     localSz=m_uipSendProcs.size()+m_uipRecvProcs.size();
     par::Mpi_Reduce<DendroIntL>(&localSz, totalProcCnt, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (totalProcCnt+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (totalProcCnt+2), 1, MPI_MAX, 0, m_mpiCommActive);
     totalProcCnt[1]=totalProcCnt[1]/size;


     // total data (send +recv)

     localSz=m_uipSendOffsets[m_uipSendOffsets.size()-1]+m_uipSendCounts[m_uipSendCounts.size()-1]+m_uipRecvOffsets[m_uipRecvOffsets.size()-1]+m_uipRecvCounts[m_uipRecvCounts.size()-1];
     par::Mpi_Reduce<DendroIntL>(&localSz, totalDataCnt, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (totalDataCnt+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (totalDataCnt+2), 1, MPI_MAX, 0, m_mpiCommActive);
     totalDataCnt[1]=totalDataCnt[1]/size;

     // scatter map nodal

     localSz=m_uipScatterMap.size();
     par::Mpi_Reduce<DendroIntL>(&localSz, scatterMp, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (scatterMp+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (scatterMp+2), 1, MPI_MAX, 0, m_mpiCommActive);
     scatterMp[1]=scatterMp[1]/size;


     // scatter map element

     localSz=m_uipElemScatterMap.size();
     par::Mpi_Reduce<DendroIntL>(&localSz, elementScatterMp, 1, MPI_MIN, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (elementScatterMp+1), 1, MPI_SUM, 0, m_mpiCommActive);
     par::Mpi_Reduce<DendroIntL>(&localSz, (elementScatterMp+2), 1, MPI_MAX, 0, m_mpiCommActive);
     elementScatterMp[1]=elementScatterMp[1]/size;


     if(!rank)
     {
         std::cout<<"=======ODA NODE STATISTICS (min mean max)=========="<<std::endl;
         std::cout<<"Node(total):\t"<<nodeTlSz[0]<<"\t"<<nodeTlSz[1]<<"\t"<<nodeTlSz[2]<<std::endl;
         std::cout<<"Boundary(global):\t"<<glbBndry[0]<<"\t"<<glbBndry[1]<<"\t"<<glbBndry[2]<<std::endl;
         std::cout<<"PreGhost:\t"<<preGhost[0]<<"\t"<<preGhost[1]<<"\t"<<preGhost[2]<<std::endl;
         std::cout<<"Independent(mine):\t"<<indNodes[0]<<"\t"<<indNodes[1]<<"\t"<<indNodes[2]<<std::endl;
         std::cout<<"PostGhost:\t"<<postGhost[0]<<"\t"<<postGhost[1]<<"\t"<<postGhost[2]<<std::endl;


         std::cout<<"===================ODA COMMUNICATION STATISTICS (min mean max)===================="<<std::endl;
         std::cout<<"SendProcessorCnt:\t"<<sendProcCnt[0]<<"\t"<<sendProcCnt[1]<<"\t"<<sendProcCnt[2]<<std::endl;

         std::cout<<"SendDataCnt(total):\t"<<sendDataCnt[0]<<"\t"<<sendDataCnt[1]<<"\t"<<sendDataCnt[2]<<std::endl;
         std::cout<<"SendDataCntMin:\t"<<sendCnt[0]<<"\t"<<sendCnt[1]<<"\t"<<sendCnt[2]<<std::endl;
         std::cout<<"SendDataCntMax:\t"<<sendCnt[3]<<"\t"<<sendCnt[4]<<"\t"<<sendCnt[5]<<std::endl;
         std::cout<<"SendDataCntRatio:\t"<<sendR[0]<<"\t"<<sendR[1]<<"\t"<<sendR[2]<<std::endl;

         std::cout<<"RecvProcessorCnt:\t"<<recvProcCnt[0]<<"\t"<<recvProcCnt[1]<<"\t"<<recvProcCnt[2]<<std::endl;
         std::cout<<"RecvDataCnt(total):\t"<<recvDataCnt[0]<<"\t"<<recvDataCnt[1]<<"\t"<<recvDataCnt[2]<<std::endl;
         std::cout<<"RecvDataCntMin:\t:"<<recvCnt[0]<<"\t"<<recvCnt[1]<<"\t"<<recvCnt[2]<<std::endl;
         std::cout<<"RecvDataCntMax:\t"<<recvCnt[3]<<"\t"<<recvCnt[4]<<"\t"<<recvCnt[5]<<std::endl;
         std::cout<<"RecvDataCntRatio:\t"<<recvR[0]<<"\t"<<recvR[1]<<"\t"<<recvR[2]<<std::endl;

         std::cout<<"MaxLocal(senCntMax,recvCntMax):\t"<<comCnt[0]<<"\t"<<comCnt[1]<<"\t"<<comCnt[2]<<std::endl;

         std::cout<<"(Send+Recv)ProcessorCnt:\t"<<totalProcCnt[0]<<"\t"<<totalProcCnt[1]<<"\t"<<totalProcCnt[2]<<std::endl;
         std::cout<<"(Send+Recv)DataCnt(total):\t"<<totalDataCnt[0]<<"\t"<<totalDataCnt[1]<<"\t"<<totalDataCnt[2]<<std::endl;
         std::cout<<"NodalScatter(size):\t"<<scatterMp[0]<<"\t"<<scatterMp[1]<<"\t"<<scatterMp[2]<<std::endl;
         std::cout<<"ElementScatter(size):\t"<<elementScatterMp[0]<<"\t"<<elementScatterMp[1]<<"\t"<<elementScatterMp[2]<<std::endl;

     }






 }


 void DA::printODANodeListStatistics(char * nlistFName)
 {

     DendroIntL diff_pre=0,diff_mine=0,diff_post=0;
     DendroIntL  pre_diff[3];  pre_diff[0]=INTMAX_MAX; pre_diff[1]=0; pre_diff[2]=0;
     DendroIntL  mine_diff[3]; mine_diff[0]=INTMAX_MAX; mine_diff[1]=0; mine_diff[2]=0;
     DendroIntL  post_diff[3]; post_diff[0]=INTMAX_MAX; post_diff[1]=0; post_diff[2]=0;


     DendroIntL  pre_diff_g[3];  pre_diff_g[0]=INTMAX_MAX; pre_diff_g[1]=0; pre_diff_g[2]=0;
     DendroIntL  mine_diff_g[3]; mine_diff_g[0]=INTMAX_MAX; mine_diff_g[1]=0; mine_diff_g[2]=0;
     DendroIntL  post_diff_g[3]; post_diff_g[0]=INTMAX_MAX; post_diff_g[1]=0; post_diff_g[2]=0;

//     FILE* outfile = fopen(nlistFName, "wb");

     int size,rank;
     MPI_Comm_rank(m_mpiCommActive,&rank);
     MPI_Comm_size(m_mpiCommActive,&size);


     unsigned int nList[8];
     unsigned int last_pre=0,last_mine=0,last_post=0;
     bool pre=true;
     bool mine=true;
     bool post=true;
     unsigned int currentIdx;
     DendroIntL preCnt=0,mineCnt=0,postCnt=0;
     for((this->init<DA_FLAGS::ALL>());(this->curr()<end<DA_FLAGS::ALL>());(this->next<DA_FLAGS::ALL>()))
     {
         currentIdx=this->curr();
         this->getNodeIndices(nList);
         pre=true;mine=true;post=true;

         pre_diff[0]=INTMAX_MAX; pre_diff[1]=0; pre_diff[2]=0;
         mine_diff[0]=INTMAX_MAX; mine_diff[1]=0; mine_diff[2]=0;
         post_diff[0]=INTMAX_MAX; post_diff[1]=0; post_diff[2]=0;

         for(int k=0;k<8;k++)
         {
           if(nList[k]<m_uiElementBegin)
           { //Pre ghost

                preCnt++;
               if(pre)
               {
                   //std::cout<<"Pre Ghost End:"<<m_uiElementBegin<<std::endl;
                   last_pre=nList[k];
                   diff_pre=0;
                   pre=false;
               }else
               {
                   diff_pre=std::labs(last_pre-nList[k]);
                   //std::cout<<"diff_pre:"<<diff_pre<<std::endl;
                   last_pre=nList[k];
               }

               if(pre_diff[0]>diff_pre)
                   pre_diff[0]=diff_pre;

               pre_diff[1]=pre_diff[1]+diff_pre;

               if(pre_diff[2]<diff_pre)
                   pre_diff[2]=diff_pre;



           }else if(nList[k]<m_uiElementEnd)
           { // mine
               mineCnt++;
               if(mine)
               {
                   last_mine=nList[k];
                   diff_mine=0;
                   mine=false;
               }else
               {
                   diff_mine=std::labs(last_mine-nList[k]);
                   last_mine=nList[k];
               }

               if(mine_diff[0]>diff_mine)
                   mine_diff[0]=diff_mine;

               mine_diff[1]=mine_diff[1]+diff_mine;

               if(mine_diff[2]<diff_mine)
                   mine_diff[2]=diff_mine;

           }else if(nList[k]>=m_uiElementEnd)
           {//post
                postCnt++;
               if(post)
               {
                   last_post=nList[k];
                   diff_post=0;
                   post=false;
               }else
               {
                   diff_post = std::labs(last_post-nList[k]);
                   last_post=nList[k];
               }

               if(post_diff[0]>diff_post)
                   post_diff[0]=diff_post;

               post_diff[1]=post_diff[1]+diff_post;

               if(post_diff[2]<diff_post)
                   post_diff[2]=diff_post;

           }



         }


         if((!pre) && (pre_diff_g[2]<pre_diff[2]))
              pre_diff_g[2]=pre_diff[2];

         if(!pre)
             pre_diff_g[1]+=pre_diff[1];

         if(!pre && (pre_diff_g[0]>pre_diff[0]))
             pre_diff_g[0]=pre_diff[0];


         if((!mine )&& (mine_diff_g[2]<mine_diff[2]))
             mine_diff_g[2]=mine_diff[2];

         if(!mine)
             mine_diff_g[1]+=mine_diff[1];

         if((!mine) && (mine_diff_g[0]>mine_diff[0]))
             mine_diff_g[0]=mine_diff[0];


         if((!post) && (post_diff_g[2]<post_diff[2]))
             post_diff_g[2]=post_diff[2];

         if(!post)
             post_diff_g[1]+=post_diff[1];

         if((!post) && (post_diff_g[0]>post_diff[0]))
             post_diff_g[0]=post_diff[0];




          /*fwrite(&currentIdx,sizeof(unsigned int),1,outfile);
          fwrite(nList,sizeof(unsigned int),8,outfile);
          if(!pre) {
              fwrite(pre_diff, sizeof(DendroIntL), 3, outfile);
          }else
          {
              int pad[3]; pad[0]=-1;pad[1]=-1;pad[2]=-1;
              fwrite(pad, sizeof(int), 3, outfile);

          }

         if(!mine) {
             fwrite(mine_diff, sizeof(DendroIntL), 3, outfile);
         }else
         {
             int pad[3]; pad[0]=-1;pad[1]=-1;pad[2]=-1;
             fwrite(pad, sizeof(int), 3, outfile);
         }
         if(!post) {
             fwrite(post_diff, sizeof(DendroIntL), 3, outfile);
         }else
         {
             int pad[3]; pad[0]=-1;pad[1]=-1;pad[2]=-1;
             fwrite(pad, sizeof(int), 3, outfile);
         }
*/


     }

     /*fwrite(&preCnt,sizeof(DendroIntL),1,outfile);
     fwrite(&mineCnt,sizeof(DendroIntL),1,outfile);
     fwrite(&postCnt,sizeof(DendroIntL ),1,outfile);

     fclose(outfile);*/

     DendroIntL diff_stat_ofMax[9];
     DendroIntL diff_stat_ofSum[3];

     DendroIntL cnt[3];
     //std::cout<<"ststs computed"<<std::endl;
     if(preCnt)
        pre_diff_g[1]=pre_diff_g[1]/preCnt;
     else
         pre_diff_g[1]=0; // this is not needed but put to make sure;

     if(mineCnt)
        mine_diff_g[1]=mine_diff_g[1]/mineCnt;
     else
         mine_diff_g[1]=0;
     if(postCnt)
        post_diff_g[1]=post_diff_g[1]/postCnt;
     else
         post_diff_g[1]=0;



     par::Mpi_Reduce((pre_diff_g+1),(diff_stat_ofMax),1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce((pre_diff_g+1),(diff_stat_ofMax+1),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce((pre_diff_g+1),(diff_stat_ofMax+2),1,MPI_MAX,0,m_mpiCommActive);
     diff_stat_ofMax[1]=diff_stat_ofMax[1]/size;

     par::Mpi_Reduce((mine_diff_g+1),(diff_stat_ofMax+3),1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce((mine_diff_g+1),(diff_stat_ofMax+4),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce((mine_diff_g+1),(diff_stat_ofMax+5),1,MPI_MAX,0,m_mpiCommActive);
     diff_stat_ofMax[4]=diff_stat_ofMax[4]/size;

     par::Mpi_Reduce((post_diff_g+1),(diff_stat_ofMax+6),1,MPI_MIN,0,m_mpiCommActive);
     par::Mpi_Reduce((post_diff_g+1),(diff_stat_ofMax+7),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce((post_diff_g+1),(diff_stat_ofMax+8),1,MPI_MAX,0,m_mpiCommActive);
     diff_stat_ofMax[7]=diff_stat_ofMax[7]/size;


     par::Mpi_Reduce(&preCnt,cnt,1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&mineCnt,(cnt+1),1,MPI_SUM,0,m_mpiCommActive);
     par::Mpi_Reduce(&postCnt,(cnt+2),1,MPI_SUM,0,m_mpiCommActive);
//
//
//     par::Mpi_Reduce((pre_diff_g+1),(diff_stat_ofSum),1,MPI_SUM,0,m_mpiCommActive);
//     par::Mpi_Reduce((mine_diff_g+1),(diff_stat_ofSum+1),1,MPI_SUM,0,m_mpiCommActive);
//     par::Mpi_Reduce((post_diff_g+1),(diff_stat_ofSum+2),1,MPI_SUM,0,m_mpiCommActive);




     if(!rank)
     {

         std::cout<<"========================NODE LIST STATISTICS OF (MAX DIFFERENCE) MIN MEAN MAX ACCESS COUNT====================================="<<std::endl;
         std::cout<<"PreGhost(MeanDiff):\t"<<diff_stat_ofMax[0]<<"\t"<<diff_stat_ofMax[1]<<"\t"<<diff_stat_ofMax[2]<<std::endl;
         std::cout<<"MyElements(MeanDiff):\t"<<diff_stat_ofMax[3]<<"\t"<<diff_stat_ofMax[4]<<"\t"<<diff_stat_ofMax[5]<<std::endl;
         std::cout<<"PostGhost(MeanDiff):\t"<<diff_stat_ofMax[6]<<"\t"<<diff_stat_ofMax[7]<<"\t"<<diff_stat_ofMax[8]<<std::endl;
         std::cout<<"PreGhostAccessCnt:"<<cnt[0]<<std::endl;
         std::cout<<"MineGhostAccessCnt:"<<cnt[1]<<std::endl;
         std::cout<<"PostGhostAccessCnt:"<<cnt[2]<<std::endl;

//         std::cout<<"========================NODE LIST STATISTICS OF (MAX DIFFERENCE) Total Diff Sum====================================="<<std::endl;
//         std::cout<<"Pre Ghost diff Sum:\t"<<diff_stat_ofSum[0]<<std::endl;
//         std::cout<<"My Element diff Sum:\t"<<diff_stat_ofSum[1]<<std::endl;
//         std::cout<<"Post Ghost diff Sum:\t"<<diff_stat_ofSum[2]<<std::endl;
//         std::cout<<"=============================================================================================="<<std::endl;


     }



 }


int DA::alignPointsWithDA(std::vector<ot::NodeAndValues<double,3>>& pts) {
  //Re-distribute pts to align with minBlocks
  
  int* sendCnts = new int[m_iNpesAll];
  int* tmpSendCnts = new int[m_iNpesAll];

  for(int i = 0; i < m_iNpesAll; i++) {
    sendCnts[i] = 0;
    tmpSendCnts[i] = 0;
  }//end for i

  int numPts = pts.size();
  unsigned int *part = NULL;
    if(numPts) {
      part = new unsigned int[numPts];
  }

  for(int i = 0; i < numPts; i++) {
    bool found = seq::maxLowerBound<ot::TreeNode>(m_tnMinAllBlocks, pts[i].node, part[i], NULL, NULL);
    assert(found);
    assert(part[i] < m_iNpesAll);
    sendCnts[part[i]]++;
  }//end for i

  int* sendDisps = new int[m_iNpesAll];

  sendDisps[0] = 0;
  for(int i = 1; i < m_iNpesAll; i++) {
    sendDisps[i] = sendDisps[i-1] + sendCnts[i-1];
  }//end for i

  std::vector<ot::NodeAndValues<double, 3> > sendList(numPts);

  unsigned int *commMap = NULL;
  if(numPts) {
    commMap = new unsigned int[numPts];
  }

  for(int i = 0; i < numPts; i++) {
      unsigned int pId = part[i];
      unsigned int sId = (sendDisps[pId] + tmpSendCnts[pId]);
      assert(sId < numPts);
      sendList[sId] = pts[i];
      commMap[sId] = i;
      tmpSendCnts[pId]++;
  }//end for i

  if(part) {
    delete [] part;
    part = NULL;
  }
  delete [] tmpSendCnts;

  int* recvCnts = new int[m_iNpesAll];
  assert(recvCnts);

  par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, m_mpiCommAll);

  int* recvDisps = new int[m_iNpesAll];

  recvDisps[0] = 0;
  for(int i = 1; i < m_iNpesAll; i++) {
    recvDisps[i] = recvDisps[i-1] + recvCnts[i-1];
  }//end for i

  std::vector<ot::NodeAndValues<double, 3> > recvList(recvDisps[m_iNpesAll - 1] + recvCnts[m_iNpesAll - 1]);

  ot::NodeAndValues<double, 3>* sendListPtr = NULL;
  ot::NodeAndValues<double, 3>* recvListPtr = NULL;

  if(!(sendList.empty())) {
    sendListPtr = (&(*(sendList.begin())));
  }

  if(!(recvList.empty())) {
    recvListPtr = (&(*(recvList.begin())));
  }

  par::Mpi_Alltoallv_sparse<ot::NodeAndValues<double, 3> >(sendListPtr, 
        sendCnts, sendDisps, recvListPtr, recvCnts, recvDisps, m_mpiCommAll);
  sendList.clear();

  std::sort(recvList.begin(), recvList.end());
  pts = std::move(recvList);

  recvList.clear();
  sendList.clear();

  // Stage 2 - send points to procs if the point is within an element that is a ghost on another proc.
  
  //== loop over pts and boundary elements simultaneously 


  // 

  
    
  // clean up.
  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;

}
 
 
int DA::alignPointsWithDA(std::vector<double>& pts, std::vector<int>& labels) {
    // MPI_Comm                        m_mpiCommAll;
    // int                             m_iRankAll;
    // int                             m_iNpesAll;

    // m_tnMinAllBlocks 
  
  unsigned int balOctMaxD = (m_uiMaxDepth - 1);
  
  int numPts = (pts.size())/3;
  double xyzFac = static_cast<double>(1u << balOctMaxD);
  
  std::vector<ot::NodeAndValues<double, 3> > ptsWrapper;
  
  for(int i = 0; i < numPts; i++) {
    ot::NodeAndValues<double, 3> tmpObj;
    unsigned int xint = static_cast<unsigned int>(pts[(3*i)]*xyzFac);
    unsigned int yint = static_cast<unsigned int>(pts[(3*i) + 1]*xyzFac);
    unsigned int zint = static_cast<unsigned int>(pts[(3*i) + 2]*xyzFac);
    tmpObj.node = ot::TreeNode(xint, yint, zint, m_uiMaxDepth, 3, m_uiMaxDepth);
    tmpObj.values[0] = pts[(3*i)];
    tmpObj.values[1] = pts[(3*i) + 1];
    tmpObj.values[2] = pts[(3*i) + 2];
    tmpObj.label = labels[i];
    ptsWrapper.push_back(tmpObj);
  }//end for i
  
  //Re-distribute ptsWrapper to align with minBlocks
  int* sendCnts = new int[m_iNpesAll];
  int* tmpSendCnts = new int[m_iNpesAll];

  for(int i = 0; i < m_iNpesAll; i++) {
    sendCnts[i] = 0;
    tmpSendCnts[i] = 0;
  }//end for i

  unsigned int *part = NULL;
    if(numPts) {
      part = new unsigned int[numPts];
  }

  for(int i = 0; i < numPts; i++) {
    bool found = seq::maxLowerBound<ot::TreeNode>(m_tnMinAllBlocks, ptsWrapper[i].node, part[i], NULL, NULL);
    assert(found);
    assert(part[i] < m_iNpesAll);
    sendCnts[part[i]]++;
  }//end for i

  int* sendDisps = new int[m_iNpesAll];

  sendDisps[0] = 0;
  for(int i = 1; i < m_iNpesAll; i++) {
    sendDisps[i] = sendDisps[i-1] + sendCnts[i-1];
  }//end for i

  std::vector<ot::NodeAndValues<double, 3> > sendList(numPts);

  unsigned int *commMap = NULL;
  if(numPts) {
    commMap = new unsigned int[numPts];
  }

  for(int i = 0; i < numPts; i++) {
      unsigned int pId = part[i];
      unsigned int sId = (sendDisps[pId] + tmpSendCnts[pId]);
      assert(sId < numPts);
      sendList[sId] = ptsWrapper[i];
      commMap[sId] = i;
      tmpSendCnts[pId]++;
  }//end for i

  if(part) {
    delete [] part;
    part = NULL;
  }
  delete [] tmpSendCnts;

  ptsWrapper.clear();

  int* recvCnts = new int[m_iNpesAll];
  assert(recvCnts);

  par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, m_mpiCommAll);

  int* recvDisps = new int[m_iNpesAll];

  recvDisps[0] = 0;
  for(int i = 1; i < m_iNpesAll; i++) {
    recvDisps[i] = recvDisps[i-1] + recvCnts[i-1];
  }//end for i

  std::vector<ot::NodeAndValues<double, 3> > recvList(recvDisps[m_iNpesAll - 1] + recvCnts[m_iNpesAll - 1]);

  ot::NodeAndValues<double, 3>* sendListPtr = NULL;
  ot::NodeAndValues<double, 3>* recvListPtr = NULL;

  if(!(sendList.empty())) {
    sendListPtr = (&(*(sendList.begin())));
  }

  if(!(recvList.empty())) {
    recvListPtr = (&(*(recvList.begin())));
  }

  par::Mpi_Alltoallv_sparse<ot::NodeAndValues<double, 3> >(sendListPtr, 
        sendCnts, sendDisps, recvListPtr, recvCnts, recvDisps, m_mpiCommAll);
  sendList.clear();
  
  std::sort(recvList.begin(), recvList.end());
  sendList = std::move(recvList);
  
  recvList.clear();
  // clear and copy to points ...
  pts.clear();
  labels.clear();
  for (auto x: sendList) {
    pts.push_back(x.values[0]);
    pts.push_back(x.values[1]);
    pts.push_back(x.values[2]);
    labels.push_back(x.label);
  }
  sendList.clear();
  
  //  Options
  // 1. Return intPts in addition
  // 2. Return only intPoints (TreeNode) ... you can compare and test for membership. Recover pt from TN 
  
  
  
  // clean up.
  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;
}

} // end namespace ot


