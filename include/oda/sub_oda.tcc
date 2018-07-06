/**
  @file sub_oda.tcc
  @author Hari Sundar, hsundar@gmail.com
 **/

namespace ot {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 
  // Implementation...

  //Next functions...
  template<ot::DA_FLAGS::loopType type>
    inline unsigned int subDA::next() {
      unsigned int current = m_da->next<type>();
      while ( ( current < m_ucpSkipList.size() ) && m_ucpSkipList[current] ) {
        // std::cout << "Skipping" << std::endl;
        current = m_da->next<type>();
      } 
      return current;
    }//end function

  //Init functions...
  template<ot::DA_FLAGS::loopType type>	
    inline void subDA::init() {
      m_da->init<type>();
      while ( m_ucpSkipList[ m_da->curr() ] )
        m_da->next<type>();

      // std::cout << "Init: " << m_ucpSkipList[]  
    }//end function


  //End functions...
  template<ot::DA_FLAGS::loopType type>
    inline unsigned int subDA::end() {
      return m_da->end<type>();
    }

  // vector functions
    template <typename T>
    int subDA::createVector(std::vector<T> &arr, bool isElemental,
        bool isGhosted, unsigned int dof) {
      // first determine the length of the vector ...
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
      
      // now create the vector
      arr.resize(sz);

      return 0;
    }

  template < typename T >
    int subDA::vecGetBuffer(std::vector<T> &in, T* &out, bool isElemental,
        bool isGhosted, bool isReadOnly, unsigned int dof) {

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
    
      unsigned int vecSz = static_cast<unsigned int>(in.size());

      if ( sz != vecSz) {
        std::cerr  << "In function subDA::" << __func__ << " sizes are unequal, sz is  " << sz
          << " and vecSz is " << vecSz << std::endl;
        assert(false);
        return -1;
      };

      // get the local Arrray,
      T *array = NULL;
      if(!in.empty()) {
        array = &(*(in.begin()));
      }

      if(isGhosted && isElemental) {
        //simply copy the pointer
        //This is the only case where the buffer will not be the size of the
        //fullLocalBufferSize. 
        out = array;
      }else {
        // First let us allocate for the buffer ... the local buffer will be of full
        // length.
        sz = dof*m_uiLocalBufferSize;
        //The default constructor of datatype T is responsible of initializing
        //out with the appropriate zero entries. 
        if(sz) {
          out = new T[sz];
          assert(out);
        } else {
          out = NULL;
        }

        //Zero Entries first if you plan to modify the buffer 
        // if(!isReadOnly) {
          for(unsigned int i = 0; i < sz; i++) {
            out[i] = 171396; // 0.0;
          }
        // }
      }

      unsigned int vecCnt=0;
      // Now we can populate the out buffer ... and that needs a loop through the
      // elements ...
      if (isGhosted) {
        if (isElemental) {
          //nothing to be done here.
        } else {
          //nodal and ghosted
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
        } // if elemental 
      } else { // not ghosted
        if (isElemental) {
          //elemental and non-ghosted
          // is a simple copy ... 
          for (unsigned int i = m_da->getIdxElementBegin(); i <          m_da->getIdxElementEnd(); i++) {
            if (m_ucpSkipList[i]) continue;
            for (unsigned int j = 0; j < dof; j++) {
              out[dof*m_uip_DA2sub_ElemMap[i]+j] = array[dof*vecCnt + j];
            }
            vecCnt++;
          }//end for i
        } else {
          //nodal and non-ghosted
          for (unsigned int i = m_da->getIdxElementBegin(); i < m_da->getIdxElementEnd(); i++) {
            // unsigned int di = m_uip_sub2DA_NodeMap[i];
            if ( ! (m_da->getFlag(i) & ot::TreeNode::NODE ) || m_ucpSkipNodeList[i] ) {
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
            if ( ! ( (m_da->getFlag(i) & ot::TreeNode::NODE ) &&
              (m_da->getFlag(i) & ot::TreeNode::BOUNDARY ) ) || m_ucpSkipNodeList[i] ) {
              continue;
            }
            for (unsigned int j=0; j<dof; j++) {
              out[dof*m_uip_DA2sub_NodeMap[i]+j] = array[dof*vecCnt + j];
            }
            vecCnt++;
          }//end for i
        }
      }
      return 0;
    }

  template < typename T >
    int subDA::vecRestoreBuffer(std::vector<T> &in, T* out, bool isElemental,
        bool isGhosted, bool isReadOnly, unsigned int dof) {
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

      unsigned int vecSz = static_cast<unsigned int>(in.size());

      if ( sz != vecSz) {
        std::cerr  << "In function subDA::STL::" << __func__ << " sizes are unequal, sz is  " <<
          sz << " and vecSz is " << vecSz << std::endl;
        assert(false);
        return -1;
      };

      unsigned int vecCnt=0;

      // if is readonly, then simply deallocate and return ...
      if ( isGhosted && isElemental ) {
        out = NULL;
      } else if ( isReadOnly ) {
        // no need to write back ... simply clean up and return
        if(out) {
          delete [] out;
          out = NULL;
        }
      } else {
        // need to write back ...
        // get the local Arrray,
        T *array = NULL;
        if(!in.empty()) {
          array = &(*in.begin());
        }

        if ( isElemental ) {
          //elemental and non-ghosted
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
            if ( ! (m_da->getFlag(i) & ot::TreeNode::NODE ) || m_ucpSkipNodeList[i] ) {
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
            if ( ! ( (m_da->getFlag(i) & ot::TreeNode::NODE ) &&
              (m_da->getFlag(i) & ot::TreeNode::BOUNDARY ) ) ||   m_ucpSkipNodeList[i] ) {
              continue;
            } 
            for (unsigned int j=0; j<dof; j++) {
              array[dof*vecCnt + j] = out[dof*m_uip_DA2sub_NodeMap[i]+j];
            }
            vecCnt++;
          }//end for i
        }

        if(out) {
          delete [] out;
          out = NULL;
        }  
      }
      return 0;
    }

  template <typename T>
    int subDA::ReadFromGhostsBegin ( T* arr, unsigned int dof) {
      PROF_READ_GHOST_NODES_BEGIN_BEGIN

      int rank, npes;
      MPI_Comm comm = m_da->getComm();
      rank = m_da->getRankAll();
      npes = m_da->getNpesAll();

      // first need to create contiguous list of boundaries ...
      T* sendK = NULL;
      if(m_uipScatterMap.size()) {
        sendK = new T[dof*m_uipScatterMap.size()];
        assert(sendK);
      }
      for (unsigned int i = 0; i < m_uipScatterMap.size(); i++ ) {
        // add dof loop ...
        for (unsigned int j = 0; j < dof; j++) {
          sendK[(dof*i) + j] = arr[(dof*m_uipScatterMap[i]) + j];
        }
      }

      // create a new context ...
      updateContext ctx;
      ctx.buffer = arr;
      ctx.keys = sendK;

      // Post Recv ...
      for (unsigned int i = 0; i < m_uipRecvProcs.size(); i++) {
        MPI_Request *req = new MPI_Request();
        assert(req);
        par::Mpi_Irecv<T>(arr + (dof*m_uipRecvOffsets[i]), (dof*m_uipRecvCounts[i]), 
            m_uipRecvProcs[i], m_uiCommTag, comm, req );
        ctx.requests.push_back(req);
      }

      //*********** Send ****************//
      for (unsigned int i = 0; i < m_uipSendProcs.size(); i++) {
        MPI_Request *req = new MPI_Request();
        assert(req);
        par::Mpi_Isend<T>( sendK + (dof*m_uipSendOffsets[i]), (dof*m_uipSendCounts[i]),
            m_uipSendProcs[i], m_uiCommTag, comm, req );
        ctx.requests.push_back(req);
      }

      // Increment tag ....
      m_uiCommTag++;

      m_mpiContexts.push_back(ctx);

      PROF_READ_GHOST_NODES_BEGIN_END
    }
 
  template <typename T>
    int subDA::ReadFromGhostsEnd(T* arr) {
      PROF_READ_GHOST_NODES_END_BEGIN

      // find the context ...
      unsigned int ctx;
      for ( ctx = 0; ctx < m_mpiContexts.size(); ctx++) {
        if ( m_mpiContexts[ctx].buffer == arr) {
          break;
        }
      }

      MPI_Status status;
      // need to wait for the commns to finish ...
      for (unsigned int i = 0; i < m_mpiContexts[ctx].requests.size(); i++) {
        MPI_Wait(m_mpiContexts[ctx].requests[i], &status);
        delete m_mpiContexts[ctx].requests[i];
      }

      // delete the sendkeys ...
      T *sendK = static_cast<T *>(m_mpiContexts[ctx].keys);

      if(sendK) {
        delete [] sendK;
        sendK = NULL;
      }

      // clear the Requests ...
      assert(ctx < m_mpiContexts.size());
      m_mpiContexts[ctx].requests.clear();

      // remove the context ...
      m_mpiContexts.erase(m_mpiContexts.begin() + ctx);

      PROF_READ_GHOST_NODES_END_END
    }

  template <typename T>
  int subDA::WriteToGhostsBegin ( T* arr, unsigned int dof) {
    PROF_WRITE_GHOST_NODES_BEGIN_BEGIN

     int rank, npes;
      MPI_Comm comm = m_da->getComm();
      rank = m_da->getRankAll();
      npes = m_da->getNpesAll();

        // first need to create contiguous list of boundaries ...
        T* recvK = NULL;
      if(m_uipScatterMap.size()) {
        recvK = new T[dof*(m_uipScatterMap.size())];
        assert(recvK);
      }

      // create a new context ...
      updateContext ctx;
      ctx.buffer = arr;
      ctx.keys = recvK;

      // Post Recv ...
      for (unsigned int i = 0; i < m_uipSendProcs.size(); i++) {
        MPI_Request *req = new MPI_Request();
        assert(req);
        par::Mpi_Irecv<T>( recvK + (dof*m_uipSendOffsets[i]), (dof*m_uipSendCounts[i]), 
            m_uipSendProcs[i], m_uiCommTag, comm, req );
        ctx.requests.push_back(req);
      }

      //The communication here is just the opposite of the communication in readFromGhosts...
      //*********** Send ****************//
      for (unsigned int i = 0; i < m_uipRecvProcs.size(); i++) {
        MPI_Request *req = new MPI_Request();
        assert(req);
        par::Mpi_Isend<T>( arr + (dof*m_uipRecvOffsets[i]), (dof*m_uipRecvCounts[i]), 
            m_uipRecvProcs[i], m_uiCommTag, comm, req );
        ctx.requests.push_back(req);
      }

      // Increment tag ....
      m_uiCommTag++;

      m_mpiContexts.push_back(ctx);

      PROF_WRITE_GHOST_NODES_BEGIN_END
  }
 
  template <typename T>
  int subDA::WriteToGhostsEnd(T* arr, unsigned int dof) {
    PROF_WRITE_GHOST_NODES_END_BEGIN

        // find the context ...
        unsigned int ctx;
      for ( ctx = 0; ctx < m_mpiContexts.size(); ctx++) {
        if ( m_mpiContexts[ctx].buffer == arr) {
          break;
        }
      }

      MPI_Status status;
      // need to wait for the commns to finish ...
      for (unsigned int i = 0; i < m_mpiContexts[ctx].requests.size(); i++) {
        MPI_Wait(m_mpiContexts[ctx].requests[i], &status);
        delete m_mpiContexts[ctx].requests[i];
      }

      //Add ghost values to the local vector.
      T *recvK = static_cast<T *>(m_mpiContexts[ctx].keys);
      for (unsigned int i=0; i<m_uipScatterMap.size(); i++ ) {
        for (unsigned int j=0; j<dof; j++) {
          arr[(dof*m_uipScatterMap[i]) + j] += recvK[(dof*i) + j];
        }
      }

      // delete the keys ...
      if(recvK) {
        delete [] recvK;
        recvK = NULL;
      }

      // clear the Requests ...
      assert(ctx < m_mpiContexts.size());
      m_mpiContexts[ctx].requests.clear();

      // remove the context ...
      m_mpiContexts.erase(m_mpiContexts.begin() + ctx);

      PROF_WRITE_GHOST_NODES_END_END
  }

}; // namespace ot
