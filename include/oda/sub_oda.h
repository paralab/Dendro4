/***
 * @file        sub_oda.h
 * @author      Hari Sundar   hsundar@gmail.com
 * @date        8 Feb 2018
 * 
 * @brief       Subdomain for Octree DA (ODA).
 * 
 ***/

#pragma once

#include <functional>
#include <iostream>

#include "oda.h"

namespace ot {
  /** 
   * @brief 		Class that manages a sub-domain within the octree mesh.
   * @author		Hari Sundar, hsundar@gmail.com 
   * @date 		  8 Feb 2018
   * 
   * Extracts a subdomain from the mesh based on a function. more later.
   * 
   **/

  class subDA {
      protected:
        ot::DA*       m_da;

        // will contain extra mapping information as compared to the DA
        std::vector<unsigned char>      m_ucpSkipList;
        // std::vector<unsigned char>      m_ucpSkipNodeList;
        unsigned char                  *m_ucpSkipNodeList;
        
        unsigned int                    m_uiElementSize;
        unsigned int                    m_uiPreGhostElementSize;

        unsigned int                    m_uiLocalBufferSize;

        unsigned int                    m_uiNodeSize;
        
        unsigned int                    m_uiPreGhostNodeSize;
        unsigned int                    m_uiPostGhostNodeSize;
        
        unsigned int                    m_uiBoundaryNodeSize;
        unsigned int                    m_uiPreGhostBoundaryNodeSize;

        unsigned int                    m_uiPostGhostBegin;

        bool                            m_bComputedLocalToGlobal;
        bool                            m_bComputedLocalToGlobalElems;

        // need estimates of local & global owned nodes

        std::vector<unsigned int>       m_uip_sub2DA_ElemMap;      
        std::vector<unsigned int>       m_uip_DA2sub_ElemMap;      

        std::vector<unsigned int>       m_uip_sub2DA_NodeMap;      
        std::vector<unsigned int>       m_uip_DA2sub_NodeMap;

        DendroIntL*                      m_dilpLocalToGlobal;
        DendroIntL*                      m_dilpLocalToGlobalElems;      

        std::vector<unsigned int>               m_uipScatterMap;
        std::vector<unsigned int>               m_uipSendOffsets;
        std::vector<unsigned int>               m_uipSendProcs;
        std::vector<unsigned int>               m_uipSendCounts;

        std::vector<unsigned int>               m_uipRecvOffsets;
        std::vector<unsigned int>               m_uipRecvProcs;
        std::vector<unsigned int>               m_uipRecvCounts;

        // tags and contexts ...
        std::vector<updateContext>              m_mpiContexts;
        unsigned int                            m_uiCommTag;

        /*  
        std::vector<unsigned int>               m_uipElemScatterMap;
        std::vector<unsigned int>               m_uipElemSendOffsets;
        std::vector<unsigned int>               m_uipElemSendProcs;
        std::vector<unsigned int>               m_uipElemSendCounts;

        std::vector<unsigned int>               m_uipElemRecvOffsets;
        std::vector<unsigned int>               m_uipElemRecvProcs;
        std::vector<unsigned int>               m_uipElemRecvCounts;
        */
       
      public:
        /**
          @name Constructors and destructors
          */
        //@{

        subDA(DA* da, std::function<double ( double, double, double ) > fx_retain, double* gSize);

        /**
          @author Hari Sundar
          @brief The destructor for the DA object
          */   
        ~subDA();
        //@}     

        /**
          @name Information about the DA / domain
          */
        //@{

       ot::DA* global_domain() { return m_da; }   

        /** 
          @brief Get the MPI communicator from the DA.
          @return MPI_Comm
          @author Hari Sundar
          */
        MPI_Comm getComm() { return m_da->getComm(); }

        /** 
          @brief Get the communicator only containing the active processors.
          @return MPI_Comm
          @author Hari Sundar
          */
        MPI_Comm getCommActive() { return m_da->getCommActive(); }

        /**
          @return the total number of processors (including inactive processors)
          @author Hari Sundar
          */
        int getNpesAll() { return m_da->getNpesAll(); }

        const std::vector<unsigned char>& getSkipList() { return m_ucpSkipNodeList; }

        const std::vector<unsigned char>& getSkipElemList() { return m_ucpSkipList; }

        /**
          @return the number of active processors
          @author Hari Sundar
          */
        int getNpesActive() { return m_da->getNpesActive(); }

        /**
          @return the rank of the calling processor, when both
          active and inactive processors are involved.  
          @author Hari Sundar
          */
        int getRankAll() { return m_da->getRankAll(); }

        /**
          @return the rank of the calling processors, when only the active processors are involved
          @author Hari Sundar
          */
        int getRankActive() { return m_da->getRankActive(); }

        /** 
          @author Hari Sundar
          @brief Get the offset for the current index. 
          @return The offset. 
          Must not be called by inactive processors

          @see getOffset() 
          @see getGhostedOffset() 
          @see iAmActive()
          */
        Point getCurrentOffset() { return m_da->getCurrentOffset(); }

        /**
          @author Hari Sundar
          @brief Get the offset for the smallest element on this processor, including ghost elements.
          @return The offset. 

          Must not be called by inactive processors
          @see getOffset() 
          @see getCurrentOffset() 
          @see iAmActive()
          */
        Point getGhostedOffset() { return m_da->getGhostedOffset(); }

        /**
          @author Hari Sundar
          @brief Get the offset for the smallest element on this processor, not including ghost elements.
          @return The offset. 

          Must not be called by inactive processors
          @see getGhostedOffset() 
          @see getCurrentOffset() 
          @see iAmActive()
          */
        Point getOffset() { return m_da->getOffset(); }

        /**
          @author Hari Sundar
          @brief Given an octant specified by a point (its anchor) and
          its level it returns the anchor of the octant
          that immediately follows the given point in the Morton ordering.
          This function assumes that the octree is linear.        
          @param p The anchor of the current octant
          @param d The level of the current octant
          @return the anchor of the next octant 
          */
          /*
        inline Point getNextOffset(Point p, unsigned char d) {
          return m_da->getNextOffset(p, d);
        }
        Point getNextOffsetByRotation(Point p, unsigned char d) {
          return m_da->getNextOffsetByRotation(p, d);
        }
        */

        /**
          @author Hari Sundar
          @brief Points to the next anchor. This function is required because we only 
          store the levels of the octants and not their anchors. So the anchors are
          computed on the fly within the loops.
          */
        void incrementCurrentOffset() { m_da->incrementCurrentOffset(); }

        /**
          @author Hari Sundar
          @brief Points to the anchor of the next pre-ghost octant.
          This function is required because we only 
          store the levels of the octants and not their anchors. So the anchors are
          computed on the fly within the loops.
          */
        void incrementPreGhostOffset() { m_da->incrementPreGhostOffset(); }

        /**
          @author Hari Sundar
          @brief Call this function to check if curr() points to an octant
          touching the domain boundaries from the inside. This function is for real octants only, 
          pseudo-octants can not be tested using this function.
          @param flags The type of boundary is returned in this variable.
          The type is one of the enumerations in BoundaryType2
          @return true if curr() points to an internal boundary octant      
          @see curr()
          @see TreeNode::BoundaryType2
          */ 
        bool isBoundaryOctant(unsigned char *flags=NULL) { return m_da->isBoundaryOctant(flags); }

        /**
          @author Hari Sundar
          @return The local to global map computed using the function computeLocalToGlobalMappings()      
          @see computeLocalToGlobalMappings
          @see computedLocalToGlobal
          */
        DendroIntL* getLocalToGlobalMap() {
          return m_dilpLocalToGlobal;
        }

        /**
          @author Hari Sundar
          @return The local to global map computed using the function computeLocalToGlobalElemMappings()      
          @see computeLocalToGlobalElemMappings
          @see computedLocalToGlobalElems
          */
        DendroIntL* getLocalToGlobalElemsMap() {
          return m_dilpLocalToGlobalElems;
        }

        /**
          @author Hari Sundar
          @brief Returns the total number of Nodes belonging to this processor. 
          This does not include the ghost nodes. This will include the 
          positive boundary nodes if any belong to this processor.
          @return The number of local nodes.
          */
        unsigned int getNodeSize() {
          return (m_uiNodeSize + m_uiBoundaryNodeSize);
        }

        /**
          @author Hari Sundar
          @brief Returns the total number of positive Boundary Nodes belonging to this processor. 
          This does not include the ghost nodes. 
          @return The number of local (positive) boundary nodes.
          */
        unsigned int getBoundaryNodeSize() {
          return m_uiBoundaryNodeSize;
        }

        /**
          @author Hari Sundar
          @brief Returns the total number of internal Nodes belonging to this processor. 
          This does not include the ghost nodes and positive boundaries . 
          @return The number of local internal nodes.
          */
        unsigned int getInternalNodeSize() {
          return m_uiNodeSize;
        }

        /**
          @author Hari Sundar
          @brief Returns the total number of elements belonging to this processor. 
          This does not include the ghost elements.  
          @return The number of local elements.
          */
        unsigned int getElementSize() {
          return m_uiElementSize;
        }

        /**
          @author Hari Sundar
          @brief Returns the total number of pre-ghost elements. 
          */
        unsigned int getPreGhostElementSize() {
          return m_uiPreGhostElementSize;
        }

        /** @author Milinda Fernando
         *  @brief Returns the number of ghost nodes. (pre ghost node + post ghost nodes)
         *
         * */

        unsigned int getPreAndPostGhostNodeSize() {
          return (m_uiPreGhostNodeSize+m_uiPostGhostNodeSize);
        }

        /**
          @author Hari Sundar
          @brief Returns the number of INDEPENDENT elements belonging to this processor. 
          @return The number of local elements.
          */
        unsigned int getIndependentSize() {
          std::cout << "[subDA::DEBUG] potential source of BUG. " << __FILE__ << ":" << __LINE__ << std::endl;
          return m_da->getIndependentSize();
        }

        /**
          @author Hari Sundar
          @brief Returns the total number of Nodes on this processor, 
          including the ghost nodes. This will include the 
          boundary nodes if any belong to this processor.
          @return The number of nodes.
          */
        unsigned int getGhostedNodeSize() {
          return (m_uiNodeSize + m_uiBoundaryNodeSize +
                  m_uiPreGhostNodeSize + m_uiPreGhostBoundaryNodeSize +
                  m_uiPostGhostNodeSize);
        }

        /**
          @author Hari Sundar
          @brief Returns the total number of elements on this processor, 
          including the ghost elements. 
          @return The number of nodes.
          */
        unsigned int getGhostedElementSize() {
          return (m_uiElementSize + m_uiPreGhostElementSize);
        }

        /**
          @author Hari Sundar
          @return the index of the first local element
          */
        unsigned int getIdxElementBegin() {
          return m_da->getIdxElementBegin();
        }

        /**
          @author Hari Sundar
          @return the index of the last local element
          */
        unsigned int getIdxElementEnd() {
          return m_da->getIdxElementEnd();
        }

        /**
          @author Hari Sundar
          @return the index of the first post ghost element
          */
        unsigned int getIdxPostGhostBegin() {
          return m_da->getIdxPostGhostBegin();
        }

        /**
          @author Hari Sundar
          @brief Returns the maximum depth (level) of the octree from which this DA was created.

          @return The maximum depth of the octree.
          The return value is the maximum depth in the modified octree that includes 'pseudo-octants' for boundary nodes. This octree has
          a maximum depth equal to 1 more than that of the input octree used to construct the finite element mesh. Hence, the value
          returned by this function will be 1 more than the true maximum depth of the input octree.
          */ 
        unsigned int getMaxDepth() {
          return m_da->getMaxDepth();
        }

        /**
          @author Hari Sundar
          @brief Returns the dimension of the octree from which this DA was created.
          @return The dimension of the octree.
          */
        unsigned int getDimension() {
          return m_da->getDimension();
        }

        unsigned int getPrePostBoundaryNodesSize() {
          return m_da->getPrePostBoundaryNodesSize();
        }
        //@}

        /** 
          @name Communication functions 
          */
        //@{

        /**
          @author Hari Sundar
          @author Hari Sundar	 
          @brief Updates the ghost values by obtaining values from the processors which own them.

          @param arr		the local buffer which needs to be updated. This must be obtained with a call to
          vecGetBuffer().
          @param isElemental	specifies whether the current buffer is elemental (true) or nodal (false).
          @param dof		The degrees of freedom for the current vector, default is 1.
          @see ReadFromGhostsEnd()
          Updates the ghost values by obtaining values from the processors which own them.
          ReadFromGhostsEnd()
          must be called before the ghosted values can be used. 
          */
        //Communicating Ghost Nodes
        template <typename T>
          int ReadFromGhostsBegin ( T* arr, unsigned int dof=1);

        /**
          @author Hari Sundar
          @author Hari Sundar
         * @brief Waits for updates of the ghost values to finish.
        **/ 
        template <typename T>
          int ReadFromGhostsEnd(T* arr);


        /**
          @author Hari Sundar
         * @brief Send the ghost values to the processors that own them so that these
         values can be added.  
        **/ 
        template <typename T>
          int WriteToGhostsBegin ( T* arr, unsigned int dof=1); 

        /**
          @author Hari Sundar
         * @brief Waits for updates of the ghost values to finish.
        **/ 
        template <typename T>
          int WriteToGhostsEnd(T* arr, unsigned int dof=1);


        /**
          @author Hari Sundar
          Counterpart of ReadFromGhostsBegin for elemental arrays
          @see ReadFromGhostsBegin()
          */
        template <typename T>
          int ReadFromGhostElemsBegin ( T* arr, unsigned int dof=1) {
            return m_da->ReadFromGhostElemsBegin<T>(arr, dof);
          }

        /**
          @author Hari Sundar
          Counterpart of ReadFromGhostsEnd() for elemental arrays
          @see ReadFromGhostsEnd()
          */
        template <typename T>
          int ReadFromGhostElemsEnd(T* arr) {
            return m_da->ReadFromGhostElemsEnd<T>(arr);
          }
        /**
          @author Hari Sundar
          Counterpart of WriteToGhostsBegin() for elemental arrays
          @see WriteToGhostsBegin()
          */
        template <typename T>
          int WriteToGhostElemsBegin ( T* arr, unsigned int dof=1) {
            return m_da->WriteToGhostElemsBegin<T>(arr, dof);
          }

        /**
          @author Hari Sundar
          Counterpart of WriteToGhostsEnd for elemental arrays
          @see WriteToGhostsEnd()
          */
        template <typename T>
          int WriteToGhostElemsEnd(T* arr, unsigned int dof=1) {
            return m_da->WriteToGhostElemsEnd<T>(arr, dof);
          }

        //@}

        std::vector<ot::TreeNode> getMinAllBlocks() {
          return m_da->getMinAllBlocks();
        }

        /**
          @name Array access functions 
          */
        //@{

        /**
          @author Hari Sundar
          @brief Returns a PETSc vector of appropriate size of the requested type.
          @param local     the local vector, a PETSc vector that may be used with the PETSc routines.
          @param isElemental true if an elemental vector is desired, 
          false for a nodal vector.
          @param isGhosted true if memory is to be allocated for ghost values.
          @param dof       the degrees of freedom for the vector. The default is 1.
          @return PETSc error code.
          */
        int createVector(Vec &local, bool isElemental, bool isGhosted, unsigned int dof=1);
        /*{
            return m_da->createVector(local, isElemental, isGhosted, dof);
          } */

        /**
          @author Hari Sundar
          @brief Similar to createVector(), except the vector is only distributed on the active processors.
          @see createVector()
          */
        int createActiveVector(Vec &local, bool isElemental, bool isGhosted, unsigned int dof=1) {
          return m_da->createActiveVector(local, isElemental, isGhosted, dof);
        }

        /**
          @author Hari Sundar
          @brief Returns a PETSc Matrix of appropriate size of the requested type.
          @param M the matrix
          @param mtype the type of matrix
          @param dof the number of degrees of freedom per node.
          */
        int createMatrix(Mat &M, MatType mtype, unsigned int dof=1); // {
          // std::cout << "HARI: " << "m_uiPre&PostGhostNodeSize: " << m_da->getPreAndPostGhostNodeSize() << std::endl; 
          // std::cout << "HARI: " << "m_uiNodeSize: " << m_da->getNodeSize() << std::endl;

          // std::cout << "creating Matrix" << std::endl;
		    //   int r = m_da->createMatrix(M, mtype, dof);
          
        //   unsigned int indices[8];
        //   std::vector<ot::MatRecord> records;
        //   ot::MatRecord mr;
		    //   // set non-dof diagonal entries to be 1
        // for ( m_da->init<ot::DA_FLAGS::ALL>(); 
        //       m_da->curr() < m_da->end<ot::DA_FLAGS::ALL>(); 
        //       m_da->next<ot::DA_FLAGS::ALL>() ) {
            
        //     m_da->getNodeIndices(indices);
        //     for (unsigned int i=0; i<8; ++i) {
        //       if ( m_ucpSkipNodeList[ indices[i] ] ) {
        //         mr.rowIdx = indices[i];
        //         mr.colIdx = indices[i];
        //         for (unsigned int j=0; j<dof; ++j) {
        //           mr.rowDim = j;
        //           mr.colDim = j;
        //           mr.val = 1.0;
        //           records.push_back(mr);
        //         } 
        //       }
        //     }
        //   }
        //   // std::cout << "setting values in Matrix" << std::endl;
        //   m_da->setValuesInMatrix(M, records, dof, ADD_VALUES);
        //   // std::cout << "done creating Matrix" << std::endl;
          
        //   return r;
        // }

        void setMatrixDiagonalForSkippedNodes(Mat M, unsigned int dof) {
          unsigned int indices[8];
          std::vector<ot::MatRecord> records;
          ot::MatRecord mr;
		  // set non-dof diagonal entries to be 1
          for ( m_da->init<ot::DA_FLAGS::ALL>(); 
              m_da->curr() < m_da->end<ot::DA_FLAGS::ALL>(); 
              m_da->next<ot::DA_FLAGS::ALL>() ) {
            
            m_da->getNodeIndices(indices);
            for (unsigned int i=0; i<8; ++i) {
              if ( m_ucpSkipNodeList[ indices[i] ] ) {
                mr.rowIdx = indices[i];
                mr.colIdx = indices[i];
                for (unsigned int j=0; j<dof; ++j) {
                  mr.rowDim = j;
                  mr.colDim = j;
                  mr.val = 1.0;
                  records.push_back(mr);
                }
              }
            }
          }
          // std::cout << "setting values in Matrix" << std::endl;
          m_da->setValuesInMatrix(M, records, dof, ADD_VALUES);
        }

        /**
          @author Hari Sundar
          @brief Similar to createMatrix, except the matrix is only distributed on the active processors.
          @see createMatrix()
          */
        int createActiveMatrix(Mat &M, MatType mtype, unsigned int dof=1) {
          return m_da->createActiveMatrix(M, mtype, dof);
        }

        /**
          @author Hari Sundar
          @brief Computes mappings between the local and global numberings for nodal buffers.
          @see setValuesInMatrix()
          Call this function only if you need to create Matrices using this mesh. This function must be called
          once before calling setValuesInMatrix(). This function should not
          be called more than once for a given mesh.
          */
        int computeLocalToGlobalMappings();

        /**
          @author Hari Sundar
          @brief Computes mappings between the local and global numberings for elemental buffers.
          This function is probably required only for developers. Typical users will not need this.
          This function should not be called more than once for a given mesh.
          */
        int computeLocalToGlobalElemMappings() {
          return m_da->computeLocalToGlobalElemMappings();
        }

        /**
          @author Hari Sundar
          @return 'true' if the function computeLocalToGlobalMappings() was called for this mesh.
          @see computeLocalToGlobalMappings()
          */
        bool computedLocalToGlobal() {
          return m_bComputedLocalToGlobal;
        }

        /**
          @author Hari Sundar
          @return 'true' if the function computeLocalToGlobalElemMappings() was called for this mesh.
          @see computeLocalToGlobalElemMappings()
          */
        bool computedLocalToGlobalElems() {
          return m_bComputedLocalToGlobalElems;
        }

        /**
          @author Hari Sundar
          @brief a wrapper for setting values into the Matrix.
          This internally calls PETSc's MatSetValues() function.
          @param mat The matrix
          @param records The values and their indices
          @param dof the number of degrees of freedom per node
          @param mode Either INSERT_VALUES or ADD_VALUES
          @return an error flag
          Call PETSc's MatAssembly routines to assemble the matrix after setting the values.
          'records' will be cleared inside the function. It would be more efficient to set values in chunks by
          calling this function multiple times with different sets of
          values instead of a single call at the end of the loop. One
          can use the size of 'records' to determine the number of
          such chunks. Calls to this function with the INSERT_VALUES and ADD_VALUES
          options cannot be mixed without intervening calls to PETSc's MatAssembly routines.
          */
        int setValuesInMatrix(Mat mat, std::vector<ot::MatRecord>& records,
            unsigned int dof, InsertMode mode); 

        /**
         * @author Hari Sundar
         * @brief zeros out specific rows of a matrix. indices are specified in local numbering and mapped to the global
         *        indices. This internally calls PETSc's
         * @param mat        the matrix
         * @param indices    the node indices that have to be zeroed out.
         * @param dof        the number of degrees of freedom per node
         * @param diag       the value to be set on the diagonal, can be 0.0
         * @param x          optional vector of solutions for zeroed rows (other entries in vector are not used)
         * @param b 	     optional vector of right hand side, that will be adjusted by provided solution
         * @return           an error flag
         */

        int zeroRowsInMatrix(Mat mat, std::vector<unsigned int>& indices, unsigned int dof,
                             double diag, Vec x, Vec b) ;

        
        /**
         * @author Hari Sundar
         * @brief  Aligns points with the DA partition so that all points will lie within local elements on any process.
         * 
         * Note that the points will be redistributed and contents will be modified.
         */ 
        int alignPointsWithDA(std::vector<double>& points, std::vector<int>& labels) {
          return m_da->alignPointsWithDA(points, labels);
        }
        
        int alignPointsWithDA(std::vector<ot::NodeAndValues<double,3>>& pts) {
          return m_da->alignPointsWithDA(pts);
        }
        
        /**
          @author Hari Sundar
          @brief Returns a std. vector of appropriate size of the requested type. 

          @param local the local vector.
          @param isElemental true if an elemental vector is desired, 
          false for a nodal vector.
          @param isGhosted true if memory is to be allocated for ghost 
          values.
          @param dof the degrees of freedom for the vector. The default is 1.
          @return PETSc error code.
          */
        template <typename T>
          int  createVector(std::vector<T> &local, bool isElemental,
              bool isGhosted, unsigned int dof=1) {
                return m_da->createVector<T>(local, isElemental, isGhosted, dof);
              }

        /**
          @author Hari Sundar
          @brief Returns a C-array of type PetscScalar from a PETSc Vec for quick local access. 
          @param in The PETSc Vec which needs to be accessed localy. 
          @param out The local C-array which is used to access data 
          localy.
          @param isElemental true if in is an elemental vector, false 
          if it is a nodal vector.
          @param isGhosted true if in contains ghost values.
          @param isReadOnly true if the buffer is required only for 
          reading, should be set to false if writes
          will be performed.
          @param dof the degrees of freedom for the vector. The default is 1.
          @see vecRestoreBuffer()
          Returns a C-array of type PetscScalar from a PETSc Vec for
          quick local access. In addition, this operation is 
          required to use the oda based indexing. vecRestoreBuffer() must be
          called when the buffer is no longer needed.
          If isReadOnly is true, this involves a simple copy of local values from in.
          The ghosts will have junk value in this case. If isReadOnly is false,
          the buffer will be zeroed out first and then the local values from in
          will be copied. The ghosts will have 0 values in this case.
          */
        int vecGetBuffer(Vec in, PetscScalar* &out, bool isElemental,
            bool isGhosted, bool isReadOnly, unsigned int dof=1);

        /**
          @author Hari Sundar
          @brief Returns a C-array of type T from a distributed std vector for quick local access. 
          @param in The std::vector which needs to be accessed localy. 
          @param out The local C-array which is used to access data 
          localy. 
          @param isElemental true if in is an elemental vector, false 
          if it is a nodal vector.
          @param isGhosted true if in contains ghost values.
          @param isReadOnly true if the buffer is required only for 
          reading, should be set to false if writes
          will be performed.
          @param dof the degrees of freedom for the vector. The default is 1.
          @see vecRestoreBuffer()
          Returns a C-array of type T from a distributed std vector 
          for quick local access. In addition, this operation is 
          required to use the oda based indexing. vecRestoreBuffer() must be 
          called when the buffer is no longer needed.
          T must have a default constructor, which should zero out the object.
          */
        template < typename T >
          int vecGetBuffer(std::vector<T> &in, T* &out, bool isElemental,
              bool isGhosted, bool isReadOnly, unsigned int dof=1) ;

        /**
          @author Hari Sundar
          @brief Restores the C-array of type PetscScalar to a PETSc Vec after quick local access. 
          @param in The PETSc Vec which was accessed localy. 
          @param out The local C-array which is used to access data 
          localy. 
          @param isElemental true if in is an elemental vector, false 
          if it is a nodal vector.
          @param isGhosted true if in contains ghost values.
          @param isReadOnly true if the buffer was used only for 
          reading, should be set to false if writes
          will be performed.
          @param dof the degrees of freedom for the vector. The default is 1.
          @see vecGetBuffer()
          Restores the C-array of type PetscScalar to a PETSc Vec after quick local access. 
          */
        int vecRestoreBuffer(Vec in, PetscScalar* out, bool isElemental, 
            bool isGhosted, bool isReadOnly, unsigned int dof=1) ;

        /**
          @author Hari Sundar
          @brief Restores the C-array of type T to a distributed std vector after quick local access. 
          @param in The std::vector which was accessed localy. 
          @param out The local C-array which is used to access data 
          localy. 
          @param isElemental true if in is an elemental vector, false 
          if it is a nodal vector.
          @param isGhosted true if in contains ghost values.
          @param isReadOnly true if the buffer was used only for 
          reading, should be set to false if writes
          will be performed.
          @param dof the degrees of freedom for the vector. The default is 1.
          @see vecGetBuffer()
          Restores the C-array of type T to a distributed std vector after quick local access. 
          */
        template < typename T >
          int vecRestoreBuffer(std::vector<T> &in, T* out, bool isElemental,
              bool isGhosted, bool isReadOnly, unsigned int dof=1) ;
        //@}

        //----------------------------------

        /**
          @name Element/Node access and iterators
          */
        //@{

        /**
          @author Hari Sundar
          @author Hari Sundar
          @brief Initializes the internal counters for a new loop. Remember that the DA
          currently only supports elemental loops.
          @param LoopType valid types are All, Local, Independent,
          Dependent, and Ghosted.

          Sample loop through the elements:

          @code
          for ( init<loopType>; curr() < end<loopType>(); next<loopType>() ) { 
        // Do whatever is required ...
        } 
        @endcode

        @see next()
        @see curr()
        @see end()
        */
        template<ot::DA_FLAGS::loopType type>
          void init();

        /**

          @author Hari Sundar
          @author Hari Sundar
          @brief Returns an index to the begining of the current loop.
          The loop needs to be initialized using a call to
          init().
          @return the index to the begining of the loop.

          Sample loop through the elements:

          @code 
          for ( init<loopType>; curr() < end<loopType>(); next<loopType>() ) { 
        // Do whatever is required ...
        } 
        @endcode 

        @see end()
        @see next()
        @see init()
        */		
        unsigned int curr() {
          return m_da->curr();
        }

        /**

          @author Hari Sundar
          @brief Returns an index to the begining of the current loop.
          The loop needs to be initialized using a call to
          init(). This also stores the current position within an
          internal structure. This can be used to re-start a loop using the FROM_STORED loopType.
          @return the index to the begining of the loop.
          @see loopType
          @see curr()            
          */
        unsigned int currWithInfo() {
          return m_da->currWithInfo();
        }

        /**
          @author Hari Sundar
          @author Hari Sundar
          @brief Returns an index to the end of the current loop.
          The loop needs to be initialized using a call to
          init().

          @return the index to the end of the loop. 

          Sample loop through the elements: 

          @code 
          for ( init<loopType>; curr() < end<loopType>(); next<loopType>() ) { 
        // Do whatever is required ...
        } 
        @endcode 

        @see curr()
        @see init()
        @see next()
        */
        template<ot::DA_FLAGS::loopType type>
          unsigned int end();

        /**
          @author Hari Sundar
          @author Hari Sundar
          @brief Returns an index to the next element of the current 
          loop. The loop needs to be initialized using a call to
          initializeLoopCounter().

          @return the index to the end of the loop.

          Sample loop through the elements:

          @code
          for ( init<loopType>; curr() < end<loopType>(); next<loopType>() ) { 
        // Do whatever is required ...
        } 
        @endcode 

        @see init()
        @see curr()
        @see end()
        */
        template<ot::DA_FLAGS::loopType type>
          unsigned int next();

        /**
          @author Hari Sundar
          @brief Returns the child number of the current element.
          @return  the child number of the current element
          */
        unsigned char getChildNumber() {
          return m_da->getChildNumber();
        }

        /**
          @author Hari Sundar
          @brief Returns the compressed representation of the octant at index i.
          @param i the index of the octant.
          @return the compressed representation of the octant at index i.
          */
        unsigned char getFlag(unsigned int i) {
          return m_da->getFlag(i);
        }

        /** 
          @author Hari Sundar
          @brief Returns true if the element specified by the index contains a hanging node.
          @param i the index to the element.
          */
        bool isHanging(unsigned int i) {
          return m_da->isHanging(i);
        }

        /**
          @author Hari Sundar
          @brief Returns true if the element/node specified by the index is a Ghost.
          @param i the index to the element/node.
          */
        bool isGhost(unsigned int i) {
          return m_da->isGhost(i);
        }

        /**
          @author Hari Sundar
          @brief Returns true if the element specified by the index corresponds to a node.
          @param i the index to the element.
          */
        bool isNode(unsigned int i) {
          return m_da->isNode(i);
        }

        /** 
          @author Hari Sundar
          @brief Returns information pertaining to which of the elements 8 nodes are hanging.
          @param i the index to the element.
          @return the bitmask specifying which of the nodes are hanging.

          Returns information pertaining to which of the elements 8 nodes are hanging.	
          */
        unsigned char getHangingNodeIndex(unsigned int i) {
          return m_da->getHangingNodeIndex(i);
        }

        /** 
          @author Hari Sundar
          @brief Returns the type mask for the given element.
          @param i the index to the element.

          Returns the type mask for the given element. The type mask is
          used to identify what kind of an element the element in 
          question is. Information can be obtained from the bits of the 
          mask. The information is stored as NHDL, where N is 1 bit to 
          identify a node, H is 1 bit to identify a hanging element, D 
          is one bit to identify a dependent element, and the remaining 
          5 bits are used to detect the level of the octant. 
          */
        unsigned char getTypeMask(unsigned int i) {
          return m_da->getTypeMask(i);
        }

        /** 
          @author Hari Sundar
          @brief Returns the level of the octant specified by the index.
          @param i the index to the element/node.
          @return the level of the octant.
          The return value is the level of the octant in the modified octree that includes 'pseudo-octants' for boundary nodes. This octree has
          a maximum depth equal to 1 more than that of the input octree used to construct the finite element mesh. Hence, the value
          returned by this function will be 1 more than the true level of the octant in the input octree.
          */
        unsigned char getLevel(unsigned int i) {
          return m_da->getLevel(i);
        }

        /**
          @author Hari Sundar
          @author Hari Sundar
          @brief Returns the indices to the nodes of the current 
          element.
          @param nodes   Indices into the nodes of the given element. Should be
          allocated by the user prior to calling.
          @return Error code.
          */ 
        int getNodeIndices(unsigned int* nodes);
        
        //@}

        /**
          @author Hari Sundar
          @return the total number of octants (local, ghosts and FOREIGN) stored on the calling processor.
          */
        unsigned int getLocalBufferSize() {
          return m_uiLocalBufferSize;
        }


        /**
          @author Hari Sundar
          @brief Call this function, if a call to getNodeIndices() 
          is skipped within the loop and if the element-to-node mappings are compressed.
          @see getNodeIndices()
          */
        void updateQuotientCounter() {
          return m_da->updateQuotientCounter();
        }

        /**
          @author Hari Sundar
          @return true if the element-to-node mappings were compressed using Goloumb-Rice encoding
          */
        bool isLUTcompressed() {
          return m_da->isLUTcompressed();
        }

        /**
          @author Hari Sundar
          @return true if the calling processor is active
          */
        bool iAmActive() {
          return m_da->iAmActive();
        }

        void printODAStatistics() {
          m_da->printODAStatistics();
        }

        void printODANodeListStatistics(char * nlistFName) {
          m_da->printODANodeListStatistics(nlistFName);
        }

      /** 
       * @author Hari Sundar
       * @date   April 2017. 
       * 
       * Support for holes on mesh
       **/
      
      void initialize_skiplist();
      void skip_current();
      void finalize_skiplist();

  };

}; // namespace ot

#include "sub_oda.tcc"
