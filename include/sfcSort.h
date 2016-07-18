//
/*
 * @author: Milinda Fernando
 * School of Computing, University of Utah
 * Created by milinda on 2/5/16.
 *
 * Contains SFC based sorting functionality for Morton and Hilbert Curve.
 *
 * */


#ifndef SFCSORTBENCH_SFCSORT_H
#define SFCSORTBENCH_SFCSORT_H

#include <iostream>
#include <vector>
#include <assert.h>
#include "hcurvedata.h"
#include "TreeNode.h"
#include "treenode2vtk.h"
#include "dendro.h"
#include "testUtils.h"
#include "ompUtils.h"

#include <mpi.h>
#include <chrono>
#include "dtypes.h"
#include "parUtils.h"

#ifdef DIM_2
    #define NUM_CHILDREN 4
    #define ROTATION_OFFSET 8
#else
    #define NUM_CHILDREN 8
    #define ROTATION_OFFSET 16
#endif



/*#define MILLISECOND_CONVERSION 1e3


*//*
 *
 * VARIABLES USED TO PROFILE THE COMMUNICATION IN SFC SORT
 *
 *
 * *//*

extern DendroIntL SEND_COUNT;
extern DendroIntL RECV_COUNT;

DendroIntL SEND_COUNT=0;
DendroIntL RECV_COUNT=0;*/

///////////////////////////////////////////////////


template <typename T>
using Iterator = typename std::vector<T>::iterator;


namespace par
{
    enum SplitterMove {NONE,MOVE_LEFT,MOVE_RIGHT};
}



template <typename T>
struct NodeInfo
{
    unsigned char rot_id;
    unsigned char lev;
    Iterator <T> begin;
    Iterator <T> end;

    NodeInfo()
    {
        rot_id=0;
        lev=0;

    }

    NodeInfo(unsigned char p_rot_id,unsigned char p_lev,Iterator <T> p_begin, Iterator <T> p_end)
    {
        rot_id=p_rot_id;
        lev=p_lev;
        begin=p_begin;
        end=p_end;

    }

};


template <typename T>
struct NodeInfo1
{
    unsigned char rot_id;
    unsigned char lev;
    DendroIntL begin;
    DendroIntL end;

    NodeInfo1()
    {
        rot_id=0;
        lev=0;

    }

    NodeInfo1(unsigned char p_rot_id,unsigned char p_lev,DendroIntL p_begin, DendroIntL p_end)
    {
        rot_id=p_rot_id;
        lev=p_lev;
        begin=p_begin;
        end=p_end;

    }

};




namespace SFC
{

    namespace seqSort
    {


        /// Function declarations.

        /*
            @param vec    The vector to be made free of duplicates.
            @param isSorted Pass 'true' if the input is sorted.

            If the vector is not sorted,
            it is first sorted before removing duplicates.
        */
        template <typename T>
        void makeVectorUnique(std::vector<T>& vec, bool isSorted) ;


        template <typename T>
        void SFC_Sort_RemoveDuplicates(std::vector<T>&pNodes,unsigned int rot_id,unsigned int pMaxDepth,bool isSorted);


        template<typename T>
        void SFC_3D_msd_sort(T *pNodes, DendroIntL n, unsigned int rot_id,unsigned int pMaxDepth);

        template<typename T>
        void SFC_3D_msd_sort_rd(T *pNodes, DendroIntL n, unsigned int rot_id, unsigned int pMaxDepthBit,
                                unsigned int pMaxDepth);

        /*
        * msd version of the bucketing function which is used in the distributed tree sort.
        *
        * */
        template<typename T>
        inline void SFC_3D_MSD_Bucketting(T *pNodes, int lev, int maxDepth,unsigned char rot_id,
                                          DendroIntL &begin, DendroIntL &end, DendroIntL *splitters);

        /*
        * msd version of the bucketing function which is used in the distributed tree sort.
        *
        * */
        template<typename T>
        inline void SFC_3D_MSD_Bucketting_rd(T *pNodes, int lev, int maxDepth,unsigned char rot_id,
                                          DendroIntL &begin, DendroIntL &end, DendroIntL *splitters);


        ////// Function definitions.


        /*
         *
         * Duplicate elements will be eliminated by a single pass after sorting is done.
         *
         * */

        template<typename T>
        void makeVectorUnique(std::vector<T>& vecT, bool isSorted) {
            if(vecT.size() < 2) { return;}
            unsigned int maxDepth=vecT[0].getMaxDepth();
            if(!isSorted) { SFC::seqSort::SFC_3D_msd_sort_rd ((&(*(vecT.begin()))),vecT.size(),0,maxDepth,maxDepth);  /*std::sort(vecT.begin(),vecT.end());*/}

            std::vector<T> tmp(vecT.size());
            //Using the array [] is faster than the vector version
            T* tmpPtr = (&(*(tmp.begin())));
            T* vecTptr = (&(*(vecT.begin())));
            tmpPtr[0] = vecTptr[0];

            unsigned int tmpSize=1;
            unsigned int vecTsz = static_cast<unsigned int>(vecT.size());

            for(unsigned int i = 1; i < vecTsz; i++) {
                if(tmpPtr[tmpSize-1] != vecTptr[i]) { // It is efficient to do this rather than marking all the elements in sorting. (Which will cause a performance degradation. )
                    tmpPtr[tmpSize] = vecTptr[i];
                    tmpSize++;
                }
            }//end for

            tmp.resize(tmpSize);
            swap(vecT, tmp);
        }//end function


        template <typename T>
        void SFC_Sort_RemoveDuplicates(std::vector<T> &pNodes,unsigned int rot_id,unsigned int pMaxDepth,bool isSorted)
        {
            if(!isSorted)
            {
                SFC_3D_msd_sort_rd((&(*(pNodes.begin()))),pNodes.size(),rot_id,pMaxDepth,pMaxDepth);
            }

            makeVectorUnique(pNodes,true);
            //std::cout<<"Seq :: pNodes: "<<pNodes.size()<<std::endl;

        }

        template<typename T>
        void SFC_3D_msd_sort(T *pNodes, DendroIntL n, unsigned int rot_id,unsigned int pMaxDepth)  {
          register unsigned int cnum;
          register unsigned int cnum_prev=0;
          unsigned int rotation=0;
          DendroIntL count[(NUM_CHILDREN+1)]={};
          pMaxDepth--;

          for (DendroIntL i=0; i<n; ++i) {
            cnum = (((pNodes[i].getZ() >> pMaxDepth) & 1u) << 2u) | (((pNodes[i].getY() >> pMaxDepth) & 1u) << 1u) | ((pNodes[i].getX() >> pMaxDepth) & 1u);
            count[cnum+1]++;
          }

          DendroIntL loc[NUM_CHILDREN];
          T unsorted[NUM_CHILDREN];
          int live = 0;

          for (unsigned int i=0; i<(NUM_CHILDREN); ++i) {
            cnum=(rotations[ROTATION_OFFSET * rot_id+i] - '0');
            (i>0)? cnum_prev = ((rotations[ROTATION_OFFSET * rot_id+i-1] - '0')+1) : cnum_prev=0;
            loc[cnum]=count[cnum_prev];
            count[cnum+1] += count[cnum_prev];
            if(loc[cnum]<n) unsorted[live] = pNodes[loc[cnum]];
            if (loc[cnum] < count[cnum+1]) {live++; /*std::cout<<i<<" Live: "<<live<<std::endl;*/}
          }
          live--;

          if(live>=0) {
              for (DendroIntL i = 0; i < n; ++i) {
                  //std::cout << i << " Live: " << live << " qqunsorted live " <<unsorted[live]<<std::endl;
                  cnum = (((unsorted[live].getZ() >> pMaxDepth) & 1u) << 2u) |
                         (((unsorted[live].getY() >> pMaxDepth) & 1u) << 1u) |
                         ((unsorted[live].getX() >> pMaxDepth) & 1u);
                  pNodes[loc[cnum]++] = unsorted[live];
                  if(loc[cnum]<n)unsorted[live] = pNodes[loc[cnum]];
                  if ((loc[cnum] == count[cnum + 1])) {
                      live--;
                  }
                  if(live<0) break;
              }
          }

          if (pMaxDepth>0) {
            for (unsigned int i=0; i<NUM_CHILDREN; i++) {
              cnum=(rotations[ROTATION_OFFSET*rot_id+i]-'0');
              (i>0)? cnum_prev = ((rotations[ROTATION_OFFSET * rot_id+i-1] - '0')+1) : cnum_prev=0;
              n = count[cnum+1] - count[cnum_prev];
              if (n > 1) {
                rotation=HILBERT_TABLE[NUM_CHILDREN * rot_id + cnum];
                SFC_3D_msd_sort(pNodes + count[cnum_prev], n,rotation,(pMaxDepth));
              } 
            }
          }
        } // msd sort


       /*
         *
         * msd version of the treeSort with remove duplicates enabled.
         * */

        template<typename T>
        void SFC_3D_msd_sort_rd(T *pNodes, DendroIntL n, unsigned int rot_id, unsigned int pMaxDepthBit,
                                unsigned int pMaxDepth)  {
            register unsigned int cnum;
            register unsigned int cnum_prev=0;
            //register unsigned int n=0;
            unsigned int rotation=0;
            DendroIntL count[(NUM_CHILDREN+2)]={};
            unsigned int lev=pMaxDepth-pMaxDepthBit;
            pMaxDepthBit--;

            count[0]=0;
            for (DendroIntL i=0; i< n; ++i) {
                if(lev==pNodes[i].getLevel())
                {
                    count[1]++;
                }else
                {
                    cnum = (((pNodes[i].getZ() >> pMaxDepthBit) & 1u) << 2u) | (((pNodes[i].getY() >> pMaxDepthBit) & 1u) << 1u) | ((pNodes[i].getX() >>pMaxDepthBit) & 1u);
                    count[cnum+2]++;
                }

            }

            DendroIntL loc[NUM_CHILDREN+1];
            T unsorted[NUM_CHILDREN+1];
            int live = 0;


            for (unsigned int i=0; i<(NUM_CHILDREN+1); ++i) {
                if(i==0)
                {
                  loc[0]=count[0];
                  count[1]+=count[0];
                  if(loc[0]<n) unsorted[live] = pNodes[loc[0]];
                  if (loc[0] < count[1]) {live++; /*std::cout<<i<<" Live: "<<live<<std::endl;*/}
                }else
                {
                    cnum=(rotations[ROTATION_OFFSET * rot_id+ i-1] - '0');
                    (i>1) ? cnum_prev = ((rotations[ROTATION_OFFSET * rot_id+i-2] - '0')+2): cnum_prev=1;
                    loc[cnum+1]=count[cnum_prev];
                    count[cnum+2] += count[cnum_prev];
                    if(loc[cnum+1]<n) unsorted[live] = pNodes[loc[cnum+1]];
                    if (loc[cnum+1] < count[cnum+2]) {live++; /*std::cout<<i<<" Live: "<<live<<std::endl;*/}
                }

            }
            live--;
            if(live>=0) {
                for (DendroIntL i = 0; i < n; ++i) {

                    if (lev == unsorted[live].getLevel()) {
                        pNodes[loc[0]++] = unsorted[live];
                        if(loc[0]<n) unsorted[live] = pNodes[loc[0]];
                        if ((loc[0] == count[1])) {
                            live--;
                        }

                    } else {

                        cnum = (((unsorted[live].getZ() >> pMaxDepthBit) & 1u) << 2u) |
                               (((unsorted[live].getY() >> pMaxDepthBit) & 1u) << 1u) |
                               ((unsorted[live].getX() >> pMaxDepthBit) & 1u);
                        pNodes[loc[(cnum + 1)]++] = unsorted[live];
                        if(loc[cnum+1]<n) unsorted[live] = pNodes[loc[cnum + 1]];
                        if ((loc[cnum + 1] == count[cnum + 2])) {
                            live--;
                        }
                    }

                    if(live<0) break;

                }
            }



            if (pMaxDepthBit > 0) {


                for (unsigned int i=1; i<(NUM_CHILDREN+1); i++) {
                    cnum=(rotations[ROTATION_OFFSET*rot_id+i-1]-'0');
                    (i>1)? cnum_prev = ((rotations[ROTATION_OFFSET * rot_id+i-2] - '0')+2) : cnum_prev=1;
                    n = count[cnum+2] - count[cnum_prev];
                    if (n > 1) {
                        rotation=HILBERT_TABLE[NUM_CHILDREN * rot_id + cnum];
                        SFC_3D_msd_sort_rd(pNodes + count[cnum_prev], n, rotation, (pMaxDepthBit), pMaxDepth);

                    }

                }
            }


        } // msd sort



        /*
         * msd version of the bucketing function which is used in the distributed tree sort.
         *
         * */
        template<typename T>
        inline void SFC_3D_MSD_Bucketting(T *pNodes, int lev, int maxDepth,unsigned char rot_id,
                                          DendroIntL &begin, DendroIntL &end, DendroIntL *splitters)
        {



            // Special case that needs to be handled when performing the load balancing.
            if ((lev >= maxDepth) || (begin == end)) {
                // Special Case when the considering level exceeds the max depth.

                for (int ii = 0; ii < NUM_CHILDREN; ii++) {
                    int index = (rotations[2 * NUM_CHILDREN * rot_id + ii] - '0');
                    int nextIndex = 0;
                    if (ii == (NUM_CHILDREN-1))
                        nextIndex = ii + 1;
                    else
                        nextIndex = (rotations[2 * NUM_CHILDREN * rot_id + ii + 1] - '0');

                    if (ii == 0) {
                        splitters[index] = begin;
                        splitters[nextIndex] = end;
                        continue;
                    }

                    splitters[nextIndex] = splitters[index];
                }
                //std::cout<<"End return "<<"maxDepth "<<maxDepth<<" Lev: "<<lev<< " Begin "<<begin <<" End "<<end<<std::endl;
                return;

            }

            register unsigned int cnum;
            register unsigned int cnum_prev=0;
            DendroIntL num_elements=0;
            unsigned int rotation=0;
            DendroIntL count[(NUM_CHILDREN+1)]={};
            //unsigned int pMaxDepth=(lev);
            //pMaxDepth--;
            unsigned int mid_bit = maxDepth - lev - 1;
            for (DendroIntL i=begin; i<end; ++i) {
                cnum = ((((pNodes[i].getZ() & (1u << mid_bit)) >> mid_bit) << 2u) | (((pNodes[i].getY() & (1u << mid_bit)) >> mid_bit) << 1u) | ((pNodes[i].getX() & (1u << mid_bit)) >> mid_bit));//(((pNodes[i].getZ() >> pMaxDepth) & 1u) << 2u) | (((pNodes[i].getY() >> pMaxDepth) & 1u) << 1u) | ((pNodes[i].getX() >> pMaxDepth) & 1u);
                count[cnum+1]++;
            }


            count[0]=begin;

            DendroIntL loc[NUM_CHILDREN];
            T unsorted[NUM_CHILDREN];
            int live = 0;

            //std::cout<<"Initial Count Ended"<<std::endl;

            for (unsigned int ii = 0; ii < NUM_CHILDREN; ii++) {
                int index = (rotations[2 * NUM_CHILDREN * rot_id + ii] - '0');
                int nextIndex = 0;
                if (ii == (NUM_CHILDREN-1))
                    nextIndex = ii + 1;
                else
                    nextIndex = (rotations[2 * NUM_CHILDREN * rot_id + ii + 1] - '0');

                if (ii == 0) {
                    splitters[index] = begin;
                }

                splitters[nextIndex] = splitters[index] + count[(index+1)];
                //std::cout<<" Spliter B:"<<index <<" "<<splitters[index]<<" Splitters E "<<nextIndex<<" "<<splitters[nextIndex]<<std::endl;

            }

            // Special case which happens only when, we have only one element in the considering bucket.
            if ((end - begin) <= 1) {
                return;
            }


            for (unsigned int  i=0; i<NUM_CHILDREN; ++i) {
                cnum=(rotations[ROTATION_OFFSET * rot_id+i] - '0');
                (i>0)? cnum_prev = ((rotations[ROTATION_OFFSET * rot_id+i-1] - '0')+1) : cnum_prev=0;
                loc[cnum]=count[cnum_prev];
                count[cnum+1] += count[cnum_prev];
                if(loc[cnum]<end) unsorted[live] = pNodes[loc[cnum]];
                if (loc[cnum] < count[cnum+1]) {live++; /*std::cout<<i<<" Live: "<<live<<std::endl;*/}
            }

            live--;

            if(live>=0) {
                for (DendroIntL i = begin; i < end; ++i) {
                    //std::cout << i << " Live: " << live << " qqunsorted live " <<unsorted[live]<<std::endl;
                    cnum = ((((unsorted[live].getZ() & (1u << mid_bit)) >> mid_bit) << 2u) |
                            (((unsorted[live].getY() & (1u << mid_bit)) >> mid_bit) << 1u) |
                            ((unsorted[live].getX() & (1u << mid_bit)) >>
                             mid_bit)); //(((unsorted[live].getZ() >> pMaxDepth) & 1u) << 2u) |  (((unsorted[live].getY() >> pMaxDepth) & 1u) << 1u) |  ((unsorted[live].getX() >> pMaxDepth) & 1u);
                    pNodes[loc[cnum]++] = unsorted[live];
                    if (loc[cnum] < end) unsorted[live] = pNodes[loc[cnum]];
                    if ((loc[cnum] == count[cnum + 1])) {
                        live--;
                    }
                    if (live < 0) break;
                }
            }

        }




        /*
        * msd version of the bucketing function which is used in the distributed tree sort.
        *
        * */
        template<typename T>
        inline void SFC_3D_MSD_Bucketting_rd(T *pNodes, int lev, int maxDepth,unsigned char rot_id,
                                             DendroIntL &begin, DendroIntL &end, DendroIntL *splitters)
        {



            // Special case that needs to be handled when performing the load balancing.
            if ((lev >= maxDepth) || (begin == end)) {
                // Special Case when the considering level exceeds the max depth.

                for (int ii = 0; ii < NUM_CHILDREN; ii++) {
                    int index = (rotations[2 * NUM_CHILDREN * rot_id + ii] - '0');
                    int nextIndex = 0;
                    if (ii == (NUM_CHILDREN-1))
                        nextIndex = ii + 1;
                    else
                        nextIndex = (rotations[2 * NUM_CHILDREN * rot_id + ii + 1] - '0');

                    if (ii == 0) {
                        splitters[index] = begin;
                        splitters[nextIndex] = end;
                        continue;
                    }

                    splitters[nextIndex] = splitters[index];
                }
                //std::cout<<"End return "<<"maxDepth "<<maxDepth<<" Lev: "<<lev<< " Begin "<<begin <<" End "<<end<<std::endl;
                return;

            }

            register unsigned int cnum;
            register unsigned int cnum_prev=0;
            DendroIntL num_elements=0;
            unsigned int rotation=0;
            DendroIntL count[(NUM_CHILDREN+2)]={};
            //unsigned int pMaxDepth=(lev);
            //pMaxDepth--;
            unsigned int mid_bit = maxDepth - lev - 1;
            count[0]=begin;
            for (DendroIntL i=begin; i<end; ++i) {
                if(lev==pNodes[i].getLevel())
                {
                    count[1]++;
                }else {
                    cnum = ((((pNodes[i].getZ() & (1u << mid_bit)) >> mid_bit) << 2u) |
                            (((pNodes[i].getY() & (1u << mid_bit)) >> mid_bit) << 1u) |
                            ((pNodes[i].getX() & (1u << mid_bit)) >>
                             mid_bit));
                    count[cnum + 2]++;
                }
            }




            DendroIntL loc[NUM_CHILDREN+1];
            T unsorted[NUM_CHILDREN+1];
            int live = 0;

            //std::cout<<"Initial Count Ended"<<std::endl;

            for (unsigned int ii = 0; ii < NUM_CHILDREN; ii++) {
                int index = (rotations[2 * NUM_CHILDREN * rot_id + ii] - '0');
                int nextIndex = 0;
                if (ii == (NUM_CHILDREN-1))
                    nextIndex = ii + 1;
                else
                    nextIndex = (rotations[2 * NUM_CHILDREN * rot_id + ii + 1] - '0');

                if (ii == 0) {
                    splitters[index] = begin;
                    splitters[nextIndex] = splitters[index]+count[1]+ count[(index+2)]; // number of elements which needs to come before the others due to level constraint.

                }else {
                    splitters[nextIndex] = splitters[index] + count[(index + 2)];
                }
                //std::cout<<" Spliter B:"<<index <<" "<<splitters[index]<<" Splitters E "<<nextIndex<<" "<<splitters[nextIndex]<<std::endl;

            }

            // Special case which happens only when, we have only one element in the considering bucket.
            if ((end - begin) <= 1) {
                return;
            }


            for (unsigned int  i=0; i<(NUM_CHILDREN+1); ++i) {
                if(i==0)
                {
                    loc[0]=count[0];
                    count[1]+=count[0];
                    if(loc[0]<end) unsorted[live] = pNodes[loc[0]];
                    if (loc[0] < count[1]) {live++; /*std::cout<<i<<" Live: "<<live<<std::endl;*/}
                }else {

                    cnum = (rotations[ROTATION_OFFSET * rot_id + i-1] - '0');
                    (i > 1) ? cnum_prev = ((rotations[ROTATION_OFFSET * rot_id + i - 2] - '0') + 2) : cnum_prev = 1;
                    loc[cnum+1] = count[cnum_prev];
                    count[cnum + 2] += count[cnum_prev];
                    if(loc[cnum+1]<end) unsorted[live] = pNodes[loc[(cnum+1)]];
                    if (loc[cnum+1] < count[cnum + 2]) { live++; /*std::cout<<i<<" Live: "<<live<<std::endl;*/}
                }
            }

            live--;

            if(live >=0) {
                for (DendroIntL i = begin; i < end; ++i) {
                    //std::cout << i << " Live: " << live << " qqunsorted live " <<unsorted[live]<<std::endl;

                    if (lev == unsorted[live].getLevel()) {
                        pNodes[loc[0]++] = unsorted[live];
                        if(loc[0]<end) unsorted[live] = pNodes[loc[0]];
                        if ((loc[0] == count[1])) {
                            live--;
                        }

                    } else {
                        cnum = ((((unsorted[live].getZ() & (1u << mid_bit)) >> mid_bit) << 2u) |
                                (((unsorted[live].getY() & (1u << mid_bit)) >> mid_bit) << 1u) |
                                ((unsorted[live].getX() & (1u << mid_bit)) >> mid_bit));
                        pNodes[loc[(cnum + 1)]++] = unsorted[live];
                        if((loc[(cnum+1)])<end) unsorted[live] = pNodes[loc[cnum + 1]];
                        if ((loc[cnum + 1] == count[cnum + 2])) {
                            live--;

                        }
                    }

                    if(live<0) break;


                }
            }

        }





    }
};


namespace SFC
{
    namespace parSort
    {



        template <typename T>
        void SFC_Sort_RemoveDuplicates(std::vector<T> &pNodes, double loadFlexibility, unsigned int pMaxDepth,bool isSorted, MPI_Comm pcomm);



        template<typename T>
        inline void SFC_3D_SplitterFix(std::vector<T>& pNodes,unsigned int pMaxDepth,double loadFlexibility,MPI_Comm comm,MPI_Comm * newComm);


        template <typename T>
        void SFC_3D_Sort(std::vector<T> &pNodes, double loadFlexibility, unsigned int pMaxDepth,MPI_Comm pcomm);





        /*
         *
         * Sort and Remove Duplicates globally.
         *
         *
         *
         * */


        template <typename T>
        void SFC_Sort_RemoveDuplicates(std::vector<T> &pNodes, double loadFlexibility, unsigned int pMaxDepth,bool isSorted, MPI_Comm pcomm)
        {

            int npes, rank;
            MPI_Comm_size(pcomm, &npes);
            MPI_Comm_rank(pcomm, &rank);


            if (!isSorted) {
                SFC_3D_Sort(pNodes,loadFlexibility,pMaxDepth,pcomm);

            }

            //Remove duplicates locally
            SFC::seqSort::makeVectorUnique<T>(pNodes,true);


            int new_rank, new_size;
            MPI_Comm new_comm;
            // very quick and dirty solution -- assert that tmpVec is non-emply at every processor (repetetive calls to splitComm2way exhaust MPI resources)
            par::splitComm2way(pNodes.empty(), &new_comm, pcomm);
            //new_comm = pcomm;
            //assert(!pNodes.empty());
            MPI_Comm_rank(new_comm, &new_rank);
            MPI_Comm_size(new_comm, &new_size);


#ifdef __DEBUG_PAR__
            MPI_Barrier(comm);
    if(!rank) {
      std::cout<<"RemDup: Stage-4 passed."<<std::endl;
    }
    MPI_Barrier(comm);
#endif

            //Checking boundaries...
            if (!pNodes.empty()) {
                T end = pNodes[pNodes.size() - 1];
                T endRecv;

                //communicate end to the next processor.
                MPI_Status status;
                par::Mpi_Sendrecv<T, T>(&end, 1, ((new_rank < (new_size - 1)) ? (new_rank + 1) : 0), 1, &endRecv,
                                        1, ((new_rank > 0) ? (new_rank - 1) : (new_size - 1)), 1, new_comm, &status);
                //Remove endRecv if it exists (There can be no more than one copy of this)
                if (new_rank) {
                    typename std::vector<T>::iterator Iter = std::find(pNodes.begin(), pNodes.end(), endRecv);
                    if (Iter != pNodes.end()) {
                        pNodes.erase(Iter);
                    }//end if found
                }//end if p not 0
            }//end if not empty


        }


        /*
         *
         * Splitter fix to reduce the network congestion when we are going for very large scale sorting.
         * This would be needed when you are going for npes > 16,000
         *
         * */

        template<typename T>
        inline void SFC_3D_SplitterFix(std::vector<T>& pNodes,unsigned int pMaxDepth,double loadFlexibility,MPI_Comm comm,MPI_Comm * newComm)
        {

#ifdef SPLITTER_SELECTION_FIX

            int rank,npes;
            MPI_Comm_rank(comm,&rank);
            MPI_Comm_size(comm,&npes);

#pragma message ("Splitter selection FIX ON")

            if(npes>NUM_NPES_THRESHOLD)
            {

                //if(!rank) std::cout<<"NUM_NPES_THRESHHOLD: "<<npes_sqrt<<std::endl;

                unsigned int npes_sqrt= 1u<<(binOp::fastLog2(npes)/2);
                unsigned int a= npes/npes_sqrt;
                unsigned int b=npes/a;

                unsigned int dim=3;
                #ifdef DIM_2
                    dim=2;
                #else
                    dim=3;
                #endif


                unsigned int firstSplitLevel = std::ceil(binOp::fastLog2(a)/(double)dim);
                unsigned int totalNumBuckets =1u << (dim* firstSplitLevel);

                //if(!rank) std::cout<<"Rank: "<<rank<<" totalNum Buckets: "<<totalNumBuckets<<std::endl;

                DendroIntL localSz=pNodes.size();
                DendroIntL globalSz=0;
                MPI_Allreduce(&localSz,&globalSz,1,MPI_LONG_LONG,MPI_SUM,comm);

                // Number of initial buckets. This should be larger than npes.

                // maintain the splitters and buckets for splitting for further splitting.
                std::vector<DendroIntL> bucketCounts;
                std::vector<NodeInfo1<T>> bucketInfo;  // Stores the buckets info of the buckets where the initial buckets was splitted.
                std::vector<DendroIntL > bucketSplitter;


                std::vector<NodeInfo1<T>> nodeStack; // rotation id stack
                NodeInfo1<T> root(0, 0, 0, pNodes.size());
                nodeStack.push_back(root);
                NodeInfo1<T> tmp = root;
                unsigned int levSplitCount = 0;

                // Used repetitively  in rotation computations.
                unsigned int hindex = 0;
                unsigned int hindexN = 0;

                unsigned int index = 0;
                //bool *updateState = new bool[pNodes.size()];
                unsigned int numLeafBuckets =0;

                unsigned int begin_loc=0;


                DendroIntL spliterstemp[(NUM_CHILDREN+1)];
                while(numLeafBuckets<totalNumBuckets) {

                    tmp = nodeStack[0];
                    nodeStack.erase(nodeStack.begin());
                    //std::cout<<"Rank "<<rank<<" nlb "<<numLeafBuckets<<" lev: "<<(int)tmp.lev<<" node_ss "<< nodeStack.size()<<std::endl;

                    //SFC::seqSort::SFC_3D_Bucketting(pNodes, tmp.lev, pMaxDepth, tmp.rot_id, tmp.begin, tmp.end, spliterstemp,updateState);
                    //SFC::seqSort::SFC_3D_MSD_Bucketting((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,  tmp.rot_id, tmp.begin, tmp.end, spliterstemp);
                    //SFC::seqSort::SFC_3D_MSD_Bucketting_rd((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,  tmp.rot_id, tmp.begin, tmp.end, spliterstemp);
#ifdef REMOVE_DUPLICATES
                    SFC::seqSort::SFC_3D_MSD_Bucketting_rd((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,  tmp.rot_id, tmp.begin, tmp.end, spliterstemp);
#else
                    SFC::seqSort::SFC_3D_MSD_Bucketting((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,  tmp.rot_id, tmp.begin, tmp.end, spliterstemp);
#endif
                    for (int i = 0; i < NUM_CHILDREN; i++) {
                        hindex = (rotations[2 * NUM_CHILDREN * tmp.rot_id + i] - '0');
                        if (i == (NUM_CHILDREN-1))
                            hindexN = i + 1;
                        else
                            hindexN = (rotations[2 * NUM_CHILDREN * tmp.rot_id + i + 1] - '0');
                        assert(spliterstemp[hindex] <= spliterstemp[hindexN]);
                        index = HILBERT_TABLE[NUM_CHILDREN * tmp.rot_id + hindex];

                        NodeInfo1<T> child(index, (tmp.lev + 1), spliterstemp[hindex], spliterstemp[hindexN]);
                        nodeStack.push_back(child);


                        /*if(spliterstemp[hindexN]-spliterstemp[hindex]>1) {
                            NodeInfo1<T> child(index, (tmp.lev + 1), spliterstemp[hindex], spliterstemp[hindexN]);
                            nodeStack.push_back(child);
                        }else if(tmp.lev<(firstSplitLevel-1))
                        {
                            NodeInfo1<T> bucket(index, (tmp.lev + 1), spliterstemp[hindex], spliterstemp[hindexN]);
                            bucketCounts.push_back((spliterstemp[hindexN] - spliterstemp[hindex]));
                            bucketSplitter.push_back(spliterstemp[hindex]);
                            bucketInfo.push_back(bucket);
                            numLeafBuckets++;

                        }*/

                        if(tmp.lev==(firstSplitLevel-1))
                        {
                            NodeInfo1<T> bucket(index, (tmp.lev + 1), spliterstemp[hindex], spliterstemp[hindexN]);
                            bucketCounts.push_back((spliterstemp[hindexN] - spliterstemp[hindex]));
                            bucketSplitter.push_back(spliterstemp[hindex]);
                            bucketInfo.push_back(bucket);
                            numLeafBuckets++;

                        }

                    }


                }



                std::vector<DendroIntL >bucketCounts_g(bucketCounts.size());
                std::vector<DendroIntL >bucketCounts_gScan(bucketCounts.size());

                par::Mpi_Allreduce<DendroIntL>(&(*(bucketCounts.begin())),&(*(bucketCounts_g.begin())),bucketCounts.size(),MPI_SUM,comm);
                //MPI_Allreduce(&bucketCounts[0], &bucketCounts_g[0], bucketCounts.size(), MPI_LONG_LONG, MPI_SUM, comm);
#ifdef DEBUG_TREE_SORT
                assert(totalNumBuckets);
#endif
                bucketCounts_gScan[0]=bucketCounts_g[0];
                for(int k=1;k<bucketCounts_g.size();k++){
                    bucketCounts_gScan[k]=bucketCounts_gScan[k-1]+bucketCounts_g[k];
                }


#ifdef DEBUG_TREE_SORT
                if(!rank) {
            for (int i = 0; i < totalNumBuckets; i++) {
                //std::cout << "Bucket count G : " << i << " : " << bucketCounts_g[i] << std::endl;
                std::cout << "Bucket initial count scan G : " << i << " : " << bucketCounts_gScan[i] << std::endl;
            }
        }
#endif
                DendroIntL * localSplitterTmp=new DendroIntL[a];
                DendroIntL idealLoadBalance=0;
                begin_loc=0;

                std::vector<unsigned int> splitBucketIndex;
                begin_loc=0;
                for(int i=0;i<a-1;i++) {
                    idealLoadBalance+=((i+1)*globalSz/a -i*globalSz/a);
                    DendroIntL toleranceLoadBalance = ((i+1)*globalSz/a -i*globalSz/a) * loadFlexibility;

                    unsigned int  loc=(std::lower_bound(bucketCounts_gScan.begin(), bucketCounts_gScan.end(), idealLoadBalance) - bucketCounts_gScan.begin());
                    //std::cout<<rank<<" Searching: "<<idealLoadBalance<<"found: "<<loc<<std::endl;

                    if(abs(bucketCounts_gScan[loc]-idealLoadBalance) > toleranceLoadBalance)
                    {

                        if(splitBucketIndex.empty()  || splitBucketIndex.back()!=loc)
                            splitBucketIndex.push_back(loc);
                        /*if(!rank)
                          std::cout<<"Bucket index :  "<<loc << " Needs a split "<<std::endl;*/
                    }else
                    {
                        if ((loc + 1) < bucketSplitter.size())
                            localSplitterTmp[i] = bucketSplitter[loc + 1];
                        else
                            localSplitterTmp[i] = bucketSplitter[loc];
                    }

                    /* if(loc+1<bucketCounts_gScan.size())
                         begin_loc=loc+1;
                     else
                         begin_loc=loc;*/

                }
                localSplitterTmp[a-1]=pNodes.size();


#ifdef DEBUG_TREE_SORT
                for(int i=0;i<splitBucketIndex.size()-1;i++)
       {
           assert(pNodes[bucketSplitter[splitBucketIndex[i]]]<pNodes[bucketSplitter[splitBucketIndex[i+1]]]);
       }
#endif


                std::vector<DendroIntL> newBucketCounts;
                std::vector<DendroIntL> newBucketCounts_g;
                std::vector<NodeInfo1<T>> newBucketInfo;
                std::vector<DendroIntL> newBucketSplitters;


                std::vector<DendroIntL> bucketCount_gMerge;
                std::vector<DendroIntL> bucketSplitterMerge;
                std::vector<NodeInfo1<T>> bucketInfoMerge;

                DendroIntL splitterTemp[(NUM_CHILDREN+1)];
                while(!splitBucketIndex.empty()) {


                    newBucketCounts.clear();
                    newBucketCounts_g.clear();
                    newBucketInfo.clear();
                    newBucketSplitters.clear();

                    bucketCount_gMerge.clear();
                    bucketSplitterMerge.clear();
                    bucketInfoMerge.clear();

                    if (bucketInfo[splitBucketIndex[0]].lev < pMaxDepth) {

                        NodeInfo1<T> tmp;
                        // unsigned int numSplitBuckets = NUM_CHILDREN * splitBucketIndex.size();

#ifdef DEBUG_TREE_SORT
                        if (!rank)
                    for (int i = 0; i < splitBucketIndex.size(); i++)
                        std::cout << "Splitter Bucket Index: " << i << "  : " << splitBucketIndex[i] << std::endl;
#endif


                        //unsigned int maxDepthBuckets = 0;

#ifdef DEBUG_TREE_SORT
                        if(!rank) {
                for (int i = 0; i <bucketSplitter.size(); i++) {
                    std::cout<<" Bucket Splitter : "<<i<<" : "<<bucketSplitter[i]<<std::endl;
                }
            }
#endif
                        for (int k = 0; k < splitBucketIndex.size(); k++) {
#ifdef DEBUG_TREE_SORT
                            if(!rank)
                          std::cout<<"Splitting Bucket index "<<splitBucketIndex[k]<<std::endl;
#endif

                            tmp = bucketInfo[splitBucketIndex[k]];
                            //SFC::seqSort::SFC_3D_Bucketting(pNodes, tmp.lev, pMaxDepth, tmp.rot_id, tmp.begin, tmp.end,splitterTemp, updateState);
                            //SFC::seqSort::SFC_3D_MSD_Bucketting((&(*(pNodes.begin()))),tmp.lev, pMaxDepth, tmp.rot_id, tmp.begin, tmp.end, splitterTemp);
                            //SFC::seqSort::SFC_3D_MSD_Bucketting_rd((&(*(pNodes.begin()))),tmp.lev, pMaxDepth, tmp.rot_id, tmp.begin, tmp.end, splitterTemp);
#ifdef REMOVE_DUPLICATES
                            SFC::seqSort::SFC_3D_MSD_Bucketting_rd((&(*(pNodes.begin()))),tmp.lev, pMaxDepth, tmp.rot_id, tmp.begin, tmp.end, splitterTemp);
#else
                            SFC::seqSort::SFC_3D_MSD_Bucketting((&(*(pNodes.begin()))),tmp.lev, pMaxDepth, tmp.rot_id, tmp.begin, tmp.end, splitterTemp);
#endif

                            for (int i = 0; i < NUM_CHILDREN; i++) {
                                hindex = (rotations[2 * NUM_CHILDREN * tmp.rot_id + i] - '0');
                                if (i == (NUM_CHILDREN-1))
                                    hindexN = i + 1;
                                else
                                    hindexN = (rotations[2 * NUM_CHILDREN * tmp.rot_id + i + 1] - '0');

                                //newBucketCounts[NUM_CHILDREN * k + i] = (splitterTemp[hindexN] - splitterTemp[hindex]);
                                newBucketCounts.push_back((splitterTemp[hindexN] - splitterTemp[hindex]));

                                index = HILBERT_TABLE[NUM_CHILDREN * tmp.rot_id + hindex];
                                NodeInfo1<T> bucket(index, (tmp.lev + 1), splitterTemp[hindex], splitterTemp[hindexN]);
//                          newBucketInfo[NUM_CHILDREN * k + i] = bucket;
//                          newBucketSplitters[NUM_CHILDREN * k + i] = splitterTemp[hindex];
                                newBucketInfo.push_back(bucket);
                                newBucketSplitters.push_back(splitterTemp[hindex]);
#ifdef DEBUG_TREE_SORT
                                assert(pNodes[splitterTemp[hindex]]<=pNodes[splitterTemp[hindexN]]);
#endif
                            }

                        }

                        newBucketCounts_g.resize(newBucketCounts.size());
                        par::Mpi_Allreduce(&(*(newBucketCounts.begin())),&(*(newBucketCounts_g.begin())),newBucketCounts.size(),MPI_SUM,comm);
                        //MPI_Allreduce(&newBucketCounts[0], &newBucketCounts_g[0], newBucketCounts.size(), MPI_LONG_LONG, MPI_SUM, comm);


#ifdef DEBUG_TREE_SORT
                        for(int i=0;i<newBucketSplitters.size()-1;i++)
                       {
                           assert(pNodes[newBucketSplitters[i]]<=pNodes[newBucketSplitters[i+1]]);
                       }
#endif



#ifdef DEBUG_TREE_SORT
                        for (int k = 0; k < splitBucketIndex.size(); k++) {
                    unsigned int sum = 0;
                    for (int i = 0; i < NUM_CHILDREN; i++)
                        sum += newBucketCounts_g[NUM_CHILDREN * k + i];
                    if (!rank) if (bucketCounts_g[splitBucketIndex[k]] != sum) {
                      assert(false);
                    }

                }
#endif

                        //int count = 0;
                        for (int k = 0; k < splitBucketIndex.size(); k++) {

                            unsigned int bucketBegin = 0;

                            (k == 0) ? bucketBegin = 0 : bucketBegin = splitBucketIndex[k - 1] + 1;


                            for (int i = bucketBegin; i < splitBucketIndex[k]; i++) {

                                bucketCount_gMerge.push_back(bucketCounts_g[i]);
                                bucketInfoMerge.push_back(bucketInfo[i]);
                                bucketSplitterMerge.push_back(bucketSplitter[i]);

                            }

                            tmp = bucketInfo[splitBucketIndex[k]];
                            for (int i = 0; i < NUM_CHILDREN; i++) {
                                bucketCount_gMerge.push_back(newBucketCounts_g[NUM_CHILDREN * k + i]);
                                bucketInfoMerge.push_back(newBucketInfo[NUM_CHILDREN * k + i]);
                                bucketSplitterMerge.push_back(newBucketSplitters[NUM_CHILDREN * k + i]);

                            }


                            if (k == (splitBucketIndex.size() - 1)) {
                                for (int i = splitBucketIndex[k] + 1; i < bucketCounts_g.size(); i++) {
                                    bucketCount_gMerge.push_back(bucketCounts_g[i]);
                                    bucketInfoMerge.push_back(bucketInfo[i]);
                                    bucketSplitterMerge.push_back(bucketSplitter[i]);
                                }


                            }


                        }

                        std::swap(bucketCounts_g,bucketCount_gMerge);
                        std::swap(bucketInfo,bucketInfoMerge);
                        std::swap(bucketSplitter,bucketSplitterMerge);


                        bucketCounts_gScan.resize(bucketCounts_g.size());

                        bucketCounts_gScan[0] = bucketCounts_g[0];
                        for (int k = 1; k < bucketCounts_g.size(); k++) {
                            bucketCounts_gScan[k] = bucketCounts_gScan[k - 1] + bucketCounts_g[k];
                        }
#ifdef DEBUG_TREE_SORT
                        assert(bucketCounts_gScan.back()==globalSz);
#endif
                        splitBucketIndex.clear();
#ifdef DEBUG_TREE_SORT
                        for(int i=0;i<bucketSplitter.size()-2;i++)
                    {
                        std::cout<<"Bucket Splitter : "<<bucketSplitter[i]<<std::endl;
                        assert( bucketSplitter[i+1]!=(pNodes.size()) && pNodes[bucketSplitter[i]]<=pNodes[bucketSplitter[i+1]]);
                    }
#endif

                        idealLoadBalance = 0;
                        //begin_loc=0;
                        for (unsigned int i = 0; i < a-1; i++) {
                            idealLoadBalance += ((i + 1) * globalSz / a - i * globalSz / a);
                            DendroIntL toleranceLoadBalance = (((i + 1) * globalSz / a - i * globalSz / a) * loadFlexibility);
                            unsigned int loc = (std::lower_bound(bucketCounts_gScan.begin(), bucketCounts_gScan.end(), idealLoadBalance) -
                                                bucketCounts_gScan.begin());

                            if (abs(bucketCounts_gScan[loc] - idealLoadBalance) > toleranceLoadBalance) {
                                if (splitBucketIndex.empty() || splitBucketIndex.back() != loc)
                                    splitBucketIndex.push_back(loc);


                            } else {
                                if ((loc + 1) < bucketSplitter.size())
                                    localSplitterTmp[i] = bucketSplitter[loc + 1];
                                else
                                    localSplitterTmp[i] = bucketSplitter[loc];

                            }

                            /* if(loc+1<bucketCounts_gScan.size())
                                 begin_loc=loc+1;
                             else
                                 begin_loc=loc;*/

                        }
                        localSplitterTmp[a-1]=pNodes.size();

                    } else {
                        //begin_loc=0;
                        idealLoadBalance = 0;
                        for (unsigned int i = 0; i < a-1; i++) {

                            idealLoadBalance += ((i + 1) * globalSz / a - i * globalSz / a);
                            //DendroIntL toleranceLoadBalance = ((i + 1) * globalSz / npes - i * globalSz / npes) * loadFlexibility;
                            unsigned int loc = (
                                    std::lower_bound(bucketCounts_gScan.begin(), bucketCounts_gScan.end(), idealLoadBalance) -
                                    bucketCounts_gScan.begin());


                            if ((loc + 1) < bucketSplitter.size())
                                localSplitterTmp[i] = bucketSplitter[loc + 1];
                            else
                                localSplitterTmp[i] = bucketSplitter[loc];

                            /* if(loc+1<bucketCounts_gScan.size())
                                 begin_loc=loc+1;
                             else
                                 begin_loc=loc;*/

                        }
                        localSplitterTmp[a-1]=pNodes.size();



                        break;
                    }

                }

                bucketCount_gMerge.clear();
                bucketCounts.clear();
                bucketCounts_gScan.clear();
                bucketInfoMerge.clear();
                bucketInfo.clear();
                bucketCounts_g.clear();
                bucketSplitter.clear();
                bucketSplitterMerge.clear();
                newBucketCounts.clear();
                newBucketCounts_g.clear();
                newBucketInfo.clear();
                newBucketSplitters.clear();



#ifdef DEBUG_TREE_SORT
                if(!rank) {
                    for (int i = 0; i <a; i++) {
                        //std::cout << "Bucket count G : " << i << " : " << bucketCounts_g[i] << std::endl;
                        std::cout << "Local Splitter Tmp : " << i << " : " << localSplitterTmp[i] << std::endl;
                    }
                }
#endif
                unsigned int * sendCnt= new unsigned int[npes];
                for(int i=0;i<npes;i++)
                    sendCnt[i]=0;

                unsigned int sendCnt_grain=0;

                for(unsigned int i=0;i<a;i++)
                {
                    //for(unsigned int j=0;j<b;j++)
                    //{
                       (i>0)? sendCnt_grain=(localSplitterTmp[i]-localSplitterTmp[i-1]):sendCnt_grain=localSplitterTmp[0];
                       sendCnt[(i*b+ (rank%b))]=sendCnt_grain;//(((j+1)*sendCnt_grain/b) - (j*sendCnt_grain/b));
                    //}
                }


#ifdef DEBUG_TREE_SORT
                if(!rank) {
                    for (int i = 0; i <npes; i++) {
                        //std::cout << "Bucket count G : " << i << " : " << bucketCounts_g[i] << std::endl;
                        std::cout << "Send Cnt : " << i << " : " << sendCnt[i] << std::endl;
                    }
                }
#endif
                unsigned int * recvCnt=new unsigned int [npes];
                par::Mpi_Alltoall(sendCnt,recvCnt,1,comm);
                /* if(rank)
                 {
                     for(int i=0;i<npes;i++)
                     {
                         std::cout<<sendCnt[i]<<" ,";
                     }
                     std::cout<<std::endl;
                 }*/
                unsigned int * sendDspl=new unsigned int [npes];
                unsigned int * recvDspl=new unsigned int [npes];
                sendDspl[0]=0;
                recvDspl[0]=0;
                omp_par::scan(sendCnt,sendDspl,npes);
                omp_par::scan(recvCnt,recvDspl,npes);
                std::vector<T> pNodesTmp(recvDspl[npes-1]+recvCnt[npes-1]);
                //par::Mpi_Alltoallv_Kway(&(*(pNodes.begin())),(int*)sendCnt,(int*)sendDspl,&(*(pNodesTmp.begin())),(int *)recvCnt,(int*)recvDspl,comm);
                par::Mpi_Alltoallv_sparse(&(*(pNodes.begin())),(int*)sendCnt,(int*)sendDspl,&(*(pNodesTmp.begin())),(int *)recvCnt,(int*)recvDspl,comm);
                //std::cout<<"All to all sparse ended"<<std::endl;
                delete[](sendCnt);
                delete[](recvCnt);
                delete[](sendDspl);
                delete[](recvDspl);
                delete[](localSplitterTmp);
                //delete[](updateState);

                //std::cout<<"Splitter Fix : "<<rank<<" pNodesTmp: "<<pNodesTmp.size()<<std::endl;
                //std::swap(pNodes,pNodesTmp);
                pNodes=pNodesTmp;
                pNodesTmp.clear();


                unsigned int col=(rank / b);
                MPI_Comm_split(comm,col,rank,newComm);



            }

#endif

            return;


        }


        /*
         *
         * Distributed version of the tree sort.
         *
         * */
        template <typename T>
        void SFC_3D_Sort(std::vector<T> &pNodes, double loadFlexibility, unsigned int pMaxDepth,MPI_Comm pcomm) {




//            MPI_Comm iActive;
//            par::splitComm2way(pNodes.empty(),&iActive,pcomm);

            MPI_Comm comm=pcomm;


            int rank, npes;

            //std::vector<double> stats_sf; // all reduce is used to fill this up due to the splitter comm.
            SFC_3D_SplitterFix(pNodes,pMaxDepth,loadFlexibility,pcomm,&comm);


            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &npes);

            //std::cout<<"rank: "<<rank<<" of "<<npes <<" pNodes: "<<pNodes.size()<<std::endl;
            //auto splitterCalculation_start=std::chrono::system_clock::now();//MPI_Wtime();
            unsigned int dim=3;
            #ifdef DIM_2
                dim=2;
            #else
                dim=3;
            #endif


            unsigned int firstSplitLevel = std::ceil(binOp::fastLog2(npes)/(double)(dim));

            unsigned int totalNumBuckets =1u << (dim * firstSplitLevel);
            DendroIntL localSz=pNodes.size();
            DendroIntL globalSz=0;
            MPI_Allreduce(&localSz,&globalSz,1,MPI_LONG_LONG,MPI_SUM,comm);

            //if(!rank) std::cout<<"First Split Level : "<<firstSplitLevel<<" Total number of buckets: "<<totalNumBuckets <<std::endl;

            //if(!rank) std::cout<<"NUM_CHILDREN: "<<NUM_CHILDREN<<std::endl;

            // Number of initial buckets. This should be larger than npes.

            // maintain the splitters and buckets for splitting for further splitting.
            std::vector<DendroIntL> bucketCounts;
            std::vector<NodeInfo1<T>> bucketInfo;  // Stores the buckets info of the buckets where the initial buckets was splitted.
            std::vector<DendroIntL > bucketSplitter;


            std::vector<NodeInfo1<T>> nodeStack; // rotation id stack
            NodeInfo1<T> root(0, 0, 0, pNodes.size());
            nodeStack.push_back(root);
            NodeInfo1<T> tmp = root;
            unsigned int levSplitCount = 0;

            // Used repetitively  in rotation computations.
            unsigned int hindex = 0;
            unsigned int hindexN = 0;

            unsigned int index = 0;
            //bool *updateState = new bool[pNodes.size()];
            unsigned int numLeafBuckets =0;

            unsigned int begin_loc=0;
              /*
             * We need to split the array into buckets until it is we number of buckets is slightly higher than the number of processors.
             */


            // 1. ===================Initial Splitting Start===============================
            DendroIntL spliterstemp[(NUM_CHILDREN+1)];
            //DendroIntL spliterstemp_msd[(NUM_CHILDREN+1)];
            while(numLeafBuckets<totalNumBuckets) {

                tmp = nodeStack[0];
                nodeStack.erase(nodeStack.begin());
                //std::cout<<"Rank "<<rank<<" nlb "<<numLeafBuckets<<" lev: "<<(int)tmp.lev<<" node_ss "<< nodeStack.size()<<std::endl;

#ifdef REMOVE_DUPLICATES
                #pragma message ("Remove Duplicates ON")
                SFC::seqSort::SFC_3D_MSD_Bucketting_rd((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,  tmp.rot_id, tmp.begin, tmp.end, spliterstemp);
#else
                SFC::seqSort::SFC_3D_MSD_Bucketting((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,  tmp.rot_id, tmp.begin, tmp.end, spliterstemp);
#endif



                /*if(!rank)
                    for(int w=0;w<(NUM_CHILDREN+1);w++)
                    {
                        std::cout<<"Splitter i: "<<w<<" Old: "<<spliterstemp[w]<<" msd : "<<spliterstemp_msd[w]<<std::endl;
                    }
                */

                for (int i = 0; i < NUM_CHILDREN; i++) {
                    hindex = (rotations[2 * NUM_CHILDREN * tmp.rot_id + i] - '0');
                    if (i == (NUM_CHILDREN-1))
                        hindexN = i + 1;
                    else
                        hindexN = (rotations[2 * NUM_CHILDREN * tmp.rot_id + i + 1] - '0');
                    assert(spliterstemp[hindex] <= spliterstemp[hindexN]);
                    index = HILBERT_TABLE[NUM_CHILDREN * tmp.rot_id + hindex];

                    NodeInfo1<T> child(index, (tmp.lev + 1), spliterstemp[hindex], spliterstemp[hindexN]);
                    nodeStack.push_back(child);


                    if(tmp.lev==(firstSplitLevel-1))
                    {
                        NodeInfo1<T> bucket(index, (tmp.lev + 1), spliterstemp[hindex], spliterstemp[hindexN]);
                        bucketCounts.push_back((spliterstemp[hindexN] - spliterstemp[hindex]));
                        bucketSplitter.push_back(spliterstemp[hindex]);
                        bucketInfo.push_back(bucket);
                        numLeafBuckets++;

                    }

                }


            }



#ifdef DEBUG_TREE_SORT
            for(int i=0;i<bucketSplitter.size()-2;i++)
            {
                std::cout<<"Bucket Splitter : "<<bucketSplitter[i]<<std::endl;
                //assert(pNodes[bucketSplitter[i]]<=pNodes[bucketSplitter[i+1]]);
            }
#endif


           // #ifdef DEBUG_TREE_SORT
            //std::cout<<rank<<" Initial Splitter Calculation ended "<<numLeafBuckets<<std::endl;
            //#endif

            //1=================== Initial Splitting END=========================================================================


            // (2) =================== Pick npes splitters form the bucket splitters.
            std::vector<DendroIntL >bucketCounts_g(bucketCounts.size());
            std::vector<DendroIntL >bucketCounts_gScan(bucketCounts.size());

            //auto ts_allReduce_start = std::chrono::system_clock::now();
            par::Mpi_Allreduce<DendroIntL>(&(*(bucketCounts.begin())),&(*(bucketCounts_g.begin())),bucketCounts.size(),MPI_SUM,comm);
            //auto ts_allReduce_end = std::chrono::system_clock::now();

            //double allReduceTime=(std::chrono::duration_cast<std::chrono::milliseconds>((ts_allReduce_end-ts_allReduce_start)).count())/(MILLISECOND_CONVERSION);

           /* if(!rank)
                std::cout<<"All Reduction Time for : "<<npes<<"  : "<<allReduceTime<<std::endl;*/

            //MPI_Allreduce(&bucketCounts[0], &bucketCounts_g[0], bucketCounts.size(), MPI_LONG_LONG, MPI_SUM, comm);
            //std::cout<<"All to all ended. "<<rank<<std::endl;
#ifdef DEBUG_TREE_SORT
            assert(totalNumBuckets);
#endif
            bucketCounts_gScan[0]=bucketCounts_g[0];
            for(int k=1;k<bucketCounts_g.size();k++){
               bucketCounts_gScan[k]=bucketCounts_gScan[k-1]+bucketCounts_g[k];
            }

#ifdef DEBUG_TREE_SORT
       if(!rank) {
            for (int i = 0; i < totalNumBuckets; i++) {
                //std::cout << "Bucket count G : " << i << " : " << bucketCounts_g[i] << std::endl;
                std::cout << "Bucket initial count scan G : " << i << " : " << bucketCounts_gScan[i] << std::endl;
            }
        }
#endif

#ifdef DEBUG_TREE_SORT
            assert(bucketCounts_gScan.back()==globalSz);
#endif
            DendroIntL* localSplitter=new DendroIntL[npes];
            std::vector<unsigned int> splitBucketIndex;
            DendroIntL idealLoadBalance=0;
            //begin_loc=0;
            for(int i=0;i<npes-1;i++) {
                idealLoadBalance+=((i+1)*globalSz/npes -i*globalSz/npes);
                DendroIntL toleranceLoadBalance = ((i+1)*globalSz/npes -i*globalSz/npes) * loadFlexibility;

                unsigned int  loc=(std::lower_bound(bucketCounts_gScan.begin(), bucketCounts_gScan.end(), idealLoadBalance) - bucketCounts_gScan.begin());
                //std::cout<<rank<<" Searching: "<<idealLoadBalance<<"found: "<<loc<<std::endl;

                if(abs(bucketCounts_gScan[loc]-idealLoadBalance) > toleranceLoadBalance)
                {

                    if(splitBucketIndex.empty()  || splitBucketIndex.back()!=loc)
                        splitBucketIndex.push_back(loc);
                     /*if(!rank)
                       std::cout<<"Bucket index :  "<<loc << " Needs a split "<<std::endl;*/
                }else
                {
                    if ((loc + 1) < bucketSplitter.size())
                        localSplitter[i] = bucketSplitter[loc + 1];
                    else
                        localSplitter[i] = bucketSplitter[loc];
                }

                /* if(loc+1<bucketCounts_gScan.size())
                     begin_loc=loc+1;
                 else
                     begin_loc=loc;*/

            }
            localSplitter[npes-1]=pNodes.size();


#ifdef DEBUG_TREE_SORT
            for(int i=0;i<splitBucketIndex.size()-1;i++)
       {
           assert(pNodes[bucketSplitter[splitBucketIndex[i]]]<pNodes[bucketSplitter[splitBucketIndex[i+1]]]);
       }
#endif


            std::vector<DendroIntL> newBucketCounts;
            std::vector<DendroIntL> newBucketCounts_g;
            std::vector<NodeInfo1<T>> newBucketInfo;
            std::vector<DendroIntL> newBucketSplitters;


            std::vector<DendroIntL> bucketCount_gMerge;
            std::vector<DendroIntL> bucketSplitterMerge;
            std::vector<NodeInfo1<T>> bucketInfoMerge;

            DendroIntL splitterTemp[(NUM_CHILDREN+1)];
            while(!splitBucketIndex.empty()) {


                newBucketCounts.clear();
                newBucketCounts_g.clear();
                newBucketInfo.clear();
                newBucketSplitters.clear();

                bucketCount_gMerge.clear();
                bucketSplitterMerge.clear();
                bucketInfoMerge.clear();

                if (bucketInfo[splitBucketIndex[0]].lev < pMaxDepth) {

                    NodeInfo1<T> tmp;
                   // unsigned int numSplitBuckets = NUM_CHILDREN * splitBucketIndex.size();

#ifdef DEBUG_TREE_SORT
                    if (!rank)
                    for (int i = 0; i < splitBucketIndex.size(); i++)
                        std::cout << "Splitter Bucket Index: " << i << "  : " << splitBucketIndex[i] << std::endl;
#endif

                    //unsigned int maxDepthBuckets = 0;

#ifdef DEBUG_TREE_SORT
                    if(!rank) {
                for (int i = 0; i <bucketSplitter.size(); i++) {
                    std::cout<<" Bucket Splitter : "<<i<<" : "<<bucketSplitter[i]<<std::endl;
                }
            }
#endif
                    for (int k = 0; k < splitBucketIndex.size(); k++) {

                        tmp = bucketInfo[splitBucketIndex[k]];
#ifdef DEBUG_TREE_SORT
                        if(!rank)
                          std::cout<<"Splitting Bucket index "<<splitBucketIndex[k]<<"begin: "<<tmp.begin <<" end: "<<tmp.end<<" rot_id: "<< (int)tmp.rot_id<<std::endl;
#endif


                        //std::vector<T> pNodes_cpy=pNodes;
                        //SFC::seqSort::SFC_3D_Bucketting(pNodes_cpy, tmp.lev, pMaxDepth, tmp.rot_id, tmp.begin, tmp.end,splitterTemp, updateState);
                        //SFC::seqSort::SFC_3D_MSD_Bucketting((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,tmp.rot_id, tmp.begin, tmp.end, splitterTemp);
                        //SFC::seqSort::SFC_3D_MSD_Bucketting_rd((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,tmp.rot_id, tmp.begin, tmp.end, splitterTemp);
#ifdef REMOVE_DUPLICATES
                        #pragma message ("Remove Duplicates ON")
                        SFC::seqSort::SFC_3D_MSD_Bucketting_rd((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,tmp.rot_id, tmp.begin, tmp.end, splitterTemp);
#else
                        SFC::seqSort::SFC_3D_MSD_Bucketting((&(*(pNodes.begin()))),tmp.lev, pMaxDepth,tmp.rot_id, tmp.begin, tmp.end, splitterTemp);
#endif

                        /*if(!rank)
                            for(int w=0;w<(NUM_CHILDREN+1);w++)
                            {
                                std::cout<<"Splitter i: "<<w<<" Old: "<<splitterTemp[w]<<" msd : "<<spliterstemp_msd[w]<<std::endl;
                            }*/


                        for (int i = 0; i < NUM_CHILDREN; i++) {
                            hindex = (rotations[2 * NUM_CHILDREN * tmp.rot_id + i] - '0');
                            if (i == (NUM_CHILDREN-1))
                                hindexN = i + 1;
                            else
                                hindexN = (rotations[2 * NUM_CHILDREN * tmp.rot_id + i + 1] - '0');

                            //newBucketCounts[NUM_CHILDREN * k + i] = (splitterTemp[hindexN] - splitterTemp[hindex]);
                            newBucketCounts.push_back((splitterTemp[hindexN] - splitterTemp[hindex]));

                            index = HILBERT_TABLE[NUM_CHILDREN * tmp.rot_id + hindex];
                            NodeInfo1<T> bucket(index, (tmp.lev + 1), splitterTemp[hindex], splitterTemp[hindexN]);
//                          newBucketInfo[NUM_CHILDREN * k + i] = bucket;
//                          newBucketSplitters[NUM_CHILDREN * k + i] = splitterTemp[hindex];
                            newBucketInfo.push_back(bucket);
                            newBucketSplitters.push_back(splitterTemp[hindex]);
#ifdef DEBUG_TREE_SORT
                            assert(pNodes[splitterTemp[hindex]]<=pNodes[splitterTemp[hindexN]]);
#endif
                        }

                    }

                    newBucketCounts_g.resize(newBucketCounts.size());
                    par::Mpi_Allreduce(&(*(newBucketCounts.begin())),&(*(newBucketCounts_g.begin())),newBucketCounts.size(),MPI_SUM,comm);
                    //MPI_Allreduce(&newBucketCounts[0], &newBucketCounts_g[0], newBucketCounts.size(), MPI_LONG_LONG, MPI_SUM,comm);


#ifdef DEBUG_TREE_SORT
                    for(int i=0;i<newBucketSplitters.size()-1;i++)
                       {
                           assert(pNodes[newBucketSplitters[i]]<=pNodes[newBucketSplitters[i+1]]);
                       }
#endif



#ifdef DEBUG_TREE_SORT
                    for (int k = 0; k < splitBucketIndex.size(); k++) {
                    unsigned int sum = 0;
                    for (int i = 0; i < NUM_CHILDREN; i++)
                        sum += newBucketCounts_g[NUM_CHILDREN * k + i];
                    if (!rank) if (bucketCounts_g[splitBucketIndex[k]] != sum) {
                      assert(false);
                    }

                }
#endif

                    //int count = 0;
                    for (int k = 0; k < splitBucketIndex.size(); k++) {

                        unsigned int bucketBegin = 0;

                        (k == 0) ? bucketBegin = 0 : bucketBegin = splitBucketIndex[k - 1] + 1;


                        for (int i = bucketBegin; i < splitBucketIndex[k]; i++) {

                            bucketCount_gMerge.push_back(bucketCounts_g[i]);
                            bucketInfoMerge.push_back(bucketInfo[i]);
                            bucketSplitterMerge.push_back(bucketSplitter[i]);

                        }

                        tmp = bucketInfo[splitBucketIndex[k]];
                        for (int i = 0; i < NUM_CHILDREN; i++) {
                            bucketCount_gMerge.push_back(newBucketCounts_g[NUM_CHILDREN * k + i]);
                            bucketInfoMerge.push_back(newBucketInfo[NUM_CHILDREN * k + i]);
                            bucketSplitterMerge.push_back(newBucketSplitters[NUM_CHILDREN * k + i]);

                        }


                        if (k == (splitBucketIndex.size() - 1)) {
                            for (int i = splitBucketIndex[k] + 1; i < bucketCounts_g.size(); i++) {
                                bucketCount_gMerge.push_back(bucketCounts_g[i]);
                                bucketInfoMerge.push_back(bucketInfo[i]);
                                bucketSplitterMerge.push_back(bucketSplitter[i]);
                            }


                        }


                    }

                    std::swap(bucketCounts_g,bucketCount_gMerge);
                    std::swap(bucketInfo,bucketInfoMerge);
                    std::swap(bucketSplitter,bucketSplitterMerge);


                    bucketCounts_gScan.resize(bucketCounts_g.size());

                    bucketCounts_gScan[0] = bucketCounts_g[0];
                    for (int k = 1; k < bucketCounts_g.size(); k++) {
                       bucketCounts_gScan[k] = bucketCounts_gScan[k - 1] + bucketCounts_g[k];
                    }
#ifdef DEBUG_TREE_SORT
                    assert(bucketCounts_gScan.back()==globalSz);
#endif
                    splitBucketIndex.clear();
#ifdef DEBUG_TREE_SORT
                    for(int i=0;i<bucketSplitter.size()-2;i++)
                    {
                        std::cout<<"Bucket Splitter : "<<bucketSplitter[i]<<std::endl;
                        assert( bucketSplitter[i+1]!=(pNodes.size()) && pNodes[bucketSplitter[i]]<=pNodes[bucketSplitter[i+1]]);
                    }
#endif

                    idealLoadBalance = 0;
                    //begin_loc=0;
                    for (unsigned int i = 0; i < npes-1; i++) {
                        idealLoadBalance += ((i + 1) * globalSz / npes - i * globalSz / npes);
                        DendroIntL toleranceLoadBalance = (((i + 1) * globalSz / npes - i * globalSz / npes) * loadFlexibility);
                        unsigned int loc = (std::lower_bound(bucketCounts_gScan.begin(), bucketCounts_gScan.end(), idealLoadBalance) -
                                            bucketCounts_gScan.begin());

                        if (abs(bucketCounts_gScan[loc] - idealLoadBalance) > toleranceLoadBalance) {
                            if (splitBucketIndex.empty() || splitBucketIndex.back() != loc)
                                splitBucketIndex.push_back(loc);


                        } else {
                            if ((loc + 1) < bucketSplitter.size())
                                localSplitter[i] = bucketSplitter[loc + 1];
                            else
                                localSplitter[i] = bucketSplitter[loc];

                        }

                        /* if(loc+1<bucketCounts_gScan.size())
                             begin_loc=loc+1;
                         else
                             begin_loc=loc;*/

                    }
                    localSplitter[npes-1]=pNodes.size();

                } else {
                    //begin_loc=0;
                    idealLoadBalance = 0;
                    for (unsigned int i = 0; i < npes-1; i++) {

                        idealLoadBalance += ((i + 1) * globalSz / npes - i * globalSz / npes);
                        //DendroIntL toleranceLoadBalance = ((i + 1) * globalSz / npes - i * globalSz / npes) * loadFlexibility;
                        unsigned int loc = (
                                std::lower_bound(bucketCounts_gScan.begin(), bucketCounts_gScan.end(), idealLoadBalance) -
                                bucketCounts_gScan.begin());


                        if ((loc + 1) < bucketSplitter.size())
                            localSplitter[i] = bucketSplitter[loc + 1];
                        else
                            localSplitter[i] = bucketSplitter[loc];

                        /* if(loc+1<bucketCounts_gScan.size())
                             begin_loc=loc+1;
                         else
                             begin_loc=loc;*/

                    }
                    localSplitter[npes-1]=pNodes.size();



                    break;
                }

            }

            //auto splitterCalculation_end=std::chrono::system_clock::now();//MPI_Wtime();

            bucketCount_gMerge.clear();
            bucketCounts.clear();
            bucketCounts_gScan.clear();
            bucketInfoMerge.clear();
            bucketInfo.clear();
            bucketCounts_g.clear();
            bucketSplitter.clear();
            bucketSplitterMerge.clear();
            newBucketCounts.clear();
            newBucketCounts_g.clear();
            newBucketInfo.clear();
            newBucketSplitters.clear();


//#ifdef DEBUG_TREE_SORT
           // if(!rank) std::cout<<"Splitter Calculation ended "<<std::endl;
//#endif

#ifdef DEBUG_TREE_SORT
            for(int i=0;i<npes;i++)
        {
            for(int j=i+1 ;j<npes -1;j++)
                assert(pNodes[localSplitter[i]]<=pNodes[localSplitter[j]]);
        }
#endif


#ifdef DEBUG_TREE_SORT
            if(!rank)
        {
            for(int i=0;i<npes;i++)
                std::cout<<"Rank "<<rank<<" Local Splitter: "<<i<<": "<<localSplitter[i]<<std::endl;
        }
#endif

// 3. All to all communication

            //auto all2all_start=std::chrono::system_clock::now();//MPI_Wtime();

            int * sendCounts = new  int[npes];
            int * recvCounts = new  int[npes];


            sendCounts[0] = localSplitter[0];

            for(unsigned int i=1;i<npes; ++i)
            {
                sendCounts[i] = localSplitter[i] - localSplitter[i-1];
            }



            par::Mpi_Alltoall(sendCounts,recvCounts,1,comm);
            //MPI_Alltoall(sendCounts, 1, MPI_INT,recvCounts,1,MPI_INT,comm);
            //std::cout<<"rank "<<rank<<" MPI_ALL TO ALL END"<<std::endl;

            int * sendDispl =new  int [npes];
            int * recvDispl =new  int [npes];


            sendDispl[0] = 0;
            recvDispl[0] = 0;



#ifdef DEBUG_TREE_SORT
            /*if (!rank)*/ std::cout << rank << " : send = " << sendCounts[0] << ", " << sendCounts[1] << std::endl;
        /*if (!rank)*/ std::cout << rank << " : recv = " << recvCounts[0] << ", " << recvCounts[1] << std::endl;

       /* if (!rank) std::cout << rank << " : send offset  = " << sendDispl[0] << ", " << sendDispl[1] << std::endl;
        if (!rank) std::cout << rank << " : recv offset  = " << recvDispl[0] << ", " << recvDispl[1] << std::endl;*/
#endif


            omp_par::scan(sendCounts,sendDispl,npes);
            omp_par::scan(recvCounts,recvDispl,npes);
            std::vector<T> pNodesRecv(recvDispl[npes-1]+recvCounts[npes-1]);


            //par::Mpi_Alltoallv(&pNodes[0],sendCounts,sendDispl,&pNodesRecv[0],recvCounts,recvDispl,comm);
            // MPI_Alltoallv(&pNodes[0],sendCounts,sendDispl,MPI_TREENODE,&pNodesRecv[0],recvCounts,recvDispl,MPI_TREENODE,comm);
            par::Mpi_Alltoallv_Kway(&(*(pNodes.begin())),sendCounts,sendDispl,&(*(pNodesRecv.begin())),recvCounts,recvDispl,comm);


#ifdef DEBUG_TREE_SORT
            if(!rank) std::cout<<"All2All Communication Ended "<<std::endl;
#endif

            delete[](localSplitter);
            localSplitter=NULL;


            std::swap(pNodes,pNodesRecv);
            pNodesRecv.clear();

            delete[](sendCounts);
            delete[](sendDispl);
            delete[](recvCounts);
            delete[](recvDispl);

            //auto all2all_end=std::chrono::system_clock::now();//MPI_Wtime();
            localSz=pNodes.size();


#ifdef PRINT_PROFILE_DATA

            MPI_Comm_size(pcomm,&npes);
            MPI_Comm_rank(pcomm,&rank);

            localSz=pNodes.size();
            DendroIntL localSzMax=0,localSzMin=0, localSzAvg=0;

            MPI_Reduce(&localSz,&localSzMin,1,MPI_LONG_LONG,MPI_MIN,0,pcomm);
            MPI_Reduce(&localSz,&localSzAvg,1,MPI_LONG_LONG,MPI_SUM,0,pcomm);
            MPI_Reduce(&localSz,&localSzMax,1,MPI_LONG_LONG,MPI_MAX,0,pcomm);

            if(!rank)
            {
                double localAVG=localSzAvg/(double)npes;
                std::cout<<YLW"====================================================== "<<std::endl;
                std::cout<<"  LOAD BALACNE SUMMARY "<<npes<<" Procs"<<std::endl;
                std::cout<<"====================================================== "<<std::endl;
                std::cout<<"LocalMin\tLocalMean\tLocalMax"<<std::endl;
                std::cout<<localSzMin<<"\t"<<localAVG<<"\t"<<localSzMax<<NRM<<std::endl;

            }

            MPI_Comm_size(comm,&npes);
            MPI_Comm_rank(comm,&rank);
#endif


            //auto localSort_start=std::chrono::system_clock::now();//MPI_Wtime();
            //SFC::seqSort::SFC_3D_Sort(pNodes,pMaxDepth);
            //SFC::seqSort::SFC_3D_msd_sort(&(*(pNodes.begin())), pNodes.size(),0, pMaxDepth);
            //SFC::seqSort::SFC_3D_msd_sort_rd(&(*(pNodes.begin())), pNodes.size(),0,pMaxDepth,pMaxDepth);
#ifdef REMOVE_DUPLICATES
            SFC::seqSort::SFC_3D_msd_sort_rd(&(*(pNodes.begin())), pNodes.size(),0,pMaxDepth,pMaxDepth);
#else
            SFC::seqSort::SFC_3D_msd_sort(&(*(pNodes.begin())), pNodes.size(),0, pMaxDepth);
#endif
            //auto localSort_end=std::chrono::system_clock::now();//MPI_Wtime();

            //auto ts_end=std::chrono::system_clock::now();//MPI_Wtime();

            //delete [] (updateState);
            //assert(par::test::isSorted(pNodes,pcomm));

        }

    }

};

#endif //SFCSORTBENCH_SFCSORT_H
