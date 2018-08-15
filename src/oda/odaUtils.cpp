
/**
  @file odaUtils.C
  @brief A collection of simple functions for supporting octree-mesh related operations.  
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  */

#include "mpi.h"
#include "odaUtils.h"
#include "TreeNode.h"
#include "nodeAndValues.h"
#include <cstdio>
#include <iostream>
#include <cassert>
#include <iomanip>
#include "oda.h"
#include "sub_oda.h"
#include "parUtils.h"
#include "seqUtils.h"


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

namespace ot {

  extern double**** ShapeFnCoeffs; 

  void interpolateData(ot::DA* da, Vec in, Vec out, Vec* gradOut,
      unsigned int dof, const std::vector<double>& pts, const double* problemSize) {

    assert(da != NULL);

    int rank = da->getRankAll();
    int npes = da->getNpesAll();
    MPI_Comm comm = da->getComm();

    std::vector<ot::TreeNode> minBlocks;
    unsigned int maxDepth;

    int npesActive;
    if(!rank) {
      minBlocks = da->getMinAllBlocks();
      maxDepth = da->getMaxDepth();
      npesActive = da->getNpesActive();
    }

    par::Mpi_Bcast<unsigned int>(&maxDepth, 1, 0, comm);
    par::Mpi_Bcast<int>(&npesActive, 1, 0, comm);

    unsigned int balOctMaxD = (maxDepth - 1);
    const unsigned int octMaxCoord = (1u << balOctMaxD);

    if(rank) {
      minBlocks.resize(npesActive);
    }

    par::Mpi_Bcast<ot::TreeNode>(&(*(minBlocks.begin())), npesActive, 0, comm);

    std::vector<ot::NodeAndValues<double, 3> > ptsWrapper;

    int numPts = (pts.size())/3;
    double xFac, yFac, zFac;
    
    if (problemSize != NULL) {
    xFac = static_cast<double>(1u << balOctMaxD)/problemSize[0];
    yFac = static_cast<double>(1u << balOctMaxD)/problemSize[1];
    zFac = static_cast<double>(1u << balOctMaxD)/problemSize[2];
    } else {
      xFac = static_cast<double>(1u << balOctMaxD);
      yFac = static_cast<double>(1u << balOctMaxD);
      zFac = static_cast<double>(1u << balOctMaxD);
    }
    for(int i = 0; i < numPts; i++) {
      ot::NodeAndValues<double, 3> tmpObj;
      unsigned int xint = static_cast<unsigned int>(pts[(3*i)]*xFac);
      unsigned int yint = static_cast<unsigned int>(pts[(3*i) + 1]*yFac);
      unsigned int zint = static_cast<unsigned int>(pts[(3*i) + 2]*zFac);

      // fix for positive boundaries...
      /*
       * Basically, if the point is on the boundary, then the interpolation needs to happen on the face that overlaps
       * with the boundary. This face can be part of 2 elements, one that is inside the domain and one that is outside.
       * By default we always go for the "right" element, but this is not correct for those on the positive boundaries,
       * hence the error. You can fix it when you compute the TreeNode corresponding to the element.
       *   Hari, May 7, 2018
       */
      if (xint == octMaxCoord)
        xint = octMaxCoord - 1;
      if (yint == octMaxCoord)
        yint = octMaxCoord - 1;
      if (zint == octMaxCoord)
        zint = octMaxCoord - 1;

      tmpObj.node = ot::TreeNode(xint, yint, zint, maxDepth, 3, maxDepth);
      tmpObj.values[0] = pts[(3*i)];
      tmpObj.values[1] = pts[(3*i) + 1];
      tmpObj.values[2] = pts[(3*i) + 2];
      ptsWrapper.push_back(tmpObj);
    }//end for i

    //Re-distribute ptsWrapper to align with minBlocks
    int* sendCnts = new int[npes];
    assert(sendCnts);

    int* tmpSendCnts = new int[npes];
    assert(tmpSendCnts);

    for(int i = 0; i < npes; i++) {
      sendCnts[i] = 0;
      tmpSendCnts[i] = 0;
    }//end for i

    unsigned int *part = NULL;
    if(numPts) {
      part = new unsigned int[numPts];
      assert(part);
    }

    for(int i = 0; i < numPts; i++) {
      bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks,
          ptsWrapper[i].node, part[i], NULL, NULL);
      assert(found);
      assert(part[i] < npes);
      sendCnts[part[i]]++;
    }//end for i

    int* sendDisps = new int[npes];
    assert(sendDisps);

    sendDisps[0] = 0;
    for(int i = 1; i < npes; i++) {
      sendDisps[i] = sendDisps[i-1] + sendCnts[i-1];
    }//end for i

    std::vector<ot::NodeAndValues<double, 3> > sendList(numPts);

    unsigned int *commMap = NULL;
    if(numPts) {
      commMap = new unsigned int[numPts];
      assert(commMap);
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

    int* recvCnts = new int[npes];
    assert(recvCnts);

    par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, comm);

    int* recvDisps = new int[npes];
    assert(recvDisps);

    recvDisps[0] = 0;
    for(int i = 1; i < npes; i++) {
      recvDisps[i] = recvDisps[i-1] + recvCnts[i-1];
    }//end for i

    std::vector<ot::NodeAndValues<double, 3> > recvList(recvDisps[npes - 1] + recvCnts[npes - 1]);

    ot::NodeAndValues<double, 3>* sendListPtr = NULL;
    ot::NodeAndValues<double, 3>* recvListPtr = NULL;

    if(!(sendList.empty())) {
      sendListPtr = (&(*(sendList.begin())));
    }

    if(!(recvList.empty())) {
      recvListPtr = (&(*(recvList.begin())));
    }

    par::Mpi_Alltoallv_sparse<ot::NodeAndValues<double, 3> >(sendListPtr, 
        sendCnts, sendDisps, recvListPtr, recvCnts, recvDisps, comm);
    sendList.clear();

    //Sort recvList but also store the mapping to the original order
    std::vector<seq::IndexHolder<ot::NodeAndValues<double, 3> > > localList(recvList.size());
    for(unsigned int i = 0; i < recvList.size(); i++) {
      localList[i].index = i;
      localList[i].value = &(*(recvList.begin() + i));
    }//end for i

    sort(localList.begin(), localList.end());

    bool computeGradient = (gradOut != NULL);

    std::vector<double> tmpOut(dof*localList.size());
    std::vector<double> tmpGradOut;
    if(computeGradient) {
      tmpGradOut.resize(3*dof*localList.size());
    }

    PetscScalar* inArr;
    da->vecGetBuffer(in, inArr, false, false, true, dof);

    if(da->iAmActive()) {
      da->ReadFromGhostsBegin<PetscScalar>(inArr, dof);
      da->ReadFromGhostsEnd<PetscScalar>(inArr);

      //interpolate at the received points
      //The pts must be inside the domain and not on the positive boundaries
      unsigned int ptsCtr = 0;
      double hxFac, hyFac, hzFac;
      if (problemSize == NULL) {
        hxFac = (1.0/static_cast<double>(1u << balOctMaxD));
        hyFac = (1.0/static_cast<double>(1u << balOctMaxD));
        hzFac = (1.0/static_cast<double>(1u << balOctMaxD));
      } else {
        hxFac = (problemSize[0]/static_cast<double>(1u << balOctMaxD));
        hyFac = (problemSize[1]/static_cast<double>(1u << balOctMaxD));
        hzFac = (problemSize[2]/static_cast<double>(1u << balOctMaxD));
      }
      for(da->init<ot::DA_FLAGS::WRITABLE>();
          (da->curr() < da->end<ot::DA_FLAGS::WRITABLE>()) && 
          (ptsCtr < localList.size()); da->next<ot::DA_FLAGS::WRITABLE>()) {

        Point pt = da->getCurrentOffset();
        unsigned int currLev = da->getLevel(da->curr());

        ot::TreeNode currOct(pt.xint(), pt.yint(), pt.zint(), currLev, 3, maxDepth);

        unsigned int indices[8];
        da->getNodeIndices(indices);

        unsigned char childNum = da->getChildNumber();
        unsigned char hnMask = da->getHangingNodeIndex(da->curr());
        unsigned char elemType = 0;
        GET_ETYPE_BLOCK(elemType, hnMask, childNum)

        double x0 = (pt.x())*hxFac;
        double y0 = (pt.y())*hyFac;
        double z0 = (pt.z())*hzFac;
        double fac = (1.0/static_cast<double>(1u << balOctMaxD));
        double hxOct = (static_cast<double>(1u << (maxDepth - currLev)))*hxFac;

        //All the recieved points lie within some octant or the other.
        //So the ptsCtr will be incremented properly inside this loop.
        //Evaluate at all points within this octant
        unsigned bdy_max = 1u << (maxDepth - 1);
        while( (ptsCtr < localList.size()) &&
            ( (currOct == ((localList[ptsCtr].value)->node)) ||
              (currOct.isAncestor(((localList[ptsCtr].value)->node))) 
              || ( (localList[ptsCtr].value)->node.getX() == bdy_max) 
              || ( (localList[ptsCtr].value)->node.getY() == bdy_max) 
              || ( (localList[ptsCtr].value)->node.getZ() == bdy_max)  
              ) ) {

          double px = ((localList[ptsCtr].value)->values)[0];
          double py = ((localList[ptsCtr].value)->values)[1];
          double pz = ((localList[ptsCtr].value)->values)[2];
          double xloc =  (2.0*(px - x0)/hxOct) - 1.0;
          double yloc =  (2.0*(py - y0)/hxOct) - 1.0;
          double zloc =  (2.0*(pz - z0)/hxOct) - 1.0;

          double ShFnVals[8];
          for(int j = 0; j < 8; j++) {
            ShFnVals[j] = ( ShapeFnCoeffs[childNum][elemType][j][0] + 
                (ShapeFnCoeffs[childNum][elemType][j][1]*xloc) +
                (ShapeFnCoeffs[childNum][elemType][j][2]*yloc) +
                (ShapeFnCoeffs[childNum][elemType][j][3]*zloc) +
                (ShapeFnCoeffs[childNum][elemType][j][4]*xloc*yloc) +
                (ShapeFnCoeffs[childNum][elemType][j][5]*yloc*zloc) +
                (ShapeFnCoeffs[childNum][elemType][j][6]*zloc*xloc) +
                (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*yloc*zloc) );
          }//end for j

          unsigned int outIdx = localList[ptsCtr].index;

          for(int k = 0; k < dof; k++) {
            tmpOut[(dof*outIdx) + k] = 0.0;
            for(int j = 0; j < 8; j++) {
              tmpOut[(dof*outIdx) + k] += (inArr[(dof*indices[j]) + k]*ShFnVals[j]);
            }//end for j
          }//end for k

          if(computeGradient) {

            double GradShFnVals[8][3];
            for(int j = 0; j < 8; j++) {
              GradShFnVals[j][0] = ( ShapeFnCoeffs[childNum][elemType][j][1] +
                  (ShapeFnCoeffs[childNum][elemType][j][4]*yloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][6]*zloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*yloc*zloc) );

              GradShFnVals[j][1] = ( ShapeFnCoeffs[childNum][elemType][j][2] +
                  (ShapeFnCoeffs[childNum][elemType][j][4]*xloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][5]*zloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*zloc) );

              GradShFnVals[j][2] = ( ShapeFnCoeffs[childNum][elemType][j][3] +
                  (ShapeFnCoeffs[childNum][elemType][j][5]*yloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][6]*xloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*yloc) );            
            }//end for j

            double gradFac = (2.0/hxOct);

            for(int k = 0; k < dof; k++) {
              for(int l = 0; l < 3; l++) {
                tmpGradOut[(3*dof*outIdx) + (3*k) + l] = 0.0;
                for(int j = 0; j < 8; j++) {
                  tmpGradOut[(3*dof*outIdx) + (3*k) + l] += (inArr[(dof*indices[j]) + k]*GradShFnVals[j][l]);
                }//end for j
                tmpGradOut[(3*dof*outIdx) + (3*k) + l] *= gradFac;
              }//end for l
            }//end for k

          }//end if need grad

          ptsCtr++;
        }//end while

      }//end writable loop
      // std::cout << "Interpolate Data: " << ptsCtr << "/" << localList.size() << std::endl;
    } else {
      assert(localList.empty());
    }//end if active

    da->vecRestoreBuffer(in, inArr, false, false, true, dof);

    recvList.clear();
    localList.clear();

    //Return the results. This communication is the exact reverse of the earlier
    //communication.

    std::vector<double> results(dof*numPts);
    for(int i = 0; i < npes; i++) {
      sendCnts[i] *= dof;
      sendDisps[i] *= dof;
      recvCnts[i] *= dof;
      recvDisps[i] *= dof;
    }//end for i

    double* tmpOutPtr = NULL;
    double* resultsPtr = NULL;

    if(!(tmpOut.empty())) {
      tmpOutPtr = (&(*(tmpOut.begin())));
    }

    if(!(results.empty())) {
      resultsPtr = (&(*(results.begin())));
    }

    par::Mpi_Alltoallv_sparse<double>( tmpOutPtr, recvCnts, recvDisps,
        resultsPtr, sendCnts, sendDisps, comm);
    tmpOut.clear();

    std::vector<double> gradResults;
    if(computeGradient) {
      for(int i = 0; i < npes; i++) {
        sendCnts[i] *= 3;
        sendDisps[i] *= 3;
        recvCnts[i] *= 3;
        recvDisps[i] *= 3;
      }//end for i
      gradResults.resize(3*dof*numPts);

      double* tmpGradOutPtr = NULL;
      double* gradResultsPtr = NULL;

      if(!(tmpGradOut.empty())) {
        tmpGradOutPtr = (&(*(tmpGradOut.begin())));
      }

      if(!(gradResults.empty())) {
        gradResultsPtr = (&(*(gradResults.begin())));
      }

      par::Mpi_Alltoallv_sparse<double >( tmpGradOutPtr, recvCnts, recvDisps,
          gradResultsPtr, sendCnts, sendDisps, comm);
      tmpGradOut.clear();
    }

    assert(sendCnts);
    delete [] sendCnts;
    sendCnts = NULL;

    assert(sendDisps);
    delete [] sendDisps;
    sendDisps = NULL;

    assert(recvCnts);
    delete [] recvCnts;
    recvCnts = NULL;

    assert(recvDisps);
    delete [] recvDisps;
    recvDisps = NULL;

    //Use commMap and re-order the results in the same order as the original
    //points
    PetscInt outSz;
    VecGetLocalSize(out, &outSz);
    assert(outSz == (dof*numPts));

    PetscScalar* outArr;
    VecGetArray(out, &outArr);
    for(int i = 0; i < numPts; i++) {
      for(int j = 0; j < dof; j++) {
        outArr[(dof*commMap[i]) + j] = results[(dof*i) + j];
      }//end for j
    }//end for i
    VecRestoreArray(out, &outArr);

    if(computeGradient) {
      PetscInt gradOutSz;
      assert(gradOut != NULL);
      VecGetLocalSize((*gradOut), &gradOutSz);
      assert(gradOutSz == (3*dof*numPts));
      PetscScalar* gradOutArr;
      VecGetArray((*gradOut), &gradOutArr);
      for(int i = 0; i < numPts; i++) {
        for(int j = 0; j < (3*dof); j++) {
          gradOutArr[(3*dof*commMap[i]) + j] = gradResults[(3*dof*i) + j];
        }//end for j
      }//end for i
      VecRestoreArray((*gradOut), &gradOutArr);
    }

    if(commMap) {
      delete [] commMap;
      commMap = NULL;
    }

  }

  void interpolateData(ot::subDA* da, Vec in, Vec out, Vec* gradOut,
      unsigned int dof, const std::vector<double>& pts, const double* problemSize, const double* subDA_max) {

    assert(da != NULL);

    int rank = da->getRankAll();
    int npes = da->getNpesAll();
    MPI_Comm comm = da->getComm();

    std::vector<ot::TreeNode> minBlocks;
    unsigned int maxDepth;

    int npesActive;
    if(!rank) {
      minBlocks = da->getMinAllBlocks();
      maxDepth = da->getMaxDepth();
      npesActive = da->getNpesActive();
    }

    par::Mpi_Bcast<unsigned int>(&maxDepth, 1, 0, comm);
    par::Mpi_Bcast<int>(&npesActive, 1, 0, comm);

    unsigned int balOctMaxD = (maxDepth - 1);
    const unsigned int octMaxCoord = (1u << balOctMaxD);

    if(rank) {
      minBlocks.resize(npesActive);
    }

    par::Mpi_Bcast<ot::TreeNode>(&(*(minBlocks.begin())), npesActive, 0, comm);

    std::vector<ot::NodeAndValues<double, 3> > ptsWrapper;

    int numPts = (pts.size())/3;
    double xFac, yFac, zFac;
    
    if (problemSize != NULL) {
      xFac = static_cast<double>(1u << balOctMaxD)/problemSize[0];
      yFac = static_cast<double>(1u << balOctMaxD)/problemSize[1];
      zFac = static_cast<double>(1u << balOctMaxD)/problemSize[2];
    } else {
      xFac = static_cast<double>(1u << balOctMaxD);
      yFac = static_cast<double>(1u << balOctMaxD);
      zFac = static_cast<double>(1u << balOctMaxD);
    }
    
    const unsigned int subDA_max_x = static_cast<unsigned int>(subDA_max[0]*xFac);
    const unsigned int subDA_max_y = static_cast<unsigned int>(subDA_max[1]*yFac);
    const unsigned int subDA_max_z = static_cast<unsigned int>(subDA_max[2]*zFac);

    for(int i = 0; i < numPts; i++) {
      ot::NodeAndValues<double, 3> tmpObj;
      unsigned int xint = static_cast<unsigned int>(pts[(3*i)]*xFac);
      unsigned int yint = static_cast<unsigned int>(pts[(3*i) + 1]*yFac);
      unsigned int zint = static_cast<unsigned int>(pts[(3*i) + 2]*zFac);

      // fix for positive boundaries...
      /*
       * Basically, if the point is on the boundary, then the interpolation needs to happen on the face that overlaps
       * with the boundary. This face can be part of 2 elements, one that is inside the domain and one that is outside.
       * By default we always go for the "right" element, but this is not correct for those on the positive boundaries,
       * hence the error. You can fix it when you compute the TreeNode corresponding to the element.
       *   Hari, May 7, 2018
       */

      // if ( (xint == octMaxCoord) ) {
      //   xint = octMaxCoord - 1;
      // }
      // if ( (yint == octMaxCoord) || ( fabs(pts[(3*i)+1] - subDASize[1]) < 1e-8 ) ) {
      //   yint = octMaxCoord - 1;
      // }
      // if ( (zint == octMaxCoord) || ( fabs(pts[(3*i)+2] - subDASize[2]) < 1e-8 ) ) {
      //   zint = octMaxCoord - 1;
      // }

      if ( xint >= subDA_max_x ) {
        xint = subDA_max_x-1;
        // std::cout << "correcting node. xint " << xint << " -> " << ((double)xint)/xFac << std::endl;
      }
      if ( yint >= subDA_max_y ) { 
        yint = subDA_max_y-1;
        // std::cout << "correcting node. yint " << yint << " -> " << ((double)yint)/yFac << std::endl;
      }
      if ( zint >= subDA_max_z ) {
        zint = subDA_max_z-1;
      }

      tmpObj.node = ot::TreeNode(xint, yint, zint, maxDepth, 3, maxDepth);
      tmpObj.values[0] = pts[(3*i)];
      tmpObj.values[1] = pts[(3*i) + 1];
      tmpObj.values[2] = pts[(3*i) + 2];
      ptsWrapper.push_back(tmpObj);
      // std::cout << "Adding node: (" << xint << ", " << yint << ", " << zint << ") " << tmpObj.node << " -> (" << pts[(3*i)] << ", " << pts[(3*i)+1]  << ", " << pts[(3*i)+2] << ")" << std::endl;
    }//end for i

    //Re-distribute ptsWrapper to align with minBlocks
    int* sendCnts = new int[npes];
    assert(sendCnts);

    int* tmpSendCnts = new int[npes];
    assert(tmpSendCnts);

    for(int i = 0; i < npes; i++) {
      sendCnts[i] = 0;
      tmpSendCnts[i] = 0;
    }//end for i

    unsigned int *part = NULL;
    if(numPts) {
      part = new unsigned int[numPts];
      assert(part);
    }

    for(int i = 0; i < numPts; i++) {
      bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks,
          ptsWrapper[i].node, part[i], NULL, NULL);
      assert(found);
      assert(part[i] < npes);
      sendCnts[part[i]]++;
    }//end for i

    int* sendDisps = new int[npes];
    assert(sendDisps);

    sendDisps[0] = 0;
    for(int i = 1; i < npes; i++) {
      sendDisps[i] = sendDisps[i-1] + sendCnts[i-1];
    }//end for i

    std::vector<ot::NodeAndValues<double, 3> > sendList(numPts);

    unsigned int *commMap = NULL;
    if(numPts) {
      commMap = new unsigned int[numPts];
      assert(commMap);
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

    int* recvCnts = new int[npes];
    assert(recvCnts);

    par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, comm);

    int* recvDisps = new int[npes];
    assert(recvDisps);

    recvDisps[0] = 0;
    for(int i = 1; i < npes; i++) {
      recvDisps[i] = recvDisps[i-1] + recvCnts[i-1];
    }//end for i

    std::vector<ot::NodeAndValues<double, 3> > recvList(recvDisps[npes - 1] + recvCnts[npes - 1]);

    ot::NodeAndValues<double, 3>* sendListPtr = NULL;
    ot::NodeAndValues<double, 3>* recvListPtr = NULL;

    if(!(sendList.empty())) {
      sendListPtr = (&(*(sendList.begin())));
    }

    if(!(recvList.empty())) {
      recvListPtr = (&(*(recvList.begin())));
    }

    par::Mpi_Alltoallv_sparse<ot::NodeAndValues<double, 3> >(sendListPtr, 
        sendCnts, sendDisps, recvListPtr, recvCnts, recvDisps, comm);
    sendList.clear();

    //Sort recvList but also store the mapping to the original order
    std::vector<seq::IndexHolder<ot::NodeAndValues<double, 3> > > localList(recvList.size());
    for(unsigned int i = 0; i < recvList.size(); i++) {
      localList[i].index = i;
      localList[i].value = &(*(recvList.begin() + i));
    }//end for i

    sort(localList.begin(), localList.end());

    bool computeGradient = (gradOut != NULL);

    std::vector<double> tmpOut(dof*localList.size());
    std::vector<double> tmpGradOut;
    if(computeGradient) {
      tmpGradOut.resize(3*dof*localList.size());
    }

    PetscScalar* inArr;
    da->vecGetBuffer(in, inArr, false, false, true, dof);

    da->ReadFromGhostsBegin<PetscScalar>(inArr, dof);
    da->ReadFromGhostsEnd<PetscScalar>(inArr);

      //interpolate at the received points
      //The pts must be inside the domain and not on the positive boundaries
      unsigned int ptsCtr = 0;
      double hxFac, hyFac, hzFac;
      if (problemSize == NULL) {
        hxFac = (1.0/static_cast<double>(1u << balOctMaxD));
        hyFac = (1.0/static_cast<double>(1u << balOctMaxD));
        hzFac = (1.0/static_cast<double>(1u << balOctMaxD));
      } else {
        hxFac = (problemSize[0]/static_cast<double>(1u << balOctMaxD));
        hyFac = (problemSize[1]/static_cast<double>(1u << balOctMaxD));
        hzFac = (problemSize[2]/static_cast<double>(1u << balOctMaxD));
      }

      // std::cout << "first point: " << (localList[ptsCtr].value)->node << " -- " << ((localList[ptsCtr].value)->values)[0] << "," << ((localList[ptsCtr].value)->values)[1] << ", " << ((localList[ptsCtr].value)->values)[2] <<  std::endl;

      for(da->init<ot::DA_FLAGS::WRITABLE>();
          (da->curr() < da->end<ot::DA_FLAGS::WRITABLE>()) && 
          (ptsCtr < localList.size()); da->next<ot::DA_FLAGS::WRITABLE>()) {


        Point pt = da->getCurrentOffset();
        unsigned int currLev = da->getLevel(da->curr());

        ot::TreeNode currOct(pt.xint(), pt.yint(), pt.zint(), currLev, 3, maxDepth);

        unsigned int outIdx = localList[ptsCtr].index;

        while ((localList[ptsCtr].value)->node < currOct ) {
          // std::cout << "skipping " << ptsCtr << ": " << (localList[ptsCtr].value)->node << " >>= " << currOct << std::endl; 
          for(int k = 0; k < dof; k++)
             tmpOut[(dof*outIdx) + k] = -2.0;
          ptsCtr++;
          outIdx = localList[ptsCtr].index;
        }

        unsigned int indices[8];
        da->getNodeIndices(indices);

        unsigned char childNum = da->getChildNumber();
        unsigned char hnMask = da->getHangingNodeIndex(da->curr());
        unsigned char elemType = 0;
        GET_ETYPE_BLOCK(elemType, hnMask, childNum)

        double x0 = (pt.x())*hxFac;
        double y0 = (pt.y())*hyFac;
        double z0 = (pt.z())*hzFac;
        double fac = (1.0/static_cast<double>(1u << balOctMaxD));
        double hxOct = (static_cast<double>(1u << (maxDepth - currLev)))*hxFac;

        //All the recieved points lie within some octant or the other.
        //So the ptsCtr will be incremented properly inside this loop.
        //Evaluate at all points within this octant
        unsigned bdy_max = 1u << (maxDepth - 1);
        //~~~~~~~~~~~~~~~
        // std::cout << "Interpolate Data: " << ptsCtr << "/" << localList.size() << std::endl;
        // std::cout << "currOct: " << currOct << std::endl;
        // std::cout << "localList"
        //~~~~~~~~~~~~~~~
        while( (ptsCtr < localList.size()) &&
            ( (currOct == ((localList[ptsCtr].value)->node)) ||
              (currOct.isAncestor(((localList[ptsCtr].value)->node))) 
              || ( (localList[ptsCtr].value)->node.getX() == bdy_max) 
              || ( (localList[ptsCtr].value)->node.getY() == bdy_max) 
              || ( (localList[ptsCtr].value)->node.getZ() == bdy_max)  
              ) ) {

          double px = ((localList[ptsCtr].value)->values)[0];
          double py = ((localList[ptsCtr].value)->values)[1];
          double pz = ((localList[ptsCtr].value)->values)[2];
          double xloc =  (2.0*(px - x0)/hxOct) - 1.0;
          double yloc =  (2.0*(py - y0)/hxOct) - 1.0;
          double zloc =  (2.0*(pz - z0)/hxOct) - 1.0;

          double ShFnVals[8];
          for(int j = 0; j < 8; j++) {
            ShFnVals[j] = ( ShapeFnCoeffs[childNum][elemType][j][0] + 
                (ShapeFnCoeffs[childNum][elemType][j][1]*xloc) +
                (ShapeFnCoeffs[childNum][elemType][j][2]*yloc) +
                (ShapeFnCoeffs[childNum][elemType][j][3]*zloc) +
                (ShapeFnCoeffs[childNum][elemType][j][4]*xloc*yloc) +
                (ShapeFnCoeffs[childNum][elemType][j][5]*yloc*zloc) +
                (ShapeFnCoeffs[childNum][elemType][j][6]*zloc*xloc) +
                (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*yloc*zloc) );
          }//end for j

          

          for(int k = 0; k < dof; k++) {
            tmpOut[(dof*outIdx) + k] = 0.0;
            for(int j = 0; j < 8; j++) {
              tmpOut[(dof*outIdx) + k] += (inArr[(dof*indices[j]) + k]*ShFnVals[j]);
            }//end for j
          }//end for k

          if(computeGradient) {

            double GradShFnVals[8][3];
            for(int j = 0; j < 8; j++) {
              GradShFnVals[j][0] = ( ShapeFnCoeffs[childNum][elemType][j][1] +
                  (ShapeFnCoeffs[childNum][elemType][j][4]*yloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][6]*zloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*yloc*zloc) );

              GradShFnVals[j][1] = ( ShapeFnCoeffs[childNum][elemType][j][2] +
                  (ShapeFnCoeffs[childNum][elemType][j][4]*xloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][5]*zloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*zloc) );

              GradShFnVals[j][2] = ( ShapeFnCoeffs[childNum][elemType][j][3] +
                  (ShapeFnCoeffs[childNum][elemType][j][5]*yloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][6]*xloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*yloc) );            
            }//end for j

            double gradFac = (2.0/hxOct);

            for(int k = 0; k < dof; k++) {
              for(int l = 0; l < 3; l++) {
                tmpGradOut[(3*dof*outIdx) + (3*k) + l] = 0.0;
                for(int j = 0; j < 8; j++) {
                  tmpGradOut[(3*dof*outIdx) + (3*k) + l] += (inArr[(dof*indices[j]) + k]*GradShFnVals[j][l]);
                }//end for j
                tmpGradOut[(3*dof*outIdx) + (3*k) + l] *= gradFac;
              }//end for l
            }//end for k

          }//end if need grad

          ptsCtr++;
          outIdx = localList[ptsCtr].index;
        }//end while

      }//end writable loop
      // std::cout << "Interpolate Data: " << ptsCtr << "/" << localList.size() << std::endl;
    // } else {
    //   assert(localList.empty());
    
    da->vecRestoreBuffer(in, inArr, false, false, true, dof);

    recvList.clear();
    localList.clear();

    //Return the results. This communication is the exact reverse of the earlier
    //communication.

    std::vector<double> results(dof*numPts);
    for(int i = 0; i < npes; i++) {
      sendCnts[i] *= dof;
      sendDisps[i] *= dof;
      recvCnts[i] *= dof;
      recvDisps[i] *= dof;
    }//end for i

    double* tmpOutPtr = NULL;
    double* resultsPtr = NULL;

    if(!(tmpOut.empty())) {
      tmpOutPtr = (&(*(tmpOut.begin())));
    }

    if(!(results.empty())) {
      resultsPtr = (&(*(results.begin())));
    }

    par::Mpi_Alltoallv_sparse<double>( tmpOutPtr, recvCnts, recvDisps,
        resultsPtr, sendCnts, sendDisps, comm);
    tmpOut.clear();

    std::vector<double> gradResults;
    if(computeGradient) {
      for(int i = 0; i < npes; i++) {
        sendCnts[i] *= 3;
        sendDisps[i] *= 3;
        recvCnts[i] *= 3;
        recvDisps[i] *= 3;
      }//end for i
      gradResults.resize(3*dof*numPts);

      double* tmpGradOutPtr = NULL;
      double* gradResultsPtr = NULL;

      if(!(tmpGradOut.empty())) {
        tmpGradOutPtr = (&(*(tmpGradOut.begin())));
      }

      if(!(gradResults.empty())) {
        gradResultsPtr = (&(*(gradResults.begin())));
      }

      par::Mpi_Alltoallv_sparse<double >( tmpGradOutPtr, recvCnts, recvDisps,
          gradResultsPtr, sendCnts, sendDisps, comm);
      tmpGradOut.clear();
    }

    assert(sendCnts);
    delete [] sendCnts;
    sendCnts = NULL;

    assert(sendDisps);
    delete [] sendDisps;
    sendDisps = NULL;

    assert(recvCnts);
    delete [] recvCnts;
    recvCnts = NULL;

    assert(recvDisps);
    delete [] recvDisps;
    recvDisps = NULL;

    //Use commMap and re-order the results in the same order as the original
    //points
    PetscInt outSz;
    VecGetLocalSize(out, &outSz);
    assert(outSz == (dof*numPts));

    PetscScalar* outArr;
    VecGetArray(out, &outArr);
    for(int i = 0; i < numPts; i++) {
      for(int j = 0; j < dof; j++) {
        outArr[(dof*commMap[i]) + j] = results[(dof*i) + j];
      }//end for j
    }//end for i
    VecRestoreArray(out, &outArr);

    if(computeGradient) {
      PetscInt gradOutSz;
      assert(gradOut != NULL);
      VecGetLocalSize((*gradOut), &gradOutSz);
      assert(gradOutSz == (3*dof*numPts));
      PetscScalar* gradOutArr;
      VecGetArray((*gradOut), &gradOutArr);
      for(int i = 0; i < numPts; i++) {
        for(int j = 0; j < (3*dof); j++) {
          gradOutArr[(3*dof*commMap[i]) + j] = gradResults[(3*dof*i) + j];
        }//end for j
      }//end for i
      VecRestoreArray((*gradOut), &gradOutArr);
    }

    if(commMap) {
      delete [] commMap;
      commMap = NULL;
    }

  }

  /*
  void interpolateData(ot::DA* da, std::vector<double> & in,
      std::vector<double> & out, std::vector<double> * gradOut,
      unsigned int dof, std::vector<double> & pts) {

    int rank = da->getRankAll();
    int npes = da->getNpesAll();
    MPI_Comm comm = da->getComm();

    std::vector<ot::TreeNode> minBlocks;
    unsigned int maxDepth;

    int npesActive;
    if(!rank) {
      minBlocks = da->getMinAllBlocks();
      maxDepth = da->getMaxDepth();
      npesActive = da->getNpesActive();
    }

    par::Mpi_Bcast<unsigned int>(&maxDepth, 1, 0, comm);
    par::Mpi_Bcast<int>(&npesActive, 1, 0, comm);

    unsigned int balOctMaxD = (maxDepth - 1);
    const unsigned int octMaxCoord = (1u << balOctMaxD);

    if(rank) {
      minBlocks.resize(npesActive);
    }

    par::Mpi_Bcast<ot::TreeNode>(&(*(minBlocks.begin())), npesActive, 0, comm);

    std::vector<ot::NodeAndValues<double, 3> > ptsWrapper;

    int numPts = (pts.size())/3;
    double xyzFac = static_cast<double>(1u << balOctMaxD);
    for(int i = 0; i < numPts; i++) {
      ot::NodeAndValues<double, 3> tmpObj;
      unsigned int xint = static_cast<unsigned int>(pts[(3*i)]*xyzFac);
      unsigned int yint = static_cast<unsigned int>(pts[(3*i) + 1]*xyzFac);
      unsigned int zint = static_cast<unsigned int>(pts[(3*i) + 2]*xyzFac);

      // fix for positive boundaries...
      //
      // Basically, if the point is on the boundary, then the interpolation needs to happen on the face that overlaps
      // with the boundary. This face can be part of 2 elements, one that is inside the domain and one that is outside.
      // By default we always go for the "right" element, but this is not correct for those on the positive boundaries,
      // hence the error. You can fix it when you compute the TreeNode corresponding to the element.
      //   Hari, May 7, 2018
      //
      if (xint == octMaxCoord)
        xint = octMaxCoord - 1;
      if (yint == octMaxCoord)
        yint = octMaxCoord - 1;
      if (zint == octMaxCoord)
        zint = octMaxCoord - 1;

      tmpObj.node = ot::TreeNode(xint, yint, zint, maxDepth, 3, maxDepth);
      tmpObj.values[0] = pts[(3*i)];
      tmpObj.values[1] = pts[(3*i) + 1];
      tmpObj.values[2] = pts[(3*i) + 2];
      ptsWrapper.push_back(tmpObj);
    }//end for i

    //Re-distribute ptsWrapper to align with minBlocks
    int* sendCnts = new int[npes];
    assert(sendCnts);
    int* tmpSendCnts = new int[npes];
    assert(tmpSendCnts);

    for(int i = 0; i < npes; i++) {
      sendCnts[i] = 0;
      tmpSendCnts[i] = 0;
    }//end for i

    unsigned int* part = NULL;
    if(numPts) {
      part = new unsigned int[numPts];
      assert(part);
    }

    for(int i = 0; i < numPts; i++) {
      bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks,
          ptsWrapper[i].node, part[i], NULL, NULL);
      assert(found);
      assert(part[i] < npes);
      sendCnts[part[i]]++;
    }//end for i

    int* sendDisps = new int[npes];
    assert(sendDisps);

    sendDisps[0] = 0;
    for(int i = 1; i < npes; i++) {
      sendDisps[i] = sendDisps[i-1] + sendCnts[i-1];
    }//end for i

    std::vector<ot::NodeAndValues<double, 3> > sendList(numPts);

    unsigned int* commMap = NULL;
    if(numPts) {
      commMap = new unsigned int[numPts];
      assert(commMap);
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

    int* recvCnts = new int[npes];
    assert(recvCnts);

    par::Mpi_Alltoall<int>(sendCnts, recvCnts, 1, comm);

    int* recvDisps = new int[npes];
    assert(recvDisps);

    recvDisps[0] = 0;
    for(int i = 1; i < npes; i++) {
      recvDisps[i] = recvDisps[i-1] + recvCnts[i-1];
    }//end for i

    std::vector<ot::NodeAndValues<double, 3> > recvList(recvDisps[npes - 1] + recvCnts[npes - 1]);

    ot::NodeAndValues<double, 3>* sendListPtr = NULL;
    ot::NodeAndValues<double, 3>* recvListPtr = NULL;

    if(!(sendList.empty())) {
      sendListPtr = (&(*(sendList.begin())));
    }

    if(!(recvList.empty())) {
      recvListPtr = (&(*(recvList.begin())));
    }

    par::Mpi_Alltoallv_sparse<ot::NodeAndValues<double, 3> >( sendListPtr, 
        sendCnts, sendDisps, recvListPtr, recvCnts, recvDisps, comm);
    sendList.clear();

    //Sort recvList but also store the mapping to the original order
    std::vector<seq::IndexHolder<ot::NodeAndValues<double, 3> > > localList(recvList.size());
    for(unsigned int i = 0; i < recvList.size(); i++) {
      localList[i].index = i;
      localList[i].value = &(*(recvList.begin() + i));
    }//end for i

    sort(localList.begin(), localList.end());

    bool computeGradient = (gradOut != NULL);

    std::vector<double> tmpOut(dof*localList.size());
    std::vector<double> tmpGradOut;
    if(computeGradient) {
      tmpGradOut.resize(3*dof*localList.size());
    }

    double* inArr;
    da->vecGetBuffer<double>(in, inArr, false, false, true, dof);

    if(da->iAmActive()) {
      da->ReadFromGhostsBegin<double>(inArr, dof);
      da->ReadFromGhostsEnd<double>(inArr);

      //interpolate at the received points
      //The pts must be inside the domain and not on the positive boundaries
      unsigned int ptsCtr = 0;
      double hxFac = (1.0/static_cast<double>(1u << balOctMaxD));
      for(da->init<ot::DA_FLAGS::WRITABLE>();
          (da->curr() < da->end<ot::DA_FLAGS::WRITABLE>()) && 
          (ptsCtr < localList.size()); da->next<ot::DA_FLAGS::WRITABLE>()) {

        Point pt = da->getCurrentOffset();
        unsigned int currLev = da->getLevel(da->curr());

        ot::TreeNode currOct(pt.xint(), pt.yint(), pt.zint(), currLev, 3, maxDepth);

        unsigned int indices[8];
        da->getNodeIndices(indices);

        unsigned char childNum = da->getChildNumber();
        unsigned char hnMask = da->getHangingNodeIndex(da->curr());
        unsigned char elemType = 0;
        GET_ETYPE_BLOCK(elemType, hnMask, childNum)

          double x0 = (pt.x())*hxFac;
        double y0 = (pt.y())*hxFac;
        double z0 = (pt.z())*hxFac;
        double hxOct = (static_cast<double>(1u << (maxDepth - currLev)))*hxFac;

        //All the recieved points lie within some octant or the other.
        //So the ptsCtr will be incremented properly inside this loop.
        //Evaluate at all points within this octant
        while( (ptsCtr < localList.size()) &&
            ( (currOct == ((localList[ptsCtr].value)->node)) ||
              (currOct.isAncestor(((localList[ptsCtr].value)->node))) ) ) {

          double px = ((localList[ptsCtr].value)->values)[0];
          double py = ((localList[ptsCtr].value)->values)[1];
          double pz = ((localList[ptsCtr].value)->values)[2];
          double xloc =  (2.0*(px - x0)/hxOct) - 1.0;
          double yloc =  (2.0*(py - y0)/hxOct) - 1.0;
          double zloc =  (2.0*(pz - z0)/hxOct) - 1.0;

          double ShFnVals[8];
          for(int j = 0; j < 8; j++) {
            ShFnVals[j] = ( ShapeFnCoeffs[childNum][elemType][j][0] + 
                (ShapeFnCoeffs[childNum][elemType][j][1]*xloc) +
                (ShapeFnCoeffs[childNum][elemType][j][2]*yloc) +
                (ShapeFnCoeffs[childNum][elemType][j][3]*zloc) +
                (ShapeFnCoeffs[childNum][elemType][j][4]*xloc*yloc) +
                (ShapeFnCoeffs[childNum][elemType][j][5]*yloc*zloc) +
                (ShapeFnCoeffs[childNum][elemType][j][6]*zloc*xloc) +
                (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*yloc*zloc) );
          }//end for j

          unsigned int outIdx = localList[ptsCtr].index;

          for(int k = 0; k < dof; k++) {
            tmpOut[(dof*outIdx) + k] = 0.0;
            for(int j = 0; j < 8; j++) {
              tmpOut[(dof*outIdx) + k] += (inArr[(dof*indices[j]) + k]*ShFnVals[j]);
            }//end for j
          }//end for k

          if(computeGradient) {

            double GradShFnVals[8][3];
            for(int j = 0; j < 8; j++) {
              GradShFnVals[j][0] = ( ShapeFnCoeffs[childNum][elemType][j][1] +
                  (ShapeFnCoeffs[childNum][elemType][j][4]*yloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][6]*zloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*yloc*zloc) );

              GradShFnVals[j][1] = ( ShapeFnCoeffs[childNum][elemType][j][2] +
                  (ShapeFnCoeffs[childNum][elemType][j][4]*xloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][5]*zloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*zloc) );

              GradShFnVals[j][2] = ( ShapeFnCoeffs[childNum][elemType][j][3] +
                  (ShapeFnCoeffs[childNum][elemType][j][5]*yloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][6]*xloc) +
                  (ShapeFnCoeffs[childNum][elemType][j][7]*xloc*yloc) );            
            }//end for j

            double gradFac = (2.0/hxOct);

            for(int k = 0; k < dof; k++) {
              for(int l = 0; l < 3; l++) {
                tmpGradOut[(3*dof*outIdx) + (3*k) + l] = 0.0;
                for(int j = 0; j < 8; j++) {
                  tmpGradOut[(3*dof*outIdx) + (3*k) + l] += (inArr[(dof*indices[j]) + k]*GradShFnVals[j][l]);
                }//end for j
                tmpGradOut[(3*dof*outIdx) + (3*k) + l] *= gradFac;
              }//end for l
            }//end for k

          }//end if need grad

          ptsCtr++;
        }//end while

      }//end writable loop
    } else {
      assert(localList.empty());
    }//end if active

    da->vecRestoreBuffer<double>(in, inArr, false, false, true, dof);

    recvList.clear();
    localList.clear();

    //Return the results. This communication is the exact reverse of the earlier
    //communication.

    std::vector<double> results(dof*numPts);
    for(int i = 0; i < npes; i++) {
      sendCnts[i] *= dof;
      sendDisps[i] *= dof;
      recvCnts[i] *= dof;
      recvDisps[i] *= dof;
    }//end for i

    double* tmpOutPtr = NULL;
    double* resultsPtr = NULL;

    if(!(tmpOut.empty())) {
      tmpOutPtr = (&(*(tmpOut.begin())));
    }

    if(!(results.empty())) {
      resultsPtr = (&(*(results.begin())));
    }

    par::Mpi_Alltoallv_sparse<double >( tmpOutPtr, recvCnts, recvDisps,
        resultsPtr, sendCnts, sendDisps, comm);
    tmpOut.clear();

    std::vector<double> gradResults;
    if(computeGradient) {
      for(int i = 0; i < npes; i++) {
        sendCnts[i] *= 3;
        sendDisps[i] *= 3;
        recvCnts[i] *= 3;
        recvDisps[i] *= 3;
      }//end for i
      gradResults.resize(3*dof*numPts);

      double* tmpGradOutPtr = NULL;
      double* gradResultsPtr = NULL;

      if(!(tmpGradOut.empty())) {
        tmpGradOutPtr = (&(*(tmpGradOut.begin())));
      }

      if(!(gradResults.empty())) {
        gradResultsPtr = (&(*(gradResults.begin())));
      }

      par::Mpi_Alltoallv_sparse<double >( tmpGradOutPtr, recvCnts, recvDisps,
          gradResultsPtr, sendCnts, sendDisps, comm);
      tmpGradOut.clear();
    }

    assert(sendCnts);
    delete [] sendCnts;
    sendCnts = NULL;

    assert(sendDisps);
    delete [] sendDisps;
    sendDisps = NULL;

    assert(recvCnts);
    delete [] recvCnts;
    recvCnts = NULL;

    assert(recvDisps);
    delete [] recvDisps;
    recvDisps = NULL;

    //Use commMap and re-order the results in the same order as the original
    //points
    out.resize(dof*numPts);
    for(int i = 0; i < numPts; i++) {
      for(int j = 0; j < dof; j++) {
        out[(dof*commMap[i]) + j] = results[(dof*i) + j];
      }//end for j
    }//end for i

    if(computeGradient) {
      gradOut->resize(3*dof*numPts);
      for(int i = 0; i < numPts; i++) {
        for(int j = 0; j < 3*dof; j++) {
          (*gradOut)[(3*dof*commMap[i]) + j] = gradResults[(3*dof*i) + j];
        }//end for j
      }//end for i
    }

    if(commMap) {
      delete [] commMap;
      commMap = NULL;
    }

  }//end function
  */

  void getNodeCoordinates(ot::DA* da, std::vector<double> &pts, const double* problemSize) {
    DendroIntL localNodeSize = da->getNodeSize();

    pts.clear();
    pts.resize(localNodeSize*3);
    // std::cout << da->getRankAll() << "pts size: " << localNodeSize << " --> " << pts.size() << "\n";
    unsigned int maxD = da->getMaxDepth();
    unsigned int lev;
    double hx, hy, hz;
    Point pt;

  
  std::vector<DendroIntL> NonGhostNodes(localNodeSize); 
  for(DendroIntL i = 0; i < localNodeSize; i++) {
    NonGhostNodes[i] = i;   
  }

  DendroIntL* node_map;
  da->vecGetBuffer<DendroIntL>(NonGhostNodes, node_map, false, false, true, 1);

  unsigned int postG_beg = da->getIdxPostGhostBegin();
  unsigned int elem_beg = da->getIdxElementBegin();

  unsigned int domain_max = 1u << (maxD - 1);
  DendroIntL index;
  double xFac = problemSize[0] / ((double) (1 << (maxD - 1)));
  double yFac = problemSize[1] / ((double) (1 << (maxD - 1)));
  double zFac = problemSize[2] / ((double) (1 << (maxD - 1)));
  double xx[8], yy[8], zz[8];
  unsigned int idx[8];
  for ( da->init<ot::DA_FLAGS::ALL>(); da->curr() < da->end<ot::DA_FLAGS::ALL>(); da->next<ot::DA_FLAGS::ALL>() ) {

    da->getNodeIndices(idx);
    pt = da->getCurrentOffset();
    unsigned char hangingMask = da->getHangingNodeIndex(da->curr());
    if (!(hangingMask & (1u << 0)) && (idx[0] >= elem_beg) && (idx[0] < postG_beg))
    {
        //! get the correct coordinates of the nodes ...
        xx[0] = pt.x() * xFac;
        yy[0] = pt.y() * yFac;
        zz[0] = pt.z() * zFac;
        index = 3*node_map[idx[0]];
        pts.at(index) = xx[0];
        pts.at(index+1) = yy[0];
        pts.at(index+2) = zz[0];
        // std::cout << da->curr() << " -> " << idx[0] << std::endl;
    }

    if (da->isBoundaryOctant())
    {
      // std::cout << "=== Boundary ===" << std::endl;
        lev = da->getLevel(da->curr());
        hx = xFac * (1 << (maxD - lev));
        hy = yFac * (1 << (maxD - lev));
        hz = zFac * (1 << (maxD - lev));
        xx[0] = pt.x() * xFac;
        yy[0] = pt.y() * yFac;
        zz[0] = pt.z() * zFac;
        xx[1] = pt.x() * xFac + hx;
        yy[1] = pt.y() * yFac;
        zz[1] = pt.z() * zFac;
        xx[2] = pt.x() * xFac;
        yy[2] = pt.y() * yFac + hy;
        zz[2] = pt.z() * zFac;
        xx[3] = pt.x() * xFac + hx;
        yy[3] = pt.y() * yFac + hy;
        zz[3] = pt.z() * zFac;

        xx[4] = pt.x() * xFac;
        yy[4] = pt.y() * yFac;
        zz[4] = pt.z() * zFac + hz;
        xx[5] = pt.x() * xFac + hx;
        yy[5] = pt.y() * yFac;
        zz[5] = pt.z() * zFac + hz;
        xx[6] = pt.x() * xFac;
        yy[6] = pt.y() * yFac + hy;
        zz[6] = pt.z() * zFac + hz;
        xx[7] = pt.x() * xFac + hx;
        yy[7] = pt.y() * yFac + hy;
        zz[7] = pt.z() * zFac + hz;

        for (int a=0; a<8; a++)
        {
            if (!(hangingMask & (1u << a)))
            {
                // boundary at x = 1, y = 1, z = 1
                if ( ( idx[a] >= elem_beg ) && ( idx[a] < postG_beg) &&  (fabs(xx[a] - problemSize[0]) < hx/2 || fabs(yy[a] - problemSize[1]) < hy/2 || fabs(zz[a] - problemSize[2]) < hz/2) )
                {
                    // add node
                    // std::cout << idx[a]*3 << " " << idx[a]*3 + 1 << " " << idx[a]*3 + 2 << "\n"; 
                    index = 3*node_map[idx[a]];
                    // std::cout <<  da->getRankAll() <<  ">>= " << da->curr() << " -> " << node_map[idx[a]] << " =<< (" << xx[a] << ", " << yy[a] << ", " << zz[a] << ") " << std::endl;
                    pts[index] = xx[a];
                    pts[index+1] = yy[a];
                    pts[index+2] = zz[a];
                }
            }
        } // for
        // std::cout << "=== ===" << std::endl;
    } // is Boundary
  } // loop over Writable

  da->vecRestoreBuffer(NonGhostNodes, node_map, false, false, true, 1);

  } // function.

void getNodeCoordinates(ot::subDA* da, std::vector<double> &pts, const double* problemSize, const double* subDA_max) {
    DendroIntL localNodeSize = da->getNodeSize();

    pts.clear();
    pts.resize(localNodeSize*3);
    // std::cout << da->getRankAll() << "pts size: " << localNodeSize << " --> " << pts.size() << "\n";
    unsigned int maxD = da->getMaxDepth();
    unsigned int lev;
    double hx, hy, hz;
    Point pt;

  
    std::vector<DendroIntL> NonGhostNodes(localNodeSize); 
    for(DendroIntL i = 0; i < localNodeSize; i++) {
      NonGhostNodes[i] = i;   
    }

    DendroIntL* node_map;
    da->vecGetBuffer<DendroIntL>(NonGhostNodes, node_map, false, false, true, 1);

    // get da2sub map

    unsigned int postG_beg = da->getIdxPostGhostBegin();
    unsigned int elem_beg = da->getIdxElementBegin();

    unsigned int domain_max = 1u << (maxD - 1);
    DendroIntL index;
    double xFac = problemSize[0] / ((double) (1 << (maxD - 1)));
    double yFac = problemSize[1] / ((double) (1 << (maxD - 1)));
    double zFac = problemSize[2] / ((double) (1 << (maxD - 1)));
    double xx[8], yy[8], zz[8];
    
    unsigned int idx[8];
    // unsigned int sub_idx[8];

    ot::DA* main_da = da->global_domain();

    // loop over DA elements
    // If regular element ... and not skipped, add node ... 
    // if boundary octact 


    for ( main_da->init<ot::DA_FLAGS::ALL>(); main_da->curr() < main_da->end<ot::DA_FLAGS::ALL>(); main_da->next<ot::DA_FLAGS::ALL>() ) {

      main_da->getNodeIndices(idx);
      // da->getNodeIndices(sub_idx);

      pt = main_da->getCurrentOffset();
      unsigned char hangingMask = main_da->getHangingNodeIndex(main_da->curr());

          //! get the correct coordinates of the nodes ...
          xx[0] = pt.x() * xFac;
          yy[0] = pt.y() * yFac;
          zz[0] = pt.z() * zFac;

        if (!da->skipElem(main_da->curr()) ) {
          unsigned int sub_idx = da->getDA2SubNode(idx[0]);
          if (!(hangingMask & (1u << 0)) && (sub_idx >= elem_beg) && (sub_idx < postG_beg))
          {        
            index = 3*node_map[sub_idx];
            pts.at(index) = xx[0];
            pts.at(index+1) = yy[0];
            pts.at(index+2) = zz[0];
          } 
        } // if not skipped

          lev = main_da->getLevel(main_da->curr());
          hx = xFac * (1 << (maxD - lev));
          hy = yFac * (1 << (maxD - lev));
          hz = zFac * (1 << (maxD - lev));
          // std::cout << da->curr() << " -> " << idx[0] << std::endl;

      // if ( main_da->isBoundaryOctant() && 
      // if ( (fabs(xx[0] - subDA_max[0]) < (hx+xFac) || fabs(yy[0] - subDA_max[1]) < (hy+yFac) || fabs(zz[0] - subDA_max[2]) < (hz + zFac) ) )
      // {
        // std::cout << "=== Boundary ===" << std::endl;
          
          xx[0] = pt.x() * xFac;
          yy[0] = pt.y() * yFac;
          zz[0] = pt.z() * zFac;

          xx[1] = pt.x() * xFac + hx;
          yy[1] = pt.y() * yFac;
          zz[1] = pt.z() * zFac;
          
          xx[2] = pt.x() * xFac;
          yy[2] = pt.y() * yFac + hy;
          zz[2] = pt.z() * zFac;
          
          xx[3] = pt.x() * xFac + hx;
          yy[3] = pt.y() * yFac + hy;
          zz[3] = pt.z() * zFac;

          xx[4] = pt.x() * xFac;
          yy[4] = pt.y() * yFac;
          zz[4] = pt.z() * zFac + hz;
          
          xx[5] = pt.x() * xFac + hx;
          yy[5] = pt.y() * yFac;
          zz[5] = pt.z() * zFac + hz;
          
          xx[6] = pt.x() * xFac;
          yy[6] = pt.y() * yFac + hy;
          zz[6] = pt.z() * zFac + hz;
          
          xx[7] = pt.x() * xFac + hx;
          yy[7] = pt.y() * yFac + hy;
          zz[7] = pt.z() * zFac + hz;

          for (int a=0; a<8; a++)
          {
            // std::cout <<  " =<< (" << xx[a] << ", " << yy[a] << ", " << zz[a] << ") " << std::endl;
              // if (!(hangingMask & (1u << a)))
              if ( fabs(xx[a] - subDA_max[0]) < xFac || fabs(yy[a] - subDA_max[1]) < yFac || fabs(zz[a] - subDA_max[2]) < zFac )
              {
                  // boundary at x = 1, y = 1, z = 1
                  unsigned int sub_idx = da->getDA2SubNode(idx[a]);
                  if ( ( sub_idx >= elem_beg ) && ( sub_idx < postG_beg) ) // &&  (fabs(xx[a] - subDA_max[0]) < hx || fabs(yy[a] - subDA_max[1]) < hy || fabs(zz[a] - subDA_max[2]) < hz) )
                  {
                      // add node
                      // std::cout << idx[a]*3 << " " << idx[a]*3 + 1 << " " << idx[a]*3 + 2 << "\n"; 
                      index = 3*node_map[sub_idx];
                      // std::cout <<  da->getRankAll() <<  ">>= " << da->curr() << " -> " << node_map[idx[a]] << " =<< (" << xx[a] << ", " << yy[a] << ", " << zz[a] << ") " << std::endl;
                      pts[index] = xx[a];
                      pts[index+1] = yy[a];
                      pts[index+2] = zz[a];
                  }
              }
          } // for
          // std::cout << "=== ===" << std::endl;
      // } // is Boundary
    } // loop over Writable

    da->vecRestoreBuffer(NonGhostNodes, node_map, false, false, true, 1);

  } // function.


  void writePartitionVTK(ot::DA* da, const char* outFileName) {
    int rank = da->getRankAll();
    //Only processor writes
    if(!rank) {
      std::vector<ot::TreeNode> minBlocks = da->getMinAllBlocks();
      unsigned int maxDepth = da->getMaxDepth();
      ot::TreeNode root(3, maxDepth);
      std::vector<ot::TreeNode> allBlocks;
      ot::completeSubtree(root, minBlocks, allBlocks, 3, maxDepth, true, true);

      FILE* outfile = fopen(outFileName,"w");

      //Set the weights of allBlocks to be the processor ids. 
      for(unsigned int i = 0; i < allBlocks.size(); i++) {
        unsigned int pId;
        bool found = seq::maxLowerBound<ot::TreeNode>(minBlocks, allBlocks[i], pId, NULL, NULL);
        assert(found);
        allBlocks[i].setWeight(pId);
      }

      unsigned int numNode = static_cast<unsigned int>(allBlocks.size());

      float coord[8][3] = {
        {0.0,0.0,0.0},
        {1.0,0.0,0.0},
        {0.0,1.0,0.0},
        {1.0,1.0,0.0},
        {0.0,0.0,1.0},
        {1.0,0.0,1.0},
        {0.0,1.0,1.0},
        {1.0,1.0,1.0}
      };

      fprintf(outfile,"# vtk DataFile Version 3.0\n");
      fprintf(outfile,"Octree field file\n");
      fprintf(outfile,"ASCII\n");
      fprintf(outfile,"DATASET UNSTRUCTURED_GRID\n");
      fprintf(outfile,"POINTS %d float\n",(numNode*8));

      for (unsigned int i = 0; i < numNode; i++) {
        unsigned int x = allBlocks[i].getX();
        unsigned int y = allBlocks[i].getY(); 
        unsigned int z = allBlocks[i].getZ();
        unsigned int d = allBlocks[i].getLevel();
        float fx, fy, fz,hx;
        fx = ((float)x)/((float)(1u<<maxDepth));
        fy = ((float)y)/((float)(1u<<maxDepth));
        fz = ((float)z)/((float)(1u<<maxDepth));
        hx = ((float)(1u<<(maxDepth-d)))/((float)(1u<<maxDepth));

        for(int j = 0; j < 8; j++)
        {
          float fxn,fyn,fzn;
          fxn = fx + coord[j][0]*hx;
          fyn = fy + coord[j][1]*hx;
          fzn = fz + coord[j][2]*hx;
          fprintf(outfile,"%f %f %f \n",fxn,fyn,fzn);
        }
      }

      fprintf(outfile,"\nCELLS %d %d\n",numNode,numNode*9);

      for(int i = 0; i < numNode; i++)
      {
        fprintf(outfile,"8 ");

        for(int j = 0; j < 8; j++)
        {
          int index = (8*i)+j;
          fprintf(outfile,"%d ",index);
        }
        fprintf(outfile,"\n");
      }

      fprintf(outfile,"\nCELL_TYPES %d\n",numNode);

      for(int i = 0; i < numNode; i++)
      {
        fprintf(outfile,"11 \n");
      }

      fprintf(outfile,"\nCELL_DATA %d\n",numNode);
      fprintf(outfile,"SCALARS scalars unsigned_int\n");
      fprintf(outfile,"LOOKUP_TABLE default\n");

      for (unsigned int i =0; i< numNode; i++) {
        unsigned int v = allBlocks[i].getWeight();
        fprintf(outfile,"%u \n", v);
      }

      fclose(outfile);

    }//end if p0
  }//end function

  unsigned int getGlobalMinLevel(ot::DA* da) {

    unsigned int myMinLev = ot::TreeNode::MAX_LEVEL;
    unsigned int globalMinLev;

    //It is sufficient to loop over the elements, since boundaries were added at the same level as some element. 
    if(da->iAmActive()) {
      for(da->init<ot::DA_FLAGS::WRITABLE>();
          da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
          da->next<ot::DA_FLAGS::WRITABLE>())  
      {
        unsigned int currLevel = da->getLevel(da->curr());
        if(currLevel < myMinLev) {
          myMinLev = currLevel;
        }
      }      
    }

    par::Mpi_Allreduce<unsigned int>(&myMinLev, &globalMinLev, 1, MPI_MIN, da->getComm() );

    //The initial octree is not root. So the min lev in the initial octree is atleast 1. So in the embedded octree the minlev is atleast 2. 
    assert(globalMinLev > 1);

    //Return the result in the original octree configuration
    return (globalMinLev -1);
  }

  unsigned int getGlobalMaxLevel(ot::DA* da) {

    unsigned int myMaxLev = 0;
    unsigned int globalMaxLev;

    //It is sufficient to loop over the elements, since boundaries were added at the same level as some element. 
    if( da->iAmActive() ) {
      for(da->init<ot::DA_FLAGS::WRITABLE>(); 
          da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
          da->next<ot::DA_FLAGS::WRITABLE>())  
      {
        unsigned int currLevel = da->getLevel(da->curr());
        if(currLevel > myMaxLev) {
          myMaxLev = currLevel;
        }
      }      
    }

    par::Mpi_Allreduce<unsigned int>(&myMaxLev, &globalMaxLev, 1, MPI_MAX, da->getComm() );

    //m_uiMaxDepth will be 0 on inActive processors.
    if(da->iAmActive()) {
      assert(globalMaxLev <= da->getMaxDepth() );
    }else {
      assert(globalMaxLev <= ot::TreeNode::MAX_LEVEL);
    }

    //Return the result in the original octree configuration
    return (globalMaxLev -1);	
  }

  unsigned int getSortOrder(unsigned int x, unsigned int y, 
      unsigned int z, unsigned int sz) {

    // first compare the x, y, and z to determine which one dominates ...
    unsigned int _x = x^(x+sz);
    unsigned int _y = y^(y+sz);
    unsigned int _z = z^(z+sz);

    // compute order ...
    if (_x > _y) {
      if ( _y > _z) {
        return ot::DA_FLAGS::ZYX;
      } else if ( _x > _z ) {
        return ot::DA_FLAGS::YZX;
      } else {
        return ot::DA_FLAGS::YXZ;
      }
    } else {
      if ( _y > _z) {
        return ot::DA_FLAGS::ZXY;
      } else if ( _x > _z ) {
        return ot::DA_FLAGS::ZXY;
      } else {
        return ot::DA_FLAGS::XYZ;
      }
    }
  }

  unsigned char getTouchConfig(const ot::TreeNode& curr, 
      const ot::TreeNode& next, unsigned int maxDepth) {
    unsigned char c = 0;
    unsigned int cx = curr.minX();
    unsigned int cy = curr.minY();
    unsigned int cz = curr.minZ();
    unsigned int nx = next.minX();
    unsigned int ny = next.minY();
    unsigned int nz = next.minZ();
    unsigned int cd = curr.getLevel();
    unsigned int nd = next.getLevel();
    unsigned int cs = (1u<<(maxDepth - cd));
    unsigned int ns = (1u<<(maxDepth - nd));

    //_zzyyxxT  
    bool isTouchingX = false;
    bool isTouchingY = false;
    bool isTouchingZ = false;

    if (cx == nx) {
      isTouchingX = true;
    }

    if (cx == (nx + ns) ) {
      isTouchingX = true;
      c += (1 << 1);
    }

    if ( (cx + cs) == nx ) {
      isTouchingX = true;
      c += (2 << 1);
    }

    if ( (cx + cs) == (nx + ns) ) {
      isTouchingX = true;
      c += (3 << 1);
    }

    if (cy == ny) {
      isTouchingY = true;
    }

    if (cy == (ny + ns) ) {
      isTouchingY = true;
      c += (1 << 3);
    }

    if ( (cy + cs) == ny ) {
      isTouchingY = true;
      c += (2 << 3);
    }

    if ( (cy + cs) == (ny + ns) ) {
      isTouchingY = true;
      c += (3 << 3);
    }

    if (cz == nz) {
      isTouchingZ = true;
    }

    if (cz == (nz + ns) ) {
      isTouchingZ = true;
      c += (1 << 5);
    }

    if ( (cz + cs) == nz ) {
      isTouchingZ = true;
      c += (2<<5);
    }

    if ( (cz + cs) == (nz + ns) ) {
      isTouchingZ = true;
      c += (3<<5);
    }

    if ( isTouchingX && isTouchingY && isTouchingZ ) {
      c += 1;
    } else {
      c = 0;
    }

    return c;
  }

  bool isRegularGrid(ot::DA* da) {
    int iHaveHanging = 0;
    if(da->iAmActive()) {
      for(da->init<ot::DA_FLAGS::WRITABLE>(); 
          da->curr() < da->end<ot::DA_FLAGS::WRITABLE>();
          da->next<ot::DA_FLAGS::WRITABLE>()) { 
        if(da->isHanging(da->curr())) {
          iHaveHanging = 1;
          std::cout<<(da->getRankActive())<<
            " found a hanging node for element with id: "
            <<(da->curr())<<std::endl;

          break;
        }
      }//end for writable
    }//end if active

    int anyOneHasHanging;
    par::Mpi_Allreduce<int>(&iHaveHanging, &anyOneHasHanging, 1, MPI_SUM, da->getComm());

    return (!anyOneHasHanging);
  }//end function

  void assignBoundaryFlags(ot::DA* da, 
      std::vector<unsigned char> & bdyFlagVec) {

    da->createVector(bdyFlagVec, false, false, 1);//Nodal, Non-ghosted, single dof

    for(int i = 0; i < bdyFlagVec.size(); i++) {
      bdyFlagVec[i] = 0;
    }//initialization loop

    unsigned char *bdyFlagArr = NULL;
    da->vecGetBuffer<unsigned char>(bdyFlagVec,bdyFlagArr,
        false,false,false,1);

    if(da->iAmActive()) {
      //We can only loop over the elements, hence the positive boundary elements
      //will add the flags for the external positive boundary nodes.
      for(da->init<ot::DA_FLAGS::ALL>(); 
          da->curr() < da->end<ot::DA_FLAGS::ALL>();
          da->next<ot::DA_FLAGS::ALL>()) { 
        unsigned char currentFlags;
        bool calledGetNodeIndices = false;
        if(da->isBoundaryOctant(&currentFlags)) {
          //The status of the anchor of any real octant is determined by the
          //negative face boundaries only
          int xNegBdy = (currentFlags & ot::TreeNode::X_NEG_BDY);
          int yNegBdy = (currentFlags & ot::TreeNode::Y_NEG_BDY);
          int zNegBdy = (currentFlags & ot::TreeNode::Z_NEG_BDY);

          unsigned char hnMask = da->getHangingNodeIndex(da->curr());
          if(!(hnMask & 1)) {
            //Anchor is not hanging           	    
            if(xNegBdy) {
              if(yNegBdy && zNegBdy){
                bdyFlagArr[da->curr()] = ot::TreeNode::CORNER_BDY;
              }else if(yNegBdy || zNegBdy) {
                bdyFlagArr[da->curr()] = ot::TreeNode::EDGE_BDY;
              }else {
                bdyFlagArr[da->curr()] = ot::TreeNode::FACE_BDY;
              }
            }else if(yNegBdy) {
              if(zNegBdy) {
                bdyFlagArr[da->curr()] = ot::TreeNode::EDGE_BDY;
              }else {
                bdyFlagArr[da->curr()] = ot::TreeNode::FACE_BDY;
              }
            }else if(zNegBdy) {
              bdyFlagArr[da->curr()] = ot::TreeNode::FACE_BDY;
            }
          }//end if anchor hanging

          if(currentFlags > ot::TreeNode::NEG_POS_DEMARCATION) {
            //Has at least one positive boundary
            //May have negative boundaries as well
            unsigned int indices[8];
            da->getNodeIndices(indices); 
            calledGetNodeIndices = true;
            int xPosBdy = (currentFlags & ot::TreeNode::X_POS_BDY);
            int yPosBdy = (currentFlags & ot::TreeNode::Y_POS_BDY);
            int zPosBdy = (currentFlags & ot::TreeNode::Z_POS_BDY);

            if(!(hnMask & (1 << 1))) {
              bool xBdy = xPosBdy;
              bool yBdy = yNegBdy;
              bool zBdy = zNegBdy;
              if(xBdy) {
                if(yBdy && zBdy){
                  bdyFlagArr[indices[1]] = ot::TreeNode::CORNER_BDY;
                }else if(yBdy || zBdy) {
                  bdyFlagArr[indices[1]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[1]] = ot::TreeNode::FACE_BDY;
                }
              }else if(yBdy) {
                if(zBdy) {
                  bdyFlagArr[indices[1]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[1]] = ot::TreeNode::FACE_BDY;
                }
              }else if(zBdy) {
                bdyFlagArr[indices[1]] = ot::TreeNode::FACE_BDY;
              }
            }

            if(!(hnMask & (1 << 2))) {
              bool xBdy = xNegBdy;
              bool yBdy = yPosBdy;
              bool zBdy = zNegBdy;
              if(xBdy) {
                if(yBdy && zBdy){
                  bdyFlagArr[indices[2]] = ot::TreeNode::CORNER_BDY;
                }else if(yBdy || zBdy) {
                  bdyFlagArr[indices[2]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[2]] = ot::TreeNode::FACE_BDY;
                }
              }else if(yBdy) {
                if(zBdy) {
                  bdyFlagArr[indices[2]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[2]] = ot::TreeNode::FACE_BDY;
                }
              }else if(zBdy) {
                bdyFlagArr[indices[2]] = ot::TreeNode::FACE_BDY;
              }
            }

            if(!(hnMask & (1 << 3))) {
              bool xBdy = xPosBdy;
              bool yBdy = yPosBdy;
              bool zBdy = zNegBdy;
              if(xBdy) {
                if(yBdy && zBdy){
                  bdyFlagArr[indices[3]] = ot::TreeNode::CORNER_BDY;
                }else if(yBdy || zBdy) {
                  bdyFlagArr[indices[3]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[3]] = ot::TreeNode::FACE_BDY;
                }
              }else if(yBdy) {
                if(zBdy) {
                  bdyFlagArr[indices[3]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[3]] = ot::TreeNode::FACE_BDY;
                }
              }else if(zBdy) {
                bdyFlagArr[indices[3]] = ot::TreeNode::FACE_BDY;
              }
            }

            if(!(hnMask & (1 << 4))) {
              bool xBdy = xNegBdy;
              bool yBdy = yNegBdy;
              bool zBdy = zPosBdy;
              if(xBdy) {
                if(yBdy && zBdy){
                  bdyFlagArr[indices[4]] = ot::TreeNode::CORNER_BDY;
                }else if(yBdy || zBdy) {
                  bdyFlagArr[indices[4]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[4]] = ot::TreeNode::FACE_BDY;
                }
              }else if(yBdy) {
                if(zBdy) {
                  bdyFlagArr[indices[4]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[4]] = ot::TreeNode::FACE_BDY;
                }
              }else if(zBdy) {
                bdyFlagArr[indices[4]] = ot::TreeNode::FACE_BDY;
              }
            }

            if(!(hnMask & (1 << 5))) {
              bool xBdy = xPosBdy;
              bool yBdy = yNegBdy;
              bool zBdy = zPosBdy;
              if(xBdy) {
                if(yBdy && zBdy){
                  bdyFlagArr[indices[5]] = ot::TreeNode::CORNER_BDY;
                }else if(yBdy || zBdy) {
                  bdyFlagArr[indices[5]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[5]] = ot::TreeNode::FACE_BDY;
                }
              }else if(yBdy) {
                if(zBdy) {
                  bdyFlagArr[indices[5]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[5]] = ot::TreeNode::FACE_BDY;
                }
              }else if(zBdy) {
                bdyFlagArr[indices[5]] = ot::TreeNode::FACE_BDY;
              }
            }

            if(!(hnMask & (1 << 6))) {
              bool xBdy = xNegBdy;
              bool yBdy = yPosBdy;
              bool zBdy = zPosBdy;
              if(xBdy) {
                if(yBdy && zBdy){
                  bdyFlagArr[indices[6]] = ot::TreeNode::CORNER_BDY;
                }else if(yBdy || zBdy) {
                  bdyFlagArr[indices[6]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[6]] = ot::TreeNode::FACE_BDY;
                }
              }else if(yBdy) {
                if(zBdy) {
                  bdyFlagArr[indices[6]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[6]] = ot::TreeNode::FACE_BDY;
                }
              }else if(zBdy) {
                bdyFlagArr[indices[6]] = ot::TreeNode::FACE_BDY;
              }
            }

            if(!(hnMask & (1 << 7))) {
              bool xBdy = xPosBdy;
              bool yBdy = yPosBdy;
              bool zBdy = zPosBdy;
              if(xBdy) {
                if(yBdy && zBdy){
                  bdyFlagArr[indices[7]] = ot::TreeNode::CORNER_BDY;
                }else if(yBdy || zBdy) {
                  bdyFlagArr[indices[7]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[7]] = ot::TreeNode::FACE_BDY;
                }
              }else if(yBdy) {
                if(zBdy) {
                  bdyFlagArr[indices[7]] = ot::TreeNode::EDGE_BDY;
                }else {
                  bdyFlagArr[indices[7]] = ot::TreeNode::FACE_BDY;
                }
              }else if(zBdy) {
                bdyFlagArr[indices[7]] = ot::TreeNode::FACE_BDY;
              }
            }
          }//end if-else has positive boundaries
        }//end if boundary
        if( (!calledGetNodeIndices) && (da->isLUTcompressed()) ) {
          da->updateQuotientCounter();
        }
      }//end for all own elements
    }//end if active

    da->vecRestoreBuffer<unsigned char>(bdyFlagVec,bdyFlagArr,false,false,false,1);
  }//end function

  void includeSiblingsOfBoundary(std::vector<ot::TreeNode>& allBoundaryLeaves, 
      const ot::TreeNode& myFirstOctant, const ot::TreeNode& myLastOctant) {
    PROF_ADD_BDY_SIBLINGS_BEGIN

      std::vector<ot::TreeNode> tmpAllBoundaryLeaves;

    for(unsigned int i = 0; i < allBoundaryLeaves.size(); i++) {
      ot::TreeNode thisOct = allBoundaryLeaves[i];
      ot::TreeNode thisOctParent = thisOct.getParent();
      //unsigned int thisCnum = ((unsigned int)(thisOct.getChildNumber()));
      // oda_change_milinda
      unsigned int thisCnum = ((unsigned int)(thisOct.getMortonIndex()));
      std::vector<ot::TreeNode> siblingsToAdd;
      thisOctParent.addChildrenMorton(siblingsToAdd);
      for(unsigned int j=0; j < 8; j++) {
        if( (j != thisCnum) && (j != (7-thisCnum)) ) {
          if( (siblingsToAdd[j] >= myFirstOctant) && 
              (siblingsToAdd[j] <= myLastOctant) ) {
            tmpAllBoundaryLeaves.push_back(siblingsToAdd[j]);
          }
        }
      }   
    }//end for i

    seq::makeVectorUnique<ot::TreeNode>(tmpAllBoundaryLeaves,false);

    std::vector<ot::TreeNode> tmp2Vec;

    unsigned int tmpCnt = 0;
    unsigned int bdyCnt = 0;

    //The two lists are independently sorted and unique, Now we do a linear
    //pass and merge them so that the result is also sorted and unique.


    assert(seq::test::isUniqueAndSorted(tmpAllBoundaryLeaves));

    while ( (tmpCnt < tmpAllBoundaryLeaves.size()) &&
        (bdyCnt < allBoundaryLeaves.size()) ) {
      if ( tmpAllBoundaryLeaves[tmpCnt] < allBoundaryLeaves[bdyCnt] ) {
        tmp2Vec.push_back(tmpAllBoundaryLeaves[tmpCnt]);
        tmpCnt++;
      }else if( tmpAllBoundaryLeaves[tmpCnt] > allBoundaryLeaves[bdyCnt] ) {
        tmp2Vec.push_back(allBoundaryLeaves[bdyCnt]);
        bdyCnt++;
      }else {
        tmp2Vec.push_back(allBoundaryLeaves[bdyCnt]);
        bdyCnt++;
        tmpCnt++; //tmpCnt must also be incremented to preserve uniqueness.
      }
    }

    while (bdyCnt < allBoundaryLeaves.size()) {
      tmp2Vec.push_back(allBoundaryLeaves[bdyCnt]);
      bdyCnt++;
    }

    while (tmpCnt < tmpAllBoundaryLeaves.size()) {
      tmp2Vec.push_back(tmpAllBoundaryLeaves[tmpCnt]);
      tmpCnt++;
    }

    allBoundaryLeaves = tmp2Vec;

    tmp2Vec.clear();
    tmpAllBoundaryLeaves.clear();

    PROF_ADD_BDY_SIBLINGS_END
  }//end function

  void prepareAprioriCommMessagesInDAtype1(const std::vector<ot::TreeNode>& in,
      std::vector<ot::TreeNode>& allBoundaryLeaves, std::vector<ot::TreeNode>& blocks,
      const std::vector<ot::TreeNode>& allBlocks, int myRank, int npes, int* sendCnt,
      std::vector<std::vector<unsigned int> >& sendNodes) {
    PROF_DA_APRIORI_COMM_BEGIN

      std::vector<unsigned int> bdy2elem;

    unsigned int bdyCnt = 0;
    unsigned int allCnt = 0;

    while (bdyCnt < allBoundaryLeaves.size()) {
      if ( allBoundaryLeaves[bdyCnt] < in[allCnt]) {
        bdyCnt++;
      } else if ( allBoundaryLeaves[bdyCnt] > in[allCnt]) {
        allCnt++;
      } else {
        //Both are equal.		
        bdy2elem.push_back(allCnt);
        bdyCnt++;
      }
    }

    //This step is necessary because some elements of allBoundaryLeaves were not
    //copied from the "in" vector, instead we generated them directly. So these
    //octants will not have correct flags set. So we find the corresponding copy
    //in the "in" vector.  
    allBoundaryLeaves.clear();
    for(unsigned int i = 0; i < bdy2elem.size(); i++) {
      allBoundaryLeaves.push_back(in[bdy2elem[i]]);
    }

    // 3. Reduce the list of global blocks to a smaller list that only has the
    // blocks which neighbour ones own blocks.

    //First mark your own blocks as being singular or not.
    unsigned int allBdyCnt = 0;
    for (unsigned int i = 0; i < blocks.size(); i++) {
      while( (allBdyCnt < allBoundaryLeaves.size()) &&
          (allBoundaryLeaves[allBdyCnt] < blocks[i]) ) {
        allBdyCnt++;
      }
      //Note, even if a block is Singular but is
      //not a ghost candidates (i.e., it is completely internal).
      // It will be treated as being NOT singular.
      bool isSingular = false;
      if( (allBdyCnt < allBoundaryLeaves.size()) &&
          (allBoundaryLeaves[allBdyCnt] == blocks[i]) ) {
        isSingular = true;
      }
      if(isSingular) {
        blocks[i].setWeight(1);
      }else {
        blocks[i].setWeight(0);
      }
    }

    std::vector<ot::TreeNode> myNhBlocks;
    for (int j = 0; j < blocks.size(); j++) {
      unsigned int myMaxX;
      unsigned int myMaxY;
      unsigned int myMaxZ;
      unsigned int myMinX;
      unsigned int myMinY;
      unsigned int myMinZ;
      if(blocks[j].getWeight()) {
        //Since the subsequent selection is done based on the octant's parent's
        //neighbours, Singular blocks must be handled differently.
        myMaxX = blocks[j].getParent().maxX();
        myMaxY = blocks[j].getParent().maxY();
        myMaxZ = blocks[j].getParent().maxZ();
        myMinX = blocks[j].getParent().minX();
        myMinY = blocks[j].getParent().minY();
        myMinZ = blocks[j].getParent().minZ();
      }else {
        myMaxX = blocks[j].maxX();
        myMaxY = blocks[j].maxY();
        myMaxZ = blocks[j].maxZ();
        myMinX = blocks[j].minX();
        myMinY = blocks[j].minY();
        myMinZ = blocks[j].minZ();
      }
      double myLenX = (double)(myMaxX-myMinX);
      double myLenY = (double)(myMaxY-myMinY);
      double myLenZ = (double)(myMaxZ-myMinZ);
      double myXc  = ((double)(myMinX + myMaxX))/2.0;
      double myYc  = ((double)(myMinY + myMaxY))/2.0;
      double myZc  = ((double)(myMinZ + myMaxZ))/2.0;
      for (int k = 0; k < allBlocks.size(); k++) {
        if ( (allBlocks[k] >= blocks[0]) && 
            (allBlocks[k] <= blocks[blocks.size()-1]) ) {
          //ignore my own blocks
          continue;
        }
        unsigned int hisMinX = allBlocks[k].minX();
        unsigned int hisMaxX = allBlocks[k].maxX();
        unsigned int hisMinY = allBlocks[k].minY();
        unsigned int hisMaxY = allBlocks[k].maxY();
        unsigned int hisMinZ = allBlocks[k].minZ();
        unsigned int hisMaxZ = allBlocks[k].maxZ();
        double hisLenX = (double)(hisMaxX-hisMinX);
        double hisLenY = (double)(hisMaxY-hisMinY);
        double hisLenZ = (double)(hisMaxZ-hisMinZ);
        double hisXc  = ((double)(hisMinX + hisMaxX))/2.0;
        double hisYc  = ((double)(hisMinY + hisMaxY))/2.0;
        double hisZc  = ((double)(hisMinZ + hisMaxZ))/2.0;
        double deltaX = ( (hisXc > myXc) ? (hisXc - myXc) : (myXc - hisXc ) );
        double deltaY = ( (hisYc > myYc) ? (hisYc - myYc) : (myYc - hisYc ) );
        double deltaZ = ( (hisZc > myZc) ? (hisZc - myZc) : (myZc - hisZc ) );

        //Note: This test will pass if the octants intersect (say one is an
        //ancestor of another) or they simply touch externally.

        if ((deltaX <= ((hisLenX+myLenX)/2.0)) &&
            (deltaY <= ((hisLenY+myLenY)/2.0)) &&
            (deltaZ <= ((hisLenZ+myLenZ)/2.0))) {
          //We touch
          myNhBlocks.push_back(allBlocks[k]);
        }//end if
      }//end for k
    }//end for j

    //This also sorts myNhBlocks
    seq::makeVectorUnique<ot::TreeNode>(myNhBlocks,false);

    sendNodes.resize(npes);
    for (int i=0; i < npes; i++) {
      sendNodes[i].clear();
      sendCnt[i] = 0;
    } 

    for (int j=0; j<allBoundaryLeaves.size(); j++) {
      //It is important to make the selection using the parent. This tackles
      //the cases where some nodes of an octant are hanging and so even if the octant itself
      //does not touch a processor, its parent does and so the replacement for
      //the hanging node will be mapped to that processor. This also handles
      //the case where two or more siblings belong to different processors. By
      //making the test using the parent they will be sent to each other.
      //Moreover, some of the nodes of these children could be hanging and the
      //replacement could belong to a third processor. Even in this case, the
      //octants will be sent to the correct processors. 

      ot::TreeNode parNode = allBoundaryLeaves[j].getParent();
      unsigned int myMaxX = parNode.maxX();
      unsigned int myMaxY = parNode.maxY();
      unsigned int myMaxZ = parNode.maxZ();
      unsigned int myMinX = parNode.minX();
      unsigned int myMinY = parNode.minY();
      unsigned int myMinZ = parNode.minZ();
      double myLenX = (double)(myMaxX-myMinX);
      double myLenY = (double)(myMaxY-myMinY);
      double myLenZ = (double)(myMaxZ-myMinZ);
      double myXc  = ((double)(myMinX + myMaxX))/2.0;
      double myYc  = ((double)(myMinY + myMaxY))/2.0;
      double myZc  = ((double)(myMinZ + myMaxZ))/2.0;
      unsigned int lastP = npes;
      for (int k = 0; k < myNhBlocks.size(); k++) {
        if (myNhBlocks[k].getWeight() == lastP) {
          continue;
        }
        unsigned int hisMinX = myNhBlocks[k].minX();
        unsigned int hisMaxX = myNhBlocks[k].maxX();
        unsigned int hisMinY = myNhBlocks[k].minY();
        unsigned int hisMaxY = myNhBlocks[k].maxY();
        unsigned int hisMinZ = myNhBlocks[k].minZ();
        unsigned int hisMaxZ = myNhBlocks[k].maxZ();
        double hisLenX = (double)(hisMaxX-hisMinX);
        double hisLenY = (double)(hisMaxY-hisMinY);
        double hisLenZ = (double)(hisMaxZ-hisMinZ);
        double hisXc  = ((double)(hisMinX + hisMaxX))/2.0;
        double hisYc  = ((double)(hisMinY + hisMaxY))/2.0;
        double hisZc  = ((double)(hisMinZ + hisMaxZ))/2.0;
        double deltaX = ( (hisXc > myXc) ? (hisXc - myXc) : (myXc - hisXc ) );
        double deltaY = ( (hisYc > myYc) ? (hisYc - myYc) : (myYc - hisYc ) );
        double deltaZ = ( (hisZc > myZc) ? (hisZc - myZc) : (myZc - hisZc ) );

        //Note: This test will pass if the octants intersect (say one is an
        //ancestor of another) or they simply touch externally.

        if ((deltaX <= ((hisLenX+myLenX)/2.0)) &&
            (deltaY <= ((hisLenY+myLenY)/2.0)) &&
            (deltaZ <= ((hisLenZ+myLenZ)/2.0))) {
          //We touch
          sendNodes[myNhBlocks[k].getWeight()].push_back(bdy2elem[j]); 
          sendCnt[myNhBlocks[k].getWeight()]++;
          lastP = myNhBlocks[k].getWeight();
        }//end if
      }//end for k
    }//end for j

    PROF_DA_APRIORI_COMM_END
  }//end function

  void prepareAprioriCommMessagesInDAtype2(const std::vector<ot::TreeNode>& in,
      std::vector<ot::TreeNode>& allBoundaryLeaves, std::vector<ot::TreeNode>& blocks,
      const std::vector<ot::TreeNode>& minsOfBlocks, int myRank, int npes, int* sendCnt,
      std::vector<std::vector<unsigned int> >& sendNodes) {
    PROF_DA_APRIORI_COMM_BEGIN

      std::vector<unsigned int> bdy2elem;

    unsigned int bdyCnt = 0;
    unsigned int allCnt = 0;
    unsigned int maxDepth = in[0].getMaxDepth();
    unsigned int dim = in[0].getDim(); 

    while (bdyCnt < allBoundaryLeaves.size()) {
      if ( allBoundaryLeaves[bdyCnt] < in[allCnt]) {
        bdyCnt++;
      } else if ( allBoundaryLeaves[bdyCnt] > in[allCnt]) {
        allCnt++;
      } else {
        //Both are equal.		
        bdy2elem.push_back(allCnt);
        bdyCnt++;
      }
    }

    //This step is necessary because some elements of allBoundaryLeaves were not
    //copied from the "in" vector, instead we generated them directly. So these
    //octants will not have correct flags set. So we find the corresponding copy
    //in the "in" vector.  
    allBoundaryLeaves.clear();
    for(unsigned int i = 0; i < bdy2elem.size(); i++) {
      allBoundaryLeaves.push_back(in[bdy2elem[i]]);
    }

    sendNodes.resize(npes);
    for (int i = 0; i < npes; i++) {
      sendNodes[i].clear();
      sendCnt[i] = 0;
    } 

    for (int j = 0; j < allBoundaryLeaves.size(); j++) {
      //It is important to make the selection using the parent. This tackles
      //the cases where some nodes of an octant are hanging and so even if the octant itself
      //does not touch a processor, its parent does and so the replacement for
      //the hanging node will be mapped to that processor. This also handles
      //the case where two or more siblings belong to different processors. By
      //making the test using the parent they will be sent to each other.
      //Moreover, some of the nodes of these children could be hanging and the
      //replacement could belong to a third processor. Even in this case, the
      //octants will be sent to the correct processors. 

      //1. We must not miss any pre-ghost elements that point to one of our own
      //octants, i.e. any pre-ghost element that we integrate over. This is
      //because we need to build the nlist for these octants as well.
      //2. We might get some extra ghosts they will be marked as FOREIGN and
      //will be ignored.
      //3. We must try to get as many post-ghost octants as possible to avoid
      //misses later and hence reduce subsequent communication. We should get
      //all direct post-ghost octants.
      //4. Post-ghosts are read-only and so it is sufficient to test for
      //post-ghost using its anchor
      std::vector<ot::TreeNode> myVertices;
      std::vector<ot::TreeNode> parVertices;
      std::vector<ot::TreeNode> anchorMirrors;

      unsigned int myX = allBoundaryLeaves[j].getX();
      unsigned int myY = allBoundaryLeaves[j].getY();
      unsigned int myZ = allBoundaryLeaves[j].getZ();

      //keys to check if you are a pre-ghost
      //Positive boundaries will not be pre-ghosts
      if(!(allBoundaryLeaves[j].getFlag() & ot::TreeNode::BOUNDARY)) {
        unsigned int myLev = allBoundaryLeaves[j].getLevel();
        unsigned int mySz = (1u<<(maxDepth - myLev));

        //All vertices except my anchor. Since my anchor belongs to my processor
        myVertices.push_back(ot::TreeNode((myX + mySz), myY, myZ, maxDepth, dim, maxDepth));
        myVertices.push_back(ot::TreeNode(myX, (myY + mySz), myZ, maxDepth, dim, maxDepth));
        myVertices.push_back(ot::TreeNode((myX + mySz), (myY + mySz), myZ, maxDepth, dim, maxDepth));
        myVertices.push_back(ot::TreeNode(myX, myY, (myZ + mySz), maxDepth, dim, maxDepth));
        myVertices.push_back(ot::TreeNode((myX + mySz), myY, (myZ + mySz), maxDepth, dim, maxDepth));
        myVertices.push_back(ot::TreeNode(myX, (myY + mySz), (myZ + mySz), maxDepth, dim, maxDepth));
        myVertices.push_back(ot::TreeNode((myX + mySz), (myY + mySz), (myZ + mySz), maxDepth, dim, maxDepth));


        ot::TreeNode parNode = allBoundaryLeaves[j].getParent();
        //@hari: Is this correct for Hilbert Ordering.
        //unsigned int myCnum = allBoundaryLeaves[j].getChildNumber();

        unsigned int myCnum = allBoundaryLeaves[j].getMortonIndex();
        unsigned int parX = parNode.getX();
        unsigned int parY = parNode.getY();
        unsigned int parZ = parNode.getZ();
        unsigned int parLev = parNode.getLevel();
        unsigned int parSz = (1u<<(maxDepth - parLev));

        //vertices numbers myCnum and (7 - myCnum) can't be hanging and so they
        //will not be mapped to the corresponding vertices of my parent
        if( (myCnum != 0) && (myCnum != 7) ) {
          parVertices.push_back(ot::TreeNode(parX, parY, parZ, maxDepth, dim, maxDepth));
        }
        if( (myCnum != 1) && (myCnum != 6) ) {
          parVertices.push_back(ot::TreeNode((parX + parSz), parY, parZ, maxDepth, dim, maxDepth));
        }
        if( (myCnum != 2) && (myCnum != 5) ) {
          parVertices.push_back(ot::TreeNode(parX, (parY + parSz), parZ, maxDepth, dim, maxDepth));
        }
        if( (myCnum != 3) && (myCnum != 4) ) {
          parVertices.push_back(ot::TreeNode((parX + parSz), (parY + parSz), parZ, maxDepth, dim, maxDepth));
        }
        if( (myCnum != 4) && (myCnum != 3) ) {
          parVertices.push_back(ot::TreeNode(parX, parY, (parZ + parSz), maxDepth, dim, maxDepth));
        }
        if( (myCnum != 5) && (myCnum != 2) ) {
          parVertices.push_back(ot::TreeNode((parX + parSz), parY, (parZ + parSz), maxDepth, dim, maxDepth));
        }
        if( (myCnum != 6) && (myCnum != 1) ) {
          parVertices.push_back(ot::TreeNode(parX, (parY + parSz), (parZ + parSz), maxDepth, dim, maxDepth));
        }
        if( (myCnum != 7) && (myCnum != 0) ) {
          parVertices.push_back(ot::TreeNode((parX + parSz), (parY + parSz),
                (parZ + parSz), maxDepth, dim, maxDepth));
        }

      }//end if positive boundary


      //Keys to check if you are a post-ghost
      //If the anchor is hanging we need not send it. When some processor searches
      //for this node as its primary key and does not find it, it will be
      //understood that this is hanging  
      if(allBoundaryLeaves[j].getFlag() & ot::TreeNode::NODE) {
        //-x
        if( myX ) {
          anchorMirrors.push_back(ot::TreeNode((myX - 1), myY, myZ, 
                maxDepth, dim, maxDepth));
        }
        //-y
        if( myY ) {
          anchorMirrors.push_back(ot::TreeNode(myX, (myY - 1), myZ, 
                maxDepth, dim, maxDepth));
        }
        //-z
        if( myZ ) {
          anchorMirrors.push_back(ot::TreeNode(myX, myY, (myZ - 1), 
                maxDepth, dim, maxDepth));
        }
        //-xy
        if( myX && myY ) {
          anchorMirrors.push_back(ot::TreeNode((myX - 1), (myY - 1), myZ, 
                maxDepth, dim, maxDepth));
        }
        //-yz
        if( myY && myZ ) {
          anchorMirrors.push_back(ot::TreeNode(myX, (myY - 1), (myZ - 1),
                maxDepth, dim, maxDepth));
        }
        //-zx
        if( myZ && myX ) {
          anchorMirrors.push_back(ot::TreeNode((myX - 1), myY, (myZ - 1),
                maxDepth, dim, maxDepth));
        }
        //-xyz
        if( myX && myY && myZ ) {
          anchorMirrors.push_back(ot::TreeNode((myX - 1), (myY - 1), (myZ - 1),
                maxDepth, dim, maxDepth));
        }
      }//end if hanging anchor


      //assert(par::test::isUniqueAndSorted(minsOfBlocks,MPI_COMM_WORLD));
      assert(seq::test::isSorted(minsOfBlocks));

      std::vector<unsigned int> pIds;
      for(int k = 0; k < parVertices.size(); k++) {
        unsigned int idx;
        seq::maxLowerBound<ot::TreeNode>(minsOfBlocks, parVertices[k], idx, NULL, NULL);
        pIds.push_back(idx);
      }

      for(int k = 0; k < anchorMirrors.size(); k++) {
        unsigned int idx;
        seq::maxLowerBound<ot::TreeNode>(minsOfBlocks, anchorMirrors[k], idx, NULL, NULL);
        pIds.push_back(idx);
      }

      for(int k = 0; k < myVertices.size(); k++) {
        unsigned int idx;
        seq::maxLowerBound<ot::TreeNode>(minsOfBlocks, myVertices[k], idx, NULL, NULL);
        pIds.push_back(idx);
      }//end for k

      //Do not send the same octant to the same processor twice 
      seq::makeVectorUnique<unsigned int>(pIds, false);

      for(int k = 0; k < pIds.size(); k++) {
        //Send to processor pIds[k] 
        //Only send to other processors  
        if(pIds[k] != myRank) {
          sendNodes[pIds[k]].push_back(bdy2elem[j]); 
          sendCnt[pIds[k]]++;
        }
      }//end for k
    }//end for j

    PROF_DA_APRIORI_COMM_END
  }//end function

  void DA_Initialize(MPI_Comm comm) {
    PROF_DA_INIT_BEGIN 

#ifdef __USE_MG_INIT_TYPE3__
      createShapeFnCoeffs_Type3(comm);
#else
#ifdef __USE_MG_INIT_TYPE2__
    createShapeFnCoeffs_Type2(comm);
#else
    createShapeFnCoeffs_Type1(comm);
#endif
#endif

    PROF_DA_INIT_END 
  }

  void DA_Finalize() {
    PROF_DA_FINAL_BEGIN 

      for(unsigned int cNum = 0; cNum < 8; cNum++) {
        for(unsigned int eType = 0; eType < 18; eType++) {
          for(unsigned int i = 0; i < 8; i++) {
            delete [] (ShapeFnCoeffs[cNum][eType][i]);
            ShapeFnCoeffs[cNum][eType][i] = NULL;
          }
          delete [] (ShapeFnCoeffs[cNum][eType]);
          ShapeFnCoeffs[cNum][eType] = NULL;
        }
        delete [] (ShapeFnCoeffs[cNum]);
        ShapeFnCoeffs[cNum] = NULL;
      }

    delete [] ShapeFnCoeffs;
    ShapeFnCoeffs = NULL;



    PROF_DA_FINAL_END 
  }

  int createShapeFnCoeffs_Type3(MPI_Comm comm) {
    FILE* infile;
    int rank, res;
    MPI_Comm_rank(comm, &rank); 
    char fname[250];
    sprintf(fname,"ShapeFnCoeffs_%d.inp", rank);
    infile = fopen(fname,"r");
    if(!infile) {
      std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
      assert(false);
    }

    typedef double* doublePtr;
    typedef doublePtr* double2Ptr;
    typedef double2Ptr* double3Ptr;

    ShapeFnCoeffs = new double3Ptr[8];
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      ShapeFnCoeffs[cNum] = new double2Ptr[18];
      for(unsigned int eType = 0; eType < 18; eType++) {
        ShapeFnCoeffs[cNum][eType] = new doublePtr[8];
        for(unsigned int i = 0; i < 8; i++) {
          ShapeFnCoeffs[cNum][eType][i] = new double[8];
          for(unsigned int j = 0; j < 8; j++) {
            res = fscanf(infile,"%lf",&(ShapeFnCoeffs[cNum][eType][i][j]));
          }
        }
      }
    }

    fclose(infile);
    return 1;
  }

  int createShapeFnCoeffs_Type2(MPI_Comm comm) {
    FILE* infile;

    int rank, npes, res;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    const int THOUSAND = 1000;
    int numGroups = (npes/THOUSAND);
    if( (numGroups*THOUSAND) < npes) {
      numGroups++;
    }

    MPI_Comm newComm;

    bool* isEmptyList = new bool[npes];
    assert(isEmptyList);
    for(int i = 0; i < numGroups; i++) {
      for(int j = 0; (j < (i*THOUSAND)) && (j < npes); j++) {
        isEmptyList[j] = true;
      }
      for(int j = (i*THOUSAND); (j < ((i+1)*THOUSAND)) && (j < npes); j++) {
        isEmptyList[j] = false;
      }
      for(int j = ((i + 1)*THOUSAND); j < npes; j++) {
        isEmptyList[j] = true;
      }
      MPI_Comm tmpComm;
      par::splitComm2way(isEmptyList, &tmpComm, comm);
      if(!(isEmptyList[rank])) {
        newComm = tmpComm;
      }
    }//end for i
    delete [] isEmptyList;

    if((rank % THOUSAND) == 0) {
      char fname[250];
      sprintf(fname,"ShapeFnCoeffs_%d.inp", (rank/THOUSAND));
      infile = fopen(fname,"r");
      if(!infile) {
        std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
        assert(false);
      }
    }

    typedef double* doublePtr;
    typedef doublePtr* double2Ptr;
    typedef double2Ptr* double3Ptr;

    ShapeFnCoeffs = new double3Ptr[8];
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      ShapeFnCoeffs[cNum] = new double2Ptr[18];
      for(unsigned int eType = 0; eType < 18; eType++) {
        ShapeFnCoeffs[cNum][eType] = new doublePtr[8];
        for(unsigned int i = 0; i < 8; i++) {
          ShapeFnCoeffs[cNum][eType][i] = new double[8];
          if((rank % THOUSAND) == 0){
            for(unsigned int j = 0; j < 8; j++) {
              res = fscanf(infile,"%lf",&(ShapeFnCoeffs[cNum][eType][i][j]));
            }
          }
        }
      }
    }

    if((rank % THOUSAND) == 0){
      fclose(infile);
    }

    double * tmpMat = new double[9216];
    assert(tmpMat);

    if((rank % THOUSAND) == 0) {
      unsigned int ctr = 0;
      for(unsigned int cNum = 0; cNum < 8; cNum++) {
        for(unsigned int eType = 0; eType < 18; eType++) {
          for(unsigned int i = 0; i < 8; i++) {
            for(unsigned int j = 0; j < 8; j++) {
              tmpMat[ctr] = ShapeFnCoeffs[cNum][eType][i][j];
              ctr++;
            }
          }
        }
      }
    }

    par::Mpi_Bcast<double>(tmpMat,9216, 0, newComm);

    if((rank % THOUSAND) != 0) {
      unsigned int ctr = 0;
      for(unsigned int cNum = 0; cNum < 8; cNum++) {
        for(unsigned int eType = 0; eType < 18; eType++) {
          for(unsigned int i = 0; i < 8; i++) {
            for(unsigned int j = 0; j < 8; j++) {
              ShapeFnCoeffs[cNum][eType][i][j] = tmpMat[ctr];
              ctr++;
            }
          }
        }
      }
    }

    delete [] tmpMat;

    return 1;
  }//end of function

  int createShapeFnCoeffs_Type1(MPI_Comm comm) {
    FILE* infile;
    int rank, res;
    MPI_Comm_rank(comm, &rank);
    if(!rank) {
      char fname[250];
      sprintf(fname,"ShapeFnCoeffs.inp");
      infile = fopen(fname,"r");
      if(!infile) {
        std::cout<<"The file "<<fname<<" is not good for reading."<<std::endl;
        assert(false);
      }
    }

    typedef double* doublePtr;
    typedef doublePtr* double2Ptr;
    typedef double2Ptr* double3Ptr;

    ShapeFnCoeffs = new double3Ptr[8];
    for(unsigned int cNum = 0; cNum < 8; cNum++) {
      ShapeFnCoeffs[cNum] = new double2Ptr[18];
      for(unsigned int eType = 0; eType < 18; eType++) {
        ShapeFnCoeffs[cNum][eType] = new doublePtr[8];
        for(unsigned int i = 0; i < 8; i++) {
          ShapeFnCoeffs[cNum][eType][i] = new double[8];
          if(!rank){
            for(unsigned int j = 0; j < 8; j++) {
              res = fscanf(infile,"%lf",&(ShapeFnCoeffs[cNum][eType][i][j]));
            }
          }
        }
      }
    }

    if(!rank){
      fclose(infile);
    }

    double * tmpMat = new double[9216];
    assert(tmpMat);

    if(!rank) {
      unsigned int ctr = 0;
      for(unsigned int cNum = 0; cNum < 8; cNum++) {
        for(unsigned int eType = 0; eType < 18; eType++) {
          for(unsigned int i = 0; i < 8; i++) {
            for(unsigned int j = 0; j < 8; j++) {
              tmpMat[ctr] = ShapeFnCoeffs[cNum][eType][i][j];
              ctr++;
            }
          }
        }
      }
    }

    par::Mpi_Bcast<double>(tmpMat,9216, 0, comm);

    if(rank) {
      unsigned int ctr = 0;
      for(unsigned int cNum = 0; cNum < 8; cNum++) {
        for(unsigned int eType = 0; eType < 18; eType++) {
          for(unsigned int i = 0; i < 8; i++) {
            for(unsigned int j = 0; j < 8; j++) {
              ShapeFnCoeffs[cNum][eType][i][j] = tmpMat[ctr];
              ctr++;
            }
          }
        }
      }
    }

    delete [] tmpMat;

    return 1;
  }//end of function

    void writeCommCountMapToFile(char * fileName, const std::vector<unsigned int>& commProcs, const std::vector<unsigned int>& commCounts, MPI_Comm comm,double threshold)
  {

      int rank=0,npes=0;
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&npes);

      unsigned int * commMap = new unsigned int[npes];

      for(unsigned int i=0;i<npes;i++)
        commMap[i]=0;

      DendroIntL sum=0;
      unsigned int deg=0;
      for(unsigned int i=0;i<commProcs.size();i++) {

          commMap[commProcs[i]] = commCounts[i];

      }

      for (unsigned int i =0; i< npes;i++)
      {
          sum+=commMap[i];
          if(commMap[i]!=0)
          {
              deg++;
          }
      }

      DendroIntL stat[3];

      par::Mpi_Reduce(&sum,stat,1,MPI_MIN,0,comm);
      par::Mpi_Reduce(&sum,stat+1,1,MPI_SUM,0,comm);
      par::Mpi_Reduce(&sum,stat+2,1,MPI_MAX,0,comm);

      if(!rank)
      {

          std::cout<<"==============================================="<<std::endl;
          std::cout<<"      Data Communicated (min total mean max)         "<<std::endl;
          std::cout<<stat[0]<<"\t"<<stat[1]<<"\t"<<stat[1]/((double)npes)<<"\t"<<stat[2]<<std::endl;
          std::cout<<"==============================================="<<std::endl;
      }



      unsigned int stat1[3];

      par::Mpi_Reduce(&deg,stat1,1,MPI_MIN,0,comm);
      par::Mpi_Reduce(&deg,stat1+1,1,MPI_SUM,0,comm);
      par::Mpi_Reduce(&deg,stat1+2,1,MPI_MAX,0,comm);

      if(!rank)
      {

          std::cout<<"==============================================="<<std::endl;
          std::cout<<" Degreee each connected to  (min total mean max) "<<std::endl;
          std::cout<<stat1[0]<<"\t"<<stat1[1]<<"\t"<<stat1[1]/((double)npes)<<"\t"<<stat1[2]<<std::endl;
          std::cout<<"==============================================="<<std::endl;
      }



 /*     unsigned int * commMapAll=new unsigned int[npes*npes];
      par::Mpi_Gather(commMap,commMapAll,npes,0,comm);*/


       /*   std::ofstream myfile;
          myfile.open(fileName);

          for(unsigned int i=0;i<npes;i++)
          {
              myfile<<commMap[i]<<"\t";

          }

          myfile.close();*/



  }// end of the function


  // ot::DA* function_to_DA(std::function<double(double,double,double)> fx, unsigned int d_min, unsigned int d_max, double* gSize, bool reject_interior, MPI_Comm comm ) {
  DA* function_to_DA (std::function<double ( double, double, double ) > fx_refine, unsigned int d_min, unsigned int d_max, double* gSize, MPI_Comm comm ) {
  // PROF_F2O_BEGIN
    int size, rank;
    unsigned int dim = 3;
    unsigned maxDepth = 30;
    bool incCorner = 1;
    bool compressLut = false;
  
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
  
    std::vector<TreeNode> nodes;
    std::vector<ot::TreeNode> nodes_new;

    unsigned int depth = 1;
    unsigned int num_intersected=1;
  
    double hx, hy, hz;
    std::array<double, 8> dist;
    Point pt;
        
    auto inside = [](double d){ return d < 0.0; };

    double xFac = gSize[0]/((double)(1<<(maxDepth)));
    double yFac = gSize[1]/((double)(1<<(maxDepth)));
    double zFac = gSize[2]/((double)(1<<(maxDepth)));

    Point p2(xFac, yFac, zFac);

    if (!rank) {
      // root does the initial refinement
      ot::TreeNode root = ot::TreeNode(dim, maxDepth);
      root.addChildren(nodes);
  
      while ( (num_intersected > 0 ) && (num_intersected < size*size ) && (depth < d_max) ) {
        // std::cout << "Depth: " << depth << " n = " << nodes.size() << std::endl;
        num_intersected = 0;
        for (auto elem: nodes ){
          if ( elem.getLevel() != depth ) {
            nodes_new.push_back(elem);
            continue;
          }
          if (depth < d_min) {
            elem.addChildren(nodes_new);
            num_intersected++;
            continue;
          }

          hx = xFac * ( 1 << (maxDepth - elem.getLevel()));
          hy = yFac * ( 1 << (maxDepth - elem.getLevel()));
          hz = zFac * ( 1 << (maxDepth - elem.getLevel()));

          // check and split
          pt = elem.getAnchor();
          pt *= p2;

          dist[0] = fx_refine(pt.x(), pt.y(), pt.z());
          dist[1] = fx_refine(pt.x()+hx, pt.y(), pt.z());
          dist[2] = fx_refine(pt.x(), pt.y()+hy, pt.z());
          dist[3] = fx_refine(pt.x()+hx, pt.y()+hy, pt.z());

          dist[4] = fx_refine(pt.x(), pt.y(), pt.z()+hz);
          dist[5] = fx_refine(pt.x()+hx, pt.y(), pt.z()+hz);
          dist[6] = fx_refine(pt.x(), pt.y()+hy, pt.z()+hz);
          dist[7] = fx_refine(pt.x()+hx, pt.y()+hy, pt.z() +hz);

          if (std::none_of(dist.begin(), dist.end(), inside)) {
            // outside, retain but do not refine
            nodes_new.push_back(elem);
          } else if (std::all_of(dist.begin(), dist.end(), inside)) {
            // if (!reject_interior)
            nodes_new.push_back(elem);
          } else {
            // intersection.
            elem.addChildren(nodes_new);
            num_intersected++;
          }
      }
      depth++;
      
      std::swap(nodes, nodes_new);
      nodes_new.clear();
    }
  } // !rank

  // now scatter the elements.
  DendroIntL totalNumOcts = nodes.size(), numOcts;
  
  par::Mpi_Bcast<DendroIntL>(&totalNumOcts, 1, 0, comm);
  
  // TODO do proper load balancing -> partitionW ?
  
  numOcts = totalNumOcts/size + (rank < totalNumOcts%size);
  // std::cout << rank << ": numOcts " <<  numOcts << std::endl;
  par::scatterValues<ot::TreeNode>(nodes, nodes_new, numOcts, comm);
  std::swap(nodes, nodes_new);
  nodes_new.clear();
  
  // std::cout << rank << ": numOcts after part " <<  nodes.size() << std::endl;
  
  // now refine in parallel.
  par::Mpi_Bcast(&depth, 1, 0, comm);
  num_intersected=1;
  
  while ( (depth < d_max) || ( (num_intersected > 0 ) && (depth < d_max) ) ) {
    // std::cout << "Depth: " << depth << " n = " << nodes.size() << std::endl;
    num_intersected = 0;
    for (auto elem: nodes ){
        if ( elem.getLevel() != depth ) {
          nodes_new.push_back(elem);
          continue;
        }
        if (depth < d_min) {
          elem.addChildren(nodes_new);
          num_intersected++;
          continue;
        }

        hx = xFac * ( 1 << (maxDepth - elem.getLevel()));
        hy = yFac * ( 1 << (maxDepth - elem.getLevel()));
        hz = zFac * ( 1 << (maxDepth - elem.getLevel()));
        
        // check and split
        pt = elem.getAnchor();
        pt *= p2;

        dist[0] = fx_refine(pt.x(), pt.y(), pt.z());
        dist[1] = fx_refine(pt.x()+hx, pt.y(), pt.z());
        dist[2] = fx_refine(pt.x(), pt.y()+hy, pt.z());
        dist[3] = fx_refine(pt.x()+hx, pt.y()+hy, pt.z());

        dist[4] = fx_refine(pt.x(), pt.y(), pt.z()+hz);
        dist[5] = fx_refine(pt.x()+hx, pt.y(), pt.z()+hz);
        dist[6] = fx_refine(pt.x(), pt.y()+hy, pt.z()+hz);
        dist[7] = fx_refine(pt.x()+hx, pt.y()+hy, pt.z() +hz);
        
        if ( std::none_of(dist.begin(), dist.end(), inside )) {
          // outside, retain but do not refine 
          nodes_new.push_back(elem);
        } else if ( std::all_of(dist.begin(), dist.end(), inside ) ) {
          // if (!reject_interior)
          nodes_new.push_back(elem);
        } else {
          // intersection.
          elem.addChildren(nodes_new);
          num_intersected++;
        }
      }
      depth++;
      
      std::swap(nodes, nodes_new);
      nodes_new.clear();
    }
    // PROF_F2O_END 
    
    // partition 
    // if(!rank){       std::cout <<"Partitioning Input... " << std::endl;    }

    par::partitionW<ot::TreeNode>(nodes, NULL, comm);
    
    // balance 
    ot::balanceOctree (nodes, nodes_new, dim, maxDepth, incCorner, comm, NULL, NULL);
  
    // build DA.
  /*  
    if(rank==0) {
      std::cout << "building DA" << std::endl;
    }
*/
    ot::DA *da = new ot::DA(nodes_new, comm, comm, compressLut);

    // if(rank==0) {
    //  std::cout << rank << ": finished building DA" << std::endl;
    // }
   
    unsigned int lev;
    /*
    double xFac = gSize[0]/((double)(1<<(maxDepth)));
    double yFac = gSize[1]/((double)(1<<(maxDepth)));
    double zFac = gSize[2]/((double)(1<<(maxDepth)));
    */

    // now process the DA to skip interior elements
    /* moved to subDA
    if (reject_interior) {
        da->initialize_skiplist();
        for ( da->init<ot::DA_FLAGS::ALL>(); da->curr() < da->end<ot::DA_FLAGS::ALL>(); da->next<ot::DA_FLAGS::ALL>() ) {
          lev = da->getLevel(da->curr());
          hx = xFac*(1<<(maxDepth - lev));
          hy = yFac*(1<<(maxDepth - lev));
          hz = zFac*(1<<(maxDepth - lev));

          pt = da->getCurrentOffset();

          dist[0] = fx_retain(pt.x()*xFac, pt.y()*yFac, pt.z()*zFac);
          dist[1] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac, pt.z()*zFac);
          dist[2] = fx_retain(pt.x()*xFac, pt.y()*yFac+hy, pt.z()*zFac);
          dist[3] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac+hy, pt.z()*zFac);

          dist[4] = fx_retain(pt.x()*xFac, pt.y()*yFac, pt.z()*zFac+hz);
          dist[5] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac, pt.z()*zFac+hz);
          dist[6] = fx_retain(pt.x()*xFac, pt.y()*yFac+hy, pt.z()*zFac+hz);
          dist[7] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac+hy, pt.z()*zFac +hz);

          / 
          *
          if (pt == Point(0,0,0)) {
            for (auto dd: dist)
              std::cout << "HS: " << dd << std::endl;
          }
          * 
          /
          
          if ( std::all_of( dist.begin(), dist.end(), inside ) ) {
            da->skip_current();
          } 
        }

        da->finalize_skiplist();
    }
    std::cout << rank << ": finished removing interior." << std::endl;
    */

    return da;
  } // end of function.
  
  
  DA* function_to_DA_bool (std::function<bool ( double, double, double ) > fx_refine, unsigned int d_min, unsigned int d_max, double* gSize, MPI_Comm comm ) {
  // PROF_F2O_BEGIN
    int size, rank;
    unsigned int dim = 3;
    unsigned maxDepth = 30;
    bool incCorner = 1;
    bool compressLut = false;
  
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
  
    std::vector<TreeNode> nodes;
    std::vector<ot::TreeNode> nodes_new;

    unsigned int depth = 1;
    unsigned int num_intersected=1;
  
    double hx, hy, hz;
    std::array<bool, 8> dist;
    Point pt;
        
    auto inside = [](bool d){ return d; };

    double xFac = gSize[0]/((double)(1<<(maxDepth)));
    double yFac = gSize[1]/((double)(1<<(maxDepth)));
    double zFac = gSize[2]/((double)(1<<(maxDepth)));

    Point p2(xFac, yFac, zFac);

    if (!rank) {
      // root does the initial refinement
      ot::TreeNode root = ot::TreeNode(dim, maxDepth);
      root.addChildren(nodes);
  
      while ( (num_intersected > 0 ) && (num_intersected < size*size ) && (depth < d_max) ) {
        // std::cout << "Depth: " << depth << " n = " << nodes.size() << std::endl;
        num_intersected = 0;
        for (auto elem: nodes ){
          if ( elem.getLevel() != depth ) {
            nodes_new.push_back(elem);
            continue;
          }
          if (depth < d_min) {
            elem.addChildren(nodes_new);
            num_intersected++;
            continue;
          }

          hx = xFac * ( 1 << (maxDepth - elem.getLevel()));
          hy = yFac * ( 1 << (maxDepth - elem.getLevel()));
          hz = zFac * ( 1 << (maxDepth - elem.getLevel()));

          // check and split
          pt = elem.getAnchor();
          pt *= p2;

          dist[0] = fx_refine(pt.x(), pt.y(), pt.z());
          dist[1] = fx_refine(pt.x()+hx, pt.y(), pt.z());
          dist[2] = fx_refine(pt.x(), pt.y()+hy, pt.z());
          dist[3] = fx_refine(pt.x()+hx, pt.y()+hy, pt.z());

          dist[4] = fx_refine(pt.x(), pt.y(), pt.z()+hz);
          dist[5] = fx_refine(pt.x()+hx, pt.y(), pt.z()+hz);
          dist[6] = fx_refine(pt.x(), pt.y()+hy, pt.z()+hz);
          dist[7] = fx_refine(pt.x()+hx, pt.y()+hy, pt.z() +hz);

          if (std::none_of(dist.begin(), dist.end(), inside)) {
            // false, don't refine
            nodes_new.push_back(elem);
          } else {
            // 
            elem.addChildren(nodes_new);
            num_intersected++;
          }
      }
      depth++;
      
      std::swap(nodes, nodes_new);
      nodes_new.clear();
    }
  } // !rank

  // now scatter the elements.
  DendroIntL totalNumOcts = nodes.size(), numOcts;
  
  par::Mpi_Bcast<DendroIntL>(&totalNumOcts, 1, 0, comm);
  
  // TODO do proper load balancing -> partitionW ?
  
  numOcts = totalNumOcts/size + (rank < totalNumOcts%size);
  // std::cout << rank << ": numOcts " <<  numOcts << std::endl;
  par::scatterValues<ot::TreeNode>(nodes, nodes_new, numOcts, comm);
  std::swap(nodes, nodes_new);
  nodes_new.clear();
  
  // std::cout << rank << ": numOcts after part " <<  nodes.size() << std::endl;
  
  // now refine in parallel.
  par::Mpi_Bcast(&depth, 1, 0, comm);
  num_intersected=1;
  
  while ( (depth < d_max) || ( (num_intersected > 0 ) && (depth < d_max) ) ) {
    // std::cout << "Depth: " << depth << " n = " << nodes.size() << std::endl;
    num_intersected = 0;
    for (auto elem: nodes ){
        if ( elem.getLevel() != depth ) {
          nodes_new.push_back(elem);
          continue;
        }
        if (depth < d_min) {
          elem.addChildren(nodes_new);
          num_intersected++;
          continue;
        }

        hx = xFac * ( 1 << (maxDepth - elem.getLevel()));
        hy = yFac * ( 1 << (maxDepth - elem.getLevel()));
        hz = zFac * ( 1 << (maxDepth - elem.getLevel()));
        
        // check and split
        pt = elem.getAnchor();
        pt *= p2;

        dist[0] = fx_refine(pt.x(), pt.y(), pt.z());
        dist[1] = fx_refine(pt.x()+hx, pt.y(), pt.z());
        dist[2] = fx_refine(pt.x(), pt.y()+hy, pt.z());
        dist[3] = fx_refine(pt.x()+hx, pt.y()+hy, pt.z());

        dist[4] = fx_refine(pt.x(), pt.y(), pt.z()+hz);
        dist[5] = fx_refine(pt.x()+hx, pt.y(), pt.z()+hz);
        dist[6] = fx_refine(pt.x(), pt.y()+hy, pt.z()+hz);
        dist[7] = fx_refine(pt.x()+hx, pt.y()+hy, pt.z() +hz);
        
        if ( std::none_of(dist.begin(), dist.end(), inside )) {
          // outside, retain but do not refine 
          nodes_new.push_back(elem);
        } else {
          // intersection.
          elem.addChildren(nodes_new);
          num_intersected++;
        }
      }
      depth++;
      
      std::swap(nodes, nodes_new);
      nodes_new.clear();
    }
    // PROF_F2O_END 
    
    // partition 
    // if(!rank){       std::cout <<"Partitioning Input... " << std::endl;    }

    par::partitionW<ot::TreeNode>(nodes, NULL, comm);
    
    // balance 
    ot::balanceOctree (nodes, nodes_new, dim, maxDepth, incCorner, comm, NULL, NULL);
  
    // build DA.
  /*  
    if(rank==0) {
      std::cout << "building DA" << std::endl;
    }
*/
    ot::DA *da = new ot::DA(nodes_new, comm, comm, compressLut);

    // if(rank==0) {
    //  std::cout << rank << ": finished building DA" << std::endl;
    // }
   
    unsigned int lev;
    /*
    double xFac = gSize[0]/((double)(1<<(maxDepth)));
    double yFac = gSize[1]/((double)(1<<(maxDepth)));
    double zFac = gSize[2]/((double)(1<<(maxDepth)));
    */

    // now process the DA to skip interior elements
    /* moved to subDA
    if (reject_interior) {
        da->initialize_skiplist();
        for ( da->init<ot::DA_FLAGS::ALL>(); da->curr() < da->end<ot::DA_FLAGS::ALL>(); da->next<ot::DA_FLAGS::ALL>() ) {
          lev = da->getLevel(da->curr());
          hx = xFac*(1<<(maxDepth - lev));
          hy = yFac*(1<<(maxDepth - lev));
          hz = zFac*(1<<(maxDepth - lev));

          pt = da->getCurrentOffset();

          dist[0] = fx_retain(pt.x()*xFac, pt.y()*yFac, pt.z()*zFac);
          dist[1] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac, pt.z()*zFac);
          dist[2] = fx_retain(pt.x()*xFac, pt.y()*yFac+hy, pt.z()*zFac);
          dist[3] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac+hy, pt.z()*zFac);

          dist[4] = fx_retain(pt.x()*xFac, pt.y()*yFac, pt.z()*zFac+hz);
          dist[5] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac, pt.z()*zFac+hz);
          dist[6] = fx_retain(pt.x()*xFac, pt.y()*yFac+hy, pt.z()*zFac+hz);
          dist[7] = fx_retain(pt.x()*xFac+hx, pt.y()*yFac+hy, pt.z()*zFac +hz);

          / 
          *
          if (pt == Point(0,0,0)) {
            for (auto dd: dist)
              std::cout << "HS: " << dd << std::endl;
          }
          * 
          /
          
          if ( std::all_of( dist.begin(), dist.end(), inside ) ) {
            da->skip_current();
          } 
        }

        da->finalize_skiplist();
    }
    std::cout << rank << ": finished removing interior." << std::endl;
    */

    return da;
  } // end of function f2DA_bool
  

  ot::DA* remesh_DA (ot::DA* da, std::vector<unsigned int> levels, double* gSize, MPI_Comm comm) {

    unsigned int dim = 3;
    unsigned int maxNumPts = 1;
    bool incCorner = 1;
    bool compressLut = false;
    unsigned int maxDepth = da->getMaxDepth();

    // elemental loop - create new points
    std::vector<double> pts;
    std::vector<ot::TreeNode> tmpNodes, linOct;
    
    for(da->init<ot::DA_FLAGS::WRITABLE>(); da->curr() < da->end<ot::DA_FLAGS::WRITABLE>(); da->next<ot::DA_FLAGS::WRITABLE>()) {
      // get coords of current element
      Point pt = da->getCurrentOffset();
      unsigned int currLev = da->getLevel(da->curr()) - 1;
      unsigned int newLev = levels[da->curr()];

      ot::TreeNode currOct(pt.xint(), pt.yint(), pt.zint(), currLev, 3, maxDepth-1);
      if (currLev > newLev ) {
        ot::TreeNode newOct = currOct.getAncestor(newLev);
        tmpNodes.push_back(newOct);
      } else if (currLev == newLev) {
        tmpNodes.push_back(currOct);
      } else  {
        currOct.addChildren(tmpNodes, newLev - currLev);
      }
    }

    par::removeDuplicates<ot::TreeNode>(tmpNodes,false,MPI_COMM_WORLD);
    std::swap(linOct, tmpNodes);
    tmpNodes.clear();
    par::partitionW<ot::TreeNode>(linOct, NULL,MPI_COMM_WORLD);

    maxDepth--; 

    pts.resize(3*(linOct.size()));
    unsigned int ptsLen = (3*(linOct.size()));
    for(int i = 0; i < linOct.size(); i++) {
      pts[3*i] = (((double)(linOct[i].getX())) + 0.5)/((double)(1u << maxDepth))* gSize[0];
      pts[(3*i)+1] = (((double)(linOct[i].getY())) +0.5)/((double)(1u << maxDepth))* gSize[1];
      pts[(3*i)+2] = (((double)(linOct[i].getZ())) +0.5)/((double)(1u << maxDepth))* gSize[2];
    }//end for i
    linOct.clear();

    ot::points2Octree(pts, gSize, linOct, dim, maxDepth, maxNumPts, comm);

    ot::balanceOctree (linOct, tmpNodes, dim, maxDepth, incCorner, comm, NULL, NULL);

    std::swap(linOct, tmpNodes);
    tmpNodes.clear();

    ot::DA *new_da = new ot::DA(linOct, comm, comm, compressLut);

    return new_da;

  } // remesh_DA

}//end namespace
