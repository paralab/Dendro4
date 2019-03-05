#include <seqUtils.h>

void makeOctreeUnique(std::vector<ot::TreeNode>& vecT, bool isSorted) {
    if(vecT.size() < 2) { return;}
    if(!isSorted) { std::sort( (&(vecT[0])), ( (&(vecT[0])) + (vecT.size()) ) ); }
    std::vector<ot::TreeNode> tmp(vecT.size());
    //Using the array [] is faster than the vector version
    ot::TreeNode* tmpPtr = (&(*(tmp.begin())));
    ot::TreeNode* vecTptr = (&(*(vecT.begin())));
    tmpPtr[0] = vecTptr[0];
    unsigned int tmpSize=1;
    unsigned int vecTsz = static_cast<unsigned int>(vecT.size());
    for(unsigned int i = 1; i < vecTsz; i++) {	
      if(tmpPtr[tmpSize-1].getAnchor() != vecTptr[i].getAnchor()) {
        tmpPtr[tmpSize] = vecTptr[i];
        tmpSize++;
      } else {
        tmpPtr[tmpSize-1] = vecTptr[i];
      }
    }//end for
    tmp.resize(tmpSize);
    swap(vecT, tmp);
  }//end function