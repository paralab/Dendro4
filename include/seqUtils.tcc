
/**
  @file seqUtils.txx
  @brief Definitions of the templated functions in the seq module.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
 */

#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <assert.h>
#include <test/testUtils.h>



namespace seq {



    /*
     * @author: Milinda Shayamal
     * Seq: Tree sort bucketting function
     * */


    template<typename T>
    inline void SFC_3D_Bucketting(std::vector<T> &pNodes, int lev, int maxDepth, unsigned char rot_id,
                                  DendroIntL &begin, DendroIntL &end, DendroIntL *splitters, bool *updateState) {


      if (lev == maxDepth || begin == end ) {
        // Special Case when the considering level exceeds the max depth.

        for (int ii = 0; ii < NUM_CHILDREN_3D; ii++) {
          int index = (rotations[2 * NUM_CHILDREN_3D * rot_id + ii] - '0');
          int nextIndex = 0;
          if (ii == 7)
            nextIndex = ii + 1;
          else
            nextIndex = (rotations[2 * NUM_CHILDREN_3D * rot_id + ii + 1] - '0');

          if (ii == 0) {
            splitters[index] = begin;
            splitters[nextIndex] = end;
            continue;
          }

          splitters[nextIndex] = splitters[index];
        }

        return;

      }

      unsigned int mid_bit = maxDepth - lev - 1;
      // Can be parallelized using OpenMP;
      if(end>pNodes.size())
        std::cout<<" end: "<<end<<" Pnodes.size(): "<<pNodes.size()<<std::endl;
      assert(end <= pNodes.size());
      register unsigned int x, y, z;
      register unsigned int index;
      register unsigned int currentIndex=0;
      register unsigned int index_tmp=0;
      DendroIntL spliterCounts[] = {0, 0, 0, 0, 0, 0, 0, 0};
      //std::cout<<"Bucketing Called For : begin: "<<(begin-pNodes.begin()) <<" end: "<<(end-pNodes.begin())<<" lev: "<<lev<<" rot_id: "<<(int)rot_id<<std::endl;

      for (DendroIntL it = begin; it < end; it++) {
        x = pNodes[it].getX();
        y = pNodes[it].getY();
        z = pNodes[it].getZ();
        if(pNodes[it].getLevel()==lev)
          index=0;
        else
          index = ((((z & (1u << mid_bit)) >> mid_bit) << 2u) | (((y & (1u << mid_bit)) >> mid_bit) << 1u) | ((x & (1u << mid_bit)) >> mid_bit));

        spliterCounts[index]++;
        updateState[(it - begin)] = false;
      }

      for (int ii = 0; ii < NUM_CHILDREN_3D; ii++) {
        int index = (rotations[2 * NUM_CHILDREN_3D * rot_id + ii] - '0');
        int nextIndex = 0;
        if (ii == 7)
          nextIndex = ii + 1;
        else
          nextIndex = (rotations[2 * NUM_CHILDREN_3D * rot_id + ii + 1] - '0');

        if (ii == 0) {
          splitters[index] = begin;
        }

        splitters[nextIndex] = splitters[index] + spliterCounts[index];
        //std::cout<<" Spliter B:"<<index <<" "<<splitters[index]<<" Splitters E "<<nextIndex<<" "<<splitters[nextIndex]<<std::endl;

      }

      if ((end - begin) <= 1) {

        return;
      }
      ot::TreeNode temp;
      DendroIntL counters[] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (DendroIntL it = begin; it < end; it++) {
        bool state = false;
        if (updateState[it - begin]) {
          continue;
        }
        do {
          x = pNodes[it].getX();
          y = pNodes[it].getY();
          z = pNodes[it].getZ();
          if(pNodes[it].getLevel()==lev)
            currentIndex=0;
          else
            currentIndex = ((((z & (1u << mid_bit)) >> mid_bit) << 2u) |(((y & (1u << mid_bit)) >> mid_bit) << 1u) | ((x & (1u << mid_bit)) >> mid_bit));
          //std::cout << "spliter: at " << currentIndex << " :" << splitters[currentIndex] << std::endl;
          assert((counters[currentIndex] <= spliterCounts[currentIndex]));
          temp =pNodes[ (splitters[currentIndex] + counters[currentIndex])];
          x = temp.getX();
          y = temp.getY();
          z = temp.getZ();

          if(temp.getLevel()==lev)
            index_tmp=0;
          else
            index_tmp = ((((z & (1u << mid_bit)) >> mid_bit) << 2u) |(((y & (1u << mid_bit)) >> mid_bit) << 1u) | ((x & (1u << mid_bit)) >> mid_bit));
          // std::cout<<"Begin of i :"<<i<<" Count["<<currentIndex<<"]:"<<counters[currentIndex]<<" Count(Index temp)["<<index_tmp<<"]:"<<counters[index_tmp]<<" Spliter Counter["<<currentIndex<<"]:"<<spliterCounts[currentIndex]<<std::endl;

          state = !(((it - begin)) == (splitters[index_tmp] + counters[index_tmp] - begin));

          std::swap(pNodes[it], pNodes[(splitters[currentIndex] + counters[currentIndex])]);

          if (currentIndex != index_tmp) {
            updateState[splitters[currentIndex] + counters[currentIndex] - begin] = true;
            counters[currentIndex]++;
            if (!state) {
              updateState[it - begin] = true;
              counters[index_tmp]++;
            }

          } else {
            updateState[splitters[currentIndex] + counters[currentIndex] - begin] = true;
            counters[currentIndex]++;
            //std::cout << "At end of i :" << i << " Count[" << currentIndex << "]:" << counters[currentIndex] << " Count(Index temp)[" << index_tmp << "]:" << counters[index_tmp] << " Spliter Counter[" <<
            //currentIndex << "]:" << spliterCounts[currentIndex] <<" State: "<<state<< std::endl;
            break;
          }

          // std::cout << "At end of i :" << i << " Count[" << currentIndex << "]:" << counters[currentIndex] << " Count(Index temp)[" << index_tmp << "]:" << counters[index_tmp] << " Spliter Counter[" <<
          // currentIndex << "]:" << spliterCounts[currentIndex] <<"State :"<<state <<std::endl;

        } while (state);

      }

    }




  /*
     * @author Milinda Shayamal
     * Sequential Tree sort implementation
     *
    */



    template<typename T>
    void SFC_3D_TreeSort(std::vector<T> &pNodes) {

      //std::cout <<" function started"<<std::endl;

      if(pNodes.empty())
        return;

      unsigned int pMaxDepth=pNodes[0].getMaxDepth();

      std::vector<NodeInfo1<T>> nodeStack; // rotation id stack
      NodeInfo1<T> root(0, 0, 0, pNodes.size());
      nodeStack.push_back(root);
      NodeInfo1<T> tmp = root;
      unsigned int levSplitCount = 1;

      unsigned int hindex = 0;
      unsigned int hindexN = 0;

      unsigned int index = 0;
      bool *updateState = new bool[pNodes.size()];

      DendroIntL splitersAll[9];
      while (!nodeStack.empty()) {
        tmp = nodeStack.back();
        nodeStack.pop_back();

        //assert(tmp.begin != tmp.end);

        seq::SFC_3D_Bucketting(pNodes, tmp.lev, pMaxDepth, tmp.rot_id, tmp.begin, tmp.end, splitersAll,
                               updateState);

        if (tmp.lev < pMaxDepth) {

          for (int i = NUM_CHILDREN_3D - 1; i >= 0; i--) {
            hindex = (rotations[2 * NUM_CHILDREN_3D * tmp.rot_id + i] - '0');
            if (i == 7)
              hindexN = i + 1;
            else
              hindexN = (rotations[2 * NUM_CHILDREN_3D * tmp.rot_id + i + 1] - '0');
            assert(splitersAll[hindex] <= splitersAll[hindexN]);

            if ((splitersAll[hindexN] - splitersAll[hindex]) > 1) {
              index = HILBERT_TABLE[NUM_CHILDREN_3D * tmp.rot_id + hindex];
              //std::cout<<" i: "<<i<<" hindex:"<<hindex<<" : "<<splitersAll[hindex]<< " hindexNext:"<<hindexN<<" : "<<splitersAll[hindexN]<<std::endl;

              NodeInfo1<T> child(index, (tmp.lev + 1), splitersAll[hindex], splitersAll[hindexN]);
              nodeStack.push_back(child);

            }

          }

        }

      }
      delete[] updateState;
    }



    /*
     * @author: Hari Sundar
     * School of Computing, University of Utah
     *
     * */

    template<typename T>
    void SFC_3D_lsd_sort(std::vector<T> &pNodes, std::vector<T> &buffer, unsigned int pMaxDepth)  {
      // std::vector<T> buffer(pNodes.size());
      // T val;
      register unsigned int cnum;

      for (unsigned int bit=0; bit<pMaxDepth; ++bit) {
        // Copy and count
        int count[9]={};
        for (int i=0; i<pNodes.size(); ++i) {
          buffer[i] = pNodes[i];
          // count
          cnum = (((pNodes[i].getZ() >> bit) & 1u) << 2u) | (((pNodes[i].getY() >> bit) & 1u) << 1u) | ((pNodes[i].getX() >> bit) & 1u);
          count[cnum+1]++;
        }
        // Init writer positions
        for (int i=0; i<8; i++) {
          count[i+1]+=count[i];
        }

        // Perform sort
        for (int i=0; i<pNodes.size(); ++i) {
          // val = buffer[i];
          cnum = (((buffer[i].getZ() >> bit) & 1u) << 2u) | (((buffer[i].getY() >> bit) & 1u) << 1u) | ((buffer[i].getX() >> bit) & 1u);
          pNodes[count[cnum]] = buffer[i];
          count[cnum]++;
        }
      }
    } // lsd



    /*template<typename T>
    void SFC_3D_msd_sort(T *pNodes, unsigned int n, unsigned int rot_id,unsigned int pMaxDepth)  {
        register unsigned int cnum;
        unsigned int count[9]={};
        for (int i=0; i<n; ++i) {
            cnum = (((pNodes[i].getZ() >> pMaxDepth) & 1u) << 2u) | (((pNodes[i].getY() >> pMaxDepth) & 1u) << 1u) | ((pNodes[i].getX() >> pMaxDepth) & 1u);
            count[cnum+1]++;
        }

        unsigned int loc[8];
        T unsorted[8];
        unsigned int live = 0;
        for (int i=0; i<8; ++i) {
            loc[i] = count[i];
            count[i+1] += count[i];
            unsorted[live] = pNodes[loc[i]];
            if (loc[i] < count[i+1]) live++;
        }
        live--;

        for (int i=0; i<n; ++i) {
            cnum = (((unsorted[live].getZ() >> pMaxDepth) & 1u) << 2u) | (((unsorted[live].getY() >> pMaxDepth) & 1u) << 1u) | ((unsorted[live].getX() >> pMaxDepth) & 1u);
            pNodes[loc[cnum]++] = unsorted[live];
            unsorted[live] = pNodes[loc[cnum]];
            if ( (loc[cnum] == count[cnum+1]) ) live--;
        }

        if (pMaxDepth>0) {
            for (int i=0; i<8; i++) {
                n = count[i+1] - count[i];
                if (n > 1) {
                    SFC_3D_msd_sort(pNodes + count[i], n,rot_id,pMaxDepth-1);
                }
            }
        }
    } // msd sort*/

    /*
    * @author: Hari Sundar
    * School of Computing, University of Utah
    *
    * */
    template<typename T>
    void SFC_3D_msd_sort(T *pNodes, unsigned int n, unsigned int rot_id,unsigned int pMaxDepth)  {

      /*std::sort(pNodes,(pNodes+n));
      return;*/

      register unsigned int cnum;
      register unsigned int cnum_prev=0;
      unsigned int rotation=0;
      unsigned int count[9]={};
      pMaxDepth--;

      for (int i=0; i<n; ++i) {
        cnum = (((pNodes[i].getZ() >> pMaxDepth) & 1u) << 2u) | (((pNodes[i].getY() >> pMaxDepth) & 1u) << 1u) | ((pNodes[i].getX() >> pMaxDepth) & 1u);
        count[cnum+1]++;
      }

      unsigned int loc[8];
      T unsorted[8];
      unsigned int live = 0;

      for (int i=0; i<8; ++i) {
        cnum=(rotations[ROTATION_OFFSET * rot_id+i] - '0');
        (i>0)? cnum_prev = ((rotations[ROTATION_OFFSET * rot_id+i-1] - '0')+1) : cnum_prev=0;
        loc[cnum]=count[cnum_prev];
        count[cnum+1] += count[cnum_prev];
        unsorted[live] = pNodes[loc[cnum]];
        if (loc[cnum] < count[cnum+1]) {live++; /*std::cout<<i<<" Live: "<<live<<std::endl;*/}
      }
      live--;

      for (int i=0; i<n; ++i) {
        //std::cout << i << " Live: " << live << " qqunsorted live " <<unsorted[live]<<std::endl;
        cnum = (((unsorted[live].getZ() >> pMaxDepth) & 1u) << 2u) |  (((unsorted[live].getY() >> pMaxDepth) & 1u) << 1u) |  ((unsorted[live].getX() >> pMaxDepth) & 1u);
        pNodes[loc[cnum]++] = unsorted[live];
        unsorted[live] = pNodes[loc[cnum]];
        if ((loc[cnum] == count[cnum + 1])) {
          live--;
        }
      }

      if (pMaxDepth>0) {
        for (int i=0; i<8; i++) {
          cnum=(rotations[ROTATION_OFFSET*rot_id+i]-'0');
          (i>0)? cnum_prev = ((rotations[ROTATION_OFFSET * rot_id+i-1] - '0')+1) : cnum_prev=0;
          n = count[cnum+1] - count[cnum_prev];
          if (n > 1) {
            rotation=HILBERT_TABLE[NUM_CHILDREN_3D*rot_id+cnum];
            SFC_3D_msd_sort(pNodes + count[cnum_prev], n,rotation,(pMaxDepth));
          }
        }
      }
    } // msd sort



  template <typename T>
    bool BinarySearch(const T* arr, unsigned int nelem, const T & key, unsigned int *ret_idx) {
      if(!nelem) {*ret_idx = nelem; return false;}
      unsigned int left = 0;
      unsigned int right = (nelem -1);	
      while (left <= right) {
        unsigned int mid =
          (unsigned int)( left + (unsigned int)(floor((double)(right-left)/2.0)) );

        if (key > arr[mid]) {
          left  = mid+1;
        } else if (key < arr[mid]) {
          if(mid>0) { right = mid-1; }
          else { right = 0; break;}
        } else {
          *ret_idx = mid;
          return true;
        }//end if-else-if
      }//end while
      *ret_idx = nelem;	
      return false;
    }//end function

  template <typename T>
    int UpperBound (unsigned int p,const T * splitt,unsigned int itr, const T & elem)
    {
      if (itr >= p) {
        return p;
      }
      while (itr < p){
        if (elem <= splitt[itr]) {
          return itr;
        } else {
          itr = itr + 1;
        }
      }//end while
      return itr;
    }//end function

  template <typename T>
    bool maxLowerBound(const std::vector<T>& arr, const T & key, unsigned int &ret_idx,
        unsigned int* leftIdx, unsigned int* rightIdx ) {


      unsigned int nelem = static_cast<unsigned int>(arr.size());
      ret_idx = 0;
      if(!nelem) { return false;}
      if(arr[0] > key) {	return false;   }
      if(arr[nelem-1] < key) {
        ret_idx = (nelem-1);
        return true;
      }//end if	
      //binary search
      unsigned int left = 0;
      unsigned int right = (nelem -1);	
      unsigned int mid = 0;
      if(leftIdx) {
        left = (*leftIdx);
      }
      if(rightIdx) {
        right = (*rightIdx);
      }
      while (left <= right) {
        mid = (unsigned int)( left + (unsigned int)(floor((double)(right-left)/2.0)) );
        if (key > arr[mid]) {
          left  = mid + (1u);
        } else if (key < arr[mid]){
          if(mid>0) {
            right = mid-1;
          }else {
            right=0;
            break;
          }
        } else {
          ret_idx = mid;
          return true;
        }//end if-else-if
      }//end while

      //If binary search did not find an exact match, it would have
      //stopped one element after or one element before. 

      if( (arr[mid] > key) && (mid > 0) ){ mid--; }	
      if(arr[mid] <= key ) { ret_idx = mid; return true; }
      else { ret_idx = 0; return false;}
    }//end function

  template<typename T> void makeVectorUnique(std::vector<T>& vecT, bool isSorted) {
    if(vecT.size() < 2) { return;}
    if(!isSorted) { std::sort( (&(vecT[0])), ( (&(vecT[0])) + (vecT.size()) ) ); }
    std::vector<T> tmp(vecT.size());
    //Using the array [] is faster than the vector version
    T* tmpPtr = (&(*(tmp.begin())));
    T* vecTptr = (&(*(vecT.begin())));
    tmpPtr[0] = vecTptr[0];
    unsigned int tmpSize=1;
    unsigned int vecTsz = static_cast<unsigned int>(vecT.size());
    for(unsigned int i = 1; i < vecTsz; i++) {	
      if(tmpPtr[tmpSize-1] != vecTptr[i]) {
        tmpPtr[tmpSize] = vecTptr[i];
        tmpSize++;
      }
    }//end for
    tmp.resize(tmpSize);
    swap(vecT, tmp);
  }//end function

  template <typename T>
    void flashsort(T* a, int n, int m, int *ctr)
    {
      const int THRESHOLD = 75;
      const int CLASS_SIZE = 75;     /* minimum value for m */

      /* declare variables */
      int *l, nmin, nmax, i, j, k, nmove, nx, mx;

      T c1,c2,flash,hold;

      /* allocate space for the l vector */
      l=(int*)calloc(m,sizeof(int));

      /***** CLASS FORMATION ****/

      nmin=nmax=0;
      for (i=0 ; i<n ; i++)
        if (a[i] < a[nmin]) nmin = i;
        else if (a[i] > a[nmax]) nmax = i;

        if ( (a[nmax]==a[nmin]) && (ctr==0) )
        {
          printf("All the numbers are identical, the list is sorted\n");
          return;
        }

        c1=(m-1.0)/(a[nmax]-a[nmin]) ;
        c2=a[nmin];

        l[0]=-1; /* since the base of the "a" (data) array is 0 */
        for (k=1; k<m ; k++) l[k]=0;

        for (i=0; i<n ; i++)
        {
          k=floor(c1*(a[i]-c2) );
          l[k]+=1;
        }

        for (k=1; k<m ; k++) l[k]+=l[k-1];

        hold=a[nmax];
        a[nmax]=a[0];
        a[0]=hold; 
        /**** PERMUTATION *****/

        nmove=0;
        j=0;
        k=m-1;

        while(nmove<n)
        {
          while  (j  >  l[k] )
          {
            j++;
            k=floor(c1*(a[j]-c2) ) ;
          }

          flash=a[ j ] ;

          while ( j <= l[k] )
          {
            k=floor(c1*(flash-c2));
            hold=a[ l[k] ];
            a[ l[k] ] = flash;
            l[k]--;
            flash=hold;
            nmove++;
          }
        }

        /**** Choice of RECURSION or STRAIGHT INSERTION *****/

        for (k=0;k<(m-1);k++)
          if ( (nx = l[k+1]-l[k]) > THRESHOLD )  /* then use recursion */
          {
            flashsort(&a[l[k]+1],nx,CLASS_SIZE,ctr);
            (*ctr)++;
          }

          else  /* use insertion sort */
            for (i=l[k+1]-1; i > l[k] ; i--)
              if (a[i] > a[i+1])
              {
                hold=a[i];
                j=i;
                while  (hold  >  a[j+1] )  a[j++]=a[j+1] ;
                a[j]=hold;
              }
        free(l);   /* need to free the memory we grabbed for the l vector */
    }

}//end namespace


 
