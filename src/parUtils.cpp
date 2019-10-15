
/**
  @file parUtils.C
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  */

#include "mpi.h"
#include "binUtils.h"
#include "dtypes.h"
#include "parUtils.h"
//#include "parUtils.tcc"
#include <execinfo.h>
#include <stdio.h>
#include <unistd.h>
#include <cxxabi.h>
#include <string>
#include <colors.h>

#ifdef __DEBUG__
#ifndef __DEBUG_PAR__
#define __DEBUG_PAR__
#endif
#endif

namespace par {

  void print_trace(void) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t i, size;
    enum Constexpr { MAX_SIZE = 1024 };
    void *array[MAX_SIZE];
    char **bt_syms;

    size_t funcnamesize = 256;
    char funcname[256];

    if (!rank) {
      size = backtrace(array, MAX_SIZE);
      bt_syms = backtrace_symbols(array, size);
      int tabb=4;
      for (unsigned int i=size-3; i>0; i--) {
        for (int j=0; j<tabb; ++j)
          std::cout << ' ';
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

	      for (char *p = bt_syms[i]; *p; ++p) {
	        if (*p == '(')
		        begin_name = p;
	        else if (*p == '+')
		        begin_offset = p;
	        else if (*p == ')' && begin_offset) {
		        end_offset = p;
		        break;
	        }
	      }

	      if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
	        *begin_name++ = '\0';
	        *begin_offset++ = '\0';
	        *end_offset = '\0';

	        int status;
	        char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);

	        if (status == 0) {
            // std::cout << GRN << "↳" << YLW << bt_syms[i] << " : " << BLU << funcname << MAG << "➠" << begin_offset << NRM << std::endl;
            std::cout << GRN << "▶" << BLU << ret << NRM << std::endl;
  	      } else {
	  	      // demangling failed. Output function name as a C function with no arguments.
            std::cout << GRN << "⚑" << YLW << bt_syms[i] << " : " << BLU << begin_name << MAG << "➠" << begin_offset << NRM << std::endl;
	        }
	      } else {
	        // couldn't parse the line? print the whole line.
          std::cout << GRN << "➥" << YLW << bt_syms[i] << NRM << std::endl;
	      }
        tabb+=2;
      }
    } // !rank
  } // print_trace


  unsigned int splitCommBinary( MPI_Comm orig_comm, MPI_Comm *new_comm) {
    int npes, rank;

    MPI_Group  orig_group, new_group;

    MPI_Comm_size(orig_comm, &npes);
    MPI_Comm_rank(orig_comm, &rank);

    unsigned int splitterRank = binOp::getPrevHighestPowerOfTwo(npes);

    int *ranksAsc, *ranksDesc;
    //Determine sizes for the 2 groups 
    ranksAsc = new int[splitterRank];
    ranksDesc = new int[( npes - splitterRank)];

    int numAsc = 0;
    int numDesc = ( npes - splitterRank - 1);

    //This is the main mapping between old ranks and new ranks.
    for(int i=0; i<npes; i++) {
      if( static_cast<unsigned int>(i) < splitterRank) {
        ranksAsc[numAsc] = i;
        numAsc++;
      }else {
        ranksDesc[numDesc] = i;
        numDesc--;
      }
    }//end for i

    MPI_Comm_group(orig_comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (static_cast<unsigned int>(rank) < splitterRank) {
      MPI_Group_incl(orig_group, splitterRank, ranksAsc, &new_group);
    }else {
      MPI_Group_incl(orig_group, (npes-splitterRank), ranksDesc, &new_group);
    }

    MPI_Comm_create(orig_comm, new_group, new_comm);

    delete [] ranksAsc;
    ranksAsc = NULL;
    
    delete [] ranksDesc;
    ranksDesc = NULL;

    return splitterRank;
  }//end function

  unsigned int splitCommBinaryNoFlip( MPI_Comm orig_comm, MPI_Comm *new_comm) {
    int npes, rank;

    MPI_Group  orig_group, new_group;

    MPI_Comm_size(orig_comm, &npes);
    MPI_Comm_rank(orig_comm, &rank);

    unsigned int splitterRank =  binOp::getPrevHighestPowerOfTwo(npes);

    int *ranksAsc, *ranksDesc;
    //Determine sizes for the 2 groups 
    ranksAsc = new int[splitterRank];
    ranksDesc = new int[( npes - splitterRank)];

    int numAsc = 0;
    int numDesc = 0; //( npes - splitterRank - 1);

    //This is the main mapping between old ranks and new ranks.
    for(int i = 0; i < npes; i++) {
      if(static_cast<unsigned int>(i) < splitterRank) {
        ranksAsc[numAsc] = i;
        numAsc++;
      }else {
        ranksDesc[numDesc] = i;
        numDesc++;
      }
    }//end for i

    MPI_Comm_group(orig_comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (static_cast<unsigned int>(rank) < splitterRank) {
      MPI_Group_incl(orig_group, splitterRank, ranksAsc, &new_group);
    }else {
      MPI_Group_incl(orig_group, (npes-splitterRank), ranksDesc, &new_group);
    }

    MPI_Comm_create(orig_comm, new_group, new_comm);

    delete [] ranksAsc;
    ranksAsc = NULL;
    
    delete [] ranksDesc;
    ranksDesc = NULL;

    return splitterRank;
  }//end function

  //create Comm groups and remove empty processors...
  int splitComm2way(bool iAmEmpty, MPI_Comm * new_comm, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_SPLIT_COMM_2WAY_BEGIN

      MPI_Group  orig_group, new_group;
    int size;
    MPI_Comm_size(comm, &size);

    bool* isEmptyList = new bool[size];
    par::Mpi_Allgather<bool>(&iAmEmpty, isEmptyList, 1, comm);

    int numActive=0, numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        numIdle++;
      }else {
        numActive++;
      }
    }//end for i

    int* ranksActive = new int[numActive];
    int* ranksIdle = new int[numIdle];

    numActive=0;
    numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        ranksIdle[numIdle] = i;
        numIdle++;
      }else {
        ranksActive[numActive] = i;
        numActive++;
      }
    }//end for i

    delete [] isEmptyList;	
    isEmptyList = NULL;

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (!iAmEmpty) {
      MPI_Group_incl(orig_group, numActive, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, numIdle, ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

    PROF_SPLIT_COMM_2WAY_END
  }//end function

  int splitCommUsingSplittingRank(int splittingRank, MPI_Comm* new_comm,
      MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_SPLIT_COMM_BEGIN

      MPI_Group  orig_group, new_group;
    int size;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int* ranksActive = new int[splittingRank];
    int* ranksIdle = new int[size - splittingRank];

    for(int i = 0; i < splittingRank; i++) {
      ranksActive[i] = i;
    }

    for(int i = splittingRank; i < size; i++) {
      ranksIdle[i - splittingRank] = i;
    }

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (rank < splittingRank) {
      MPI_Group_incl(orig_group, splittingRank, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, (size - splittingRank), ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

    PROF_SPLIT_COMM_END
  }//end function

  //create Comm groups and remove empty processors...
  int splitComm2way(const bool* isEmptyList, MPI_Comm * new_comm, MPI_Comm comm) {
      
    MPI_Group  orig_group, new_group;
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int numActive=0, numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        numIdle++;
      }else {
        numActive++;
      }
    }//end for i

    int* ranksActive = new int[numActive];
    int* ranksIdle = new int[numIdle];

    numActive=0;
    numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        ranksIdle[numIdle] = i;
        numIdle++;
      }else {
        ranksActive[numActive] = i;
        numActive++;
      }
    }//end for i

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (!isEmptyList[rank]) {
      MPI_Group_incl(orig_group, numActive, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, numIdle, ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

    return 0;
  }//end function


}// end namespace

