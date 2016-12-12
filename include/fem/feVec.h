#ifndef __FE_VEC_H_
#define __FE_VEC_H_

#include "petscda.h"
#include "oda.h"


class feVec {
  public:
  /// typedefs and enums
  enum daType {
    PETSC, OCT
  }; 

  /// Contructors 
  feVec() { };
  feVec(daType da) {
#ifdef __DEBUG__
  assert ( ( da == PETSC ) || ( da == OCT ) );
#endif
  m_daType = da;


  }
  virtual ~feVec() {

  }
  
  void setDA (DA da) { m_DA = da; }
  void setDA (ot::DA* da) { m_octDA = da; }

  DA getDA() { return m_DA; }
  ot::DA* getOctDA() { return m_octDA; }
  
  //  virtual bool addVec(Vec _in, double scale=1.0) = 0;
  virtual bool addVec(Vec _in, double scale=1.0, int indx = -1) = 0;
  virtual bool computeVec(Vec _in, Vec _out,double scale=1.0) = 0;

  void setProblemDimensions(double x, double y, double z) {
    m_dLx = x;
    m_dLy = y;
    m_dLz = z;
  }
protected:

  daType          m_daType;
	
  DA              m_DA;
  ot::DA*         m_octDA;
  /// The dimensions of the problem.
  double m_dLx;
  double m_dLy;
  double m_dLz;
  int    m_iCurrentDynamicIndex;
};

#endif

