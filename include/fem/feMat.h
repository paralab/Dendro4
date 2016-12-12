#ifndef __FE_MAT_H_
#define __FE_MAT_H_

#include "petscda.h"
#include "oda.h"

#define sh1 0.7886751345948129  //  ( 1 + psi(q1))/2
#define sh2 0.2113248654051871  //  ( 1 + psi(q2))/2
#define sh3 0.8943375672974064  //  ( 3 + psi(q1))/4
#define sh4 0.6056624327025936  //  ( 3 + psi(q2))/4

//#define sh1 1.577350269189626  //  ( 1 + psi(q1))
//#define sh2 0.4226497308103743 //  ( 1 + psi(q2))
//#define sh3 1.788675134594813  //   (3 + psi(q1))/2
//#define sh4 1.211324865405187 //   (3 + psi(q2))/2


class feMat {
  public:
  /// typedefs and enums
  enum daType {
    PETSC, OCT
  }; 

  /// Contructors 
  feMat() { };
  feMat(daType da) {
#ifdef __DEBUG__
  assert ( ( da == PETSC ) || ( da == OCT ) );
#endif
  m_daType = da;


  }
  virtual ~feMat() {

  }
  
  int getDAtype() {
    return m_daType;
  }

  void setDA (DA da) { m_DA = da; }
  void setDA (ot::DA* da) { m_octDA = da; }

  DA getDA() { return m_DA; }
  ot::DA* getOctDA() { return m_octDA; }
  /**
   * 	@brief		The matrix-vector multiplication routine that is used by
   * 				matrix-free methods. 
   * 	@param		_in	PETSc Vec which is the input vector with whom the 
   * 				product is to be calculated.
   * 	@param		_out PETSc Vec, the output of M*_in
   * 	@return		bool true if successful, false otherwise.
   * 
   *  The matrix-vector multiplication routine that is used by matrix-free 
   * 	methods. The product is directly calculated from the elemental matrices,
   *  which are computed by the ElementalMatVec() function. Use the Assemble()
   *  function for matrix based methods.
   **/ 
  virtual bool MatVec(Vec _in, Vec _out, double scale=1.0) = 0;
  virtual bool MatGetDiagonal(Vec _diag, double scale=1.0) = 0;

  virtual bool GetAssembledMatrix(Mat *J, MatType mtype) = 0;


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
};

#endif

