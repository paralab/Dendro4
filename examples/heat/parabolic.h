/**
 *
 *	@file parabolic.h
 * @brief Main class for a linear parabolic problem
 * @author Santi Swaroop Adavani
 * @date   5/7/07
 * 
 *	Main class for parabolic problem where the Jacobian matmult
 * is done for the backward scheme and the right hand side is
 * also set the same way
 *
 */

#ifndef _PARABOLIC_H_
#define _PARABOLIC_H_

#include "timeStepper.h"
#include <sstream>
#include "rhs.h"

/**
 *	@brief Main class for a linear parabolic problem
 * @author Santi Swaroop Adavani
 * @date 5/7/07
 **/


class parabolic : public timeStepper //<parabolic>
{
 public:

  virtual int init();

  virtual int solve();
  /**
	*	@brief The Jacobian matmult routine by matrix free method using the given stiffness, damping and mass matrices
	*  @param m_matJacobian PETSC Matrix, this is of type matrix shell
	*  @param In  PETSC Vec, input vector
	*  @param Out PETSC Vec, Output vector
	**/
  virtual void jacobianMatMult(Vec In, Vec Out);
  virtual void jacobianGetDiagonal(Vec diag) {};

  virtual void mgjacobianMatMult(DM da, Vec In, Vec Out);
  /**
	*	@brief Sets the right hand side vector given the current solution
	*  @param m_vecCurrentSolution PETSC Vec, current solution
	*  @param m_vecNextRHS PETSC Vec, next right hand side
	**/
  virtual bool setRHS();

  virtual bool setRHSFunction(Vec In, Vec Out);

    // set the number of levels for multigrid
  void setLevels(int levels)
	 {
		m_inlevels = levels;
	 }

  // set time frames for monitor
  int setTimeFrames(int mon){
	 m_iMon = mon;
  }

  // monitor routine which saves solution
  int monitor();
  std::vector<Vec> getSolution()
	 {
		return m_solVector;
	 }

  inline void setDAForMonitor(DM da) {
    m_da = da;
  }

 private:
  int m_inlevels;
  int m_iMon;
  DM m_da;
  std::vector<Vec> m_solVector;
};


/**
 *	@brief The initialization function where the Jacobian matrix,
 *         KSP context to be used at every time step are created
 * @return bool true if successful, false otherwise
 * Matrix of shell type is created which does only a matvec.
 * This requires the mass matrix, stiffness matrix, damping matrix to do the matvec			 
 **/
int parabolic::init()
{

  int matsize;

  int ierr;
  // Allocate memory for working vectors
  ierr = VecDuplicate(m_vecInitialSolution,&m_vecSolution); CHKERRQ(ierr);
  ierr = VecDuplicate(m_vecInitialSolution,&m_vecRHS); CHKERRQ(ierr);

  ierr = VecGetLocalSize(m_vecInitialSolution,&matsize); CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,matsize,matsize,PETSC_DETERMINE,PETSC_DETERMINE,this,&m_matJacobian); CHKERRQ(ierr);
  ierr = MatShellSetOperation(m_matJacobian,MATOP_MULT,(void(*)(void))(MatMult)); CHKERRQ(ierr);

  // Create a KSP context to solve  @ every timestep
  ierr = KSPCreate(PETSC_COMM_WORLD,&m_ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(m_ksp,m_matJacobian,m_matJacobian); CHKERRQ(ierr);
  ierr = KSPSetType(m_ksp,KSPCG); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(m_ksp); CHKERRQ(ierr);


  return(0);
}

/**
 *	@brief Solve the timestepping problem
 * @return bool, return true if successful , false otherwise
 **/
int parabolic::solve()
{
  // Time info

  //  m_ti.start = m_dStartTime;
  //  m_ti.stop = m_dStopTime;
  //  m_ti.step = m_dTimeStep;
  int ierr;	
  unsigned int NT = (int)(ceil( m_ti->stop - m_ti->start)/m_ti->step);
  double temprtol;
  ierr = KSPGetTolerances(m_ksp,&temprtol,0,0,0); CHKERRQ(ierr);

  // Set initial conditions
  ierr = VecCopy(m_vecInitialSolution,m_vecSolution); CHKERRQ(ierr);
  // Time stepping

  if(!m_bIsAdjoint) // FORWARD PROBLEM
  {
    m_ti->current = m_ti->start;
    m_ti->currentstep = 0;
    while(m_ti->current < m_ti->stop)
    {
      m_ti->currentstep++;
      m_ti->current += m_ti->step;

      // Get the Right hand side of the ksp solve using the current solution
      setRHS();


//#ifdef __DEBUG__
      double norm;
      ierr = VecNorm(m_vecSolution,NORM_INFINITY,&norm); CHKERRQ(ierr);
      std::cout << "norm of the solution @ " << m_ti->current  << " = " << norm << std::endl;
//#endif

      // Solve the ksp using the current rhs and non-zero initial guess
      ierr = KSPSetInitialGuessNonzero(m_ksp,PETSC_TRUE); CHKERRQ(ierr);
      ierr = KSPSolve(m_ksp,m_vecRHS,m_vecSolution); CHKERRQ(ierr);

		if(m_iMon > 0){
		  monitor();
		}

#ifdef __DEBUG__
      ierr = VecNorm(m_vecSolution,NORM_INFINITY,&norm); CHKERRQ(ierr);
      std::cout << "norm of the solution @ " << m_ti->current  << " = " << norm << std::endl;
#endif
    }
  }else  // ADJOINT PROBLEM
  {
    m_ti->currentstep = NT;
    m_ti->current = m_ti->stop;
    while(m_ti->current > m_ti->start)
    {
      m_ti->currentstep--;
      m_ti->current -= m_ti->step;

      // Get the right hand side of the ksp solve for the adjoint problem
      setRHS();

      // Solve ksp using the current rhs and non-zero initial guess
			ierr = KSPSetInitialGuessNonzero(m_ksp,PETSC_TRUE); CHKERRQ(ierr);
      ierr = KSPSolve(m_ksp,m_vecRHS,m_vecSolution); CHKERRQ(ierr);
    }
  }

  return(0);

}

/**
 *	@brief Jacobian matrix-vec product done at every timestep
 * @param In, PETSC Vec which is the current solution
 * @param Out, PETSC Vec which is Out = J*in, J is the Jacobian (here J = Mass + dt*Stiffness)
 **/
void parabolic::jacobianMatMult(Vec In, Vec Out)
{
  VecZeroEntries(Out); /* Clear to zeros*/
  //m_Damping->MatVec(In, Out); /* Matvec */
  m_Mass->MatVec(In, Out);
  m_Stiffness->MatVec(In, Out, -m_ti->step); /* -dt factor for stiffness*/
  //m_Qtype->MatVec(In,Out,m_ti->step); /* dt factor for qtype matrix*/
}

void parabolic::mgjacobianMatMult(DM da, Vec In, Vec Out){
  assert(false);
}
/**
 *	@brief Set right hand side used in stepping at every time step
 * @return true if successful, false otherwise
 **/

bool parabolic::setRHS()
{
  VecZeroEntries(m_vecRHS);

  ((forceVector*)m_Force)->setPrevTS(m_vecSolution);
  m_Force->addVec(m_vecRHS);
  //m_Damping->MatVec(m_vecSolution,m_vecRHS); /* Set right hand side*/
  //m_Force->addVec(m_vecRHS,m_ti->step);  /* set Force term*/
  return true;
}

bool parabolic::setRHSFunction(Vec In, Vec Out)
{
  return true;
}

int parabolic::monitor()
{
  if(fmod(m_ti->currentstep,(double)(m_iMon)) < 0.0001)
	 {
		std::cout << "current step in the monitor " << m_ti->currentstep << std::endl;
		double norm;
		int ierr;
		Vec tempSol;
		ierr = VecDuplicate(m_vecSolution,&tempSol);  CHKERRQ(ierr);
#ifdef __DEBUG__
		//		VecNorm(m_vecSolution,NORM_INFINITY,&norm);
		//		PetscPrintf(0,"solution norm b4 push back %f\n",norm);
#endif
		ierr = VecCopy(m_vecSolution,tempSol); CHKERRQ(ierr);
		m_solVector.push_back(tempSol);

    std::stringstream ss;
    ss << "timestep_" << m_ti->currentstep << ".plt";
    // write_vector(ss.str().c_str(), m_vecSolution, m_da);

    /*PetscViewer view;
    PetscViewerCreate(PETSC_COMM_WORLD, &view);
    PetscViewerPushFormat(view, PETSC_VIEWER_ASCII_MATLAB); 
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, ss.str().c_str(), &view);
    VecView(m_vecSolution, view);
    PetscViewerDestroy(&view);*/

#ifdef __DEBUG__
		//		VecNorm(m_solVector[m_solVector.size()-1],NORM_INFINITY,&norm);
		//		PetscPrintf(0,"solution norm after push back %f\n",norm);
#endif
	 }
  return(0);
}
#endif

