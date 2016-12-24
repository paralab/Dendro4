#include "timeStepper.h"

//static int _internal_ierr;
// #define iC(fun) {_internal_ierr = fun; CHKERRQ(_internal_ierr);}

timeStepper::timeStepper()
{
  // MASS, STIFFNESS, DAMPING
  m_Mass = NULL;
  m_Damping = NULL;
  m_Stiffness = NULL;

  // Set initial displacement and velocity to null
  m_vecInitialSolution = NULL;
  m_vecInitialVelocity = NULL; 

  // Matrix 
  m_matJacobian = NULL;

  m_vecRHS = NULL;
  m_vecSolution = NULL;

  // Linear Solver
  m_ksp = NULL;

  // Adjoint flag false
  m_bIsAdjoint = false;


}
timeStepper::~timeStepper()
{
}

/**
 *	@brief  set the initial displacement for the problem
 * @param  PETSC Vec InitialDisplacement
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setInitialTemperature(Vec initialTemperature)
{
  m_vecInitialSolution = initialTemperature;
  return(0);
}
/**
 *	@brief set the initial velocity for the problem
 * @param PETSC Vec InitialVelocity
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setInitialVelocity(Vec InitialVelocity)
{
  m_vecInitialVelocity = InitialVelocity;
  return(0);
}
/**
 *	@brief This function sets the Mass Matrix
 * @param Mass operator
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setMassMatrix(feMat* Mass)
{
  m_Mass = Mass;
  return(0);
}

/**
 *	@brief This function sets the Damping Matrix
 * @param Damping operator
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setDampingMatrix(feMat* Damping)
{
  m_Damping = Damping;
  return(0);
}

/**
 *	@brief This function sets the Stiffness Matrix
 * @param Stiffness operator
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setStiffnessMatrix(feMat* Stiffness)
{
  m_Stiffness = Stiffness;
  return(0);
}

/**
 *	@brief This function sets the Qtype Matrix
 * @param Qtype operator
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setQtypeMatrix(feMat* Qtype)
{
  m_Qtype = Qtype;
  return(0);
}

/**
 *	@brief This function sets the Force vector
 * @param Force vector
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setForceVector(feVec* Force)
{
  m_Force = Force;
  return(0);
}

/**
 *	@brief This function sets the Reaction term
 * @param Reaction vector
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setReaction(feVec* Reaction)
{
  m_Reaction = Reaction;
  return(0);
}

/**
 *	@brief This function sets all the time parameters
 * @param StartTime double starting time of the timestepper
 * @param StopTime  double stopping time of the timestepper
 * @param TimeStep  double timestep used in the simulation
 * @param TimeStepRatio double timestep ratio used for reaction...not used now
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setTimeParameters(double StartTime, double StopTime, double TimeStep, double TimeStepRatio)
{
  m_dStartTime = StartTime;
  m_dStopTime  = StopTime;
  m_dTimeStep = TimeStep;
  m_dTimeStepRatio = TimeStepRatio;

  return(0);
}
/**
 *	@brief This function sets the time Info
 * @param *ti, pointer to ti
 * @return bool true if successful, false otherwise
 **/
int timeStepper::setTimeInfo(timeInfo *ti)
{
  m_ti = ti;
  return(0);
}
/**
 *	@brief This function sets the solve direction for the adjoint
 * @param flag, bool true implies adjoint is on, by default adjoint is false
 * @return bool, true if successuful, false otherwise
 **/

int timeStepper::setAdjoint(bool flag)
{
  m_bIsAdjoint= flag;
  return(0);
}

/// Jacobian matmult, setRhs will be in the derived class

