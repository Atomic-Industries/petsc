#include "petsc/finclude/petscts.h"

      type tTS
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tTS
      type tTSTrajectory
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tTSTrajectory
      type tTSAdapt
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tTSAdapt
      type tTSGLLEAdapt
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tTSGLLEAdapt

      TS, parameter :: PETSC_NULL_TS = tTS(0)
      TSTrajectory, parameter :: PETSC_NULL_TSTRAJECTORY = tTSTrajectory(0)
      TSAdapt, parameter :: PETSC_NULL_TSADAPT = tTSAdapt(0)
      TSGLLEAdapt, parameter :: PETSC_NULL_TSGLLEADAPT = tTSGLLEAdapt(0)

      ! TSProblemType
      PetscEnum, parameter :: TS_LINEAR = 0
      PetscEnum, parameter :: TS_NONLINEAR = 1

      ! TSEquationType
      PetscEnum, parameter :: TS_EQ_UNSPECIFIED = -1
      PetscEnum, parameter :: TS_EQ_EXPLICIT = 0
      PetscEnum, parameter :: TS_EQ_ODE_EXPLICIT = 1
      PetscEnum, parameter :: TS_EQ_DAE_SEMI_EXPLICIT_INDEX1 = 100
      PetscEnum, parameter :: TS_EQ_DAE_SEMI_EXPLICIT_INDEX2 = 200
      PetscEnum, parameter :: TS_EQ_DAE_SEMI_EXPLICIT_INDEX3 = 300
      PetscEnum, parameter :: TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI = 500
      PetscEnum, parameter :: TS_EQ_IMPLICIT = 1000
      PetscEnum, parameter :: TS_EQ_ODE_IMPLICIT = 1001
      PetscEnum, parameter :: TS_EQ_DAE_IMPLICIT_INDEX1 = 1100
      PetscEnum, parameter :: TS_EQ_DAE_IMPLICIT_INDEX2 = 1200
      PetscEnum, parameter :: TS_EQ_DAE_IMPLICIT_INDEX3 = 1300
      PetscEnum, parameter :: TS_EQ_DAE_IMPLICIT_INDEXHI = 1500

      ! TSConvergedReason
      PetscEnum, parameter :: TS_CONVERGED_ITERATING = 0
      PetscEnum, parameter :: TS_CONVERGED_TIME = 1
      PetscEnum, parameter :: TS_CONVERGED_ITS = 2
      PetscEnum, parameter :: TS_CONVERGED_USER = 3
      PetscEnum, parameter :: TS_CONVERGED_EVENT = 4
      PetscEnum, parameter :: TS_CONVERGED_PSEUDO_FATOL = 5
      PetscEnum, parameter :: TS_CONVERGED_PSEUDO_FRTOL = 6
      PetscEnum, parameter :: TS_DIVERGED_NONLINEAR_SOLVE = -1
      PetscEnum, parameter :: TS_DIVERGED_STEP_REJECTED = -2
      PetscEnum, parameter :: TSFORWARD_DIVERGED_LINEAR_SOLVE = -3
      PetscEnum, parameter :: TSADJOINT_DIVERGED_LINEAR_SOLVE = -4

      ! TSExactFinalTimeOption
      PetscEnum, parameter :: TS_EXACTFINALTIME_UNSPECIFIED = 0
      PetscEnum, parameter :: TS_EXACTFINALTIME_STEPOVER = 1
      PetscEnum, parameter :: TS_EXACTFINALTIME_INTERPOLATE = 2
      PetscEnum, parameter :: TS_EXACTFINALTIME_MATCHSTEP = 3

      ! TSSundialsLmmType
      PetscEnum, parameter :: SUNDIALS_ADAMS = 1
      PetscEnum, parameter :: SUNDIALS_BDF = 2

      ! TSSundialsGramSchmidtType
      PetscEnum, parameter :: SUNDIALS_MODIFIED_GS = 1
      PetscEnum, parameter :: SUNDIALS_CLASSICAL_GS = 2
#define SUNDIALS_UNMODIFIED_GS SUNDIALS_CLASSICAL_GS

