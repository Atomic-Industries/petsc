!
!  $Id: snes.h,v 1.14 1998/03/24 16:11:20 balay Exp balay $;
!
!  Include file for Fortran use of the SNES package in PETSc
!
#define SNES            PETScAddr
#define SNESProblemType integer

!
!  SNESType
!
#define SNES_EQ_LS          'ls'
#define SNES_EQ_TR          'tr'
#define SNES_EQ_TR_DOG_LEG  
#define SNES_EQ_TR2_LIN
#define SNES_EQ_TEST        'test'
#define SNES_UM_LS          'umls'
#define SNES_UM_TR          'umtr'
#define SNES_LS_LM          'lslm'

!
!  Two classes of nonlinear solvers
!
      integer SNES_NONLINEAR_EQUATIONS,
     *        SNES_UNCONSTRAINED_MINIMIZATION

      parameter (SNES_NONLINEAR_EQUATIONS = 0,
     *           SNES_UNCONSTRAINED_MINIMIZATION = 1)

!
!  End of Fortran include file for the SNES package in PETSc




