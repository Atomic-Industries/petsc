-include ../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_HYPRE'

CFLAGS    = ${HYPRE_INCLUDE}
SOURCEC   = hypre.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/hypre/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

