-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_MKL_CPARDISO'

CFLAGS   = ${MKL_CPARDISO_INCLUDE}
SOURCEC	 = mkl_cpardiso.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/mkl_cpardiso/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
