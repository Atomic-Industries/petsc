-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_VIENNACL'
#requiresdefine 'PETSC_HAVE_VIENNACL_NO_CUDA'

SOURCECXX= mpiaijviennacl.cxx
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/mpi/mpiviennacl/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
