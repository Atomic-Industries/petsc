#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zao.c,v 1.4 1997/07/01 19:32:56 bsmith Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "ao.h"


#ifdef HAVE_FORTRAN_CAPS
#define aocreatedebug_ AOCREATEDEBUG
#define aocreatedebugis_ AOCREATEDEBUGIS
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define aocreatedebug_ aocreatedebug
#define aocreatedebugis_ aocreatedebugis
#endif

/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
void aocreatedebug_(MPI_Comm *comm,int *napp,int *myapp,int *mypetsc,AO *aoout, int *__ierr ){
*__ierr = AOCreateDebug(
	(MPI_Comm)PetscToPointerComm( *comm ),*napp,myapp,mypetsc,aoout);
}
void aocreatedebugis_(MPI_Comm *comm,IS isapp,IS ispetsc,AO *aoout, int *__ierr ){
*__ierr = AOCreateDebugIS(
	(MPI_Comm)PetscToPointerComm( *comm ),
	(IS)PetscToPointer( *(int*)(isapp) ),
	(IS)PetscToPointer( *(int*)(ispetsc) ),aoout);
}
#if defined(__cplusplus)
}
#endif
