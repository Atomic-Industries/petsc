
#include "src/fortran/custom/zpetsc.h" 
#include "sys.h"

#ifdef HAVE_FORTRAN_CAPS
#define petscinitializefortran_       PETSCINITIALIZEFORTRAN
#define petscsetcommonblock_          PETSCSETCOMMONBLOCK
#define petscsetfortranbasepointers_  PETSCSETFORTRANBASEPOINTERS
#define petsc_null_function_          PETSC_NULL_FUNCTION
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define petscinitializefortran_       petscinitializefortran
#define petscsetcommonblock_          petscsetcommonblock
#define petscsetfortranbasepointers_  petscsetfortranbasepointers
#define petsc_null_function_          petsc_null_function
#endif

#if defined(HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define petsc_null_function_  petsc_null_function__
#endif

EXTERN_C_BEGIN
extern void petscsetcommonblock_(Viewer*,Viewer*,Viewer*,int*,int*);
EXTERN_C_END

/*@
   PetscInitializeFortran - Routine that should be called from C after
   the call to PetscInitialize() if one is using a C main program
   that calls Fortran routines that in turn call PETSc routines.

   Notes:
   PetscInitializeFortran() initializes some of the default viewers,
   communicators, etc. for use in the Fortran if a user's main program is
   written in C.  PetscInitializeFortran() is NOT needed if a user's main
   program is written in Fortran; in this case, just calling
   PetscInitialize() in the main (Fortran) program is sufficient.

.seealso:  PetscInitialize()

.keywords: Mixing C and Fortran, passing PETSc objects to Fortran
@*/

void PetscInitializeFortran(void)
{
  int c1,c2;
  /*
  s1 = PetscFromPointer(VIEWER_STDOUT_SELF);
  s2 = PetscFromPointer(VIEWER_STDERR_SELF);
  s3 = PetscFromPointer(VIEWER_STDOUT_WORLD);
  */
  c1 = PetscFromPointerComm(PETSC_COMM_WORLD);
  c2 = PetscFromPointerComm(PETSC_COMM_SELF);
  petscsetcommonblock_(&VIEWER_STDOUT_SELF,&VIEWER_STDERR_SELF,&VIEWER_STDOUT_WORLD,&c1,&c2);
}
  
EXTERN_C_BEGIN

void petscinitializefortran_(void)
{
  PetscInitializeFortran();
}

#if defined(USES_CPTOFCD)
void petscsetfortranbasepointers_(_fcd fnull_character,void *fnull_integer,
                                  void *fnull_scalar,void * fnull_double,
                                  void *fnull_function)
{
  PETSC_NULL_CHARACTER_Fortran  = _fcdtocp(fnull_character);
  PETSC_NULL_INTEGER_Fortran    = fnull_integer;
  PETSC_NULL_SCALAR_Fortran     = fnull_scalar;
  PETSC_NULL_DOUBLE_Fortran     = fnull_double;
  PETSC_NULL_FUNCTION_Fortran   = fnull_function;
}
#else
void petscsetfortranbasepointers_(char *fnull_character,void *fnull_integer,
                                  void *fnull_scalar,void * fnull_double,
                                  void *fnull_function)
{
  PETSC_NULL_CHARACTER_Fortran  = fnull_character;
  PETSC_NULL_INTEGER_Fortran    = fnull_integer;
  PETSC_NULL_SCALAR_Fortran     = fnull_scalar;
  PETSC_NULL_DOUBLE_Fortran     = fnull_double;
  PETSC_NULL_FUNCTION_Fortran   = fnull_function;
}
#endif 

/*
      Dummy function so this file is never empty
*/
void petsc_null_function_(void)
{
  return;
}

EXTERN_C_END

