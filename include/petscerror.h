/* $Id: petscerror.h,v 1.26 1998/09/23 15:48:45 bsmith Exp bsmith $ */
/*
    Contains all error handling code for PETSc.
*/
#if !defined(__PETSCERROR_H)
#define __PETSCERROR_H

#include "petsc.h"

#if defined(HAVE_AMS)
#include "ams.h"
#endif

/*
   Defines the directory where the compiled source is located; used
   in printing error messages. Each makefile has an entry 
   LOCDIR	  =  thedirectory
   and bmake/common includes in CCPPFLAGS -D__SDIR__='"${LOCDIR}"'
   which is a flag passed to the C/C++ compilers.
*/
#if !defined(__SDIR__)
#define __SDIR__ "unknowndirectory/"
#endif

/*
   Defines the function where the compiled source is located; used 
   in printing error messages.
*/
#if !defined(__FUNC__)
#define __FUNC__ "unknownfunction"
#endif

/* 
     These are the generic error codes. These error codes are used
     many different places in the PETSc source code.

     In addition, each specific error in the code has an error
     message: a specific, unique error code.  (The specific error
     code is not yet in use; these will be generated automatically and
     embed an integer into the PetscError() calls. For non-English
     error messages, that integer will be extracted and used to look up the
     appropriate error message in the local language from a file.)

*/
#define PETSC_ERR_MEM             55   /* unable to allocate requested memory */
#define PETSC_ERR_SUP             56   /* no support for requested operation */
#define PETSC_ERR_SIG             59   /* signal received */
#define PETSC_ERR_FP              72   /* floating point exception */
#define PETSC_ERR_COR             74   /* corrupted PETSc object */
#define PETSC_ERR_LIB             76   /* error in library called by PETSc */
#define PETSC_ERR_PLIB            77   /* PETSc library generated inconsistent data */
#define PETSC_ERR_MEMC            78   /* memory corruption */

#define PETSC_ERR_ARG_SIZ         60   /* nonconforming object sizes used in operation */
#define PETSC_ERR_ARG_IDN         61   /* two arguments not allowed to be the same */
#define PETSC_ERR_ARG_WRONG       62   /* wrong argument (but object probably ok) */
#define PETSC_ERR_ARG_CORRUPT     64   /* null or corrupted PETSc object as argument */
#define PETSC_ERR_ARG_OUTOFRANGE  63   /* input argument, out of range */
#define PETSC_ERR_ARG_BADPTR      68   /* invalid pointer argument */
#define PETSC_ERR_ARG_NOTSAMETYPE 69   /* two args must be same object type */
#define PETSC_ERR_ARG_WRONGSTATE  73   /* object in argument is in wrong state, e.g. unassembled mat */
#define PETSC_ERR_ARG_INCOMP      75   /* two arguments are incompatible */

#define PETSC_ERR_FILE_OPEN       65   /* unable to open file */
#define PETSC_ERR_FILE_READ       66   /* unable to read from file */
#define PETSC_ERR_FILE_WRITE      67   /* unable to write to file */
#define PETSC_ERR_FILE_UNEXPECTED 79   /* unexpected data in file */

#define PETSC_ERR_KSP_BRKDWN      70   /* break down in a Krylov method */

#define PETSC_ERR_MAT_LU_ZRPVT    71   /* detected a zero pivot during LU factorization */
#define PETSC_ERR_MAT_CH_ZRPVT    71   /* detected a zero pivot during Cholesky factorization */

#if defined(USE_PETSC_DEBUG)
#define SETERRQ(n,p,s) {return PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s);}
#define SETERRA(n,p,s) {int _ierr = PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s);\
                          MPI_Abort(PETSC_COMM_WORLD,_ierr);}
#define CHKERRQ(n)     {if (n) SETERRQ(n,0,(char *)0);}
#define CHKERRA(n)     {if (n) SETERRA(n,0,(char *)0);}
#define CHKPTRQ(p)     if (!p) SETERRQ(PETSC_ERR_MEM,0,(char*)0);
#define CHKPTRA(p)     if (!p) SETERRA(PETSC_ERR_MEM,0,(char*)0);
#else
#define SETERRQ(n,p,s) ;
#define SETERRA(n,p,s) ;
#define CHKERRQ(n)     ;
#define CHKERRA(n)     ;
#define CHKPTRQ(p)     ;
#define CHKPTRA(p)     ;
#endif

extern int PetscTraceBackErrorHandler(int,char*,char*,char*,int,int,char*,void*);
extern int PetscStopErrorHandler(int,char*,char*,char*,int,int,char*,void*);
extern int PetscAbortErrorHandler(int,char*,char*,char*,int,int,char*,void* );
extern int PetscAttachDebuggerErrorHandler(int,char*,char*,char*,int,int,char*,void*); 
extern int PetscError(int,char*,char*,char*,int,int,char*);
extern int PetscPushErrorHandler(int (*handler)(int,char*,char*,char*,int,int,char*,void*),void*);
extern int PetscPopErrorHandler(void);

extern int PetscDefaultSignalHandler(int,void*);
extern int PetscPushSignalHandler(int (*)(int,void *),void*);
extern int PetscPopSignalHandler(void);
#define PETSC_FP_TRAP_OFF    0
#define PETSC_FP_TRAP_ON     1
extern int PetscSetFPTrap(int);
extern int PetscInitializeNans(Scalar*,int);
extern int PetscInitializeLargeInts(int *,int);

/*
      Allows the code to build a stack frame as it runs
*/
#if defined(USE_PETSC_STACK)

typedef struct  {
  char **function;
  char **file;
  char **directory;
  int  *line;
} PetscStack;

extern int        petscstacksize_max;
extern int        petscstacksize;
extern PetscStack *petscstack;

#if !defined(HAVE_AMS)

#define PetscFunctionBegin \
  {\
   if (petscstack && (petscstacksize < petscstacksize_max)) {    \
    petscstack->function[petscstacksize]  = __FUNC__; \
    petscstack->file[petscstacksize]      = __FILE__; \
    petscstack->directory[petscstacksize] = __SDIR__; \
    petscstack->line[petscstacksize]      = __LINE__; \
    petscstacksize++; \
  }}

#define PetscStackPush(n) \
  {if (petscstack && (petscstacksize < petscstacksize_max)) {    \
    petscstack->function[petscstacksize]  = n; \
    petscstack->file[petscstacksize]      = "unknown"; \
    petscstack->directory[petscstacksize] = "unknown"; \
    petscstack->line[petscstacksize]      = 0; \
    petscstacksize++; \
  }}

#define PetscStackPop \
  {if (petscstack && petscstacksize > 0) {     \
    petscstacksize--; \
    petscstack->function[petscstacksize]  = 0; \
    petscstack->file[petscstacksize]      = 0; \
    petscstack->directory[petscstacksize] = 0; \
    petscstack->line[petscstacksize]      = 0; \
  }};

#define PetscFunctionReturn(a) \
  {\
  PetscStackPop; \
  return(a);}

#define PetscStackActive (petscstack != 0)

#else

/*
    Duplicate Code for when the ALICE Memory Snooper (AMS)
  is being used. When HAVE_AMS is defined.

     stack_mem is the AMS memory that contains fields for the 
               number of stack frames and names of the stack frames
*/

extern AMS_Memory stack_mem;
extern int        stack_err;

#define PetscFunctionBegin \
  {\
   if (petscstack && (petscstacksize < petscstacksize_max)) {    \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_take_access(stack_mem);\
    petscstack->function[petscstacksize]  = __FUNC__; \
    petscstack->file[petscstacksize]      = __FILE__; \
    petscstack->directory[petscstacksize] = __SDIR__; \
    petscstack->line[petscstacksize]      = __LINE__; \
    petscstacksize++; \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_grant_access(stack_mem);\
  }}

#define PetscStackPush(n) \
  {if (petscstack && (petscstacksize < petscstacksize_max)) {    \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_take_access(stack_mem);\
    petscstack->function[petscstacksize]  = n; \
    petscstack->file[petscstacksize]      = "unknown"; \
    petscstack->directory[petscstacksize] = "unknown"; \
    petscstack->line[petscstacksize]      = 0; \
    petscstacksize++; \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_grant_access(stack_mem);\
  }}

#define PetscStackPop \
  {if (petscstack && petscstacksize > 0) {     \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_take_access(stack_mem);\
    petscstacksize--; \
    petscstack->function[petscstacksize]  = 0; \
    petscstack->file[petscstacksize]      = 0; \
    petscstack->directory[petscstacksize] = 0; \
    petscstack->line[petscstacksize]      = 0; \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_grant_access(stack_mem);\
  }};

#define PetscFunctionReturn(a) \
  {\
  PetscStackPop; \
  return(a);}

#define PetscStackActive (petscstack != 0)

#endif

#else

#define PetscFunctionBegin 
#define PetscFunctionReturn(a)  return(a)
#define PetscStackPop 
#define PetscStackPush(f) 
#define PetscStackActive        0

#endif

extern int PetscStackCreate(int);
extern int PetscStackView(Viewer);
extern int PetscStackDestroy(void);

/*
          For locking and unlocking AMS memories associated with 
    PETSc objects
*/

#if defined(HAVE_AMS)
#define PetscAMSTakeAccess(obj)   \
    ((((PetscObject)(obj))->amem == -1) ? 0 : AMS_Memory_take_access(((PetscObject)(obj))->amem));
#define PetscAMSGrantAccess(obj)  \
    ((((PetscObject)(obj))->amem == -1) ? 0 : AMS_Memory_grant_access(((PetscObject)(obj))->amem));
#else
#define PetscAMSTakeAccess(obj) 
#define PetscAMSGrantAccess(obj)
#endif

#endif

