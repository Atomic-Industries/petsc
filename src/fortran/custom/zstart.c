#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zstart.c,v 1.47 1998/06/26 23:45:52 balay Exp balay $";
#endif

/*
  This file contains Fortran stubs for PetscInitialize and Finalize.
*/

/*
    This is to prevent the Cray T3D version of MPI (University of Edinburgh)
  from stupidly redefining MPI_INIT(). They put this in to detect errors
  in C code, but here I do want to be calling the Fortran version from a
  C subroutine. 
*/
#define T3DMPI_FORTRAN
#define T3EMPI_FORTRAN

#include "src/fortran/custom/zpetsc.h" 
#include "sys.h"
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
extern int          PetscBeganMPI;

#if defined(HAVE_NAGF90)
#define iargc_  f90_unix_MP_iargc
#define getarg_ f90_unix_MP_getarg
#endif

#ifdef HAVE_FORTRAN_CAPS
#define petscinitialize_              PETSCINITIALIZE
#define petscfinalize_                PETSCFINALIZE
#define aliceinitialize_              ALICEINITIALIZE
#define alicefinalize_                ALICEFINALIZE
#define petscsetcommworld_            PETSCSETCOMMWORLD
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#if defined(PARCH_nt)
#define IARGC                        NARGS
#endif

#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define petscfinalize_                petscfinalize
#define aliceinitialize_              aliceinitialize
#define alicefinalize_                alicefinalize
#define petscsetcommworld_            petscsetcommworld
#define mpi_init_                     mpi_init
/*
    HP-UX does not have Fortran underscore but iargc and getarg 
  do have underscores????
*/
#if !defined(PARCH_hpux)
#define iargc_                        iargc
#define getarg_                       getarg
#endif

#endif

/*
    The extra _ is because the f2c compiler puts an
  extra _ at the end if the original routine name 
  contained any _.
*/
#if defined(HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define mpi_init_             mpi_init__
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void mpi_init_(int*);

/*
     Different Fortran compilers handle command lines in different ways
*/
#if defined(PARCH_nt)
/*
extern short  __declspec(dllimport) __stdcall iargc_();
extern void __declspec(dllimport) __stdcall  getarg_(short*,char*,int,short *);
*/
extern short __stdcall iargc_();
extern void __stdcall  getarg_(short*,char*,int,short *);

#else
extern int  iargc_();
extern void getarg_(int*,char*,int);
/*
      The Cray T3D/T3E use the PXFGETARG() function
*/
#if defined(HAVE_PXFGETARG)
extern void PXFGETARG(int *,_fcd,int*,int*);
#endif
#endif
#if defined(__cplusplus)
}
#endif

extern int OptionsCheckInitial_Alice(void);

/*
    Reads in Fortran command line argments and sends them to 
  all processors and adds them to Options database.
*/

int PETScParseFortranArgs_Private(int *argc,char ***argv)
{
#if defined (PARCH_nt)
  short i,flg;
#else
  int  i;
#endif
  int warg = 256,rank,ierr;
  char *p;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (!rank) {
    *argc = 1 + iargc_();
  }
  ierr = MPI_Bcast(argc,1,MPI_INT,0,PETSC_COMM_WORLD); if (ierr) return ierr;

  *argv = (char **) PetscMalloc((*argc+1)*(warg*sizeof(char)+sizeof(char*)));CHKPTRQ(*argv);
  (*argv)[0] = (char*) (*argv + *argc + 1);

  if (!rank) {
    PetscMemzero((*argv)[0],(*argc)*warg*sizeof(char));
    for ( i=0; i<*argc; i++ ) {
      (*argv)[i+1] = (*argv)[i] + warg;
#if defined(HAVE_PXFGETARG)
      {char *tmp = (*argv)[i]; 
       int  ierr,ilen;
       PXFGETARG(&i, _cptofcd(tmp,warg),&ilen,&ierr); CHKERRQ(ierr);
       tmp[ilen] = 0;
      } 
#elif defined (PARCH_nt)
      getarg_( &i, (*argv)[i],warg,&flg );
#else
      getarg_( &i, (*argv)[i], warg );
#endif
      /* zero out garbage at end of each argument */
      p = (*argv)[i] + warg-1;
      while (p > (*argv)[i]) {
        if (*p == ' ') *p = 0; 
        p--;
      }
    }
  }
  ierr = MPI_Bcast((*argv)[0],*argc*warg,MPI_CHAR,0,PETSC_COMM_WORLD); if (ierr) return ierr; 
  if (rank) {
    for ( i=0; i<*argc; i++ ) {
      (*argv)[i+1] = (*argv)[i] + warg;
    }
  } 
  return 0;   
}

#if defined(__cplusplus)
extern "C" {
#endif

#undef __FUNC__  
#define __FUNC__ "aliceinitialize"
/*
    aliceinitialize - Version called from Fortran.

    Notes:
      Since this is called from Fortran it does not return error codes
      
*/
void aliceinitialize_(CHAR filename,int *__ierr,int len)
{
#if defined (PARCH_nt)
  short  flg,i;
#else
  int i;
#endif
  int  j,flag,argc = 0,dummy_tag;
  char **args = 0,*t1, name[256];

  *__ierr = 1;
  PetscMemzero(name,256);
  if (PetscInitializedCalled) {*__ierr = 0; return;}
  
  *__ierr = OptionsCreate(); 
  if (*__ierr) return;
  i = 0;
#if defined(HAVE_PXFGETARG)
  { int ilen;
    PXFGETARG(&i, _cptofcd(name,256),&ilen,__ierr); 
    if (*__ierr) return;
    name[ilen] = 0;
  }
#elif defined (PARCH_nt)
  getarg_( &i, name, 256, &flg);
#else
  getarg_( &i, name, 256);
  /* Eliminate spaces at the end of the string */
  for ( j=254; j>=0; j-- ) {
    if (name[j] != ' ') {
      name[j+1] = 0;
      break;
    }
  }
#endif
  PetscSetProgramName(name);

  MPI_Initialized(&flag);
  if (!flag) {
    mpi_init_(__ierr);
    if (*__ierr) {(*PetscErrorPrintf)("PetscInitialize:");return;}
    PetscBeganMPI    = 1;
  }
  PetscInitializedCalled = 1;

  if (!PETSC_COMM_WORLD) {
    PETSC_COMM_WORLD          = MPI_COMM_WORLD;
  }

#if defined(USE_PETSC_COMPLEX)
  /* 
     Initialized the global variable; this is because with 
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
    Scalar ic(0.0,1.0);
    PETSC_i = ic;
  }
  MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_COMPLEX);
  MPI_Type_commit(&MPIU_COMPLEX);
#endif

  /*
     PetscInitializeFortran() is called twice. Here it initializes
     PETSC_NULLCHARACTOR_Fortran. Below it initializes the VIEWERs.
     The VIEWERs have not been created yet, so they must be initialized
     below.
  */
  PetscInitializeFortran();

  PETScParseFortranArgs_Private(&argc,&args);
  FIXCHAR(filename,len,t1);
  *__ierr = OptionsInsert(&argc,&args,t1); 
  FREECHAR(filename,t1);
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Creating options database");return;}
  PetscFree(args);
  *__ierr = OptionsCheckInitial_Alice(); 
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Checking initial options");return;}

  /*
       Initialize PETSC_COMM_SELF as a MPI_Comm with the PETSc 
     attribute.
  */
  *__ierr = PetscCommDup_Private(MPI_COMM_SELF,&PETSC_COMM_SELF,&dummy_tag);
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up PETSC_COMM_SELF");return;}
  *__ierr = PetscCommDup_Private(PETSC_COMM_WORLD,&PETSC_COMM_WORLD,&dummy_tag); 
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up PETSC_COMM_WORLD");return;}
#if defined (LAM_MPI)
  {
    int c1 = lam_F_typefind(MPI_COMM_SELF);
    int c2 = lam_F_typefind(PETSC_COMM_WORLD);
    lam_F_maketype(&c1, __ierr,PETSC_COMM_SELF);
    lam_F_maketype(&c2, __ierr,PETSC_COMM_WORLD);
  }
#endif

  *__ierr = ViewerInitialize_Private(); 
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up default viewers");return;}
  PetscInitializeFortran();

  if (PetscBeganMPI) {
    int size;

    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    PLogInfo(0,"PetscInitialize(Fortran):PETSc successfully started: procs %d\n",size);
  }

  *__ierr = 0;
}

void alicefinalize_(int *__ierr)
{
#if defined(HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *__ierr = AliceFinalize();
}

#if defined(__cplusplus)
}
#endif

/* -----------------------------------------------------------------------------------------------*/

extern int OptionsCheckInitial_Components(void);
extern int PetscInitialize_DynamicLibraries(void);

#if defined(__cplusplus)
extern "C" {
#endif

#undef __FUNC__  
#define __FUNC__ "petscinitialize"
/*
    petscinitialize - Version called from Fortran.

    Notes:
      Since this is called from Fortran it does not return error codes
      
*/
void petscinitialize_(CHAR filename,int *__ierr,int len)
{
  aliceinitialize_(filename,__ierr,len); 
  if (*__ierr) return;
  
  *__ierr = OptionsCheckInitial_Components(); 
  if (*__ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Checking initial options");return;}

  *__ierr = PetscInitialize_DynamicLibraries(); 
  if (*__ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Initializing dynamic libraries");return;}
}

void petscfinalize_(int *__ierr)
{
#if defined(HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *__ierr = PetscFinalize();
}

void petscsetcommworld_(MPI_Comm *comm,int *__ierr)
{
  *__ierr = PetscSetCommWorld((MPI_Comm)PetscToPointerComm( *comm )  );
}

#if defined(__cplusplus)
}
#endif
