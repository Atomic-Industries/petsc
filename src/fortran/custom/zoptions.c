#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zoptions.c,v 1.47 1998/06/03 22:06:28 bsmith Exp bsmith $";
#endif

/*
  This file contains Fortran stubs for Options routines. 
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include "src/fortran/custom/zpetsc.h" 
#include "sys.h"
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
extern int          PetscBeganMPI;

#ifdef HAVE_FORTRAN_CAPS
#define petscgetarchtype_             PETSCGETARCHTYPE
#define optionsgetintarray_           OPTIONSGETINTARRAY
#define optionssetvalue_              OPTIONSSETVALUE
#define optionsclearvalue_            OPTIONSCLEARVALUE
#define optionshasname_               OPTIONSHASNAME
#define optionsgetint_                OPTIONSGETINT
#define optionsgetdouble_             OPTIONSGETDOUBLE
#define optionsgetdoublearray_        OPTIONSGETDOUBLEARRAY
#define optionsgetstring_             OPTIONSGETSTRING
#define petscgetprogramname           PETSCGETPROGRAMNAME
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define petscgetarchtype_             petscgetarchtype
#define optionssetvalue_              optionssetvalue
#define optionsclearvalue_            optionsclearvalue
#define optionshasname_               optionshasname
#define optionsgetint_                optionsgetint
#define optionsgetdouble_             optionsgetdouble
#define optionsgetdoublearray_        optionsgetdoublearray
#define optionsgetstring_             optionsgetstring
#define optionsgetintarray_           optionsgetintarray
#define petscgetprogramname_          petscgetprogramname
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* ---------------------------------------------------------------------*/

void optionssetvalue_(CHAR name,CHAR value,int *__ierr, int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(name,len1,c1);
  FIXCHAR(value,len2,c2);
  ierr = OptionsSetValue(c1,c2);
  FREECHAR(name,c1);
  FREECHAR(value,c2);
  *__ierr = ierr;
}

void optionsclearvalue_(CHAR name,int *__ierr, int len1)
{
  char *c1;
  int  ierr;

  FIXCHAR(name,len1,c1);
  ierr = OptionsClearValue(c1);
  FREECHAR(name,c1);
  *__ierr = ierr;
}

void optionshasname_(CHAR pre,CHAR name,int *flg,int *__ierr,int len1,int len2){
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsHasName(c1,c2,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *__ierr = ierr;
}

void optionsgetint_(CHAR pre,CHAR name,int *ivalue,int *flg,int *__ierr,int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetInt(c1,c2,ivalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *__ierr = ierr;
}

void optionsgetdouble_(CHAR pre,CHAR name,double *dvalue,int *flg,int *__ierr,
                       int len1,int len2){
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetDouble(c1,c2,dvalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *__ierr = ierr;
}

void optionsgetdoublearray_(CHAR pre,CHAR name,
              double *dvalue,int *nmax,int *flg,int *__ierr,int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetDoubleArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *__ierr = ierr;
}

void optionsgetintarray_(CHAR pre,CHAR name,int *dvalue,int *nmax,int *flg,
                         int *__ierr,int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetIntArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *__ierr = ierr;
}

void optionsgetstring_(CHAR pre,CHAR name,CHAR string,int *flg,
                       int *__ierr, int len1, int len2,int len){
  char *c1,*c2,*c3;
  int  ierr,len3;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
#if defined(USES_CPTOFCD)
    c3   = _fcdtocp(string);
    len3 = _fcdlen(string) - 1;
#else
    c3   = string;
    len3 = len - 1;
#endif

  ierr = OptionsGetString(c1,c2,c3,len3,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *__ierr = ierr;
}

void petscgetarchtype_(CHAR str,int *__ierr,int len)
{
#if defined(USES_CPTOFCD)
  char *tstr = _fcdtocp(str); 
  int  len1 = _fcdlen(str);
  *__ierr = PetscGetArchType(tstr,len1);
#else
  *__ierr = PetscGetArchType(str,len);
#endif
}

void petscgetprogramname_(CHAR name, int *__ierr,int len_in )
{
  char *tmp;
  int len;
#if defined(USES_CPTOFCD)
  tmp = _fcdtocp(name);
  len = _fcdlen(name) - 1;
#else
  tmp = name;
  len = len_in - 1;
#endif
  *__ierr = PetscGetProgramName(tmp,len);
}

#if defined(__cplusplus)
}
#endif


/*
    This is code for translating PETSc memory addresses to integer offsets 
    for Fortran.
*/
char   *PETSC_NULL_CHARACTER_Fortran;
void   *PETSC_NULL_INTEGER_Fortran;
void   *PETSC_NULL_SCALAR_Fortran;
void   *PETSC_NULL_DOUBLE_Fortran;
void   *PETSC_NULL_FUNCTION_Fortran;

long PetscIntAddressToFortran(int *base,int *addr)
{
  unsigned long tmp1 = (unsigned long) base,tmp2 = tmp1/sizeof(int);
  unsigned long tmp3 = (unsigned long) addr;
  long          itmp2;

  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1)/sizeof(int);
    itmp2 = (long) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3)/sizeof(int);
    itmp2 = -((long) tmp2);
  }
  if (base + itmp2 != addr) {
    (*PetscErrorPrintf)("PetscIntAddressToFortran:C and Fortran arrays are\n");
    (*PetscErrorPrintf)("not commonly aligned or are too far apart to be indexed \n");
    (*PetscErrorPrintf)("by an integer. Locations: C %ld Fortran %ld\n",tmp1,tmp3);
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  return itmp2;
}

int *PetscIntAddressFromFortran(int *base,long addr)
{
  return base + addr;
}

/*
       obj - PETSc object on which request is made
       base - Fortran array address
       addr - C array address
       res  - will contain offset from C to Fortran
       shift - number of bytes that prevent base and addr from being commonly aligned
*/
int PetscScalarAddressToFortran(PetscObject obj,Scalar *base,Scalar *addr,int N,long *res)
{
  unsigned long tmp1 = (unsigned long) base,tmp2 = tmp1/sizeof(Scalar);
  unsigned long tmp3 = (unsigned long) addr;
  long          itmp2;
  int           shift;

  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1)/sizeof(Scalar);
    itmp2 = (long) tmp2;
    shift = (sizeof(Scalar) - (int) ((tmp3 - tmp1) % sizeof(Scalar))) % sizeof(Scalar);
  } else {  
    tmp2  = (tmp1 - tmp3)/sizeof(Scalar);
    itmp2 = -((long) tmp2);
    shift = (int) ((tmp1 - tmp3) % sizeof(Scalar));
  }
  
  if (shift) { 
    /* 
        Fortran and C not Scalar aligned, recover by copying values into
        memory that is aligned with the Fortran
    */
    int                  ierr;
    Scalar               *work;
    PetscObjectContainer container;

    work = (Scalar *) PetscMalloc((N+1)*sizeof(Scalar));CHKPTRQ(work); 

    /* shift work by that number of bytes */
    work = (Scalar *) (((char *) work) + shift);
    PetscMemcpy(work,addr,N*sizeof(Scalar));

    /* store in the first location in addr how much you shift it */
    ((int *)addr)[0] = shift;
 
    ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscObjectContainerSetPointer(container,addr);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",(PetscObject)container);CHKERRQ(ierr);

    tmp3 = (unsigned long) work;
    if (tmp3 > tmp1) {  /* C is bigger than Fortran */
      tmp2  = (tmp3 - tmp1)/sizeof(Scalar);
      itmp2 = (long) tmp2;
      shift = (sizeof(Scalar) - (int) ((tmp3 - tmp1) % sizeof(Scalar))) % sizeof(Scalar);
    } else {  
      tmp2  = (tmp1 - tmp3)/sizeof(Scalar);
      itmp2 = -((long) tmp2);
      shift = (int) ((tmp1 - tmp3) % sizeof(Scalar));
    }
    if (shift) {
      (*PetscErrorPrintf)("PetscScalarAddressToFortran:C and Fortran arrays are\n");
      (*PetscErrorPrintf)("not commonly aligned.\n");
      (*PetscErrorPrintf)("Locations/sizeof(Scalar): C %f Fortran %f\n",
                         ((double) tmp3)/sizeof(Scalar),((double) tmp1)/sizeof(Scalar));
      MPI_Abort(PETSC_COMM_WORLD,1);
    }
    PLogInfo((void *)obj,"Efficiency warning, copying array in XXXGetArray() due\n\
    to alignment differences between C and Fortran\n");
  }
  *res = itmp2;
  return 0;
}

/*
    obj - the PETSc object where the scalar pointer came from
    base - the Fortran array address
    addr - the Fortran offset from base
    N    - the amount of data

    lx   - the array space that is to be passed to XXXXRestoreArray()
*/     
int PetscScalarAddressFromFortran(PetscObject obj,Scalar *base,long addr,int N,Scalar **lx)
{
  int                  ierr,shift;
  PetscObjectContainer container;
  Scalar               *tlx;

  ierr = PetscObjectQuery(obj,"GetArrayPtr",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscObjectContainerGetPointer(container,(void **) lx);CHKERRQ(ierr);
    tlx   = base + addr;

    shift = *(int *)*lx;
    PetscMemcpy(*lx,tlx,N*sizeof(Scalar));
    tlx  = (Scalar *) (((char *)tlx) - shift);
    PetscFree(tlx);
    ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",0);CHKERRQ(ierr);
  } else {
    *lx = base + addr;
  }
  return 0;
}

/*@
    MPICCommToFortranComm - Converts a MPI_Comm represented
    in C to one appropriate to pass to a Fortran routine.

    Not collective

    Input Parameter:
.   cobj - the C MPI_Comm

    Output Parameter:
.   fobj - the Fortran MPI_Comm

    Notes:
    MPICCommToFortranComm() must be called in a C/C++ routine.
    MPI 1 does not provide a standard for mapping between
    Fortran and C MPI communicators; this routine handles the
    mapping correctly on all machines.

.keywords: Fortran, C, MPI_Comm, convert, interlanguage

.seealso: MPIFortranCommToCComm()
@*/
int MPICCommToFortranComm(MPI_Comm comm,int *fcomm)
{
  *fcomm = PetscFromPointerComm(comm);
  PetscFunctionReturn(0);
}

/*@
    MPIFortranCommToCComm - Converts a MPI_Comm represented
    int Fortran (as an integer) to a MPI_Comm in C.

    Not collective

    Input Parameter:
.   fcomm - the Fortran MPI_Comm (an integer)

    Output Parameter:
.   comm - the C MPI_Comm

    Notes:
    MPIFortranCommToCComm() must be called in a C/C++ routine.
    MPI 1 does not provide a standard for mapping between
    Fortran and C MPI communicators; this routine handles the
    mapping correctly on all machines.

.keywords: Fortran, C, MPI_Comm, convert, interlanguage

.seealso: MPICCommToFortranComm()
@*/
int MPIFortranCommToCComm(int fcomm,MPI_Comm *comm)
{
  *comm = (MPI_Comm)PetscToPointerComm(fcomm);
  PetscFunctionReturn(0);
}



