#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mtr.c,v 1.105 1998/04/21 15:08:01 curfman Exp bsmith $";
#endif
/*
     PETSc's interface to malloc() and free(). This code allows for 
  logging of memory usage and some error checking 
*/
#include "petsc.h"           /*I "petsc.h" I*/
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/petscfix.h"


/*
     These are defined in mal.c and ensure that malloced space is Scalar aligned
*/
extern void *PetscMallocAlign(int);
extern int  PetscFreeAlign(void *);

void *PetscTrMallocDefault(unsigned int, int, char *,char *,char *);
int  PetscTrFreeDefault( void *, int, char *,char *,char *);

/*
  Code for checking if a pointer is out of the range 
  of malloced memory. This will only work on flat memory models and 
  even then is suspicious.
*/
void *PetscLow = (void *) 0x0  , *PetscHigh = (void *) 0xEEEEEEEE;
static int TrUseNan;   /* unitialize Scalar arrays with Nans */

#undef __FUNC__  
#define __FUNC__ "PetscSetUseTrMalloc_Private"
int PetscSetUseTrMalloc_Private(int usenan)
{
  int ierr;

  PetscFunctionBegin;
#if !defined(PETSC_INSIGHT)
  PetscLow     = (void *) 0xEEEEEEEE;
  PetscHigh    = (void *) 0x0;
#endif
  ierr         = PetscSetMalloc(PetscTrMallocDefault,PetscTrFreeDefault); CHKERRQ(ierr);
  TrUseNan     = usenan;
  PetscFunctionReturn(0);
}

/*
    PetscTrSpace - Routines for tracing space usage.

    Description:
    PetscTrMalloc replaces malloc and PetscTrFree replaces free.  These routines
    have the same syntax and semantics as the routines that they replace,
    In addition, there are routines to report statistics on the memory
    usage, and to report the currently allocated space.  These routines
    are built on top of malloc and free, and can be used together with
    them as long as any space allocated with PetscTrMalloc is only freed with
    PetscTrFree.
 */

/* HEADER_DOUBLES is the number of doubles in a PetscTrSpace header */
/* We have to be careful about alignment rules here */

#define HEADER_DOUBLES      8

#if defined(HAVE_64BITS)
#define TR_ALIGN_BYTES      8
#define TR_ALIGN_MASK       0x7
#else
#define TR_ALIGN_BYTES      4
#define TR_ALIGN_MASK       0x3
#endif

#define COOKIE_VALUE   0xf0e0d0c9
#define ALREADY_FREED  0x0f0e0d9c
#define MAX_TR_STACK   20
#define TR_MALLOC      0x1
#define TR_FREE        0x2

typedef struct _trSPACE {
    unsigned long   size;
    int             id;
    int             lineno;
    char            *filename;
    char            *functionname;
    char            *dirname;
    unsigned long   cookie;        
    struct _trSPACE *next, *prev;
} TRSPACE;

/* This union is used to insure that the block passed to the user is
   aligned on a double boundary */
typedef union {
    TRSPACE sp;
    double  v[HEADER_DOUBLES];
} TrSPACE;

static long    allocated    = 0, frags = 0;
static TRSPACE *TRhead      = 0;
static int     TRid         = 0;
static int     TRdebugLevel = 0;
static long    TRMaxMem     = 0;

#if defined(HAVE_MALLOC_VERIFY) && defined(__cplusplus)
extern "C" {
  extern int malloc_verify();
}
#elif defined(HAVE_MALLOC_VERIFY)
  extern int malloc_verify();
#endif


#undef __FUNC__  
#define __FUNC__ "PetscTrValid"
/*
   PetscTrValid - Test the allocated blocks for validity.  This can be used to
   check for memory overwrites.

   Input Parameter:
.  line, file - line number and filename where call originated.

   Return value:
   The number of errors detected.
   
   Output Effect:
   Error messages are written to stdout.  

   No output is generated if there are no problems detected.
*/
int PetscTrValid(int line,char *function,char *file,char *dir )
{
  TRSPACE  *head;
  char     *a;
  unsigned long *nend;

  PetscFunctionBegin;
  head = TRhead;
  while (head) {
    if (head->cookie != COOKIE_VALUE) {
      (*PetscErrorPrintf)("called from %s() line %d in %s%s\n",function,line,dir,file );
      (*PetscErrorPrintf)("Block at address %p is corrupted\n", head );
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      SETERRQ(PETSC_ERR_MEMC,0,"");
    }
    if (head->size <=0) {
      (*PetscErrorPrintf)("called from %s() line %d in %s%s\n",function,line,dir,file );
      (*PetscErrorPrintf)("Block at address %p is corrupted\n", head );
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      SETERRQ(PETSC_ERR_MEMC,0,"");
    }
    a    = (char *)(((TrSPACE*)head) + 1);
    nend = (unsigned long *)(a + head->size);
    if (nend[0] != COOKIE_VALUE) {
      (*PetscErrorPrintf)("called from %s() line %d in %s%s\n",function,line,dir,file );
      if (nend[0] == ALREADY_FREED) {
        (*PetscErrorPrintf)("Block [id=%d(%lx)] at address %p already freed\n",head->id,head->size, a );
        SETERRQ(PETSC_ERR_MEMC,0,"Freed block in memory list, corrupted memory");
      } else {
        (*PetscErrorPrintf)("Block [id=%d(%lx)] at address %p is corrupted (probably write past end)\n", 
	        head->id, head->size, a );
        (*PetscErrorPrintf)("Block allocated in %s() line %d in %s%s\n",head->functionname,
                head->lineno,head->dirname,head->filename);
        SETERRQ(PETSC_ERR_MEMC,0,"Corrupted memory");
      }
    }
    head = head->next;
  }
#if defined(HAVE_MALLOC_VERIFY) && defined(USE_PETSC_BOPT_g)
  malloc_verify();
#endif

  PetscFunctionReturn(0);
}

/*
      Arrays to log information on all Mallocs
*/
static int  PetscLogMallocMax = 10000, PetscLogMalloc = -1, *PetscLogMallocLength;
static char **PetscLogMallocDirectory, **PetscLogMallocFile,**PetscLogMallocFunction;


#undef __FUNC__  
#define __FUNC__ "PetscTrMallocDefault"
/*
    PetscTrMallocDefault - Malloc with tracing.

    Input Parameters:
.   a   - number of bytes to allocate
.   lineno - line number where used.  Use __LINE__ for this
.   function - function calling routine. Use __FUNC__ for this
.   filename  - file name where used.  Use __FILE__ for this
.   dir - directory where file is. Use __SDIR__ for this

    Returns:
    double aligned pointer to requested storage, or null if not
    available.
 */
void *PetscTrMallocDefault(unsigned int a,int lineno,char *function,char *filename,char *dir)
{
  TRSPACE          *head;
  char             *inew;
  unsigned long    *nend;
  unsigned int     nsize;
  int              ierr;

  PetscFunctionBegin;
  if (TRdebugLevel > 0) {
    ierr = PetscTrValid(lineno,function,filename,dir); if (ierr) PetscFunctionReturn(0);
  }

  if (a == 0) {
    (*PetscErrorPrintf)("PETSC ERROR: PetscTrMalloc: malloc zero length, this is illegal!\n");
    PetscFunctionReturn(0);
  }
  nsize = a;
  if (nsize & TR_ALIGN_MASK) nsize += (TR_ALIGN_BYTES - (nsize & TR_ALIGN_MASK));
  inew = (char *) PetscMallocAlign((unsigned)(nsize+sizeof(TrSPACE)+sizeof(Scalar)));  
  if (!inew) PetscFunctionReturn(0);

  
  /*
   Keep track of range of memory locations we have malloced in 
  */
#if !defined(PETSC_INSIGHT)
  if (PetscLow > (void *) inew) PetscLow = (void *) inew;
  if (PetscHigh < (void *) (inew+nsize+sizeof(TrSPACE)+sizeof(unsigned long)))
      PetscHigh = (void *) (inew+nsize+sizeof(TrSPACE)+sizeof(unsigned long));
#endif

  head   = (TRSPACE *)inew;
  inew  += sizeof(TrSPACE);

  if (TRhead) TRhead->prev = head;
  head->next     = TRhead;
  TRhead         = head;
  head->prev     = 0;
  head->size     = nsize;
  head->id       = TRid;
  head->lineno   = lineno;

  head->filename     = filename;
  head->functionname = function;
  head->dirname      = dir;
  head->cookie       = COOKIE_VALUE;
  nend               = (unsigned long *)(inew + nsize);
  nend[0]            = COOKIE_VALUE;

  allocated += nsize;
  if (allocated > TRMaxMem) {
    TRMaxMem   = allocated;
  }
  frags++;

  if (TrUseNan && sizeof(Scalar)*(nsize/sizeof(Scalar)) == nsize) {
    ierr = PetscInitializeNans((Scalar*) inew,nsize/sizeof(Scalar)); 
    if (ierr) PetscFunctionReturn(0);
  } else if (TrUseNan && sizeof(int)*(nsize/sizeof(int)) == nsize) {
    ierr = PetscInitializeLargeInts((int*) inew,nsize/sizeof(int)); 
    if (ierr) PetscFunctionReturn(0);
  }

  /*
         Allow logging of all mallocs made
  */
  if (PetscLogMalloc > -1 && PetscLogMalloc < PetscLogMallocMax) {
    if (PetscLogMalloc == 0) {
      PetscLogMallocLength    = (int *) malloc( PetscLogMallocMax*sizeof(int));
      if (!PetscLogMallocLength) PetscFunctionReturn(0);
      PetscLogMallocDirectory = (char **) malloc( PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocDirectory) PetscFunctionReturn(0);
      PetscLogMallocFile      = (char **) malloc( PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocFile) PetscFunctionReturn(0);
      PetscLogMallocFunction  = (char **) malloc( PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocFunction) PetscFunctionReturn(0);
    }
    PetscLogMallocLength[PetscLogMalloc]      = nsize;
    PetscLogMallocDirectory[PetscLogMalloc]   = dir;
    PetscLogMallocFile[PetscLogMalloc]        = filename;
    PetscLogMallocFunction[PetscLogMalloc++]  = function; 
  }

  PetscFunctionReturn((void *)inew);
}


#undef __FUNC__  
#define __FUNC__ "PetscTrFreeDefault"
/*
   PetscTrFreeDefault - Free with tracing.

   Input Parameters:
.   a    - pointer to a block allocated with PetscTrMalloc
.   lineno - line number where used.  Use __LINE__ for this
.   function - function calling routine. Use __FUNC__ for this
.   file  - file name where used.  Use __FILE__ for this
.   dir - directory where file is. Use __SDIR__ for this
 */
int PetscTrFreeDefault( void *aa, int line, char *function, char *file, char *dir )
{
  char     *a = (char *) aa;
  TRSPACE  *head;
  char     *ahead;
  int      ierr;
  unsigned long *nend;

  PetscFunctionBegin;
  /* Don't try to handle empty blocks */
  if (!a) {
    (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Trying to free null block");
  }

  if (TRdebugLevel > 0) {
    ierr = PetscTrValid(line,function,file,dir); CHKERRQ(ierr);
  }

#if !defined(PETSC_INSIGHT)
  if (PetscLow > aa || PetscHigh < aa){
    (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"PetscTrFreeDefault called with address not allocated by PetscTrMallocDefault");
  } 
#endif

  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE *)a;

  if (head->cookie != COOKIE_VALUE) {
    (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
    (*PetscErrorPrintf)("Block at address %p is corrupted; cannot free;\n\
may be block not allocated with PetscTrMalloc or PetscMalloc\n", a );
    SETERRQ(PETSC_ERR_MEMC,0,"Bad location or corrupted memory");
  }
  nend = (unsigned long *)(ahead + head->size);
  if (*nend != COOKIE_VALUE) {
    if (*nend == ALREADY_FREED) {
      (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Block [id=%d(%lx)] at address %p was already freed\n", 
              head->id, head->size, a + sizeof(TrSPACE) );
      if (head->lineno > 0) {
	(*PetscErrorPrintf)("Block freed in %s() line %d in %s%s\n", head->functionname,
                 head->lineno,head->dirname,head->filename);	
      } else {
        (*PetscErrorPrintf)("Block allocated in %s() line %d in %s%s\n", head->functionname,
                -head->lineno,head->dirname,head->filename);	
      }
      SETERRQ(PETSC_ERR_ARG_WRONG,0,"Memory already freed");
    } else {
      /* Damaged tail */
      (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Block [id=%d(%lx)] at address %p is corrupted (probably write past end)\n", 
		head->id, head->size, a );
      (*PetscErrorPrintf)("Block allocated in %s() line %d in %s%s\n", head->functionname,
                 head->lineno,head->dirname,head->filename);
      SETERRQ(PETSC_ERR_MEMC,0,"Corrupted memory");
    }
  }
  /* Mark the location freed */
  *nend        = ALREADY_FREED;
  /* Save location where freed.  If we suspect the line number, mark as 
     allocated location */
  if (line > 0 && line < 50000) {
    head->lineno       = line;
    head->filename     = file;
    head->functionname = function;
    head->dirname      = dir;
  } else {
    head->lineno = - head->lineno;
  }
  /* zero out memory - helps to find some reuse of already freed memory */
  PetscMemzero(aa,head->size);

  allocated -= head->size;
  frags     --;
  if (head->prev) head->prev->next = head->next;
  else TRhead = head->next;

  if (head->next) head->next->prev = head->prev;
  PetscFreeAlign(a);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscTrSpace"
/*@
    PetscTrSpace - Returns space statistics.
   
    Output Parameters:
.   space - number of bytes currently allocated
.   frags - number of blocks currently allocated
.   maxs - maximum number of bytes ever allocated

    Not Collective

.keywords: memory, allocation, tracing, space, statistics

.seealso: PetscTrDump()
 @*/
int PetscTrSpace( PLogDouble *space, PLogDouble *fr, PLogDouble *maxs )
{
  PetscFunctionBegin;

  if (space) *space = (PLogDouble) allocated;
  if (fr)    *fr    = (PLogDouble) frags;
  if (maxs)  *maxs  = (PLogDouble) TRMaxMem;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscTrDump"
/*@C
   PetscTrDump - Dumps the allocated memory blocks to a file. The information 
   printed is: size of space (in bytes), address of space, id of space, 
   file in which space was allocated, and line number at which it was 
   allocated.

   Input Parameter:
.  fp  - file pointer.  If fp is NULL, stderr is assumed.

   Collective on PETSC_COMM_WORLD

   Options Database Key:
.  -trdump - Dumps unfreed memory during call to PetscFinalize()

   Notes: The calling sequence in Fortran is PetscTrDump(integer ierr)

.keywords: memory, allocation, tracing, space, statistics

.seealso:  PetscTrSpace()
 @*/
int PetscTrDump( FILE *fp )
{
  TRSPACE *head;
  int     rank;

  PetscFunctionBegin;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (fp == 0) fp = stderr;
  if (allocated > 0) {fprintf(fp,"[%d]Total space allocated %d bytes\n",rank,(int)allocated);}
  head = TRhead;
  while (head) {
    fprintf(fp,"[%2d]%8d bytes %s() line %d in %s%s\n",rank,(int) head->size,
            head->functionname,head->lineno,head->dirname,head->filename);
    head = head->next;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

#undef __FUNC__  
#define __FUNC__ "PetscTrLog"
/*@C
    PetscTrLog - Activates logging of all calls to malloc.

    Not Collective

    Options Database Key:
.  -trmalloc_log - Activates PetscTrLog() and PetscTrLogDump()

.seealso: PetscTrLogDump()
@*/
int PetscTrLog(void)
{
  PetscFunctionBegin;

  PetscLogMalloc = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscTrLogDump"
/*@C
    PetscTrLogDump - Dumps the log of all calls to malloc; also calls 
    PetscGetResidentSetSize().

    Input Parameters:
.   fp - file pointer; or PETSC_NULL

    Collective on PETSC_COMM_WORLD

    Options Database Key:
.  -trmalloc_log - Activates PetscTrLog() and PetscTrLogDump()

.seealso: PetscTrLog()
@*/
int PetscTrLogDump(FILE *fp)
{
  int        i,rank,j,n,*shortlength,ierr,dummy,size, tag = 1212 /* very bad programming */;
  char       **shortfunction;
  PLogDouble rss;
  MPI_Status status;

  PetscFunctionBegin;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  /*
       Try to get the data printed in order by processor. This will only sometimes work 
  */  
  fflush(fp);
  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  if (rank) {
    ierr = MPI_Recv(&dummy,1,MPI_INT,rank-1,tag,MPI_COMM_WORLD,&status);CHKERRQ(ierr);
  }


  if (fp == 0) fp = stderr;
  ierr = PetscGetResidentSetSize(&rss); CHKERRQ(ierr);
  fprintf(fp,"[%d] Maximum memory used %d Size of entire process %d\n",rank,(int)TRMaxMem,(int)rss);

  /*
  for ( i=0; i<PetscLogMalloc; i++ ) {
    fprintf(fp,"[%d] %d %s%s %s()\n",rank,PetscLogMallocLength[i],PetscLogMallocDirectory[i],
            PetscLogMallocFile[i],PetscLogMallocFunction[i]);
  }
  */

  shortlength   = (int *) malloc(PetscLogMalloc*sizeof(int)); CHKPTRQ(shortlength);
  shortfunction = (char**) malloc(PetscLogMalloc*sizeof(char *));CHKPTRQ(shortfunction);
  shortfunction[0] = PetscLogMallocFunction[0];
  shortlength[0]   = PetscLogMallocLength[0]; 
  n = 1;
  for ( i=1; i<PetscLogMalloc; i++ ) {
    for ( j=0; j<n; j++ ) {
      if (!PetscStrcmp(shortfunction[j],PetscLogMallocFunction[i])) {
        shortlength[j] += PetscLogMallocLength[i];
        goto foundit;
      }
    }
    shortfunction[n] = PetscLogMallocFunction[i];
    shortlength[n]   = PetscLogMallocLength[i]; 
    n++;
    foundit:;
  }

  fprintf(fp,"[%d] Memory usage sorted by function\n",rank);
  for ( i=0; i<n; i++ ) {
    fprintf(fp,"[%d] %d %s()\n",rank,shortlength[i],shortfunction[i]);
  }
  free(shortlength);
  free(shortfunction);
  fflush(fp);
  if (size > 1 && rank != size-1) {
    ierr = MPI_Send(&dummy,1,MPI_INT,rank+1,tag,MPI_COMM_WORLD);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

#undef __FUNC__  
#define __FUNC__ "PetscTrDebugLevel"
/*
    PetscTrDebugLevel - Set the level of debugging for the space management 
                   routines.

    Input Parameter:
.   level - level of debugging.  Currently, either 0 (no checking) or 1
    (use PetscTrValid at each PetscTrMalloc or PetscTrFree).
*/
int  PetscTrDebugLevel(int level )
{
  PetscFunctionBegin;

  TRdebugLevel = level;
  PetscFunctionReturn(0);
}

#if defined(PARCH_IRIX) || defined(PARCH_IRIX5)
static long nanval[2] = {0x7fffffff,0xffffffff };  /* Signaling nan */
/* static long nanval[2] = {0x7ff7ffff,0xffffffff };  Quiet nan */
#elif defined(PARCH_sun4)
#elif defined(PARCH_rs6000)
struct sigcontext;
#include <fpxcp.h>
#else
static long nanval[2] = {-1,-1}; /* Probably a bad floating point value */
#endif

typedef union { long l[2]; double d; } NANDouble;

#include <math.h>
#undef __FUNC__  
#define __FUNC__ "PetscInitializeNans"
/*@
   PetscInitializeNans - Intialize certain memory locations with NANs.

   Input parameters:
.  p   - pointer to data
.  n   - length of data (in Scalars)

   Not Collective

   Options Database Key:
.  -trmalloc_nan - Activates PetscInitializeLargeInts() and PetscInitializeNans()

   Notes:
   This routine is used to mark an array as being uninitialized, so that
   if values are used for computation without first having been set,
   a floating point exception is generated.

   This routine is useful for tracking down the use of uninitialized
   array values.  If the code is run with the -fp_trap option, it will
   stop if one of the "unitialized" values is used in a computation.

.seealso: PetscInitializeLargeInts()
@*/
int PetscInitializeNans(Scalar *p,int n )
{
  double     *pp,nval;

  PetscFunctionBegin;
#if defined(PARCH_sun4) 
  nval = signaling_nan();
#elif defined(PARCH_rs6000)
  nval = FP_INV_SNAN;
#else
  { 
    NANDouble  nd;
    nd.l[0] = nanval[0];
    nd.l[1] = nanval[1];
    nval    = nd.d;
  }
#endif
  pp = (double *) p;
#if defined(USE_PETSC_COMPLEX)
  n *= 2;
#endif
  while (n--) *pp++   = nval;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscInitializeLargeInts"
/*@
   PetscInitializeLargeInts - Intializes an array of integers
   with very large values.

   Input Parameters:
.  p   - pointer to data
.  n   - length of data (in ints)

   Not Collective

   Options Database Key:
.  -trmalloc_nan - Activates PetscInitializeLargeInts() and PetscInitializeNans()

   Notes:
   This routine is useful for tracking down the use of uninitialized
   array values.  If an integer array value is absurdly large, then
   there's a good chance that it is being used before having been set.

.seealso: PetscInitializeNans()
@*/
int PetscInitializeLargeInts(int *p,int n )
{
  PetscFunctionBegin;

  while (n--) *p++   = 1073741824;
  PetscFunctionReturn(0);
}






