#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mprint.c,v 1.17 1998/08/26 22:01:40 balay Exp balay $";
#endif
/*
      Some PETSc utilites routines to add simple IO capability.
*/
#include "petsc.h"        
#include "sys.h"             /*I    "sys.h"   I*/
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

/*
   If petsc_history is on, then all Petsc*Printf() results are saved
   if the appropriate (usually .petschistory) file.
*/
extern FILE *petsc_history;

/* ----------------------------------------------------------------------- */

typedef struct _PrintfQueue *PrintfQueue;
struct _PrintfQueue {
  char        string[256];
  PrintfQueue next;
};
static PrintfQueue queue       = 0,queuebase = 0;
static int         queuelength = 0;
static FILE        *queuefile  = PETSC_NULL;

#undef __FUNC__  
#define __FUNC__ "PetscSynchronizedPrintf" 
/*@C
    PetscSynchronizedPrintf - Prints synchronized output from several processors.
    Output of the first processor is followed by that of the second, etc.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string 

    Notes:
    Usage of PetscSynchronizedPrintf() with different MPI communicators
    REQUIRES an intervening call to PetscSynchronizedFlush().
    The length of the formatted message cannot exceed 256 charactors.

.seealso: PetscSynchronizedFlush(), PetscSynchronizedFPrintf(), PetscFPrintf(), 
          PetscPrintf()
@*/
int PetscSynchronizedPrintf(MPI_Comm comm,const char format[],...)
{
  int rank;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  
  /* First processor prints immediately to stdout */
  if (!rank) {
    va_list Argp;
    va_start( Argp, format );
#if defined(HAVE_VFPRINTF_CHAR)
    vfprintf(stdout,format,(char*)Argp);
#else
    vfprintf(stdout,format,Argp);
#endif
    fflush(stdout);
    if (petsc_history) {
#if defined(HAVE_VFPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end( Argp );
  } else { /* other processors add to local queue */
    int         len;
    va_list     Argp;
    PrintfQueue next = PetscNew(struct _PrintfQueue); CHKPTRQ(next);
    if (queue) {queue->next = next; queue = next;}
    else       {queuebase   = queue = next;}
    queuelength++;
    va_start( Argp, format );
#if defined(HAVE_VFPRINTF_CHAR)
    vsprintf(next->string,format,(char *)Argp);
#else
    vsprintf(next->string,format,Argp);
#endif
    va_end( Argp );
    len = PetscStrlen(next->string);
    if (len > 256) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Formated string longer than 256 bytes");
  }
    
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ "PetscSynchronizedFPrintf" 
/*@C
    PetscSynchronizedFPrintf - Prints synchronized output to the specified file from
    several processors.  Output of the first processor is followed by that of the 
    second, etc.

    Not Collective

    Input Parameters:
+   comm - the communicator
.   fd - the file pointer
-   format - the usual printf() format string 

    Notes:
    Usage of PetscSynchronizedFPrintf() with different MPI communicators
    REQUIRES an intervening call to PetscSynchronizedFlush().
    The length of the formatted message cannot exceed 256 charactors.

    Contributed by: Matthew Knepley

.seealso: PetscSynchronizedPrintf(), PetscSynchronizedFlush(), PetscFPrintf(),
          PetscFOpen()

@*/
int PetscSynchronizedFPrintf(MPI_Comm comm,FILE* fp,const char format[],...)
{
  int rank;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  
  /* First processor prints immediately to fp */
  if (!rank) {
    va_list Argp;
    va_start( Argp, format );
#if defined(HAVE_VPRINTF_CHAR)
    vfprintf(fp,format,(char*)Argp);
#else
    vfprintf(fp,format,Argp);
#endif
    fflush(fp);
    queuefile = fp;
    if (petsc_history) {
#if defined(HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end( Argp );
  } else { /* other processors add to local queue */
    int         len;
    va_list     Argp;
    PrintfQueue next = PetscNew(struct _PrintfQueue); CHKPTRQ(next);
    if (queue) {queue->next = next; queue = next;}
    else       {queuebase   = queue = next;}
    queuelength++;
    va_start( Argp, format );
#if defined(HAVE_VPRINTF_CHAR)
    vsprintf(next->string,format,(char *)Argp);
#else
    vsprintf(next->string,format,Argp);
#endif
    va_end( Argp );
    len = PetscStrlen(next->string);
    if (len > 256) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Formated string longer then 256 bytes");
  }
    
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSynchronizedFlush" 
/*@C
    PetscSynchronizedFlush - Flushes to the screen output from all processors 
    involved in previous PetscSynchronizedPrintf() calls.

    Collective on MPI_Comm

    Input Parameters:
.   comm - the communicator

    Notes:
    Usage of PetscSynchronizedPrintf() and PetscSynchronizedFPrintf() with
    different MPI communicators REQUIRES an intervening call to PetscSynchronizedFlush().

.seealso: PetscSynchronizedPrintf(), PetscFPrintf(), PetscPrintf()
@*/
int PetscSynchronizedFlush(MPI_Comm comm)
{
  int        rank,size,i,j,n, tag = 12341,ierr;
  char       message[256];
  MPI_Status status;
  FILE       *fd;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);
  
  /* First processor waits for messages from all other processors */
  if (!rank) {
    if (queuefile != PETSC_NULL) {
      fd = queuefile;
    } else {
      fd = stdout;
    }
    for ( i=1; i<size; i++ ) {
      ierr = MPI_Recv(&n,1,MPI_INT,i,tag,comm,&status);CHKERRQ(ierr);
      for ( j=0; j<n; j++ ) {
        ierr = MPI_Recv(message,256,MPI_CHAR,i,tag,comm,&status);CHKERRQ(ierr);
        fprintf(fd,"%s",message);
        if (petsc_history) {
          fprintf(petsc_history,"%s",message);
        }
      }
    }
    fflush(fd);
    if (petsc_history) fflush(petsc_history);
    queuefile = PETSC_NULL;
  } else { /* other processors send queue to processor 0 */
    PrintfQueue next = queuebase,previous;

    ierr = MPI_Send(&queuelength,1,MPI_INT,0,tag,comm);CHKERRQ(ierr);
    for ( i=0; i<queuelength; i++ ) {
      ierr     = MPI_Send(next->string,256,MPI_CHAR,0,tag,comm);CHKERRQ(ierr);
      previous = next; 
      next     = next->next;
      PetscFree(previous);
    }
    queue       = 0;
    queuelength = 0;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PetscFPrintf" 
/*@C
    PetscFPrintf - Prints to a file, only from the first
    processor in the communicator.

    Not Collective

    Input Parameters:
+   comm - the communicator
.   fd - the file pointer
-   format - the usual printf() format string 

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, fprintf

.seealso: PetscPrintf(), PetscSynchronizedPrintf()
@*/
int PetscFPrintf(MPI_Comm comm,FILE* fd,const char format[],...)
{
  int rank;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    va_list Argp;
    va_start( Argp, format );
#if defined(HAVE_VPRINTF_CHAR)
    vfprintf(fd,format,(char*)Argp);
#else
    vfprintf(fd,format,Argp);
#endif
    fflush(fd);
    if (petsc_history) {
#if defined(HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end( Argp );
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscPrintf" 
/*@C
    PetscPrintf - Prints to standard out, only from the first
    processor in the communicator.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string 

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, printf

.seealso: PetscFPrintf(), PetscSynchronizedPrintf()
@*/
int PetscPrintf(MPI_Comm comm,const char format[],...)
{
  int rank;

  PetscFunctionBegin;
  if (!comm) comm = PETSC_COMM_WORLD;
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    va_list Argp;
    va_start( Argp, format );
#if defined(HAVE_VPRINTF_CHAR)
    vfprintf(stdout,format,(char *)Argp);
#else
    vfprintf(stdout,format,Argp);
#endif
    fflush(stdout);
    if (petsc_history) {
#if defined(HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end( Argp );
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PetscHelpPrintfDefault" 
/*@C
    PetscHelpPrintfDefault - Prints to standard out, only from the first
    processor in the communicator.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string 

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, printf

.seealso: PetscFPrintf(), PetscSynchronizedPrintf()
@*/
int PetscHelpPrintfDefault(MPI_Comm comm,const char format[],...)
{
  int rank;

  PetscFunctionBegin;
  if (!comm) comm = PETSC_COMM_WORLD;
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    va_list Argp;
    va_start( Argp, format );
#if defined(HAVE_VPRINTF_CHAR)
    vfprintf(stdout,format,(char *)Argp);
#else
    vfprintf(stdout,format,Argp);
#endif
    fflush(stdout);
    if (petsc_history) {
#if defined(HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end( Argp );
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PetscErrorPrintfDefault" 
/*@C
    PetscErrorPrintfDefault - Prints error messages.

    Not Collective

    Input Parameters:
.   format - the usual printf() format string 

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, printf

.seealso: PetscFPrintf(), PetscSynchronizedPrintf()
@*/
int PetscErrorPrintfDefault(const char format[],...)
{
  va_list     Argp;
  static  int PetscErrorPrintfCalled = 0;

  /*
      This function does not call PetscFunctionBegin and PetscFunctionReturn() because
    it may be called by PetscStackView()
  */

  /*
       On the SGI machines and Cray T3E, if errors are generated  "simultaneously" by
    different processors, the messages are printed all jumbled up; to try to 
    prevent this we have each processor wait based on their rank
  */
  if (!PetscErrorPrintfCalled) {
    int  rank;
    char arch[10],hostname[64],username[16],pname[256];

    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    if (rank > 8) rank = 8;
    PetscSleep(rank);
    PetscGetArchType(arch,10);
    PetscGetHostName(hostname,64);
    PetscGetUserName(username,16);
    PetscGetProgramName(pname,256);
    fprintf(stderr,"--------------------------------------------\
------------------------------\n");
    fprintf(stderr,"%s\n",PETSC_VERSION_NUMBER);
    fprintf(stderr,"Satish Balay, Bill Gropp, Lois Curfman McInnes, Barry Smith.\n");
    fprintf(stderr,"Bug reports, questions: petsc-maint@mcs.anl.gov\n");
    fprintf(stderr,"Web page: http://www.mcs.anl.gov/petsc/\n");
    fprintf(stderr,"See docs/copyright.html for copyright information.\n");
    fprintf(stderr,"See docs/changes.html for recent updates.\n");
    fprintf(stderr,"See docs/troubleshooting.html for hints about trouble shooting.\n");
    fprintf(stderr,"See docs/manualpages/manualpages.html for manual pages.\n");
    fprintf(stderr,"--------------------------------------------\
---------------------------\n");
    fprintf(stderr,"%s \n on a %s named %s by %s %s",pname,arch,hostname,username,PetscGetDate());
#if !defined (PARCH_win32)
    fprintf(stderr,"Libraries linked from %s\n",PETSC_LDIR);
#endif
    fprintf(stderr,"--------------------------------------------\
---------------------------\n");
    fflush(stderr);
    PetscErrorPrintfCalled = 1;
  }

  va_start( Argp, format );
#if defined(HAVE_VPRINTF_CHAR)
  vfprintf(stderr,format,(char *)Argp);
#else
  vfprintf(stderr,format,Argp);
#endif
  fflush(stderr);
  va_end( Argp );
  return 0;
}

