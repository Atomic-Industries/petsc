
#ifndef lint
static char vcid[] = "$Id: options.c,v 1.104 1996/10/28 15:00:05 bsmith Exp bsmith $";
#endif
/*
   These routines simplify the use of command line, file options, etc.,
   and are used to manipulate the options database.

  This file uses regular malloc and free because it cannot know 
  what malloc is being used until it has already processed the input.
*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include <stdio.h>
#include <math.h>
#include "sys.h"
#include "src/sys/nreg.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"

/* 
    For simplicity, we begin with a static size database
*/
#define MAXOPTIONS 256
#define MAXALIASES 25

typedef struct {
  int  N,argc,Naliases;
  char **args,*names[MAXOPTIONS],*values[MAXOPTIONS];
  char *aliases1[MAXALIASES],*aliases2[MAXALIASES];
  int  used[MAXOPTIONS];
  int  namegiven;
  char programname[256]; /* HP includes entire path in name */
} OptionsTable;

static OptionsTable *options = 0;
       int          PetscBeganMPI = 0;

int        OptionsCheckInitial_Private(),
           OptionsCreate_Private(int*,char***,char*),
           OptionsSetAlias_Private(char *,char *);
static int OptionsDestroy_Private();

#if defined(PETSC_COMPLEX)
MPI_Datatype  MPIU_COMPLEX;
Scalar PETSC_i(0.0,1.0);
#else
Scalar PETSC_i = 0.0;
#endif

/* 
   Optional file where all PETSc output from MPIU_*printf() is saved. 
*/
FILE *petsc_history = 0;

int PLogOpenHistoryFile(char *filename,FILE **fd)
{
  int  ierr,rank,size;
  char pfile[256];

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 
  if (!rank) {
    char arch[10];
    PetscGetArchType(arch,10);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    if (!filename) {
      ierr = PetscGetHomeDirectory(240,pfile); CHKERRQ(ierr);
      PetscStrcat(pfile,"/.petschistory");
      filename = pfile;
    }
    *fd = fopen(filename,"a"); if (!fd) SETERRQ(1,"PLogOpenHistoryFile:");
    fprintf(*fd,"---------------------------------------------------------\n");
    fprintf(*fd,"%s %s ",PETSC_VERSION_NUMBER,PetscGetDate());
    fprintf(*fd,"%s on a %s, %d proc. with options:\n",
            options->programname,arch,size);
    OptionsPrint(*fd);
    fprintf(*fd,"---------------------------------------------------------\n");
    fflush(*fd);
  }
  return 0; 
}

static int PLogCloseHistoryFile(FILE **fd)
{
  int  rank;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 
  if (rank) return 0;
  fprintf(*fd,"---------------------------------------------------------\n");
  fprintf(*fd,"Finished at %s",PetscGetDate());
  fprintf(*fd,"---------------------------------------------------------\n");
  fflush(*fd);
  fclose(*fd);
  return 0; 
}

int      PetscInitializedCalled = 0;
int      PetscGlobalRank = -1, PetscGlobalSize = -1;
MPI_Comm PETSC_COMM_WORLD = 0;

/* ---------------------------------------------------------------------------*/
int    PetscCompare          = 0;
double PetscCompareTolerance = 1.e-10;
/*@C
   PetscCompareInt - Compares intss while running with PETSc's incremental
        debugger; triggered with the -compare option.

  Input Parameter:
.    d - integer to compare

@*/
int PetscCompareInt(int d)
{
  int work = d;

  MPI_Bcast(&work,1,MPI_INT,0,MPI_COMM_WORLD);
  if (d != work) {
    SETERRQ(1,"PetscCompareInt:Inconsistent integer");
  }
  return 0;
}

/*@C
   PetscCompareDouble - Compares doubles while running with PETSc's incremental
        debugger; triggered with the -compare <tol> flag.

  Input Parameter:
.    d - double precision number to compare

@*/
int PetscCompareDouble(double d)
{
  double work = d;

  MPI_Bcast(&work,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  if (!d && !work) return 0;
  if (PetscAbsDouble(work - d)/PetscMax(PetscAbsDouble(d),PetscAbsDouble(work)) 
      > PetscCompareTolerance) {
    SETERRQ(1,"PetscCompareDouble:Inconsistent double");
  }
  return 0;
}

/*@C
   PetscCompareScalar - Compares scalars while running with PETSc's incremental
        debugger; triggered with the -compare <tol> flag.

  Input Parameter:
.    d - scalar to compare

@*/
int PetscCompareScalar(Scalar d)
{
  Scalar work = d;

  MPI_Bcast(&work,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
  if (!PetscAbsScalar(d) && !PetscAbsScalar(work)) return 0;
  if (PetscAbsScalar(work - d)/PetscMax(PetscAbsScalar(d),PetscAbsScalar(work)) 
      >= PetscCompareTolerance) {
    SETERRQ(1,"PetscCompareScalar:Inconsistent scalar");
  }
  return 0;
}

/*
    PetscCompareInitialize - If there is a command line option -compare then
       this routine calls MPI_Init() and sets up two PETSC_COMM_WORLD, one for 
       each program being compared.

    Note: Only works with C programs.

*/
int PetscCompareInitialize(double tol)
{
  int       i, len,rank,work,*gflag,size,mysize;
  char      *programname = options->programname, basename[256];
  MPI_Group group_all,group_sub;

  PetscCompareTolerance = tol;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if (!rank) {
    PetscStrcpy(basename,programname);
    len = PetscStrlen(basename);
  }

  /* broadcase name from first processor to all processors */
  MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(basename,len+1,MPI_CHAR,0,MPI_COMM_WORLD);

  /* determine what processors belong to my group */
  if (!PetscStrcmp(programname,basename)) work = 1;
  else                                    work = 0;
  gflag = (int *) malloc( size*sizeof(int) ); CHKPTRQ(gflag);
  MPI_Allgather(&work,1,MPI_INT,gflag,1 ,MPI_INT,MPI_COMM_WORLD); 
  mysize = 0;
  for ( i=0; i<size; i++ ) {
    if (work == gflag[i]) gflag[mysize++] = i;
  }
  /*   printf("[%d] my name %s basename %s mysize %d\n",rank,programname,basename,mysize); */

  if (mysize == 0 || mysize == size) {
    SETERRQ(1,"PetscCompareInitialize:Need two different programs to compare");
  }

  /* create a new communicator for each program */
  MPI_Comm_group(MPI_COMM_WORLD,&group_all);
  MPI_Group_incl(group_all,mysize,gflag,&group_sub);
  MPI_Comm_create(MPI_COMM_WORLD,group_sub,&PETSC_COMM_WORLD);
  MPI_Group_free(&group_all);
  MPI_Group_free(&group_sub);
  free(gflag);

  PetscCompare = 1;
  PLogInfo(0,"Configured to compare two programs\n",rank);
  return 0;
}
/* ------------------------------------------------------------------------------------*/
int PetscInitializeOptions()
{
  options = (OptionsTable*) malloc(sizeof(OptionsTable)); CHKPTRQ(options);
  PetscMemzero(options->used,MAXOPTIONS*sizeof(int));
  return 0;
}

/*@C
   PetscInitialize - Initializes the PETSc database and MPI. 
   PetscInitialize calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of 
   your program -- usually the very first line! 

   Input Parameters:
.  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use PETSC_NULL for default)
.  help - [optional] Help message to print, use PETSC_NULL for no message

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   Options Database Keys:
$  -start_in_debugger [noxterm,dbx,xdb,...]
$  -debugger_nodes node1,node2,...
$  -debugger_pause sleeptime (in seconds)
$  -trdebug
$  -trmalloc
$  -trmalloc_off
$  -no_signal_handler : Turns off the signal handler
$  -fp_trap : Stops on floating point exceptions
$      Note: On the IBM RS6000 this slows code by
$            at least a factor of 10.


   Fortran Version:
   In Fortran this routine has the format
$       call PetscInitialize(file,ierr)

.   ierr - error return code
.   file - [optional] PETSc database file name, defaults to 
           ~username/.petscrc (use PETSC_NULL_CHARACTER for default)
           
   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use PETSC_NULL as 
   in the C version.  See the users manual for details.

.keywords: initialize, options, database, startup

.seealso: PetscFinalize()
@*/
int PetscInitialize(int *argc,char ***args,char *file,char *help)
{
  int        ierr,flag,flg;

  if (PetscInitializedCalled) return 0;

  ierr = PetscInitializeOptions(); CHKERRQ(ierr);

  /*
     We initialize the program name here because MPICH has a bug in 
     it that it sets args[0] on all processors to be args[0]
     on the first processor.
  */
  if (argc && *argc) {
    options->namegiven = 1;
    PetscStrncpy(options->programname,**args,256);
  }
  else {options->namegiven = 0;}


  MPI_Initialized(&flag);
  if (!flag) {
    ierr = MPI_Init(argc,args); CHKERRQ(ierr);
    PetscBeganMPI    = 1;
    PETSC_COMM_WORLD = MPI_COMM_WORLD;
  } else if (!PETSC_COMM_WORLD) PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitializedCalled = 1;

  MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);
  MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);
#if defined(PETSC_COMPLEX)
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
  ierr = OptionsCreate_Private(argc,args,file); CHKERRQ(ierr);
  ierr = OptionsCheckInitial_Private(); CHKERRQ(ierr);
  ierr = ViewerInitialize_Private(); CHKERRQ(ierr);
  if (PetscBeganMPI) {
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    PLogInfo(0,"PETSc successfully started: number of processors = %d\n",size);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (help && flg) {
    PetscPrintf(PETSC_COMM_WORLD,help);
  }
  return 0;
}

/*@C 
   PetscFinalize - Checks for options to be called at the conclusion
   of the program and calls MPI_Finalize().

   Options Database Keys:
$  -optionstable : Calls OptionsPrint()
$  -optionsleft : Prints unused options that remain in 
$     the database
$  -log_all : Prints extensive log information (for
$      code compiled with PETSC_LOG)
$  -log : Prints basic log information (for code 
$      compiled with PETSC_LOG)
$  -log_summary : Prints summary of flop and timing
$      information to screen (for code compiled with 
$      PETSC_LOG)
$  -log_mpe : creates a logfile viewable by the 
$      utility upshot/nupshot (in MPICH distribution)
$  -mpidump : Calls PetscMPIDump()
$  -trdump : Calls PetscTrDump()
$  -trinfo : Prints total memory usage
$  -trdebug : Calls malloc_debug(2) to activate memory
$      allocation diagnostics (used by PETSC_ARCH=sun4, 
$      BOPT=[g,g_c++,g_complex] only!)

.keywords: finalize, exit, end

.seealso: PetscInitialize(), OptionsPrint(), PetscTrDump(), PetscMPIDump()
@*/
int PetscFinalize()
{
  int  ierr,i,rank = 0,flg1,flg2;

  ViewerDestroy_Private();
  ViewerDestroyDrawX_Private();
  ViewerDestroyMatlab_Private();
#if defined(PETSC_LOG)
  {
    char mname[64];
#if defined (HAVE_MPE)
    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_mpe",mname,64,&flg1); CHKERRQ(ierr);
    if (flg1){
      if (mname[0]) PLogMPEDump(mname); 
      else          PLogMPEDump(0);
    }
#endif
    ierr = OptionsHasName(PETSC_NULL,"-log_summary",&flg1); CHKERRQ(ierr);
    if (flg1) { PLogPrintSummary(PETSC_COMM_WORLD,stdout); }
    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_all",mname,64,&flg1); CHKERRQ(ierr);
    ierr = OptionsGetString(PETSC_NULL,"-log",mname,64,&flg2); CHKERRQ(ierr);
    if (flg1 || flg2){
      if (mname[0]) PLogDump(mname); 
      else          PLogDump(0);
    }
    PLogDestroy();
  }
#endif
  ierr = OptionsHasName(PETSC_NULL,"-no_signal_handler",&flg1);CHKERRQ(ierr);
  if (!flg1) { PetscPopSignalHandler(); }
  ierr = OptionsHasName(PETSC_NULL,"-mpidump",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscMPIDump(stdout); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-trdump",&flg1); CHKERRQ(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ierr = OptionsHasName(PETSC_NULL,"-optionstable",&flg1); CHKERRQ(ierr);
  if (flg1) {
    if (!rank) OptionsPrint(stdout);
  }
  ierr = OptionsHasName(PETSC_NULL,"-optionsleft",&flg1); CHKERRQ(ierr);
  if (flg1) {
    if (!rank) {
      int nopt = OptionsAllUsed();
      OptionsPrint(stdout);
      if (nopt == 0) 
        fprintf(stdout,"There are no unused options.\n");
      else if (nopt == 1) 
        fprintf(stdout,"There is one unused database option. It is:\n");
      else
        fprintf(stdout,"There are %d unused database options. They are:\n",nopt);
      for ( i=0; i<options->N; i++ ) {
        if (!options->used[i]) {
          fprintf(stdout,"Option left: name:-%s value: %s\n",options->names[i],
                                                           options->values[i]);
        }
        fflush(stdout);
      }
    } 
  }
  ierr = OptionsHasName(PETSC_NULL,"-log_history",&flg1); CHKERRQ(ierr);
  if (flg1) {
    PLogCloseHistoryFile(&petsc_history);
    petsc_history = 0;
  }
  ierr = OptionsHasName(PETSC_NULL,"-trdump",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trinfo",&flg2); CHKERRQ(ierr);
  if (flg1) {
    OptionsDestroy_Private();
    NRDestroyAll();
    PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);
      ierr = PetscTrDump(stderr); CHKERRQ(ierr);
    PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);
  }
  else if (flg2) {
    double maxm;
    OptionsDestroy_Private();
    NRDestroyAll();
    ierr = PetscTrSpace(PETSC_NULL,PETSC_NULL,&maxm); CHKERRQ(ierr);
    PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);
      fprintf(stdout,"[%d] Maximum memory used %g\n",rank,maxm);
      fflush(stdout);
    PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);
  }
  else {
    OptionsDestroy_Private();
    NRDestroyAll(); 
  }
  if (PetscBeganMPI) {
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    PLogInfo(0,"PETSc successfully ended!\n");
    ierr = MPI_Finalize(); CHKERRQ(ierr);
  }
  return 0;
}
 
/* 
   This is ugly and probably belongs somewhere else, but I want to 
  be able to put a true MPI abort error handler with command line args.

    This is so MPI errors in the debugger will leave all the stack 
  frames. The default abort cleans up and exits.
*/

void abort_function(MPI_Comm *comm,int *flag) 
{
  fprintf(stderr,"MPI error %d\n",*flag);
  abort();
}

#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
  extern int malloc_debug(int);
}
#elif defined(PARCH_sun4)
  extern int malloc_debug(int);
#endif

extern int PLogInfoAllow(PetscTruth);
extern int PetscSetUseTrMalloc_Private();

#include "snes.h" /* so that cookies are defined */

int OptionsCheckInitial_Private()
{
  char     string[64];
  MPI_Comm comm = PETSC_COMM_WORLD;
  int      flg1,flg2,flg3, ierr,*nodes,flag,i,rank;

#if defined(PETSC_BOPT_g)
  ierr = OptionsHasName(PETSC_NULL,"-trmalloc_off", &flg1); CHKERRQ(ierr);
  if (!flg1) { ierr = PetscSetUseTrMalloc_Private(); CHKERRQ(ierr); }
#else
  ierr = OptionsHasName(PETSC_NULL,"-trdump",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trmalloc",&flg2); CHKERRQ(ierr);
  if (flg1 || flg2) { ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr); }
#endif
  ierr = OptionsHasName(PETSC_NULL,"-trdebug",&flg1); CHKERRQ(ierr);
  if (flg1) { 
    ierr = PetscTrDebugLevel(1);CHKERRQ(ierr);
#if defined(PARCH_sun4) && defined(PETSC_BOPT_g)
    malloc_debug(2);
#endif
  }
  ierr = PetscSetDisplay(); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-v",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-version",&flg2); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg3); CHKERRQ(ierr);
  if (flg1 || flg2 || flg3 ){
    PetscPrintf(comm,"--------------------------------------------\
------------------------------\n");
    PetscPrintf(comm,"\t   %s\n",PETSC_VERSION_NUMBER);
    PetscPrintf(comm,"Satish Balay, Bill Gropp, Lois Curfman McInnes, Barry Smith.\n");
    PetscPrintf(comm,"Bug reports, questions: petsc-maint@mcs.anl.gov\n");
    PetscPrintf(comm,"Web page: http://www.mcs.anl.gov/petsc/petsc.html\n");
    PetscPrintf(comm,"See petsc/COPYRIGHT for copyright information,\
 petsc/Changes for recent updates.\n");
    PetscPrintf(comm,"--------------------------------------------\
---------------------------\n");
  }
  ierr = OptionsHasName(PETSC_NULL,"-fp_trap",&flg1); CHKERRQ(ierr);
  if (flg1) { ierr = PetscSetFPTrap(PETSC_FP_TRAP_ON); CHKERRQ(ierr); }
  ierr = OptionsHasName(PETSC_NULL,"-on_error_abort",&flg1); CHKERRQ(ierr);
  if (flg1) { PetscPushErrorHandler(PetscAbortErrorHandler,0); } 
  ierr = OptionsHasName(PETSC_NULL,"-on_error_stop",&flg1); CHKERRQ(ierr);
  if (flg1) { PetscPushErrorHandler(PetscStopErrorHandler,0); }
  /*
     Set default debugger on solaris and rs6000 to dbx because gdb doesn't
    work with the native compilers.
  */
#if defined(PARCH_solaris) || defined(PARCH_rs6000)
  ierr = PetscSetDebugger("dbx",1,0); CHKERRQ(ierr);
#endif
  ierr = OptionsGetString(PETSC_NULL,"-on_error_attach_debugger",string,64, 
                          &flg1); CHKERRQ(ierr);
  if (flg1) {
    char *debugger = 0, *display = 0;
    int  xterm     = 1;
    if (PetscStrstr(string,"noxterm")) xterm = 0;
#if defined(PARCH_hpux)
    if (PetscStrstr(string,"xdb"))     debugger = "xdb";
#else
    if (PetscStrstr(string,"dbx"))     debugger = "dbx";
#endif
#if defined(PARCH_rs6000)
    if (PetscStrstr(string,"xldb"))    debugger = "xldb";
#endif
    if (PetscStrstr(string,"xxgdb"))   debugger = "xxgdb";
    if (PetscStrstr(string,"ups"))     debugger = "ups";
    display = (char *) malloc(128*sizeof(char)); CHKPTRQ(display);
    PetscGetDisplay(display,128);
    PetscSetDebugger(debugger,xterm,display);
    PetscPushErrorHandler(PetscAttachDebuggerErrorHandler,0);
  }
  ierr = OptionsGetString(PETSC_NULL,"-start_in_debugger",string,64,&flg1);
         CHKERRQ(ierr);
  if (flg1) {
    char           *debugger = 0, *display = 0;
    int            xterm     = 1,size;
    MPI_Errhandler abort_handler;
    /*
       we have to make sure that all processors have opened 
       connections to all other processors, otherwise once the 
       debugger has stated it is likely to receive a SIGUSR1
       and kill the program. 
    */
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    if (size > 2) {
      int        dummy;
      MPI_Status status;
      for ( i=0; i<size; i++) {
        MPI_Send(&dummy,1,MPI_INT,i,109,PETSC_COMM_WORLD);
      }
      for ( i=0; i<size; i++) {
        MPI_Recv(&dummy,1,MPI_INT,i,109,PETSC_COMM_WORLD,&status);
      }
    }
    /* check if this processor node should be in debugger */
    nodes = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(nodes);
    ierr = OptionsGetIntArray(PETSC_NULL,"-debugger_nodes",nodes,&size,&flag);
           CHKERRQ(ierr);
    if (flag) {
      for (i=0; i<size; i++) {
        if (nodes[i] == rank) { flag = 0; break; }
      }
    }
    if (!flag) {        
      if (PetscStrstr(string,"noxterm")) xterm = 0;
#if defined(PARCH_hpux)
      if (PetscStrstr(string,"xdb"))     debugger = "xdb";
#else
      if (PetscStrstr(string,"dbx"))     debugger = "dbx";
#endif
#if defined(PARCH_rs6000)
      if (PetscStrstr(string,"xldb"))    debugger = "xldb";
#endif
      if (PetscStrstr(string,"xxgdb"))   debugger = "xxgdb";
      if (PetscStrstr(string,"ups"))     debugger = "ups";
      display = (char *) malloc( 128*sizeof(char) ); CHKPTRQ(display);
      PetscGetDisplay(display,128);
      PetscSetDebugger(debugger,xterm,display);
      PetscPushErrorHandler(PetscAbortErrorHandler,0);
      PetscAttachDebugger();
      MPI_Errhandler_create((MPI_Handler_function*)abort_function,&abort_handler);
      MPI_Errhandler_set(comm,abort_handler);
    }
    PetscFree(nodes);
  }
  ierr = OptionsHasName(PETSC_NULL,"-no_signal_handler", &flg1); CHKERRQ(ierr);
  if (!flg1) { PetscPushSignalHandler(PetscDefaultSignalHandler,(void*)0); }
#if defined(PETSC_LOG)
  {
    char mname[256];
    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_history",mname,256, &flg1);CHKERRQ(ierr);
    if(flg1) {
      if (mname[0]) {
        ierr = PLogOpenHistoryFile(mname,&petsc_history); CHKERRQ(ierr);
      }
      else {
        ierr = PLogOpenHistoryFile(0,&petsc_history); CHKERRQ(ierr);
      }
    }
  }
  ierr = OptionsHasName(PETSC_NULL,"-log_info", &flg1); CHKERRQ(ierr);
  if (flg1) { 
    char mname[256];
    PLogInfoAllow(PETSC_TRUE); 
    ierr = OptionsGetString(PETSC_NULL,"-log_info",mname,256, &flg1);CHKERRQ(ierr);
    if (flg1) {
      if (PetscStrstr(mname,"no_mat")) {
        PLogInfoDeactivateClass(MAT_COOKIE);
      }
      if (PetscStrstr(mname,"no_sles")) {
        PLogInfoDeactivateClass(SLES_COOKIE);
      }
    }
  }
#if defined (HAVE_MPE)
  ierr = OptionsHasName(PETSC_NULL,"-log_mpe", &flg1); CHKERRQ(ierr);
  if (flg1) PLogMPEBegin();
#endif
  ierr = OptionsHasName(PETSC_NULL,"-log_all", &flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-log", &flg2); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-log_summary", &flg3); CHKERRQ(ierr);
  if (flg1)              {  PLogAllBegin();  }
  else if (flg2 || flg3) {  PLogBegin(); }
#endif
  ierr = OptionsHasName(PETSC_NULL,"-help", &flg1); CHKERRQ(ierr);
  if (flg1) {
    PetscPrintf(comm,"Options for all PETSc programs:\n");
    PetscPrintf(comm," -on_error_abort: cause an abort when an error is");
    PetscPrintf(comm," detected. Useful \n       only when run in the debugger\n");
    PetscPrintf(comm," -on_error_attach_debugger [dbx,xxgdb,ups,noxterm]\n"); 
    PetscPrintf(comm,"       start the debugger (gdb by default) in new xterm\n");
    PetscPrintf(comm,"       unless noxterm is given\n");
    PetscPrintf(comm," -start_in_debugger [dbx,xxgdb,ups,noxterm]\n");
    PetscPrintf(comm,"       start all processes in the debugger\n");
    PetscPrintf(comm," -debugger_nodes [n1,n2,..] Nodes to start in debugger\n");
    PetscPrintf(comm," -debugger_pause m (in seconds) delay to get attached\n");
    PetscPrintf(comm," -display display: Location where graphics and debuggers are displayed\n");
    PetscPrintf(comm," -no_signal_handler: do not trap error signals\n");
    PetscPrintf(comm," -fp_trap: stop on floating point exceptions\n");
    PetscPrintf(comm,"           note on IBM RS6000 this slows run greatly\n");
    PetscPrintf(comm," -trdump: dump list of unfreed memory at conclusion\n");
    PetscPrintf(comm," -trmalloc: use our error checking malloc\n");
    PetscPrintf(comm," -trmalloc_off: don't use error checking malloc\n");
    PetscPrintf(comm," -trinfo: prints total memory usage\n");
    PetscPrintf(comm," -trdebug: enables extended checking for memory corruption\n");
    PetscPrintf(comm," -optionstable: dump list of options inputted\n");
    PetscPrintf(comm," -optionsleft: dump list of unused options\n");
    PetscPrintf(comm," -log[_all _summary]: logging objects and events\n");
#if defined (HAVE_MPE)
    PetscPrintf(comm," -log_mpe: Also create logfile viewable through upshot\n");
#endif
    PetscPrintf(comm," -log_info: print informative messages about the calculations\n");
    PetscPrintf(comm," -v: prints PETSc version number and release date\n");
    PetscPrintf(comm,"-----------------------------------------------\n");
  }
  ierr = OptionsHasName(PETSC_NULL,"-compare",&flg1); CHKERRQ(ierr);
  if (flg1) {
     double tol = 1.e-12;
     ierr = OptionsGetDouble(PETSC_NULL,"-compare",&tol,&flg1);CHKERRQ(ierr); 
     ierr = PetscCompareInitialize(tol); CHKERRQ(ierr);
  }
  return 0;
}

char *OptionsGetProgramName()
{
  if (!options) return (char *) 0;
  if (!options->namegiven) return (char *) 0;
  return options->programname;
}

/*
   OptionsCreate_Private - Creates a database of options.

   Input Parameters:
.  argc - count of number of command line arguments
.  args - the command line arguments
.  file - optional filename, defaults to ~username/.petscrc

   Note:
   Since OptionsCreate_Private() is automatically called by PetscInitialize(),
   the user does not typically need to call this routine. OptionsCreate_Private()
   can be called several times, adding additional entries into the database.

.keywords: options, database, create

.seealso: OptionsDestroy_Private(), OptionsPrint()
*/
int OptionsCreate_Private(int *argc,char ***args,char* file)
{
  int  ierr,rank;
  char pfile[128];

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (!file) {
    if ((ierr = PetscGetHomeDirectory(120,pfile))) return ierr;
    PetscStrcat(pfile,"/.petscrc");
    file = pfile;
  }

  options->N = 0;
  options->Naliases = 0;
  options->argc = (argc) ? *argc : 0;
  options->args = (args) ? *args : 0;

  /* insert file options */
  {
    char string[128],*first,*second,*third,*final;
    int   len;
    FILE *fd = fopen(file,"r"); 
    if (fd) {
      while (fgets(string,128,fd)) {
        /* Comments are indicated by #, ! or % in the first column */
        if (string[0] == '#') continue;
        if (string[0] == '!') continue;
        if (string[0] == '%') continue;
        first = PetscStrtok(string," ");
        second = PetscStrtok(0," ");
        if (first && first[0] == '-') {
          if (second) {final = second;} else {final = first;}
          len = PetscStrlen(final);
          while (len > 0 && (final[len-1] == ' ' || final[len-1] == '\n')) {
            len--; final[len] = 0;
          }
          OptionsSetValue(first,second);
        }
        else if (first && !PetscStrcmp(first,"alias")) {
          third = PetscStrtok(0," ");
          if (!third) SETERRQ(1,"OptionsCreate_Private:Error in options file:alias");
          len = PetscStrlen(third); 
          if (third[len-1] == '\n') third[len-1] = 0;
          ierr = OptionsSetAlias_Private(second,third); CHKERRQ(ierr);
        }
      }
      fclose(fd);
    }
  }
  /* insert environmental options */
  {
    char *eoptions = 0, *second, *first;
    int  len;
    if (!rank) {
      eoptions = (char *) getenv("PETSC_OPTIONS");
      len      = PetscStrlen(eoptions);
      MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);
    } else {
      MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);
      if (len) {
        eoptions = (char *) PetscMalloc((len+1)*sizeof(char *));CHKPTRQ(eoptions);
      }
    }
    if (len) {
      MPI_Bcast(eoptions,len,MPI_CHAR,0,PETSC_COMM_WORLD); 
      eoptions[len] = 0;
      first    = PetscStrtok(eoptions," ");
      while (first) {
        if (first[0] != '-') {first = PetscStrtok(0," "); continue;}
        second = PetscStrtok(0," ");
        if ((!second) || ((second[0] == '-') && (second[1] > '9'))) {
          OptionsSetValue(first,(char *)0);
          first = second;
        }
        else {
          OptionsSetValue(first,second);
          first = PetscStrtok(0," ");
        }
      }
      if (rank) PetscFree(eoptions);
    }
  }

  /* insert command line options */
  if (argc && args && *argc) {
    int   left = *argc - 1;
    char  **eargs = *args + 1;
    while (left) {
      if (eargs[0][0] != '-') {
        eargs++; left--;
      }
      else if ((left < 2) || ((eargs[1][0] == '-') && 
               ((eargs[1][1] > '9') || (eargs[1][1] < '0')))) {
        OptionsSetValue(eargs[0],(char *)0);
        eargs++; left--;
      }
      else {
        OptionsSetValue(eargs[0],eargs[1]);
        eargs += 2; left -= 2;
      }
    }
  }
  return 0;
}

/*@C
   OptionsPrint - Prints the options that have been loaded. This is
        useful for debugging purposes.

   Input Parameter:
.  FILE fd - location to print options (usually stdout or stderr)

   Options Database Key:
$  -optionstable : checked within PetscFinalize()

.keywords: options, database, print, table

.seealso: OptionsAllUsed()
@*/
int OptionsPrint(FILE *fd)
{
  int i;
  if (!fd) fd = stdout;
  if (!options) OptionsCreate_Private(0,0,0);
  for ( i=0; i<options->N; i++ ) {
    if (options->values[i]) {
      fprintf(fd,"OptionTable: -%s %s\n",options->names[i],options->values[i]);
    }
    else {
      fprintf(fd,"OptionTable: -%s\n",options->names[i]);
    }
  }
  fflush(fd);
  return 0;
}

/*
    OptionsDestroy_Private - Destroys the option database. 

    Note:
    Since OptionsDestroy_Private() is called by PetscFinalize(), the user 
    typically does not need to call this routine.

.keywords: options, database, destroy

.seealso: OptionsCreate_Private()
*/
static int OptionsDestroy_Private()
{
  int i;
  if (!options) return 0;
  for ( i=0; i<options->N; i++ ) {
    if (options->names[i]) free(options->names[i]);
    if (options->values[i]) free(options->values[i]);
  }
  for ( i=0; i<options->Naliases; i++ ) {
    free(options->aliases1[i]);
    free(options->aliases2[i]);
  }
  free(options);
  options = 0;
  return 0;
}

/*@C
   OptionsSetValue - Sets an option name-value pair in the options 
   database, overriding whatever is already present.

   Input Parameters:
.  name - name of option, this SHOULD have the - prepended
.  value - the option value (not used for all options)

   Note:
   Only some options have values associated with them, such as
   -ksp_rtol tol.  Other options stand alone, such as -ksp_monitor.

.keywords: options, database, set, value

.seealso: OptionsCreate_Private()
@*/
int OptionsSetValue(char *name,char *value)
{
  int  len, N, n, i;
  char **names;
  if (!options) OptionsCreate_Private(0,0,0);

  /* this is so that -h and -help are equivalent (p4 don't like -help)*/
  if (!PetscStrcmp(name,"-h")) name = "-help";

  name++;
  /* first check against aliases */
  N = options->Naliases; 
  for ( i=0; i<N; i++ ) {
    if (!PetscStrcmp(options->aliases1[i],name)) {
      name = options->aliases2[i];
      break;
    }
  }

  N = options->N; n = N;
  names = options->names; 
 
  for ( i=0; i<N; i++ ) {
    if (PetscStrcmp(names[i],name) == 0) {
      if (options->values[i]) free(options->values[i]);
      len = PetscStrlen(value);
      if (len) {
        options->values[i] = (char *) malloc( len ); 
        CHKPTRQ(options->values[i]);
        PetscStrcpy(options->values[i],value);
      }
      else { options->values[i] = 0;}
      return 0;
    }
    else if (PetscStrcmp(names[i],name) > 0) {
      n = i;
      break;
    }
  }
  if (N >= MAXOPTIONS) {
    fprintf(stderr,"No more room in option table, limit %d\n",MAXOPTIONS);
    fprintf(stderr,"recompile options/src/options.c with larger ");
    fprintf(stderr,"value for MAXOPTIONS\n");
    return 0;
  }
  /* shift remaining values down 1 */
  for ( i=N; i>n; i-- ) {
    names[i] = names[i-1];
    options->values[i] = options->values[i-1];
  }
  /* insert new name and value */
  len = (PetscStrlen(name)+1)*sizeof(char);
  names[n] = (char *) malloc( len ); CHKPTRQ(names[n]);
  PetscStrcpy(names[n],name);
  if (value) {
    len = (PetscStrlen(value)+1)*sizeof(char);
    options->values[n] = (char *) malloc( len ); CHKPTRQ(options->values[n]);
    PetscStrcpy(options->values[n],value);
  }
  else {options->values[n] = 0;}
  options->N++;
  return 0;
}

int OptionsSetAlias_Private(char *newname,char *oldname)
{
  int len,n = options->Naliases;

  if (newname[0] != '-') SETERRQ(1,"OptionsSetAlias_Private:aliased must have -");
  if (oldname[0] != '-') SETERRQ(1,"OptionsSetAlias_Private:aliasee must have -");
  if (n >= MAXALIASES) {SETERRQ(1,"OptionsSetAlias_Private:Aliases overflow");}

  newname++; oldname++;
  len = (PetscStrlen(newname)+1)*sizeof(char);
  options->aliases1[n] = (char *) malloc( len ); CHKPTRQ(options->aliases1[n]);
  PetscStrcpy(options->aliases1[n],newname);
  len = (PetscStrlen(oldname)+1)*sizeof(char);
  options->aliases2[n] = (char *) malloc( len );CHKPTRQ(options->aliases2[n]);
  PetscStrcpy(options->aliases2[n],oldname);
  options->Naliases++;
  return 0;
}

static int OptionsFindPair_Private( char *pre,char *name,char **value,int *flg)
{
  int  i, N,ierr;
  char **names,tmp[128];

  if (!options) {ierr = OptionsCreate_Private(0,0,0); CHKERRQ(ierr);}
  N = options->N;
  names = options->names;

  if (name[0] != '-') SETERRQ(1,"OptionsFindPair_Private:Name must begin with -");

  /* append prefix to name */
  if (pre) {
    PetscStrcpy(tmp,pre); PetscStrcat(tmp,name+1);
  }
  else PetscStrcpy(tmp,name+1);

  /* slow search */
  *flg = 0;
  for ( i=0; i<N; i++ ) {
    if (!PetscStrcmp(names[i],tmp)) {
       *value = options->values[i];
       options->used[i]++;
       *flg = 1;
       break;
     }
  }
  return 0;
}

/*@C
   OptionsHasName - Determines whether a certain option is given in the database.

   Input Parameters:
.  name - the option one is seeking 
.  pre - string to prepend to the name or PETSC_NULL

   Output Parameters:
.   flg - 1 if found else 0.


.keywords: options, database, has, name

.seealso: OptionsGetInt(), OptionsGetDouble(),
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsHasName(char* pre,char *name,int *flg)
{
  char *value;
  return OptionsFindPair_Private(pre,name,&value,flg);
}

/*@C
   OptionsGetInt - Gets the integer value for a particular option in the 
                   database.

   Input Parameters:
.  name - the option one is seeking
.  pre - the string to prepend to the name or PETSC_NULL

   Output Parameter:
.  ivalue - the integer value to return
.  flg - 1 if found, else 0

.keywords: options, database, get, int

.seealso: OptionsGetDouble(), OptionsHasName(), OptionsGetString(),
          OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetInt(char*pre,char *name,int *ivalue,int *flg)
{
  char *value;
  int  ierr;

  ierr = OptionsFindPair_Private(pre,name,&value,flg); CHKERRQ(ierr);
  if (!*flg) return 0;
  if (!value) {*flg = 0; *ivalue = 0;}
  else        *ivalue = atoi(value);
  return 0; 
} 

/*@C
   OptionsGetDouble - Gets the double precision value for a particular 
   option in the database.

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL

   Output Parameter:
.  dvalue - the double value to return
.  flg - 1 if found, 0 if not found

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetDouble(char* pre,char *name,double *dvalue,int *flg)
{
  char *value;
  int  ierr;
  ierr = OptionsFindPair_Private(pre,name,&value,flg); CHKERRQ(ierr);
  if (!*flg) return 0;
  if (!value) {*flg = 0; *dvalue = 0.0;}
  else *dvalue = atof(value);
  return 0; 
} 

/*@C
   OptionsGetScalar - Gets the scalar value for a particular 
   option in the database. At the moment can get only a Scalar with 
   0 imaginary part.

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL

   Output Parameter:
.  dvalue - the double value to return
.  flg - 1 if found, else 0

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetScalar(char* pre,char *name,Scalar *dvalue,int *flg)
{
  char *value;
  int  ierr;
  
  ierr = OptionsFindPair_Private(pre,name,&value,flg); CHKERRQ(ierr);
  if (!*flg) return 0;
  if (!value) {*flg = 0; *dvalue = 0;}
  else        *dvalue = atof(value);
  return 0; 
} 

/*@C
   OptionsGetDoubleArray - Gets an array of double precision values for a 
   particular option in the database.  The values must be separated with 
   commas with no intervening spaces.

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL
.  nmax - maximum number of values to retrieve

   Output Parameters:
.  dvalue - the double value to return
.  nmax - actual number of values retreived
.  flg - 1 if found, else 0

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray()
@*/
int OptionsGetDoubleArray(char* pre,char *name,double *dvalue, int *nmax,int *flg)
{
  char *value,*cpy;
  int  n = 0,ierr,len;

  ierr = OptionsFindPair_Private(pre,name,&value,flg); CHKERRQ(ierr);
  if (!*flg)  {*nmax = 0; return 0;}
  if (!value) {*nmax = 0; return 0;}

  /* make a copy of the values, otherwise we destroy the old values */
  len = PetscStrlen(value) + 1; 
  cpy = (char *) PetscMalloc(len*sizeof(char));
  PetscStrcpy(cpy,value);
  value = cpy;

  value = PetscStrtok(value,",");
  while (n < *nmax) {
    if (!value) break;
    *dvalue++ = atof(value);
    value = PetscStrtok(0,",");
    n++;
  }
  *nmax = n;
  PetscFree(cpy);
  return 0; 
} 

/*@C
   OptionsGetIntArray - Gets an array of integer values for a particular 
   option in the database.  The values must be separated with commas with 
   no intervening spaces. 

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL
.  nmax - maximum number of values to retrieve

   Output Parameter:
.  dvalue - the integer values to return
.  nmax - actual number of values retreived
.  flg - 1 if found, else 0

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetDoubleArray()
@*/
int OptionsGetIntArray(char* pre,char *name,int *dvalue,int *nmax,int *flg)
{
  char *value,*cpy;
  int  n = 0,ierr,len;

  ierr = OptionsFindPair_Private(pre,name,&value,flg); CHKERRQ(ierr);
  if (!*flg) {*nmax = 0; return 0;}
  if (!value) {*nmax = 0; return 0;}

  /* make a copy of the values, otherwise we destroy the old values */
  len = PetscStrlen(value) + 1; 
  cpy = (char *) PetscMalloc(len*sizeof(char));
  PetscStrcpy(cpy,value);
  value = cpy;

  value = PetscStrtok(value,",");
  while (n < *nmax) {
    if (!value) break;
    *dvalue++ = atoi(value);
    value = PetscStrtok(0,",");
    n++;
  }
  *nmax = n;
  PetscFree(cpy);
  return 0; 
} 

/*@C
   OptionsGetString - Gets the string value for a particular option in
   the database.

   Input Parameters:
.  name - the option one is seeking
.  len - maximum string length
.  pre - string to prepend to name or PETSC_NULL

   Output Parameter:
.  string - location to copy string
.  flg - 1 if found, else 0

.keywords: options, database, get, string

.seealso: OptionsGetInt(), OptionsGetDouble(),  
           OptionsHasName(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetString(char *pre,char *name,char *string,int len, int *flg)
{
  char *value;
  int  ierr;

  ierr = OptionsFindPair_Private(pre,name,&value,flg); CHKERRQ(ierr); 
  if (!*flg) {return 0;}
  if (value) PetscStrncpy(string,value,len);
  else PetscMemzero(string,len);
  return 0; 
}

/*@C
   OptionsAllUsed - Returns a count of the number of options in the 
   database that have never been selected.

   Options Database Key:
$  -optionsleft : checked within PetscFinalize()

.keywords: options, database, missed, unused, all, used

.seealso: OptionsPrint()
@*/
int OptionsAllUsed()
{
  int  i,n = 0;
  for ( i=0; i<options->N; i++ ) {
    if (!options->used[i]) { n++; }
  }
  return n;
}
