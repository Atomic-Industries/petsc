#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: plog.c,v 1.170 1997/09/26 02:21:43 bsmith Exp curfman $";
#endif
/*
      PETSc code to log object creation and destruction and PETSc events.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include "ts.h"      /* This include is to define all the PETSc cookies */
#include "ec.h"
#if defined(HAVE_MPE)
#include "mpe.h"
#endif
#include <stdio.h>
#include <stdarg.h>
#include <sys/types.h>
#include "sys.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
#include "pinclude/ptime.h"

/*
    The next two variables determine which, if any, PLogInfo() calls are used.
  If PlogPrintInfo is zero, no info messages are printed. 
  If PLogInfoFlags[OBJECT_COOKIE - PETSC_COOKIE] is zero, no messages related
  to that object are printed. OBJECT_COOKIE is, for example, MAT_COOKIE.
*/
int PLogPrintInfo = 0;
static int PLogInfoFlags[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              1,1,1,1,1,1,1,1,1,1,1,1};
#undef __FUNC__  
#define __FUNC__ "PLogInfoAllow"
/*@C
    PLogInfoAllow - Causes PLogInfo() messages to be printed to standard output.

    Input Parameter:
.   flag - PETSC_TRUE or PETSC_FALSE

    Options Database Key:
$    -log_info 

.keywords: allow, information, printing, monitoring

.seealso: PLogInfo()
@*/
int PLogInfoAllow(PetscTruth flag)
{
  PLogPrintInfo = (int) flag;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogInfoDeactivateClass"
/*@
    PLogInfoDeactivateClass - Deactivates PlogInfo() messages for a PETSc 
                              object class.

    Input Parameter:
.   objclass - for example, MAT_COOKIE, SNES_COOKIE, etc.

.seealso: PLogInfoActivateClass(), PLogInfo(), PLogInfoAllow()
@*/
int PLogInfoDeactivateClass(int objclass)
{
  PLogInfoFlags[objclass - PETSC_COOKIE - 1] = 0;
  if (objclass == SLES_COOKIE) {
    PLogInfoFlags[PC_COOKIE - PETSC_COOKIE - 1]  = 0;
    PLogInfoFlags[KSP_COOKIE - PETSC_COOKIE - 1] = 0;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogInfoActivateClass"
/*@
    PLogInfoActivateClass - Activates PlogInfo() messages for a PETSc 
                            object class.

    Input Parameter:
.   objclass - for example, MAT_COOKIE, SNES_COOKIE, etc.

.seealso: PLogInfoDeactivateClass(), PLogInfo(), PLogInfoAllow()
@*/
int PLogInfoActivateClass(int objclass)
{
  PLogInfoFlags[objclass - PETSC_COOKIE - 1] = 1;
  if (objclass == SLES_COOKIE) {
    PLogInfoFlags[PC_COOKIE - PETSC_COOKIE - 1]  = 1;
    PLogInfoFlags[KSP_COOKIE - PETSC_COOKIE - 1] = 1;
  }
  return 0;
}

/*
   If the option -log_history was used, then all printed PLogInfo() 
  messages are also printed to the history file, called by default
  .petschistory in ones home directory.
*/
extern FILE *petsc_history;

#undef __FUNC__  
#define __FUNC__ "PLogInfo"
/*@C
    PLogInfo - Logs informative data, which is printed to standard output
    when the option -log_info is specified.

    Input Parameter:
.   vobj - object most closely associated with the logging statement
.   message - logging message, using standard "printf" format

    Options Database Key:
$    -log_info : activates printing of PLogInfo() messages 

    Fortran Note:
    This routine is not supported in Fortran.

    Example of Usage:
$
$     Mat A
$     double alpha
$     PLogInfo(A,"Matrix uses parameter alpha=%g\n",alpha);
$

.keywords: information, printing, monitoring

.seealso: PLogInfoAllow()
@*/
int PLogInfo(void *vobj,char *message,...)
{
  va_list     Argp;
  int         rank,urank,len;
  PetscObject obj = (PetscObject) vobj;
  char        string[256];

  if (obj) PetscValidHeader(obj);
  if (!PLogPrintInfo) return 0;
  if (obj && !PLogInfoFlags[obj->cookie - PETSC_COOKIE - 1]) return 0;
  if (!obj) rank = 0;
  else      {MPI_Comm_rank(obj->comm,&rank);} 
  if (rank) return 0;

  MPI_Comm_rank(PETSC_COMM_WORLD,&urank);
  va_start( Argp, message );
  sprintf(string,"[%d]",urank); len = PetscStrlen(string);
#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7 && defined(PARCH_freebsd) )
  vsprintf(string+len,message,(char *)Argp);
#else
  vsprintf(string+len,message,Argp);
#endif
  fprintf(stdout,"%s",string);
  fflush(stdout);
  if (petsc_history) {
#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7 && defined(PARCH_freebsd) )
    vfprintf(petsc_history,message,(char *)Argp);
#else
    vfprintf(petsc_history,message,Argp);
#endif
  }
  va_end( Argp );
  return 0;
}

/* -------------------------------------------------------------------*/
#if defined(PETSC_LOG)
static int PLOG_USER_EVENT_LOW = PLOG_USER_EVENT_LOW_STATIC;

/* 
   Make sure that all events used by PETSc have the
   corresponding flags set here: 
     1 - activated for PETSc logging
     0 - not activated for PETSc logging
 */
int PLogEventFlags[] = {1,1,1,1,1,  /* 0 - 24*/
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,  /* 25 -49 */
                        1,1,1,1,1,
                        1,1,1,1,0,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 50 - 74 */
                        1,1,1,1,1,
                        1,1,1,1,0,
                        0,0,0,0,0,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 75 - 99 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,0,0,1,1, /* 100 - 124 */ 
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 125 - 149 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 150 - 174 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 175 - 199 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1};

static char *(oname[]) = {"Viewer           ",
                          "Index set        ",
                          "Vector           ",
                          "Vector Scatter   ",
                          "Matrix           ",
                          "Draw             ",
                          "Line graph       ",
                          "Krylov Solver    ",
                          "Preconditioner   ",
                          "SLES             ",  /* 10 */
                          "EC               ",  /* 11 */
                          "                 ",
                          "SNES             ",
                          "Distributed array",
                          "DF               ", /* 15 */
                          "Axis             ", /* 16 */
                          "Null Space       ",
                          "TS               ",
                          "Random           ",
                          "AO               ", /* 20 */
                          "DC               ",
                          "FD Coloring      ",
                          "Grid             ", /* 23 */
                          "                 ",
                          "                 ",
                          "                 ",
                          "Draw SP          ", /* 27 */                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
			  "                 "};

char *(PLogEventName[]) = {"MatMult         ",
                         "MatMatFreeMult  ",
                         "MatAssemblyBegin",
                         "MatAssemblyEnd  ",
                         "MatGetReordering",
                         "MatMultTrans    ",
                         "MatMultAdd      ",
                         "MatMultTransAdd ",
                         "MatLUFactor     ",
                         "MatCholeskyFctr ",
                         "MatLUFctrSymbol ",
                         "MatILUFctrSymbol",
                         "MatCholeskyFctr ",
                         "MatIncompleteCho",
                         "MatLUFactorNumer",
                         "MatCholeskyFact ",
                         "MatRelax        ",
                         "MatCopy         ",
                         "MatConvert      ",
                         "MatScale        ",
                         "MatZeroEntries  ",
                         "MatSolve        ",
                         "MatSolveAdd     ",
                         "MatSolveTrans   ",
                         "MatSolveTransAdd",
                         "MatSetValues    ",
                         "MatForwardSolve ",
                         "MatBackwardSolve",
                         "MatLoad         ",
                         "MatView         ",
                         "MatILUFactor    ",
                         "MatGetColoring  ",
                         "MatGetSubMatrice",
                         "MatGetValues    ",
                         "MatIncreaseOvlap",
                         "MatGetRow       ",
                         "                ",
                         "                ",
                         "                ",
                         "VecScatterBarrie",
                         "VecDot          ",
                         "VecNorm         ",
                         "VecMax          ",
                         "VecMin          ",
                         "VecTDot         ",
                         "VecScale        ",
                         "VecCopy         ",
                         "VecSet          ",
                         "VecAXPY         ",
                         "VecAYPX         ",
                         "VecSwap         ",
                         "VecWAXPY        ",
                         "VecAssemblyBegin",
                         "VecAssemblyEnd  ",
                         "VecMTDot        ",
                         "VecMDot         ",
                         "VecMAXPY        ",
                         "VecPointwiseMult",
                         "VecSetValues    ",
                         "VecLoad         ",
                         "VecView         ",
                         "VecScatterBegin ",
                         "VecScatterEnd   ",
                         "VecSetRandom    ",
                         "VecNormBarrier  ",
                         "VecNormComm     ",
                         "VecDotBarrier   ",
                         "VecDotComm      ",
                         "VecMDotBarrier  ",
                         "VecMDotComm     ",
                         "SLESSolve       ",
                         "SLESSetUp       ",
                         "KSPGMRESOrthog  ",
                         " ",
                         "PCModifySubMat  ",
                         "PCSetUp         ",
                         "PCSetUpOnBlocks ",
                         "PCApply         ",
                         "PCApplySymmLeft ",
                         "PCApplySymmRight",
                         "SNESSolve       ",
                         "SNESLineSearch  ",
                         "SNESFunctionEval",
                         "SNESJacobianEval",
                         "SNESMinFunctnEvl",
                         "SNESGradientEval",
                         "SNESHessianEval ",
                         " ",
                         " ",
                         " ",
                         "TSStep          ",
                         "TSPseudoCmptTStp",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         "PetscBarrier    ", /* 100 */
                         "                ",
                         "                ",
                         " ",
                         " ",
                         "ECSetUp         ",
                         "ECSolve         ",
                         " ",
                         " ",
                         " ",
                         "DFVecRefineVecto",
                         "DFVec_AssembleFu",
                         "DFVec_GetCompone",
                         "DFVec_DrawContou",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," "};

#define CHUNCK       1000
#define CREATE       0
#define DESTROY      1
#define ACTIONBEGIN  2
#define ACTIONEND    3

/*
    flops contains cumulative flops 
    mem contains current memory usage
    memmax contains maximum memory usage so far
*/
typedef struct {
  PLogDouble      time,flops,mem,maxmem;
  int             cookie,type,event,id1,id2,id3;
} Events;

typedef struct {
  int         parent;
  PLogDouble  mem;
  char        string[64];
  char        name[32];
  PetscObject obj;
} Objects;

/* 
    Global counters 
*/
PLogDouble _TotalFlops = 0.0;
PLogDouble irecv_ct = 0.0,isend_ct = 0.0,wait_ct = 0.0,wait_any_ct = 0.0;
PLogDouble irecv_len = 0.0,isend_len = 0.0,recv_len = 0.0, send_len = 0.0;
PLogDouble send_ct = 0.0,recv_ct = 0.0;
PLogDouble wait_all_ct = 0.0,allreduce_ct = 0.0,sum_of_waits_ct = 0.0;

/* used in the MPI_Allreduce() macro */
int PETSC_DUMMY;

/*
    Log counters in this file only 
*/
static PLogDouble  BaseTime;
static Events  *events = 0;
static Objects *objects = 0;

static int     nobjects = 0, nevents = 0, objectsspace = CHUNCK;
static int     ObjectsDestroyed = 0, eventsspace = CHUNCK;
/* make sure the 50 below is larger then any cookie - PETSC_COOKIE */
static PLogDouble  ObjectsType[10][50][4];

static int     EventsStage = 0;    /* which log sessions are we using */
static int     EventsStageMax = 0; /* highest event log used */ 
static int     EventsStagePushed = 0;
static int     EventsStageStack[100];
static char    *(EventsStageName[]) = {0,0,0,0,0,0,0,0,0,0};
static PLogDouble  EventsStageFlops[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static PLogDouble  EventsStageTime[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static PLogDouble  EventsStageMessageCounts[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static PLogDouble  EventsStageMessageLengths[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static PLogDouble  EventsStageReductions[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
#define COUNT      0
#define FLOPS      1
#define TIME       2
#define MESSAGES   3
#define LENGTHS    4
#define REDUCTIONS 5
static PLogDouble  EventsType[10][PLOG_USER_EVENT_HIGH][6];


#undef __FUNC__  
#define __FUNC__ "PLogStageRegister"
/*@C
    PLogStageRegister - Attaches a charactor string name to a logging stage.

    Input Parameters:
.   stage - the stage from 0 to 9 inclusive
.   sname - the name to associate with that stage

    Notes:
    The string information (for stage names) is not copied, so the user
    should NOT change any strings specified here.

.seealso: PLogStagePush(), PLogStagePop()
@*/
int PLogStageRegister(int stage, char *sname)
{
  if (stage < 0 || stage > 10) SETERRQ(1,0,"Out of range");
  EventsStageName[stage] = sname;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogStagePush"
/*@C
   PLogStagePush - Users can log up to 10 stages within a code by using
   -log_summary in conjunction with PLogStagePush() and PLogStagePop().

   Input Parameters:
.  stage - stage on which to log (0 <= stage <= 9)

   Example of Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 3 sets of summary data will be printed during
   PetscFinalize().
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     [stage 0 of code]   
$     for (i=0; i<ntimes; i++) {
$        PLogStagePush(1);
$        [stage 1 of code]
$        PLogStagePop()
$        PLogStagePush(2);
$        [stage 2 of code]
$        PLogStagePop()
$     }
$     PetscFinalize();
$

   Notes:  
   Use PLogStageRegister() to register a stage.

.keywords: log, push, stage

.seealso: PLogStagePop(), PLogStageRegister()
@*/
int PLogStagePush(int stage)
{
  if (stage < 0 || stage > 10) SETERRQ(1,0,"Out of range");
  /* record flops/time of previous stage */
  if (EventsStagePushed) {
    PetscTimeAdd(EventsStageTime[EventsStage]);
    EventsStageFlops[EventsStage]          += _TotalFlops;
    EventsStageMessageCounts[EventsStage]  += irecv_ct + isend_ct + recv_ct + send_ct;
    EventsStageMessageLengths[EventsStage] += irecv_len + isend_len + recv_len + send_len;
    EventsStageReductions[EventsStage]     += allreduce_ct;
  }
  EventsStageStack[EventsStagePushed] = EventsStage;
  if (EventsStagePushed++ > 99) SETERRQ(1,0,"Too many pushes");
  EventsStage = stage;
  if (stage > EventsStageMax) EventsStageMax = stage;
  PetscTimeSubtract(EventsStageTime[EventsStage]);
  EventsStageFlops[EventsStage]          -= _TotalFlops;
  EventsStageMessageCounts[EventsStage]  -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsStageMessageLengths[EventsStage] -= irecv_len + isend_len + recv_len + send_len;
  EventsStageReductions[EventsStage]     -= allreduce_ct;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogStagePop"
/*@C
   PLogStagePop - Users can log up to 10 stages within a code by using
   -log_summary in conjunction with PLogStagePush() and PLogStagePop().

   Example of Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 2 sets of summary data will be printed during
   PetscFinalize().
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     [stage 0 of code]   
$     PLogStagePush(1);
$     [stage 1 of code]
$     [some code (stage 1)]
$     PLogStagePop();
$     [more stage 0 of code]   
$     PetscFinalize();
$

   Notes:  
   Use PLogStageRegister() to register a stage.

.keywords: log, pop, stage

.seealso: PLogStagePush(), PLogStageRegister()
@*/
int PLogStagePop()
{
  PetscTimeAdd(EventsStageTime[EventsStage]);
  EventsStageFlops[EventsStage]          += _TotalFlops;
  EventsStageMessageCounts[EventsStage]  += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsStageMessageLengths[EventsStage] += irecv_len + isend_len + recv_len + send_len;
  EventsStageReductions[EventsStage]     += allreduce_ct;
  if (EventsStagePushed < 1) SETERRQ(1,0,"Too many pops\n");
  EventsStage = EventsStageStack[--EventsStagePushed];
  if (EventsStagePushed) {
    PetscTimeSubtract(EventsStageTime[EventsStage]);
    EventsStageFlops[EventsStage]          -= _TotalFlops;
    EventsStageMessageCounts[EventsStage]  -= irecv_ct + isend_ct + recv_ct + send_ct;
    EventsStageMessageLengths[EventsStage] -= irecv_len + isend_len + recv_len + send_len;
    EventsStageReductions[EventsStage]     -= allreduce_ct;
  }
  return 0;
}

/* --------------------------------------------------------------------------------*/

int (*_PLogPHC)(PetscObject) = 0;
int (*_PLogPHD)(PetscObject) = 0;
int (*_PLogPLB)(int,int,PetscObject,PetscObject,PetscObject,PetscObject) = 0;
int (*_PLogPLE)(int,int,PetscObject,PetscObject,PetscObject,PetscObject) = 0;

/*
      Default object create logger 
*/
#undef __FUNC__  
#define __FUNC__ "PLogDefaultPHC"
int PLogDefaultPHC(PetscObject obj)
{
  if (nevents >= eventsspace) {
    Events *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp = (Events *) malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  if (nobjects >= objectsspace) {
    Objects *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp = (Objects *) malloc((objectsspace+CHUNCK)*sizeof(Objects));CHKPTRQ(tmp);
    PetscMemcpy(tmp,objects,objectsspace*sizeof(Objects));
    free(objects);
    objects = tmp; objectsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  PetscTime(events[nevents].time); events[nevents].time -= BaseTime;
  events[nevents].cookie  = obj->cookie - PETSC_COOKIE - 1;
  events[nevents].type    = obj->type;
  events[nevents].id1     = nobjects;
  events[nevents].id2     = -1;
  events[nevents].id3     = -1;
  events[nevents].flops   = _TotalFlops;
  PetscTrSpace(&events[nevents].mem,PETSC_NULL,&events[nevents].maxmem);
  events[nevents++].event = CREATE;
  objects[nobjects].parent= -1;
  objects[nobjects].obj   = obj;
  PetscMemzero(objects[nobjects].string,64*sizeof(char));
  PetscMemzero(objects[nobjects].name,16*sizeof(char));
  obj->id = nobjects++;
  ObjectsType[EventsStage][obj->cookie - PETSC_COOKIE-1][0]++;
  return 0;
}
/*
      Default object destroy logger 
*/
#undef __FUNC__  
#define __FUNC__ "PLogDefaultPHD"
int PLogDefaultPHD(PetscObject obj)
{
  PetscObject parent;
  if (nevents >= eventsspace) {
    Events *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp = (Events *) malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  PetscTime(events[nevents].time); events[nevents].time -= BaseTime;
  events[nevents].event     = DESTROY;
  events[nevents].cookie    = obj->cookie - PETSC_COOKIE - 1;
  events[nevents].type      = obj->type;
  events[nevents].id1       = obj->id;
  events[nevents].id2       = -1;
  events[nevents].flops   = _TotalFlops;
  PetscTrSpace(&events[nevents].mem,PETSC_NULL,&events[nevents].maxmem);
  events[nevents++].id3     = -1;
  if (obj->parent) {objects[obj->id].parent   = obj->parent->id;}
  else {objects[obj->id].parent   = -1;}
  if (obj->name) { PetscStrncpy(objects[obj->id].name,obj->name,16);}
  objects[obj->id].obj      = 0;
  objects[obj->id].mem      = obj->mem;
  ObjectsType[EventsStage][obj->cookie - PETSC_COOKIE-1][1]++;
  ObjectsType[EventsStage][obj->cookie - PETSC_COOKIE-1][2] += obj->mem;
  /*
     Credit all ancestors with your memory 
  */
  parent = obj->parent;
  while (parent) {
    int exists;
    PetscObjectExists(parent,&exists);
    if (!exists) break;
    ObjectsType[EventsStage][parent->cookie - PETSC_COOKIE-1][3] += obj->mem;   
    parent = parent->parent;
  } 
  ObjectsDestroyed++;
  return 0;
}
/*
    Event begin logger with complete logging
*/
#undef __FUNC__  
#define __FUNC__ "PLogDefaultPLBAll"
int PLogDefaultPLBAll(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
 PLogDouble ltime;
 if (nevents >= eventsspace) {
    Events *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp = (Events *) malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  PetscTime(ltime);
  events[nevents].time = ltime - BaseTime;
  if (o1) events[nevents].id1     = o1->id; else events[nevents].id1 = -1;
  if (o2) events[nevents].id2     = o2->id; else events[nevents].id2 = -1;
  if (o3) events[nevents].id3     = o3->id; else events[nevents].id3 = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents].flops   = _TotalFlops;
  PetscTrSpace(&events[nevents].mem,PETSC_NULL,&events[nevents].maxmem);
  events[nevents++].event= ACTIONBEGIN;
  EventsType[EventsStage][event][COUNT]++;
  EventsType[EventsStage][event][TIME]        -= ltime;
  EventsType[EventsStage][event][FLOPS]       -= _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     -= irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  -= allreduce_ct;
  return 0;
}
/*
     Event end logger with complete logging
*/
#undef __FUNC__  
#define __FUNC__ "PLogDefaultPLEAll"
int PLogDefaultPLEAll(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
 PLogDouble ltime;
 if (nevents >= eventsspace) {
    Events *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp = (Events *) malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  PetscTime(ltime);
  events[nevents].time   = ltime - BaseTime;
  if (o1) events[nevents].id1    = o1->id; else events[nevents].id1 = -1;
  if (o2) events[nevents].id2    = o2->id; else events[nevents].id2 = -1;
  if (o3) events[nevents].id3    = o3->id; else events[nevents].id3 = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents].flops   = _TotalFlops;
  PetscTrSpace(&events[nevents].mem,PETSC_NULL,&events[nevents].maxmem);
  events[nevents++].event= ACTIONEND;
  EventsType[EventsStage][event][TIME]        += ltime;
  EventsType[EventsStage][event][FLOPS]       += _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     += irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  += allreduce_ct;
  return 0;
}
/*
     Default event begin logger
*/
#undef __FUNC__  
#define __FUNC__ "PLogDefaultPLB"
int PLogDefaultPLB(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  EventsType[EventsStage][event][COUNT]++;
  PetscTimeSubtract(EventsType[EventsStage][event][TIME]);
  EventsType[EventsStage][event][FLOPS]       -= _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     -= irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  -= allreduce_ct;
  return 0;
}

/*
     Default event end logger
*/
#undef __FUNC__  
#define __FUNC__ "PLogDefaultPLE"
int PLogDefaultPLE(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscTimeAdd(EventsType[EventsStage][event][TIME]);
  EventsType[EventsStage][event][FLOPS]       += _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     += irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  += allreduce_ct;
  return 0;
}

/*
     Default trace event logging routines
*/
FILE   *tracefile = 0;
int    tracelevel = 0;
char   *traceblanks = "                                                                    ";
char   tracespace[72];
PLogDouble tracetime = 0.0;

#undef __FUNC__  
#define __FUNC__ "PLogDefaultPLBTrace"
int PLogDefaultPLBTrace(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  int  rank;

  if (!tracetime) { tracetime = PetscGetTime(); }

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  PetscStrncpy(tracespace,traceblanks,2*tracelevel);
  tracespace[2*tracelevel] = 0;
  fprintf(tracefile,"%s[%d] %g Event begin: %s\n",tracespace,rank,PetscGetTime()-tracetime,PLogEventName[event]);
  fflush(tracefile);
  tracelevel++;

  return 0;
}

/*
     Default trace event logging
*/
#undef __FUNC__  
#define __FUNC__ "PLogDefaultPLETrace"
int PLogDefaultPLETrace(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  int rank;

  tracelevel--;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  PetscStrncpy(tracespace,traceblanks,2*tracelevel);
  tracespace[2*tracelevel] = 0;
  fprintf(tracefile,"%s[%d] %g Event end: %s\n",tracespace,rank,PetscGetTime()-tracetime,PLogEventName[event]);
  fflush(tracefile);
  return 0;
}

/* -------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PLogObjectState"
int PLogObjectState(PetscObject obj,char *format,...)
{
  va_list Argp;
  if (!objects) return 0;
  va_start( Argp, format );
#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7 && defined(PARCH_freebsd) )
  vsprintf(objects[obj->id].string,format,(char *)Argp);
#else
  vsprintf(objects[obj->id].string,format,Argp);
#endif
  va_end( Argp );
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogSet"
/*@C
    PLogSet - Sets the logging functions called at the beginning and ending 
              of every event.

   Input Parameters:
.    b - function called at beginning of event
.    e - function called at end of event

.seealso: PLogDump(), PLogBegin(), PLogAllBegin(), PLogTraceBegin()

@*/
int PLogSet(int (*b)(int,int,PetscObject,PetscObject,PetscObject,PetscObject),
            int (*e)(int,int,PetscObject,PetscObject,PetscObject,PetscObject))
{
  _PLogPLB    = b;
  _PLogPLE    = e;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogAllBegin"
/*@C
   PLogAllBegin - Turns on extensive logging of objects and events. Logs 
   all events. This creates large log files and slows the program down.

   Options Database Keys:
$  -log_all : Prints extensive log information (for code compiled
$      with PETSC_LOG)

   Notes:
   A related routine is PLogBegin (with the options key -log), which is 
   intended for production runs since it logs only flop rates and object
   creation (and shouldn't significantly slow the programs).

.keywords: log, all, begin

.seealso: PLogDump(), PLogBegin(), PLogTraceBegin()
@*/
int PLogAllBegin()
{
  int ierr;
  objects  = (Objects*) malloc(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events   = (Events*) malloc(CHUNCK*sizeof(Events));CHKPTRQ(events);
  _PLogPHC = PLogDefaultPHC;
  _PLogPHD = PLogDefaultPHD;
  ierr     = PLogSet(PLogDefaultPLBAll,PLogDefaultPLEAll); CHKERRQ(ierr);
  /* all processors sync here for more consistent logging */
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(BaseTime);
  PLogStagePush(0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogDestroy"
/*@C
   PLogDestroy - Destroys the object and event logging data and resets the 
   global counters. 

   Notes:
   This routine should not usually be used by programmers. Instead employ 
   PLogStagePush() and PLogStagePop().

.keywords: log, destroy

.seealso: PLogDump(), PLogAllBegin(), PLogPrintSummary(), PLogStagePush(), PlogStagePop()
@*/
int PLogDestroy()
{
  int ierr;

  if (objects) {free(objects); objects = 0;}
  if (events)  {free(events); events = 0;}
  ierr    = PLogSet(0,0); CHKERRQ(ierr);

  /* Resetting phase */
  PetscMemzero(EventsType,sizeof(EventsType));
  PetscMemzero(ObjectsType,sizeof(ObjectsType));
  _TotalFlops      = 0.0;
  nobjects         = 0;
  nevents          = 0;
  ObjectsDestroyed = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogBegin"
/*@C
    PLogBegin - Turns on logging of objects and events. This logs flop
    rates and object creation and should not slow programs down too much.
    This routine may be called more than once.

   Options Database Keys:
$  -log : Prints basic log information (for code compiled 
$      with PETSC_LOG)
$  -log_summary : Prints summary of flop and timing information 
$      to screen (for code compiled with PETSC_LOG)

.keywords: log, begin

.seealso: PLogDump(), PLogAllBegin(), PLogPrintSummary(), PLogTraceBegin()
@*/
int PLogBegin()
{
  int ierr;

  objects  = (Objects*) malloc(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events   = (Events*) malloc(CHUNCK*sizeof(Events));CHKPTRQ(events);
  _PLogPHC = PLogDefaultPHC;
  _PLogPHD = PLogDefaultPHD;
  ierr     = PLogSet(PLogDefaultPLB,PLogDefaultPLE); CHKERRQ(ierr);
  /* all processors sync here for more consistent logging */
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(BaseTime);
  PLogStagePush(0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogTraceBegin"
/*@
    PLogTraceBegin - Activates trace logging.  Every time a PETSc event
    begins or ends, the event name is printed.

    Input Parameter:
.   file - file to print trace in (e.g. stdout)

    Options Database Keys:
$   -log_trace [filename]:

    Notes:
    PLogTraceBegin() prints the processor number, the execution time (sec),
    then "Event begin:" or "Event end:" followed by the event name.

    PLogTraceBegin() allows tracing of all PETSc calls, which is useful
    to determine where a program is hanging without running in the 
    debugger.  Can be used in conjunction with the -log_info option. 

.seealso: PLogDump(), PLogAllBegin(), PLogPrintSummary(), PLogBegin()
@*/
int PLogTraceBegin(FILE *file)
{
  int ierr;

  ierr      = PLogSet(PLogDefaultPLBTrace,PLogDefaultPLETrace); CHKERRQ(ierr);
  tracefile = file;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogDump"
/*@C
   PLogDump - Dumps logs of objects to a file. This file is intended to 
   be read by petsc/bin/petscview.

   Input Parameter:
.  name - an optional file name

   Options Database Keys:
$  -log : Prints basic log information (for code compiled 
$      with PETSC_LOG)
$  -log_all : Prints extensive log information (for code compiled
$      with PETSC_LOG)
   
   Notes:
   The default file name is 
$      Log.<rank>
   where <rank> is the processor number. If no name is specified, 
   this file will be used.

.keywords: log, dump

.seealso: PLogBegin(), PLogPrintSummary()
@*/
int PLogDump(char* sname)
{
  int    i,rank;
  FILE   *fd;
  char   file[64];
  PLogDouble flops,_TotalTime;
  
  PetscTime(_TotalTime);
  _TotalTime -= BaseTime;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (sname) sprintf(file,"%s.%d",sname,rank);
  else  sprintf(file,"Log.%d",rank);
  fd = fopen(file,"w"); if (!fd) SETERRQ(1,0,"cannot open file");

  fprintf(fd,"Objects created %d Destroyed %d\n",nobjects,
                                                 ObjectsDestroyed);
  fprintf(fd,"Clock Resolution %g\n",0.0);
  fprintf(fd,"Events %d\n",nevents);
  for ( i=0; i<nevents; i++ ) {
    fprintf(fd,"%g %d %d %d %d %d %d %g %g %g\n",events[i].time,
                              events[i].event,
                              events[i].cookie,events[i].type,events[i].id1,
                              events[i].id2,events[i].id3,
                              events[i].flops,events[i].mem,
                              events[i].maxmem);
  }
  for ( i=0; i<nobjects; i++ ) {
    fprintf(fd,"%d %d\n",objects[i].parent,(int)objects[i].mem);
    if (!objects[i].string[0]) {fprintf(fd,"No Info\n");}
    else fprintf(fd,"%s\n",objects[i].string);
    if (!objects[i].name[0]) {fprintf(fd,"No Name\n");}
    else fprintf(fd,"%s\n",objects[i].name);
  }
  for ( i=0; i<PLOG_USER_EVENT_HIGH; i++ ) {
    flops = 0.0;
    if (EventsType[0][i][TIME]){flops = EventsType[0][i][FLOPS]/EventsType[0][i][TIME];}
    fprintf(fd,"%d %16g %16g %16g %16g\n",i,EventsType[0][i][COUNT],
                      EventsType[0][i][FLOPS],EventsType[0][i][TIME],flops);
  }
  fprintf(fd,"Total Flops %14e %16.8e\n",_TotalFlops,_TotalTime);
  fclose(fd);
  return 0;
}

extern char *PLogEventColor[];
extern int  PLogEventColorMalloced[];

#undef __FUNC__  
#define __FUNC__ "PLogEventRegister"
/*@C
    PLogEventRegister - Registers an event name for logging operations in 
    an application code. 

    Input Parameter:
.   string - name associated with the event
.   color - (optional) string specifying a color and display pattern
            for viewing an event, used by Upshot/Nupshot to view logs
            generated by -log_mpe (e.g., - "red:", "green:vlines3");
            use PETSC_NULL to let PETSc assign a color.
            
    Output Parameter:
.   e -  event id for use with PLogEventBegin() and PLogEventEnd().

    Notes: 
    PETSc automatically logs library events if the code has been
    compiled with -DPETSC_LOG (which is the default) and -log,
    -log_summary, or -log_all are specified.  PLogEventRegister() is
    intended for logging user events to supplement this PETSc
    information. 

    PETSc can gather data for use with the utilities Upshot/Nupshot
    (part of the MPICH distribution).  If PETSc has been compiled
    with flag -DHAVE_MPE (MPE is an additional utility within
    MPICH), the user can employ another command line option, -log_mpe,
    to create a logfile, "mpe.log", which can be visualized
    Upshot/Nupshot. The color argument is used by this utility
    in forming the display of this event; the standard X-windows
    color names should be used.

    Example of Usage:
$     #include "petsclog.h"
$     int USER_EVENT;
$     int user_event_flops;
$     PLogEventRegister(&USER_EVENT,"User event name","EventColor");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.keywords: log, event, register

.seealso: PLogEventBegin(), PLogEventEnd(), PLogFlops(),
          PLogEventMPEActivate(), PLogEventMPEDeactivate(),
          PLogEventActivate(), PLogEventDeactivate()
@*/
int PLogEventRegister(int *e,char *string,char *color)
{
  char *cstring;
  *e = PLOG_USER_EVENT_LOW++;
  if (*e > PLOG_USER_EVENT_HIGH) { 
    *e = 0;
    SETERRQ(1,0,"Out of event IDs");
  }
  cstring = (char *) PetscMalloc( PetscStrlen(string)+1 );CHKPTRQ(cstring);
  PetscStrcpy(cstring,string);
  PLogEventName[*e] = cstring;
#if defined(HAVE_MPE)
  if (UseMPE) {
    int   rank;
    char* ccolor;

    PLogEventMPEFlags[*e]       = 1;
    if (color != PETSC_NULL) {
      ccolor = (char *) PetscMalloc( PetscStrlen(color)+1 );CHKPTRQ(ccolor);
      PetscStrcpy(ccolor,color);
      PLogEventColor[*e]         = ccolor;
      PLogEventColorMalloced[*e] = 1;
    }
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    if (!rank) {
      MPE_Describe_state(MPEBEGIN+2*(*e),MPEBEGIN+2*(*e)+1,cstring,PLogEventColor[*e]);
    }
  }
#endif
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogEventRegisterDestroy_Private"
/*
   PLogEventRegisterDestroy_Private - Destroy the memory allocated during calls to 
        PLogEventRegister().

*/
int PLogEventRegisterDestroy_Private()
{
  int i;
  
  for (i=PLOG_USER_EVENT_LOW-1; i>=PLOG_USER_EVENT_LOW_STATIC; i--) {
    PetscFree(PLogEventName[i]);
#if defined(HAVE_MPE)
    if (PLogEventColorMalloced[i]) PetscFree(PLogEventColor[i]);
#endif
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogEventDeactivate"
/*@
    PLogEventDeactivate - Indicates that a particular event should not be
       logged. Note: the event may be either a pre-defined
       PETSc event (found in include/petsclog.h) or an event number obtained
       with PLogEventRegister().

  Input Parameter:
.   event - integer indicating event

   Example of Usage:
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     PLogEventDeactivate(VEC_SetValues);
$      code where you do not want to log VecSetValues() 
$     PLogEventActivate(VEC_SetValues);
$      code where you do want to log VecSetValues() 
$     .......
$     PetscFinalize();
$

.seealso: PLogEventMPEDeactivate(),PLogEventMPEActivate(),PlogEventActivate()
@*/
int PLogEventDeactivate(int event)
{
  PLogEventFlags[event] = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogEventActivate"
/*@
    PLogEventActivate - Indicates that a particular event should be
       logged. Note: the event may be either a pre-defined
       PETSc event (found in include/petsclog.h) or an event number obtained
       with PLogEventRegister().

  Input Parameter:
.   event - integer indicating event

   Example of Usage:
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     PLogEventDeactivate(VEC_SetValues);
$      code where you do not want to log VecSetValues() 
$     PLogEventActivate(VEC_SetValues);
$      code where you do want to log VecSetValues() 
$     .......
$     PetscFinalize();
$

.seealso: PLogEventMPEDeactivate(),PLogEventMPEActivate(),PlogEventDeactivate()
@*/
int PLogEventActivate(int event)
{
  PLogEventFlags[event] = 1;
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "PLogPrintSummary"
/*@C
   PLogPrintSummary - Prints a summary of the logging.

   Input Parameter:
.  file - an optional file name
.  comm - MPI communicator (one processor prints)

   Options Database Keys:
$  -log_summary : Prints summary of log information (for code
   compiled with PETSC_LOG)

   Notes:
   By defult the summary is printed to stdout.
   More extensive examination of the log information can be done with 
   PLogDump(), which is activated by the option -log or -log_all, in 
   combination with petsc/bin/petscview.
   
.keywords: log, dump, print

.seealso: PLogBegin(), PLogDump()
@*/
int PLogPrintSummary(MPI_Comm comm,char* filename)
{
  PLogDouble maxo,mino,aveo,mem,totmem,maxmem,minmem,mlensmcounts;
  PLogDouble maxf,minf,avef,totf,_TotalTime,maxt,mint,avet,tott,ratio;
  PLogDouble fmin,fmax,ftot,wdou,totts,totff,rat,sstime,sflops,ratf;
  PLogDouble ptotts,ptotff,ptotts_stime,ptotff_sflops,rat1,rat2,rat3;
  PLogDouble minm,maxm,avem,totm,minr,maxr,maxml,minml,totml,aveml,totr;
  PLogDouble rp,mp,lp,rpg,mpg,lpg,totms,totmls,totrs,mps,lps,rps,lpmp;
  PLogDouble pstime,psflops1,psflops,flopr,mict,mact,rct;
  int    size,rank,i,j;
  char   arch[10],hostname[64],username[16];
  FILE   *fd = stdout;

  /* pop off any stages the user forgot to remove */
  while (EventsStagePushed) PLogStagePop();


  PetscTime(_TotalTime);  _TotalTime -= BaseTime;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  /* Open the summary file */
  if (filename && !rank) {
    fd = fopen(filename,"w"); 
    if (!fd) SETERRQ(1,0,"cannot open file");
  }

  PetscFPrintf(comm,fd,"************************************************************************************************************************\n");
  PetscFPrintf(comm,fd,"***             WIDEN YOUR WINDOW TO 120 CHARACTERS.  Use 'enscript -r -fCourier9' to print this document            ***\n");
  PetscFPrintf(comm,fd,"************************************************************************************************************************\n");

  PetscFPrintf(comm,fd,"\n---------------------------------------------- PETSc Performance Summary: ----------------------------------------------\n\n");
  PetscGetArchType(arch,10);
  PetscGetHostName(hostname,64);
  PetscGetUserName(username,16);
  if (size == 1)
    PetscFPrintf(comm,fd,"%s on a %s named %s with %d processor, by %s %s",
                 OptionsGetProgramName(),arch,hostname,size,username,PetscGetDate());
  else
    PetscFPrintf(comm,fd,"%s on a %s named %s with %d processors, by %s %s",
                 OptionsGetProgramName(),arch,hostname,size,username, PetscGetDate());

  wdou = _TotalFlops; 
  MPI_Allreduce(&wdou,&minf,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxf,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&totf,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
  avef = (totf)/((PLogDouble) size);
  wdou = nobjects;
  MPI_Allreduce(&wdou,&mino,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxo,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&aveo,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
  aveo = (aveo)/((PLogDouble) size);
  wdou = _TotalTime;
  MPI_Allreduce(&wdou,&mint,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxt,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&tott,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
  avet = (tott)/((PLogDouble) size);

  PetscFPrintf(comm,fd,"\n                         Max       Max/Min      Avg      Total \n");
  if (mint) ratio = maxt/mint; else ratio = 0.0;
  PetscFPrintf(comm,fd,"Time (sec):           %5.3e   %10.5f   %5.3e\n",maxt,ratio,avet);
  if (mino) ratio = maxo/mino; else ratio = 0.0;
  PetscFPrintf(comm,fd,"Objects:              %5.3e   %10.5f   %5.3e\n",maxo,ratio,aveo);
  if (minf) ratio = maxf/minf; else ratio = 0.0;
  PetscFPrintf(comm,fd,"Flops:                %5.3e   %10.5f   %5.3e  %5.3e\n",maxf,ratio,avef,totf);

  if (mint) fmin = minf/mint; else fmin = 0;
  if (maxt) fmax = maxf/maxt; else fmax = 0;
  if (maxt) ftot = totf/maxt; else ftot = 0;
  if (fmin) ratio = fmax/fmin; else ratio = 0.0;
  PetscFPrintf(comm,fd,"Flops/sec:            %5.3e   %10.5f              %5.3e\n",fmax,ratio,ftot);
  PetscTrSpace(PETSC_NULL,PETSC_NULL,&mem);
  if (mem > 0.0) {
    MPI_Allreduce(&mem,&maxmem,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
    MPI_Allreduce(&mem,&minmem,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
    MPI_Allreduce(&mem,&totmem,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
    if (minmem) ratio = maxmem/minmem; else ratio = 0.0;
    PetscFPrintf(comm,fd,"Memory:               %5.3e   %8.3f              %5.3e\n",maxmem,ratio,totmem);
  }
  wdou = .5*(irecv_ct + isend_ct + recv_ct + send_ct);
  MPI_Allreduce(&wdou,&minm,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxm,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&totm,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
  avem = (totm)/((PLogDouble) size);
  wdou = .5*(irecv_len + isend_len + recv_len + send_len);
  MPI_Allreduce(&wdou,&minml,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxml,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&totml,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
  if (totm) aveml = (totml)/(totm); else aveml = 0;
  if (minm) ratio = maxm/minm; else ratio = 0.0;
  PetscFPrintf(comm,fd,"MPI Messages:         %5.3e   %8.3f   %5.3e  %5.3e\n",maxm,ratio,avem,totm);
  if (minml) ratio = maxml/minml; else ratio = 0.0;
  PetscFPrintf(comm,fd,"MPI Message Lengths:  %5.3e   %8.3f   %5.3e  %5.3e\n",maxml,ratio,aveml,totml);
  MPI_Allreduce(&allreduce_ct,&minr,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&allreduce_ct,&maxr,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&allreduce_ct,&totr,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
  if (minr) ratio = maxr/minr; else ratio = 0.0;
  PetscFPrintf(comm,fd,"MPI Reductions:       %5.3e   %8.3f\n",maxr,ratio);
  PetscFPrintf(comm,fd,"\nFlop counting convention: 1 flop = 1 real number operation of type (multiply/divide/add/subtract)\n");
  PetscFPrintf(comm,fd,"                            e.g., VecAXPY() for real vectors of length N --> 2N flops\n");
  PetscFPrintf(comm,fd,"                            and VecAXPY() for complex vectors of length N --> 8N flops\n");

  if (EventsStageMax) {
    PLogDouble mcounts,mlens,rcounts;

    PetscFPrintf(comm,fd,"\nSummary of Stages:  ---- Time ------     ----- Flops -------    -- Messages -- -- Message-lengths -- Reductions --\n");
    PetscFPrintf(comm,fd,"                      Avg      %%Total        Avg       %%Total   counts   %%Total    avg      %%Total   counts  %%Total \n");
    for ( j=0; j<=EventsStageMax; j++ ) {
      MPI_Allreduce(&EventsStageFlops[j],&sflops,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageTime[j],&sstime,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      if (tott > 0.) pstime = 100.0*sstime/tott; else pstime = 0.0;
      if (pstime >= 99.9 ) pstime = 99.9;
      if (totf > 0.) psflops = 100.*sflops/totf; else psflops = 0.0; 
      if (sstime > 0) psflops1 = (size*sflops)/sstime; else psflops1 = 0.0;

      MPI_Allreduce(&EventsStageMessageCounts[j],&mcounts,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageMessageLengths[j],&mlens,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageReductions[j],&rcounts,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      mcounts = .5*mcounts; mlens = .5*mlens; rcounts = rcounts/size;
      if (totm)  rat1 = 100.*mcounts/totm; else rat1 = 0.0; if (rat1 >= 99.9) rat1 = 99.9;
      if (totml) rat2 = 100.*mlens/totml; else rat2 = 0.0;  if (rat2 >= 99.9) rat2 = 99.9;
      if (totr)  rat3 = 100.*size*rcounts/totr; else rat3 = 0.0;if (rat2 >= 100.0) rat2 = 99.9;
      if (mcounts) mlensmcounts = mlens/mcounts; else mlensmcounts = 0.0; 
      if (EventsStageName[j]) {
        PetscFPrintf(comm,fd," %d: %15s: %6.4e    %4.1f%%     %6.4e      %4.1f%%  %5.3e   %4.1f%%  %5.3e  %4.1f%%  %5.3e  %4.1f%% \n",
                j,EventsStageName[j],sstime/size,pstime,psflops1,psflops,mcounts,rat1,mlensmcounts,rat2,
                rcounts,rat3);
      } else {
        PetscFPrintf(comm,fd," %d:                 %6.4e    %4.1f%%     %6.4e      %4.1f%%  %5.3e   %4.1f%%  %5.3e  %4.1f%%  %5.3e  %4.1f%% \n",
                j,sstime/size,pstime,psflops1,psflops,mcounts,rat1,mlensmcounts,rat2,rcounts,rat3);
      }
    }
  }


  PetscFPrintf(comm,fd,  
    "\n------------------------------------------------------------------------------------------------------------------------\n"); 
  PetscFPrintf(comm,fd,"See the 'Profiling' chapter of the users' manual for details on interpreting output.\n");
  PetscFPrintf(comm,fd,"Phase summary info:\n");
  PetscFPrintf(comm,fd,"   Count: number of times phase was executed\n");
  PetscFPrintf(comm,fd,"   Time and Flops/sec: Max - maximum over all processors\n");
  PetscFPrintf(comm,fd,"                       Ratio - ratio of maximum to minimum over all processors\n");
  PetscFPrintf(comm,fd,"   Mess: number of messages sent\n");
  PetscFPrintf(comm,fd,"   Avg. len: average message length\n");
  PetscFPrintf(comm,fd,"   Reduct: number of global reductions\n");
  PetscFPrintf(comm,fd,"   Global: entire computation\n");
  PetscFPrintf(comm,fd,"   Stage: optional user-defined stages of a computation. Set stages with PLogStagePush() and PLogStagePop().\n");
  PetscFPrintf(comm,fd,"      %%T - percent time in this phase         %%F - percent flops in this phase\n");
  PetscFPrintf(comm,fd,"      %%M - percent messages in this phase     %%L - percent message lengths in this phase\n");
  PetscFPrintf(comm,fd,"      %%R - percent reductions in this phase\n");
  PetscFPrintf(comm,fd,"   Total Mflop/s: 10e-6 * (sum of flops over all processors)/(max time over all processors)\n");
  PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n"); 

#if defined(PETSC_BOPT_g)
  PetscFPrintf(comm,fd,"\n\n");
  PetscFPrintf(comm,fd,"      ##########################################################\n");
  PetscFPrintf(comm,fd,"      #                                                        #\n");
  PetscFPrintf(comm,fd,"      #                          WARNING!!!                    #\n");
  PetscFPrintf(comm,fd,"      #                                                        #\n");
  PetscFPrintf(comm,fd,"      #   This code was compiled with a debugging option,      #\n");
  PetscFPrintf(comm,fd,"      #   BOPT=<g,g_c++,g_complex>.   To get timing results    #\n");
  PetscFPrintf(comm,fd,"      #   ALWAYS compile your code with an optimized version,  #\n");
  PetscFPrintf(comm,fd,"      #   BOPT=<O,O_c++,O_complex>;  the performance wil       #\n");
  PetscFPrintf(comm,fd,"      #   be generally two or three times faster.              #\n");
  PetscFPrintf(comm,fd,"      #                                                        #\n");
  PetscFPrintf(comm,fd,"      ##########################################################\n\n\n");
#endif
#if defined(PETSC_COMPLEX) && !defined(USE_FORTRAN_KERNELS)
  PetscFPrintf(comm,fd,"\n\n");
  PetscFPrintf(comm,fd,"      ##########################################################\n");
  PetscFPrintf(comm,fd,"      #                                                        #\n");
  PetscFPrintf(comm,fd,"      #                          WARNING!!!                    #\n");
  PetscFPrintf(comm,fd,"      #                                                        #\n");
  PetscFPrintf(comm,fd,"      #   The code for various complex numbers numerical       #\n");
  PetscFPrintf(comm,fd,"      #   kernels uses C++, which generally is not well        #\n");
  PetscFPrintf(comm,fd,"      #   optimized.  For faster performance (about 4-5 times  #\n");
  PetscFPrintf(comm,fd,"      #   faster), specify the flag -DUSE_FORTRAN_KERNELS in   #\n");
  PetscFPrintf(comm,fd,"      #   base.O_complex and recompile the PETSc libraries.    #\n");
  PetscFPrintf(comm,fd,"      #                                                        #\n");
  PetscFPrintf(comm,fd,"      ##########################################################\n\n\n");
#endif

  /* loop over operations looking for interesting ones */
  PetscFPrintf(comm,fd,"Phase                  Count      Time (sec)        Flops/sec \
                          --- Global ---  --- Stage ---   Total\n");
  PetscFPrintf(comm,fd,"                    Max  Ratio  Max     Ratio      Max     Ratio\
  Mess  Avg len  Reduct %%T %%F %%M %%L %%R  %%T %%F %%M %%L %%R Mflop/s\n");
  PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n"); 
  for ( j=0; j<=EventsStageMax; j++ ) {
    MPI_Allreduce(&EventsStageFlops[j],&sflops,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
    MPI_Allreduce(&EventsStageTime[j],&sstime,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
    if (EventsStageMax) {
      if (EventsStageName[j]) {
        PetscFPrintf(comm,fd,"\n--- Event Stage %d: %s\n\n",j,EventsStageName[j]);
      } else {
        PetscFPrintf(comm,fd,"\n--- Event Stage %d:\n\n",j);
      }
    }
    /* This loop assumes that PLOG_USER_EVENT_HIGH is the max event number */
    for ( i=0; i<PLOG_USER_EVENT_HIGH; i++ ) {  
      if (EventsType[j][i][TIME]) {
        wdou = EventsType[j][i][FLOPS]/EventsType[j][i][TIME];
      }
      else wdou = 0.0;
      MPI_Allreduce(&wdou,&minf,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
      MPI_Allreduce(&wdou,&maxf,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
      MPI_Allreduce(&EventsType[j][i][FLOPS],&totff,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsType[j][i][TIME],&mint,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
      MPI_Allreduce(&EventsType[j][i][TIME],&maxt,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);
      MPI_Allreduce(&EventsType[j][i][TIME],&totts,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsType[j][i][MESSAGES],&mp,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsType[j][i][LENGTHS],&lp,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsType[j][i][REDUCTIONS],&rp,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);

      MPI_Allreduce(&EventsType[j][i][COUNT],&mict,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
      MPI_Allreduce(&EventsType[j][i][COUNT],&mact,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);
     
      if (maxt) flopr = totff/maxt; else flopr = 0.0;
      MPI_Allreduce(&EventsStageMessageCounts[j],&totms,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageMessageLengths[j],&totmls,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageReductions[j],&totrs,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);

      if (EventsType[j][i][COUNT]) {
        if (mint > 0.0) rat = maxt/mint; else rat = 0.0;
        if (minf > 0.0) ratf = maxf/minf; else ratf = 0.0;
        if (tott > 0.0) ptotts = 100.*totts/tott; else ptotts = 0.0;
        if (ptotts >= 99. ) ptotts = 99.;
        if (totf > 0.0) ptotff = 100.*totff/totf; else ptotff = 0.0;
        if (ptotff >= 99. ) ptotff = 99.;
        if (sstime > 0.0) ptotts_stime = 100.*totts/sstime; else  ptotts_stime = 0.0;
        if (ptotts_stime >= 99. ) ptotts_stime = 99.;
        if (sflops > 0.0) ptotff_sflops = 100.*totff/sflops; else ptotff_sflops = 0.0;
        if (ptotff_sflops >= 99. ) ptotff_sflops = 99.;
        if (totm) mpg = 100.*mp/totm; else mpg = 0.0; if (mpg >= 99.) mpg = 99.;
        if (totml)  lpg = 100.*lp/totml; else lpg = 0.0; if (lpg >= 99.) lpg = 99.;
        if (totr)  rpg = 100.*rp/totr; else rpg = 0.0;if (rpg >= 99.) rpg = 99.;
        if (totms) mps = 100.*mp/totms; else mps = 0.0; if (mps >= 99.) mps = 99.;
        if (totmls)  lps = 100.*lp/totmls; else lps = 0.0; if (lps >= 99.) lps = 99.;
        if (totrs)  rps = 100.*rp/totrs; else rps = 0.0;if (rps >= 99.) rps = 99.;
        if (mp) lpmp = lp/mp; else lpmp = 0.0;
        if (mict) rct = mact/mict; else rct = 0.0;
        mp = mp/2.0;
        rp = rp/((PLogDouble) size);
        PetscFPrintf(comm,fd,"%s %7d %3.1f  %5.4e %5.1f  %3.2e %6.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f  %2.0f %2.0f %2.0f %2.0f %2.0f %5.0f\n",
                    PLogEventName[i],(int)mact,rct,maxt,rat,maxf,ratf,
                    mp,lpmp,rp,ptotts,ptotff,mpg,lpg,rpg,ptotts_stime,ptotff_sflops,mps,lps,rps,flopr/1.e6);
      }
    }
  }

  PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n"); 
  PetscFPrintf(comm,fd,"\n"); 
  PetscFPrintf(comm,fd,"Memory usage is given in bytes:\n\n");

  /* loop over objects looking for interesting ones */
  PetscFPrintf(comm,fd,"Object Type      Creations   Destructions   Memory  Descendants' Mem.\n");
  for ( j=0; j<=EventsStageMax; j++ ) {
    if (EventsStageMax) {
      if (EventsStageName[j]) {
        PetscFPrintf(comm,fd,"\n--- Event Stage %d: %s\n\n",j,EventsStageName[j]);
      } else {
        PetscFPrintf(comm,fd,"\n--- Event Stage %d:\n\n",j);
      }
    }
    for ( i=0; i<50; i++ ) {
      if (ObjectsType[j][i][0]) {
        PetscFPrintf(comm,fd,"%s %5d          %5d  %9d     %g\n",oname[i],(int) 
            ObjectsType[j][i][0],(int)ObjectsType[j][i][1],(int)ObjectsType[j][i][2],
            ObjectsType[j][i][3]);
      }
    }
  }
  PetscFPrintf(comm,fd,"\n");
  if (filename && !rank) fclose(fd);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscGetFlops"
/*@C
   PetscGetFlops - Returns the number of flops used on this processor 
   since the program began. 

   Output Parameter:
   returns the number of flops as a double.

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

.keywords: log, flops, floating point operations

.seealso: PetscGetTime(), PLogFlops()
@*/
PLogDouble PetscGetFlops()
{
  return _TotalFlops;
}

/* --------- Activate version -------------  */

#undef __FUNC__  
#define __FUNC__ "PLogEventActivateClass"
/*@
    PLogEventActivateClass - Activates event logging for a PETSc object
        class.

  Input Parameter:
.    cookie - for example MAT_COOKIE, SNES_COOKIE,

.seealso: PLogInfoActivate(),PLogInfo(),PLogInfoAllow(),PLogEventDeactivateClass(),
          PLogEventActivate(),PLogEventDeactivate()
@*/
int PLogEventActivateClass(int cookie)
{
  if (cookie == SNES_COOKIE) {
    PLogEventActivate(SNES_Solve);
    PLogEventActivate(SNES_LineSearch);
    PLogEventActivate(SNES_FunctionEval);
    PLogEventActivate(SNES_JacobianEval);
    PLogEventActivate(SNES_MinimizationFunctionEval);
    PLogEventActivate(SNES_GradientEval);
    PLogEventActivate(SNES_HessianEval);
  } else if (cookie == SLES_COOKIE || cookie == PC_COOKIE || cookie == KSP_COOKIE) {
    PLogEventActivate(SLES_Solve);
    PLogEventActivate(SLES_SetUp);
    PLogEventActivate(KSP_GMRESOrthogonalization);
    PLogEventActivate(PC_ModifySubMatrices);
    PLogEventActivate(PC_SetUp);
    PLogEventActivate(PC_SetUpOnBlocks);
    PLogEventActivate(PC_Apply);
    PLogEventActivate(PC_ApplySymmetricLeft);
    PLogEventActivate(PC_ApplySymmetricRight);
  } else if (cookie == MAT_COOKIE) {
    PLogEventActivate(MAT_Mult);
    PLogEventActivate(MAT_MatrixFreeMult);
    PLogEventActivate(MAT_AssemblyBegin);
    PLogEventActivate(MAT_AssemblyEnd);
    PLogEventActivate(MAT_GetReordering);
    PLogEventActivate(MAT_MultTrans);
    PLogEventActivate(MAT_MultAdd);
    PLogEventActivate(MAT_MultTransAdd);
    PLogEventActivate(MAT_LUFactor);
    PLogEventActivate(MAT_CholeskyFactor);
    PLogEventActivate(MAT_LUFactorSymbolic);
    PLogEventActivate(MAT_ILUFactorSymbolic);
    PLogEventActivate(MAT_CholeskyFactorSymbolic);
    PLogEventActivate(MAT_IncompleteCholeskyFactorSymbolic);
    PLogEventActivate(MAT_LUFactorNumeric);
    PLogEventActivate(MAT_CholeskyFactorNumeric);
    PLogEventActivate(MAT_CholeskyFactorNumeric);
    PLogEventActivate(MAT_Relax);
    PLogEventActivate(MAT_Copy);
    PLogEventActivate(MAT_Convert);
    PLogEventActivate(MAT_Scale);
    PLogEventActivate(MAT_ZeroEntries);
    PLogEventActivate(MAT_Solve);
    PLogEventActivate(MAT_SolveAdd);
    PLogEventActivate(MAT_SolveTrans);
    PLogEventActivate(MAT_SolveTransAdd);
    PLogEventActivate(MAT_SetValues);
    PLogEventActivate(MAT_ForwardSolve);
    PLogEventActivate(MAT_BackwardSolve);
    PLogEventActivate(MAT_Load);
    PLogEventActivate(MAT_View);
    PLogEventActivate(MAT_ILUFactor);

    PLogEventActivate(MAT_GetValues);
    PLogEventActivate(MAT_IncreaseOverlap);
    PLogEventActivate(MAT_GetRow);
  } else if (cookie == VEC_COOKIE) {
    PLogEventActivate(VEC_Dot);
    PLogEventActivate(VEC_Norm);
    PLogEventActivate(VEC_Max);
    PLogEventActivate(VEC_Min);
    PLogEventActivate(VEC_TDot);
    PLogEventActivate(VEC_Scale);
    PLogEventActivate(VEC_Copy);
    PLogEventActivate(VEC_Set);
    PLogEventActivate(VEC_AXPY);
    PLogEventActivate(VEC_AYPX);
    PLogEventActivate(VEC_Swap);
    PLogEventActivate(VEC_WAXPY);
    PLogEventActivate(VEC_AssemblyBegin);
    PLogEventActivate(VEC_AssemblyEnd);
    PLogEventActivate(VEC_MTDot);
    PLogEventActivate(VEC_MDot);
    PLogEventActivate(VEC_MAXPY);
    PLogEventActivate(VEC_PMult);
    PLogEventActivate(VEC_SetValues);
    PLogEventActivate(VEC_Load);
    PLogEventActivate(VEC_View);
    PLogEventActivate(VEC_ScatterBegin);
    PLogEventActivate(VEC_ScatterEnd);
    PLogEventActivate(VEC_SetRandom);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PLogEventDeactivateClass"
/*@
    PLogEventDeactivateClass - Deactivates event logging for a PETSc object
        class.

  Input Parameter:
.    cookie - for example MAT_COOKIE, SNES_COOKIE,

.seealso: PLogInfoActivate(),PLogInfo(),PLogInfoAllow(),PLogEventActivateClass(),
          PLogEventActivate(),PLogEventDeactivate()
@*/
int PLogEventDeactivateClass(int cookie)
{
  if (cookie == SNES_COOKIE) {
    PLogEventDeactivate(SNES_Solve);
    PLogEventDeactivate(SNES_LineSearch);
    PLogEventDeactivate(SNES_FunctionEval);
    PLogEventDeactivate(SNES_JacobianEval);
    PLogEventDeactivate(SNES_MinimizationFunctionEval);
    PLogEventDeactivate(SNES_GradientEval);
    PLogEventDeactivate(SNES_HessianEval);
  } else if (cookie == SLES_COOKIE || cookie == PC_COOKIE || cookie == KSP_COOKIE) {
    PLogEventDeactivate(SLES_Solve);
    PLogEventDeactivate(SLES_SetUp);
    PLogEventDeactivate(KSP_GMRESOrthogonalization);
    PLogEventDeactivate(PC_ModifySubMatrices);
    PLogEventDeactivate(PC_SetUp);
    PLogEventDeactivate(PC_SetUpOnBlocks);
    PLogEventDeactivate(PC_Apply);
    PLogEventDeactivate(PC_ApplySymmetricLeft);
    PLogEventDeactivate(PC_ApplySymmetricRight);
  } else if (cookie == MAT_COOKIE) {
    PLogEventDeactivate(MAT_Mult);
    PLogEventDeactivate(MAT_MatrixFreeMult);
    PLogEventDeactivate(MAT_AssemblyBegin);
    PLogEventDeactivate(MAT_AssemblyEnd);
    PLogEventDeactivate(MAT_GetReordering);
    PLogEventDeactivate(MAT_MultTrans);
    PLogEventDeactivate(MAT_MultAdd);
    PLogEventDeactivate(MAT_MultTransAdd);
    PLogEventDeactivate(MAT_LUFactor);
    PLogEventDeactivate(MAT_CholeskyFactor);
    PLogEventDeactivate(MAT_LUFactorSymbolic);
    PLogEventDeactivate(MAT_ILUFactorSymbolic);
    PLogEventDeactivate(MAT_CholeskyFactorSymbolic);
    PLogEventDeactivate(MAT_IncompleteCholeskyFactorSymbolic);
    PLogEventDeactivate(MAT_LUFactorNumeric);
    PLogEventDeactivate(MAT_CholeskyFactorNumeric);
    PLogEventDeactivate(MAT_CholeskyFactorNumeric);
    PLogEventDeactivate(MAT_Relax);
    PLogEventDeactivate(MAT_Copy);
    PLogEventDeactivate(MAT_Convert);
    PLogEventDeactivate(MAT_Scale);
    PLogEventDeactivate(MAT_ZeroEntries);
    PLogEventDeactivate(MAT_Solve);
    PLogEventDeactivate(MAT_SolveAdd);
    PLogEventDeactivate(MAT_SolveTrans);
    PLogEventDeactivate(MAT_SolveTransAdd);
    PLogEventDeactivate(MAT_SetValues);
    PLogEventDeactivate(MAT_ForwardSolve);
    PLogEventDeactivate(MAT_BackwardSolve);
    PLogEventDeactivate(MAT_Load);
    PLogEventDeactivate(MAT_View);
    PLogEventDeactivate(MAT_ILUFactor);

    PLogEventDeactivate(MAT_GetValues);
    PLogEventDeactivate(MAT_IncreaseOverlap);
    PLogEventDeactivate(MAT_GetRow);
  } else if (cookie == VEC_COOKIE) {
    PLogEventDeactivate(VEC_Dot);
    PLogEventDeactivate(VEC_Norm);
    PLogEventDeactivate(VEC_Max);
    PLogEventDeactivate(VEC_Min);
    PLogEventDeactivate(VEC_TDot);
    PLogEventDeactivate(VEC_Scale);
    PLogEventDeactivate(VEC_Copy);
    PLogEventDeactivate(VEC_Set);
    PLogEventDeactivate(VEC_AXPY);
    PLogEventDeactivate(VEC_AYPX);
    PLogEventDeactivate(VEC_Swap);
    PLogEventDeactivate(VEC_WAXPY);
    PLogEventDeactivate(VEC_AssemblyBegin);
    PLogEventDeactivate(VEC_AssemblyEnd);
    PLogEventDeactivate(VEC_MTDot);
    PLogEventDeactivate(VEC_MDot);
    PLogEventDeactivate(VEC_MAXPY);
    PLogEventDeactivate(VEC_PMult);
    PLogEventDeactivate(VEC_SetValues);
    PLogEventDeactivate(VEC_Load);
    PLogEventDeactivate(VEC_View);
    PLogEventDeactivate(VEC_ScatterBegin);
    PLogEventDeactivate(VEC_ScatterEnd);
    PLogEventDeactivate(VEC_SetRandom);
  }
  return 0;
}



/* end of -DPETSC_LOG section */
#else  /* -------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PLogObjectState"
int PLogObjectState(PetscObject obj,char *format,...)
{
  return 0;
}

#endif

#undef __FUNC__  
#define __FUNC__ "PetscGetTime"
/*@C
   PetscGetTime - Returns the current time of day in seconds. This 
     returns wall-clock time.  

   Output Parameter:
.  v - time counter

   Usage: 
$     PLogDouble v;
$     v = PetscGetTime();
$     .... perform some calculation ...
$     v = PetscGetTime() -v;

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscGetTime() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(),  PLogStagePush(), 
          PLogStagePop(), PLogStageRegister(), PetscGetFlops()

.keywords:  get, time
@*/
PLogDouble PetscGetTime()
{
  PLogDouble t;
  PetscTime(t);
  return t;
}
