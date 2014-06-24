#include <petsc-private/threadcommimpl.h>      /*I "petscthreadcomm.h" I*/
#include <petscviewer.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

PetscMPIInt     Petsc_ThreadComm_keyval = MPI_KEYVAL_INVALID;

/* Logging support */
PetscLogEvent ThreadComm_RunKernel, ThreadComm_Barrier;

static PetscErrorCode PetscThreadCommRunKernel0_Private(PetscThreadComm tcomm,PetscErrorCode (*func)(PetscInt,...));

#undef __FUNCT__
#define __FUNCT__ "PetscCommGetThreadComm"
/*@C
  PetscCommGetThreadComm - Gets the thread communicator
                           associated with the MPI communicator

  Not Collective

  Input Parameters:
. comm - the MPI communicator

  Output Parameters:
. tcommp - pointer to the thread communicator

  Notes: If no thread communicator is on the MPI_Comm then the global thread communicator
         is returned.
  Level: Intermediate

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy()
@*/
PetscErrorCode PetscCommGetThreadComm(MPI_Comm comm,PetscThreadComm *tcomm)
{
  PetscErrorCode  ierr;
  PetscMPIInt     flg;
  void            *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,(PetscThreadComm*)&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    *tcomm = PETSC_NULL;
  } else *tcomm = (PetscThreadComm)ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAlloc"
/*
   PetscThreadCommAlloc - Allocates a thread communicator object

   Not Collective

   Output Parameters:
.  tcomm - pointer to the thread communicator object

   Level: developer

.seealso: PetscThreadCommDestroy()
*/
PetscErrorCode PetscThreadCommAlloc(PetscThreadComm *tcomm)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcommout;

  PetscFunctionBegin;
  PetscValidPointer(tcomm,2);

  *tcomm = NULL;
  ierr                      = PetscNew(&tcommout);CHKERRQ(ierr);

  tcommout->refct           = 0;
  tcommout->leader          = 0;
  tcommout->isnothread      = PETSC_TRUE;
  tcommout->thread_start    = 0;
  tcommout->red             = NULL;
  tcommout->active          = PETSC_FALSE;

  tcommout->ncommthreads    = -1;
  tcommout->nthreads        = 0;
  tcommout->commthreads     = NULL;

  tcommout->barrier_threads = 0;
  tcommout->wait1           = PETSC_TRUE;
  tcommout->wait2           = PETSC_TRUE;

  *tcomm                    = tcommout;

  PetscFunctionReturn(0);
}

/* #if defined(PETSC_USE_DEBUG) */

/* static PetscErrorCode PetscThreadCommStackCreate_kernel(PetscInt trank) */
/* { */
/*   if (trank && !PetscStackActive()) { */
/*     PetscStack *petscstack_in; */
/*     petscstack_in = (PetscStack*)malloc(sizeof(PetscStack)); */
/*     petscstack_in->currentsize = 0; */
/*     PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,petscstack_in); */
/*   } */
/*   return 0; */
/* } */

/* /\* Creates stack frames for threads other than the main thread *\/ */
/* #undef __FUNCT__ */
/* #define __FUNCT__ "PetscThreadCommStackCreate" */
/* PetscErrorCode  PetscThreadCommStackCreate(void) */
/* { */
/*   PetscErrorCode ierr; */
/*   ierr = PetscThreadCommRunKernel0(PETSC_COMM_SELF,(PetscThreadKernel)PetscThreadCommStackCreate_kernel);CHKERRQ(ierr); */
/*   ierr = PetscThreadCommBarrier(PETSC_COMM_SELF);CHKERRQ(ierr); */
/*   return 0; */
/* } */

/* static PetscErrorCode PetscThreadCommStackDestroy_kernel(PetscInt trank) */
/* { */
/*   if (trank && PetscStackActive()) { */
/*     PetscStack *petscstack_in; */
/*     petscstack_in = (PetscStack*)PetscThreadLocalGetValue(petscstack); */
/*     free(petscstack_in); */
/*     PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,(PetscStack*)0); */
/*   } */
/*   return 0; */
/* } */

/* #undef __FUNCT__ */
/* #define __FUNCT__ "PetscThreadCommStackDestroy" */
/* /\* Destroy stack frames for threads other than main thread */
/*  * */
/*  * The keyval may have been destroyed by the time this function is called, thus we must call */
/*  * PetscThreadCommRunKernel0_Private so that we never reference an MPI_Comm. */
/*  *\/ */
/* PetscErrorCode  PetscThreadCommStackDestroy(void) */
/* { */
/*   PetscErrorCode ierr; */
/*   PetscFunctionBegin; */
/*   ierr = PetscThreadCommRunKernel0_Private(PETSC_THREAD_COMM_WORLD,(PetscThreadKernel)PetscThreadCommStackDestroy_kernel);CHKERRQ(ierr); */
/*   PETSC_THREAD_COMM_WORLD = NULL; */
/*   PetscFunctionReturn(0); */
/*   return 0; */
/* } */
/* #else */
/* #undef __FUNCT__ */
/* #define __FUNCT__ "PetscThreadCommStackCreate" */
/* PetscErrorCode  PetscThreadCommStackCreate(void) */
/* { */
/*   PetscFunctionBegin; */
/*   PetscFunctionReturn(0); */
/* } */

/* #undef __FUNCT__ */
/* #define __FUNCT__ "PetscThreadCommStackDestroy" */
/* PetscErrorCode  PetscThreadCommStackDestroy(void) */
/* { */
/*   PetscFunctionBegin; */
/*   PETSC_THREAD_COMM_WORLD = NULL; */
/*   PetscFunctionReturn(0); */
/* } */

/* #endif */

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy"
/*
  PetscThreadCommDestroy - Frees a thread communicator object

  Not Collective

  Input Parameters:
. tcomm - the PetscThreadComm object

  Level: developer

.seealso: PetscThreadCommCreate()
*/
PetscErrorCode PetscThreadCommDestroy(PetscThreadComm *tcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("In ThreadCommDestroy refct=%d\n",(*tcomm)->refct);
  if (!*tcomm) PetscFunctionReturn(0);
  if (!--(*tcomm)->refct) {
    printf("Destroying ThreadComm\n");
    //if(pool->model==THREAD_MODEL_LOOP) {
    //ierr = PetscThreadCommStackDestroy();CHKERRQ(ierr);
    //}

    ierr = PetscThreadPoolDestroy(*tcomm);CHKERRQ(ierr);
    ierr = PetscThreadCommReductionDestroy((*tcomm)->red);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm));CHKERRQ(ierr);
  }
  *tcomm = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommView"
/*@C
   PetscThreadCommView - view a thread communicator

   Collective on comm

   Input Parameters:
+  comm - MPI communicator
-  viewer - viewer to display, for example PETSC_VIEWER_STDOUT_WORLD

   Level: developer

.seealso: PetscThreadCommCreate()
@*/
PetscErrorCode PetscThreadCommView(MPI_Comm comm,PetscViewer viewer)
{
  PetscThreadPool pool;
  PetscErrorCode  ierr;
  PetscBool       iascii;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Thread Communicator\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Number of threads = %D\n",tcomm->ncommthreads);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Type = %s\n",pool->type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    if (pool->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*pool->ops->view)(tcomm,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetNThreads"
/*@C
   PetscThreadCommGetNThreads - Gets the thread count from the thread communicator
                                associated with the MPI communicator

   Not collective

   Input Parameters:
.  comm - the MPI communicator

   Output Parameters:
.  nthreads - number of threads

   Level: developer

.seealso: PetscThreadCommSetNThreads()
@*/
PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm comm,PetscInt *nthreads)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr      = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  *nthreads = tcomm->ncommthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetAffinities"
/*@C
   PetscThreadCommGetAffinities - Returns the core affinities set for the
                                  thread communicator associated with the MPI_Comm

    Not collective

    Input Parameters:
.   comm - MPI communicator

    Output Parameters:
.   affinities - thread affinities

    Level: developer

    Notes:
    The user must allocate space (nthreads PetscInts) for the
    affinities. Must call PetscThreadCommSetAffinities before.

*/
PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm comm,PetscInt affinities[])
{
  PetscThreadPool pool;
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;
  PetscValidIntPointer(affinities,2);
  ierr = PetscMemcpy(affinities,pool->affinities,tcomm->ncommthreads*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier"
/*  PetscThreadCommBarrier - Apply a barrier on the thread communicator
                             associated with the MPI communicator

    Not collective

    Input Parameters:
.   comm - the MPI communicator

    Level: developer

    Notes:
    This routine provides an interface to put an explicit barrier between
    successive kernel calls to ensure that the first kernel is executed
    by all the threads before calling the next one.

    Called by the main thread only.

    May not be applicable to all types.
*/
PetscErrorCode PetscThreadCommBarrier(MPI_Comm comm)
{
  PetscThreadPool pool;
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;
  if (pool->ops->kernelbarrier) {
    ierr = (*pool->ops->kernelbarrier)(tcomm);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetScalars"
/*@C
   PetscThreadCommGetScalars - Gets pointers to locations for storing three PetscScalars that may be passed
                               to PetscThreadCommRunKernel to ensure that the scalar values remain valid
                               even after the main thread exits the calling function.

   Input Parameters:
+  comm - the MPI communicator having the thread communicator
.  val1 - pointer to store the first scalar value
.  val2 - pointer to store the second scalar value
-  val3 - pointer to store the third scalar value

   Level: developer

   Notes:
   This is a utility function to ensure that any scalars passed to PetscThreadCommRunKernel remain
   valid even after the main thread exits the calling function. If any scalars need to passed to
   PetscThreadCommRunKernel then these should be first stored in the locations provided by PetscThreadCommGetScalars()

   Pass NULL if any pointers are not needed.

   Called by the main thread only, not from within kernels

   Typical usage:

   PetscScalar *valptr;
   PetscThreadCommGetScalars(comm,&valptr,NULL,NULL);
   *valptr = alpha;   (alpha is the scalar you wish to pass in PetscThreadCommRunKernel)

   PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,valptr);

.seealso: PetscThreadCommRunKernel()
@*/
PetscErrorCode PetscThreadCommGetScalars(MPI_Comm comm,PetscScalar **val1, PetscScalar **val2, PetscScalar **val3)
{
  PetscErrorCode          ierr;
  PetscThreadComm         tcomm;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscInt                job_num, trank;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetRank(tcomm,&trank);CHKERRQ(ierr);
  pool = tcomm->pool;
  jobqueue = tcomm->pool->poolthreads[trank]->jobqueue;
  job_num = jobqueue->ctr%pool->nkernels;
  job     = &jobqueue->jobs[job_num];
  if (val1) *val1 = &job->scalars[0];
  if (val2) *val2 = &job->scalars[1];
  if (val3) *val3 = &job->scalars[2];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetInts"
/*@C
   PetscThreadCommGetInts - Gets pointers to locations for storing three PetscInts that may be passed
                               to PetscThreadCommRunKernel to ensure that the scalar values remain valid
                               even after the main thread exits the calling function.

   Input Parameters:
+  comm - the MPI communicator having the thread communicator
.  val1 - pointer to store the first integer value
.  val2 - pointer to store the second integer value
-  val3 - pointer to store the third integer value

   Level: developer

   Notes:
   This is a utility function to ensure that any scalars passed to PetscThreadCommRunKernel remain
   valid even after the main thread exits the calling function. If any scalars need to passed to
   PetscThreadCommRunKernel then these should be first stored in the locations provided by PetscThreadCommGetInts()

   Pass NULL if any pointers are not needed.

   Called by the main thread only, not from within kernels

   Typical usage:

   PetscScalar *valptr;
   PetscThreadCommGetScalars(comm,&valptr,NULL,NULL);
   *valptr = alpha;   (alpha is the scalar you wish to pass in PetscThreadCommRunKernel)

   PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,valptr);

.seealso: PetscThreadCommRunKernel()
@*/
PetscErrorCode PetscThreadCommGetInts(MPI_Comm comm,PetscInt **val1, PetscInt **val2, PetscInt **val3)
{
  PetscErrorCode          ierr;
  PetscThreadComm         tcomm;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscInt                job_num,trank;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr    = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetRank(tcomm,&trank);CHKERRQ(ierr);
  pool = tcomm->pool;
  jobqueue = tcomm->pool->poolthreads[trank]->jobqueue;
  job_num = jobqueue->ctr%pool->nkernels;
  job     = &jobqueue->jobs[job_num];
  if (val1) *val1 = &job->ints[0];
  if (val2) *val2 = &job->ints[1];
  if (val3) *val3 = &job->ints[2];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel"
/*@C
   PetscThreadCommRunKernel - Runs the kernel using the thread communicator
                              associated with the MPI communicator

   Not Collective

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  nargs - Number of input arguments for the kernel
-  ...   - variable list of input arguments

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,z);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x, PetscScalar* y, PetscReal* z)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),PetscInt nargs,...)
{
  PetscErrorCode          ierr;
  va_list                 argptr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool;

  PetscFunctionBegin;
  if (nargs > PETSC_KERNEL_NARGS_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Requested %D input arguments for kernel, max. limit %D",nargs,PETSC_KERNEL_NARGS_MAX);
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;

  // Loop over each thread in threadcomm
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    job  = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel
    job->tcomm          = tcomm;
    tcomm->commthreads[i]->job_ctr = jobqueue->ctr;
    job->nargs          = nargs;
    job->pfunc          = (PetscThreadKernel)func;
    va_start(argptr,nargs);
    for (i=0; i < nargs; i++) job->args[i] = va_arg(argptr,void*);
    va_end(argptr);
    job->job_status = THREAD_JOB_POSTED;
  }

  jobqueue = tcomm->commthreads[0]->jobqueue;
  job  = &jobqueue->jobs[jobqueue->ctr];

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
    jobqueue->kernel_ctr++;
  }

  // Run Kernel for main thread
  if (tcomm->isnothread) {
    ierr = PetscRunKernel(0,job->nargs,job);CHKERRQ(ierr);
    job->job_status = THREAD_JOB_COMPLETED;
  } else {
    ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel0_Private"
/* The zero-argument kernel needs to be callable with an unwrapped PetscThreadComm after ThreadComm keyval has been freed. */
static PetscErrorCode PetscThreadCommRunKernel0_Private(PetscThreadComm tcomm,PetscErrorCode (*func)(PetscInt,...))
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool = tcomm->pool;

  PetscFunctionBegin;
  if (tcomm->isnothread) {
    ierr = (*func)(0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Loop over each thread in threadcomm
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;

    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
    // Make sure previous job completed
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    // Prepare to run kernel
    job->tcomm          = tcomm;
    tcomm->commthreads[i]->job_ctr = jobqueue->ctr;
    job->nargs          = 1;
    job->pfunc          = (PetscThreadKernel)func;
    job->job_status = THREAD_JOB_POSTED;
  }

  jobqueue = tcomm->commthreads[0]->jobqueue;
  job = &jobqueue->jobs[jobqueue->ctr];

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
    jobqueue->kernel_ctr++;
  }

  // Run kernel for main thread
  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel0"
/*@C
   PetscThreadCommRunKernel0 - PetscThreadCommRunKernel version for kernels with no
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
-  func  - the kernel (needs to be cast to PetscThreadKernel)

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel0(comm,(PetscThreadKernel)kernel_func);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel0(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...))
{
  PetscErrorCode        ierr;
  PetscThreadComm       tcomm=0;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel0_Private(tcomm,func);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel1"
/*@C
   PetscThreadCommRunKernel1 - PetscThreadCommRunKernel version for kernels with 1
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
-  in1   - input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel1(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;

  if (tcomm->isnothread) {
    ierr = (*func)(0,in1);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Loop over each thread in threadcomm
  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;

    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    job->tcomm          = tcomm;
    tcomm->commthreads[i]->job_ctr = jobqueue->ctr;
    job->nargs          = 1;
    job->pfunc          = (PetscThreadKernel)func;
    job->args[0]        = in1;
    job->job_status = THREAD_JOB_POSTED;
  }

  jobqueue = tcomm->commthreads[0]->jobqueue;
  job  = &jobqueue->jobs[jobqueue->ctr];

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
    jobqueue->kernel_ctr++;
  }

  // Run kernel for main thread
  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel2"
/*@C
   PetscThreadCommRunKernel2 - PetscThreadCommRunKernel version for kernels with 2
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - 1st input argument for the kernel
-  in2   - 2nd input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt *x,PetscInt *y)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel2(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;

  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    job->tcomm          = tcomm;
    tcomm->commthreads[i]->job_ctr = jobqueue->ctr;
    job->nargs          = 2;
    job->pfunc          = (PetscThreadKernel)func;
    job->args[0]        = in1;
    job->args[1]        = in2;
    job->job_status = THREAD_JOB_POSTED;
  }

  // Get job for main thread
  jobqueue = tcomm->commthreads[0]->jobqueue;
  job  = &jobqueue->jobs[jobqueue->ctr];

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
    jobqueue->kernel_ctr++;
  }

  // Run kernel for main thread
  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel3"
/*@C
   PetscThreadCommRunKernel3 - PetscThreadCommRunKernel version for kernels with 3
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
-  in3   - third input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel3(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;

  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    job->tcomm          = tcomm;
    tcomm->commthreads[i]->job_ctr = jobqueue->ctr;
    job->nargs          = 3;
    job->pfunc          = (PetscThreadKernel)func;
    job->args[0]        = in1;
    job->args[1]        = in2;
    job->args[2]        = in3;
    job->job_status = THREAD_JOB_POSTED;
  }

  jobqueue = tcomm->commthreads[0]->jobqueue;
  job  = &jobqueue->jobs[jobqueue->ctr];

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
    jobqueue->kernel_ctr++;
  }

  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel4"
/*@C
   PetscThreadCommRunKernel4 - PetscThreadCommRunKernel version for kernels with 4
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
.  in3   - third input argument for the kernel
-  in4   - fourth input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel4(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3,void *in4)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;

  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3,in4);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    job->tcomm          = tcomm;
    tcomm->commthreads[i]->job_ctr = jobqueue->ctr;
    job->nargs          = 4;
    job->pfunc          = (PetscThreadKernel)func;
    job->args[0]        = in1;
    job->args[1]        = in2;
    job->args[2]        = in3;
    job->args[3]        = in4;
    job->job_status = THREAD_JOB_POSTED;
  }

  jobqueue = tcomm->commthreads[0]->jobqueue;
  job  = &jobqueue->jobs[jobqueue->ctr];

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
    jobqueue->kernel_ctr++;
  }

  // Run kernel for main thread
  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel6"
/*@C
   PetscThreadCommRunKernel6 - PetscThreadCommRunKernel version for kernels with 6
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
.  in3   - third input argument for the kernel
.  in4   - fourth input argument for the kernel
.  in5   - fifth input argument for the kernel
-  in6   - sixth input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel6(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3,void *in4,void *in5,void *in6)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  pool = tcomm->pool;

  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3,in4,in5,in6);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    if (!jobqueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
    job = &jobqueue->jobs[jobqueue->ctr]; /* Get the job context from the queue to launch this job */
    if (job->job_status != THREAD_JOB_NONE) {
      while (PetscReadOnce(int,job->job_status) != THREAD_JOB_COMPLETED) ;
    }

    job->tcomm          = tcomm;
    tcomm->commthreads[i]->job_ctr = jobqueue->ctr;
    job->nargs          = 6;
    job->pfunc          = (PetscThreadKernel)func;
    job->args[0]        = in1;
    job->args[1]        = in2;
    job->args[2]        = in3;
    job->args[3]        = in4;
    job->args[4]        = in5;
    job->args[5]        = in6;
    job->job_status = THREAD_JOB_POSTED;
  }

  jobqueue = tcomm->commthreads[0]->jobqueue;
  job  = &jobqueue->jobs[jobqueue->ctr];

  for(i=0; i<tcomm->ncommthreads; i++) {
    jobqueue = tcomm->commthreads[i]->jobqueue;
    jobqueue->ctr = (jobqueue->ctr+1)%pool->nkernels; /* Increment the queue ctr to point to the next available slot */
    jobqueue->kernel_ctr++;
  }

  // Run kernel for main thread
  ierr = (*pool->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Detaches the thread communicator from the MPI communicator if it exists
*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDetach"
PetscErrorCode PetscThreadCommDetach(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  PetscThreadComm tcomm;
  PetscCommGetThreadComm(comm,&tcomm);
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (flg) {
    printf("Detaching comm refct=%d\n",tcomm->refct);
    ierr = MPI_Attr_delete(comm,Petsc_ThreadComm_keyval);CHKERRQ(ierr);
    printf("After attr_delete refct=%d\n",tcomm->refct);
  }
  PetscFunctionReturn(0);
}

/*
   This routine attaches the thread communicator to the MPI communicator if it does not
   exist already.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAttach"
PetscErrorCode PetscThreadCommAttach(MPI_Comm comm,PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    tcomm->refct++;
    printf("Attaching comm refct=%d\n",tcomm->refct);
    ierr = MPI_Attr_put(comm,Petsc_ThreadComm_keyval,tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate"
/*
  PetscThreadCommInitialize - Initializes a thread communicator object

  PetscThreadCommInitialize() defaults to using the nonthreaded communicator.
*/
PetscErrorCode PetscThreadCommCreate(MPI_Comm comm,PetscInt nthreads,PetscBool createthreads,MPI_Comm *mpicomm)
{
  PetscInt i, *granks;
  PetscThreadComm tcomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Allocate space for ThreadComm
  ierr = PetscThreadCommAlloc(&tcomm);
  // Create ThreadPool
  ierr = PetscThreadPoolCreate(tcomm,nthreads,createthreads);
  // Set thread ranks
  PetscMalloc1(nthreads,&granks);
  for(i=0; i<nthreads; i++) {
    granks[i] = i;
  }
  // Initialize ThreadComm
  ierr = PetscThreadCommInitialize(nthreads,granks,tcomm);
  // Duplicate MPI_Comm
  ierr = PetscCommDuplicate(comm,mpicomm,PETSC_NULL);
  // Attach ThreadComm to new MPI_Comm
  ierr = PetscThreadCommAttach(*mpicomm,tcomm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreateShare"
/*
  PetscThreadCommInitialize - Initializes a thread communicator object

  PetscThreadCommInitialize() defaults to using the nonthreaded communicator.
*/
PetscErrorCode PetscThreadCommCreateShare(MPI_Comm comm,PetscInt nthreads,PetscInt *granks,MPI_Comm *mpicomm)
{
  PetscThreadComm tcomm,incomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Get input comm
  ierr = PetscCommGetThreadComm(comm,&incomm);
  // Allocate space for new ThreadComm
  ierr = PetscThreadCommAlloc(&tcomm);
  // Set new threadcomm to use input threadpool
  tcomm->pool = incomm->pool;
  // Initialize ThreadComm
  ierr = PetscThreadCommInitialize(nthreads,granks,tcomm);
  // Duplicate MPI_Comm
  ierr = PetscCommDuplicate(comm,mpicomm,PETSC_NULL);
  incomm->refct++;
  // Remove previous threadcomm
  ierr = PetscThreadCommDetach(*mpicomm);
  // Attach ThreadComm to new MPI_Comm
  ierr = PetscThreadCommAttach(*mpicomm,tcomm);
  PetscFunctionReturn(0);
}

/* #undef __FUNCT__ */
/* #define __FUNCT__ "PetscThreadCommCreateAttach" */
/* /\* */
/*   PetscThreadCommInitialize - Initializes a thread communicator object */

/*   PetscThreadCommInitialize() defaults to using the nonthreaded communicator. */
/* *\/ */
/* PetscErrorCode PetscThreadCommCreateAttach(MPI_Comm comm,PetscInt nthreads,MPI_Comm *mpicomm,PetscThreadComm *tcomm) */
/* { */
/*   PetscInt i, *granks; */
/*   PetscErrorCode ierr; */

/*   PetscFunctionBegin; */
/*   // Allocate space for ThreadComm */
/*   ierr = PetscThreadCommAlloc(tcomm); */
/*   // Create ThreadPool */
/*   ierr = PetscThreadPoolCreate(nthreads,&(*tcomm)->pool); */
/*   // Set thread ranks */
/*   PetscMalloc1(nthreads,&granks); */
/*   for(i=0; i<nthreads; i++) { */
/*     granks[i] = i; */
/*   } */
/*   // Initialize ThreadComm */
/*   ierr = PetscThreadCommInitialize(nthreads,granks,*tcomm); */
/*   // Duplicate MPI_Comm */
/*   ierr = PetscCommDuplicate(comm,mpicomm,PETSC_NULL); */
/*   // Attach ThreadComm to new MPI_Comm */
/*   ierr = PetscThreadCommAttach(*mpicomm,*tcomm); */
/*   PetscFunctionReturn(0); */
/* } */

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitialize"
/*
  PetscThreadCommInitialize - Initializes a thread communicator object

  PetscThreadCommInitialize() defaults to using the nonthreaded communicator.
*/
PetscErrorCode PetscThreadCommInitialize(PetscInt nthreads,PetscInt *granks,PetscThreadComm tcomm)
{
  PetscInt                i;
  PetscErrorCode          ierr;
  PetscThreadPool         pool;

  PetscFunctionBegin;
  pool = tcomm->pool;
  //ierr = PetscThreadCommCreate(&tcomm);CHKERRQ(ierr);

  if (pool->ismainworker) {
    tcomm->thread_start = 0;
    tcomm->ncommthreads = nthreads;
  } else {
    tcomm->thread_start = 1;
    tcomm->ncommthreads = nthreads-1;
  }
  ierr = PetscStrcmp(NOTHREAD,pool->type,&tcomm->isnothread);CHKERRQ(ierr);

  ierr = PetscMalloc1(tcomm->ncommthreads,&tcomm->commthreads);
  for(i=tcomm->thread_start; i<tcomm->ncommthreads; i++) {
    tcomm->commthreads[i] = pool->poolthreads[granks[i]];
  }

  /* Set the leader thread rank */
  if (pool->npoolthreads) {
    if (pool->ismainworker) tcomm->leader = granks[0];
    else tcomm->leader = granks[1];
  }

  ierr = PetscThreadCommReductionCreate(tcomm,&tcomm->red);CHKERRQ(ierr);
  //if(pool->model==THREAD_MODEL_LOOP) {
  //ierr = PetscThreadCommStackCreate();CHKERRQ(ierr);
  //}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetOwnershipRanges"
/*
   PetscThreadCommGetOwnershipRanges - Given the global size of an array, compute the local sizes
                                       and sets the starting array indices

   Input Parameters:
+  comm - the MPI communicator which holds the thread communicator
-  N    - the global size of the array

   Output Parameters:
.  trstarts - The starting array indices for each thread. the size of trstarts is nthreads+1

   Notes:
   trstarts is malloced in this routine
*/
PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm comm,PetscInt N,PetscInt *trstarts[])
{
  PetscErrorCode  ierr;
  PetscInt        Q,R;
  PetscBool       S;
  PetscThreadComm tcomm = NULL;
  PetscInt        *trstarts_out,nloc,i;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  ierr            = PetscMalloc1((tcomm->ncommthreads+1),&trstarts_out);CHKERRQ(ierr);
  trstarts_out[0] = 0;
  Q               = N/tcomm->ncommthreads;
  R               = N - Q*tcomm->ncommthreads;
  for (i=0; i<tcomm->ncommthreads; i++) {
    S                 = (PetscBool)(i < R);
    nloc              = S ? Q+1 : Q;
    trstarts_out[i+1] = trstarts_out[i] + nloc;
  }

  *trstarts = trstarts_out;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetRank"
/*
   PetscThreadCommGetRank - Gets the rank of the calling thread

   Input Parameters:
.  tcomm - the thread communicator

   Output Parameters:
.  trank - The rank of the calling thread

*/
PetscErrorCode PetscThreadCommGetRank(PetscThreadComm tcomm,PetscInt *trank)
{
  PetscThreadPool pool;
  PetscErrorCode ierr;
  PetscInt       rank = 0;

  PetscFunctionBegin;
  pool = tcomm->pool;
  if (pool->ops->getrank) {
    ierr = (*pool->ops->getrank)(&rank);CHKERRQ(ierr);
  }
  *trank = rank;
  PetscFunctionReturn(0);
}
