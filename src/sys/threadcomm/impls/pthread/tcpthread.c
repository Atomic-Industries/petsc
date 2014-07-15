/* Define feature test macros to make sure CPU_SET and other functions are available
 */
#define PETSC_DESIRE_FEATURE_TEST_MACROS

#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#if defined PETSC_HAVE_MALLOC_H
#include <malloc.h>
#endif

#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank;
#else
pthread_key_t PetscPThreadRankkey;
#endif

PetscErrorCode PetscThreadCommGetRank_PThread(PetscInt *trank)
{
#if defined(PETSC_PTHREAD_LOCAL)
  *trank = PetscPThreadRank;
#else
  *trank = *((PetscInt*)pthread_getspecific(PetscPThreadRankkey));
#endif
  return 0;
}

/* Sets the attributes for threads */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetAffinity_PThread"
PetscErrorCode PetscThreadCommSetAffinity_PThread(PetscThreadPool pool,PetscThread thread)
{
  PetscErrorCode      ierr;
  PetscThread_PThread ptcomm;
  PetscBool           set;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  cpu_set_t           cpuset;
#endif

  PetscFunctionBegin;
  printf("in setaff_pthread\n");
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  /* Set affinity for workers */
  ptcomm = (PetscThread_PThread)thread->data;
  ierr = pthread_attr_init(&ptcomm->attr);CHKERRQ(ierr);
  PetscThreadPoolSetAffinity(pool,&cpuset,thread->affinity,&set);
  if(set) pthread_attr_setaffinity_np(&ptcomm->attr,sizeof(cpu_set_t),&cpuset);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommJoinPThreads"
PetscErrorCode PetscThreadCommJoinPThreads(PetscThreadPool pool)
{
  PetscErrorCode          ierr;
  void                    *jstatus;
  PetscThread_PThread ptcomm;
  PetscInt                i;

  PetscFunctionBegin;
  for (i=0; i<pool->npoolthreads; i++) {
    printf("Terminating thread=%d\n",i);
    pool->poolthreads[i]->status = THREAD_TERMINATE;
  }
  for (i=1; i<pool->npoolthreads; i++) {
    ptcomm = (PetscThread_PThread)pool->poolthreads[i]->data;
    ierr = pthread_join(ptcomm->tid,&jstatus);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy_PThread"
PetscErrorCode PetscThreadCommDestroy_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ptcomm) PetscFunctionReturn(0);
  /* Destroy pthread threadcomm data */
  if(tcomm->model==THREAD_MODEL_USER || tcomm->model==THREAD_MODEL_AUTO) {
    ierr = pthread_barrier_destroy(&ptcomm->barr);CHKERRQ(ierr);
    ierr = pthread_mutex_destroy(&ptcomm->threadmutex);CHKERRQ(ierr);
  }
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolDestroy_PThread"
PetscErrorCode PetscThreadPoolDestroy_PThread(PetscThreadPool pool)
{
  PetscThread_PThread pt;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* Terminate the thread pool */
  if(pool->model==THREAD_MODEL_LOOP || pool->model==THREAD_MODEL_AUTO) {
    ierr = PetscThreadCommJoinPThreads(pool);CHKERRQ(ierr);
  }
  /* Destroy pthread thread data */
  for(i=0; i<pool->npoolthreads; i++) {
    pt = (PetscThread_PThread)pool->poolthreads[i]->data;
    ierr = PetscFree(pt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCreate_PThread"
PETSC_EXTERN PetscErrorCode PetscThreadCreate_PThread(PetscThread thread)
{
  PetscThread_PThread ptcomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Create pthread\n");
  ierr = PetscNew(&ptcomm);
  thread->data = (void*)ptcomm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInit_PThread"
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_PThread(PetscThreadPool pool)
{
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("Init PThread\n");
  ierr = PetscStrcpy(pool->type,PTHREAD);CHKERRQ(ierr);
  pool->threadtype = THREAD_TYPE_PTHREAD;
  pool->ops->createthread = PetscThreadCreate_PThread;
  pool->ops->startthreads = PetscThreadCommInitialize_PThread;
  pool->ops->setaffinities = PetscThreadCommSetAffinity_PThread;
  pool->ops->pooldestroy = PetscThreadPoolDestroy_PThread;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThread"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("Creating PThread\n");
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);

  pthread_barrier_init(&ptcomm->barr,PETSC_NULL,tcomm->ncommthreads);
  pthread_mutex_init(&ptcomm->threadmutex,PETSC_NULL);

  tcomm->data             = (void*)ptcomm;
  tcomm->ops->commdestroy = PetscThreadCommDestroy_PThread;
  tcomm->ops->runkernel   = PetscThreadCommRunKernel_PThread;
  tcomm->ops->barrier     = PetscThreadCommBarrier_PThread;
  tcomm->ops->getrank     = PetscThreadCommGetRank_PThread;

  if (tcomm->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
    PetscPThreadRank=0; /* Main thread rank */
#else
    ierr = pthread_key_create(&PetscPThreadRankkey,PETSC_NULL);CHKERRQ(ierr);
    ierr = pthread_setspecific(PetscPThreadRankkey,&tcomm->commthreads[0]->grank);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_PThread"
PetscErrorCode PetscThreadCommRunKernel_PThread(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadCommJobQueue jobqueue;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("rank=%d running kernel\n",0);
  // Do work for main thread
  if (tcomm->ismainworker) {
    job->job_status   = THREAD_JOB_RECIEVED;
    tcomm->commthreads[0]->jobdata = job;
    PetscRunKernel(job->commrank,job->nargs, tcomm->commthreads[0]->jobdata);
    job->job_status   = THREAD_JOB_COMPLETED;
    jobqueue = tcomm->commthreads[tcomm->lleader]->jobqueue;
    jobqueue->current_job_index = (jobqueue->current_job_index+1)%tcomm->nkernels;
    jobqueue->completed_jobs_ctr++;
  }
  // Synchronize
  if (tcomm->syncafter) {
    ierr = PetscThreadCommJobBarrier(tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitialize_PThread"
PetscErrorCode PetscThreadCommInitialize_PThread(PetscThreadPool pool)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThread_PThread ptcomm;

  PetscFunctionBegin;
  /* Init thread structs */
  for (i=0; i<pool->npoolthreads; i++) {
    printf("Creating thread=%d\n",i);
    pool->poolthreads[i]->status = THREAD_CREATED;
    pool->poolthreads[i]->pool = pool;
  }

  /* Create threads */
  for (i=pool->thread_start; i<pool->npoolthreads; i++) {
    printf("Creating thread %d\n",i);
    ptcomm = (PetscThread_PThread)pool->poolthreads[i]->data;
    ierr = pthread_create(&ptcomm->tid,&ptcomm->attr,&PetscThreadPoolFunc,&pool->poolthreads[i]);CHKERRQ(ierr);
  }

  if (pool->ismainworker) pool->poolthreads[0]->status = THREAD_INITIALIZED;

  PetscInt threads_initialized=0;
  /* Wait till all threads have been initialized */
  while (threads_initialized != pool->npoolthreads) {
    threads_initialized=0;
    for (i=0; i<pool->npoolthreads; i++) {
      if (!pool->poolthreads[i]->status) break;
      threads_initialized++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_PThread"
PetscErrorCode PetscThreadCommBarrier_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  pthread_barrier_wait(&ptcomm->barr);
  ierr = PetscLogEventEnd(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
