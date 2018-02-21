
/*
    Defines parallel vector scatters.
*/

#include <../src/vec/vec/impls/dvecimpl.h>         /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscsf.h>

PetscErrorCode VecScatterView_MPI(VecScatter ctx,PetscViewer viewer)
{
  VecScatter_MPI_General *to  =(VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from=(VecScatter_MPI_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i;
  PetscMPIInt            rank;
  PetscViewerFormat      format;
  PetscBool              iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ctx),&rank);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format ==  PETSC_VIEWER_ASCII_INFO) {
      PetscInt nsend_max,nrecv_max,lensend_max,lenrecv_max,alldata,itmp;

      ierr = MPI_Reduce(&to->n,&nsend_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      ierr = MPI_Reduce(&from->n,&nrecv_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      itmp = to->starts[to->n+1];
      ierr = MPI_Reduce(&itmp,&lensend_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      itmp = from->starts[from->n+1];
      ierr = MPI_Reduce(&itmp,&lenrecv_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      ierr = MPI_Reduce(&itmp,&alldata,1,MPIU_INT,MPI_SUM,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"VecScatter statistics\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Blocksize %D\n",to->bs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number sends %D\n",nsend_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number receives %D\n",nrecv_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data sent %D\n",(int)(lensend_max*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data received %D\n",(int)(lenrecv_max*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Total data sent %D\n",(int)(alldata*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  VecScatter Blocksize %D\n",to->bs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number sends = %D; Number to self = %D\n",rank,to->n,to->local.n);CHKERRQ(ierr);
      if (to->n) {
        for (i=0; i<to->n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]   %D length = %D to whom %d\n",rank,i,to->starts[i+1]-to->starts[i],to->procs[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Now the indices for all remote sends (in order by process sent to)\n",rank);CHKERRQ(ierr);
        for (i=0; i<to->starts[to->n]; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D \n",rank,to->indices[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number receives = %D; Number from self = %D\n",rank,from->n,from->local.n);CHKERRQ(ierr);
      if (from->n) {
        for (i=0; i<from->n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D length %D from whom %d\n",rank,i,from->starts[i+1]-from->starts[i],from->procs[i]);CHKERRQ(ierr);
        }

        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Now the indices for all remote receives (in order by process received from)\n",rank);CHKERRQ(ierr);
        for (i=0; i<from->starts[from->n]; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D \n",rank,from->indices[i]);CHKERRQ(ierr);
        }
      }
      if (to->local.n) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Indices for local part of scatter\n",rank);CHKERRQ(ierr);
        for (i=0; i<to->local.n; i++) {  /* the to and from have the opposite meaning from what you would expect */
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] From %D to %D \n",rank,to->local.vslots[i],from->local.vslots[i]);CHKERRQ(ierr);
        }
      }

#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
      if (to->use_intranodeshmem) {
        for (i=0; i<to->shmn; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Via shared memory to local memory partner %d count %d\n",rank,to->shmprocs[i],to->shmstarts[i+1]-to->shmstarts[i]);CHKERRQ(ierr);
        }
      }
      if (from->use_intranodeshmem) {
        for (i=0; i<from->shmn; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Via shared memory from local memory partner %d count %d\n",rank,from->shmprocs[i],from->shmstarts[i+1]-from->shmstarts[i]);CHKERRQ(ierr);
        }
      }
#endif

      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"Method used to implement the VecScatter: ");CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
      if (to->use_alltoallw) {
        ierr = PetscViewerASCIIPrintf(viewer,"Uses MPI_alltoallw if INSERT_MODE\n");CHKERRQ(ierr);
      } else
#endif
      if (ctx->packtogether || to->use_alltoallv || to->use_window) {
        if (to->use_alltoallv) {
          ierr = PetscViewerASCIIPrintf(viewer,"Uses MPI MPI_alltoallv\n");CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
        } else if (to->use_window) {
          ierr = PetscViewerASCIIPrintf(viewer,"Uses MPI window\n");CHKERRQ(ierr);
#endif
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"Packs all messages and then sends them\n");CHKERRQ(ierr);
        }
      }  else {
        ierr = PetscViewerASCIIPrintf(viewer,"Packs and sends messages one at a time\n");CHKERRQ(ierr);
      }

#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
      if (to->use_intranodeshmem) {
        ierr = PetscViewerASCIIPrintf(viewer,"Uses MPI-3.0 process shared-memory for intra-node communication\n");CHKERRQ(ierr);
      }
#endif

    }
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/
/*
      The next routine determines what part of  the local part of the scatter is an
  exact copy of values into their current location. We check this here and
  then know that we need not perform that portion of the scatter when the vector is
  scattering to itself with INSERT_VALUES.

     This is currently not used but would speed up, for example DMLocalToLocalBegin/End()

*/
PetscErrorCode VecScatterLocalOptimize_Private(VecScatter scatter,VecScatter_Seq_General *to,VecScatter_Seq_General *from)
{
  PetscInt       n = to->n,n_nonmatching = 0,i,*to_slots = to->vslots,*from_slots = from->vslots;
  PetscErrorCode ierr;
  PetscInt       *nto_slots,*nfrom_slots,j = 0;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (to_slots[i] != from_slots[i]) n_nonmatching++;
  }

  if (!n_nonmatching) {
    to->nonmatching_computed = PETSC_TRUE;
    to->n_nonmatching        = from->n_nonmatching = 0;
    ierr = PetscInfo1(scatter,"Reduced %D to 0\n", n);CHKERRQ(ierr);
  } else if (n_nonmatching == n) {
    to->nonmatching_computed = PETSC_FALSE;
    ierr = PetscInfo(scatter,"All values non-matching\n");CHKERRQ(ierr);
  } else {
    to->nonmatching_computed= PETSC_TRUE;
    to->n_nonmatching       = from->n_nonmatching = n_nonmatching;

    ierr = PetscMalloc1(n_nonmatching,&nto_slots);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_nonmatching,&nfrom_slots);CHKERRQ(ierr);

    to->slots_nonmatching   = nto_slots;
    from->slots_nonmatching = nfrom_slots;
    for (i=0; i<n; i++) {
      if (to_slots[i] != from_slots[i]) {
        nto_slots[j]   = to_slots[i];
        nfrom_slots[j] = from_slots[i];
        j++;
      }
    }
    ierr = PetscInfo2(scatter,"Reduced %D to %D\n",n,n_nonmatching);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

/* -------------------------------------------------------------------------------------*/
PetscErrorCode VecScatterDestroy_PtoP(VecScatter ctx)
{
  VecScatter_MPI_General *to   = (VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from = (VecScatter_MPI_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i;

  PetscFunctionBegin;
  if (to->use_readyreceiver) {
    /*
       Since we have already posted sends we must cancel them before freeing
       the requests
    */
    for (i=0; i<from->n; i++) {
      ierr = MPI_Cancel(from->requests+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Cancel(to->rev_requests+i);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(from->n,from->requests,to->rstatus);CHKERRQ(ierr);
    ierr = MPI_Waitall(to->n,to->rev_requests,to->rstatus);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_MPI_ALLTOALLW) && !defined(PETSC_USE_64BIT_INDICES)
  if (to->use_alltoallw) {
    for (i=0; i<to->n; i++) {
      ierr = MPI_Type_free(to->types+to->procs[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree3(to->wcounts,to->wdispls,to->types);CHKERRQ(ierr);
    if (!from->contiq) {
      for (i=0; i<from->n; i++) {
        ierr = MPI_Type_free(from->types+from->procs[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree3(from->wcounts,from->wdispls,from->types);CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  if (to->use_window) {
    ierr = MPI_Win_free(&from->window);CHKERRQ(ierr);
    ierr = MPI_Win_free(&to->window);CHKERRQ(ierr);
    ierr = PetscFree(from->winstarts);CHKERRQ(ierr);
    ierr = PetscFree(to->winstarts);CHKERRQ(ierr);
  }
#endif

  if (to->use_alltoallv) {
    ierr = PetscFree2(to->counts,to->displs);CHKERRQ(ierr);
    ierr = PetscFree2(from->counts,from->displs);CHKERRQ(ierr);
  }

  /* release MPI resources obtained with MPI_Send_init() and MPI_Recv_init() */
  /*
     IBM's PE version of MPI has a bug where freeing these guys will screw up later
     message passing.
  */
#if !defined(PETSC_HAVE_BROKEN_REQUEST_FREE)
  if (!to->use_alltoallv && !to->use_window) {   /* currently the to->requests etc are ALWAYS allocated even if not used */
    if (to->requests) {
      for (i=0; i<to->n; i++) {
        ierr = MPI_Request_free(to->requests + i);CHKERRQ(ierr);
      }
    }
    if (to->rev_requests) {
      for (i=0; i<to->n; i++) {
        ierr = MPI_Request_free(to->rev_requests + i);CHKERRQ(ierr);
      }
    }
  }
  /*
      MPICH could not properly cancel requests thus with ready receiver mode we
    cannot free the requests. It may be fixed now, if not then put the following
    code inside a if (!to->use_readyreceiver) {
  */
  if (!to->use_alltoallv && !to->use_window) {    /* currently the from->requests etc are ALWAYS allocated even if not used */
    if (from->requests) {
      for (i=0; i<from->n; i++) {
        ierr = MPI_Request_free(from->requests + i);CHKERRQ(ierr);
      }
    }

    if (from->rev_requests) {
      for (i=0; i<from->n; i++) {
        ierr = MPI_Request_free(from->rev_requests + i);CHKERRQ(ierr);
      }
    }
  }
#endif

#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  if (to->use_intranodeshmem) {
    ierr = MPI_Win_free(&to->shmwin);CHKERRQ(ierr); /* free shmwin and the associated shared memory. from->shmwin = to->shmwin, so do not free the window twice */
    ierr = PetscFree4(to->shmprocs,to->shmflags,to->shmstarts,to->shmindices);CHKERRQ(ierr);
    ierr = PetscFree5(from->shmprocs,from->shmspaces,from->shmflags,from->shmstarts,from->shmindices);CHKERRQ(ierr);
  }
#endif

  ierr = PetscFree(to->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree(from->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree2(to->local.copy_starts,to->local.copy_lengths);CHKERRQ(ierr);
  ierr = PetscFree2(from->local.copy_starts,from->local.copy_lengths);CHKERRQ(ierr);
  ierr = PetscFree2(to->counts,to->displs);CHKERRQ(ierr);
  ierr = PetscFree2(from->counts,from->displs);CHKERRQ(ierr);
  ierr = PetscFree(to->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree(from->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree(to->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(from->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(to->requests);CHKERRQ(ierr);
  ierr = PetscFree(from->requests);CHKERRQ(ierr);
  ierr = PetscFree4(to->values,to->indices,to->starts,to->procs);CHKERRQ(ierr);
  ierr = PetscFree2(to->sstatus,to->rstatus);CHKERRQ(ierr);
  ierr = PetscFree4(from->values,from->indices,from->starts,from->procs);CHKERRQ(ierr);
  ierr = PetscFree(from);CHKERRQ(ierr);
  ierr = PetscFree(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
/*
    Special optimization to see if the local part of the scatter is actually
    made of copies. The scatter routines call PetscMemcpy() instead.

*/
PetscErrorCode VecScatterLocalOptimizeCopy_Private(VecScatter scatter,VecScatter_Seq_General *to,VecScatter_Seq_General *from,PetscInt bs)
{
  PetscInt       n = to->n,i,*to_slots = to->vslots,*from_slots = from->vslots;
  PetscInt       j,n_copies;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* count number of copies, which runs from 1 to n */
  n_copies = 1;
  for (i=0; i<n-1; i++) {
    if (to_slots[i]+bs != to_slots[i+1] || from_slots[i]+bs != from_slots[i+1]) n_copies++;
  }

  /* if average copy size >= 256 bytes, use memcpy instead of load/store */
  if (bs*n*sizeof(PetscScalar)/n_copies >= 256) {
    ierr = PetscMalloc2(n_copies,&to->copy_starts,n_copies,&to->copy_lengths);CHKERRQ(ierr);
    ierr = PetscMalloc2(n_copies,&from->copy_starts,n_copies,&from->copy_lengths);CHKERRQ(ierr);

    /* set up copy_starts[] & copy_lenghts[] of to and from */
    to->copy_starts[0]   = to_slots[0];
    from->copy_starts[0] = from_slots[0];

    if (n_copies != 1) { /* one copy is trival and we can save some work */
      j = 0;  /* j-th copy */
      for (i=0; i<n-1; i++) {
        if (to_slots[i]+bs != to_slots[i+1] || from_slots[i]+bs != from_slots[i+1]) {
          to->copy_lengths[j]    = sizeof(PetscScalar)*(to_slots[i]+bs-to->copy_starts[j]);
          from->copy_lengths[j]  = sizeof(PetscScalar)*(from_slots[i]+bs-from->copy_starts[j]);
          to->copy_starts[j+1]   = to_slots[i+1];
          from->copy_starts[j+1] = from_slots[i+1];
          j++;
        }
      }
    }

    /* set up copy_lengths[] of the last copy */
    to->copy_lengths[n_copies-1]   = sizeof(PetscScalar)*(to_slots[n-1]+bs-to->copy_starts[n_copies-1]);
    from->copy_lengths[n_copies-1] = sizeof(PetscScalar)*(from_slots[n-1]+bs-from->copy_starts[n_copies-1]);

    to->made_of_copies   = PETSC_TRUE;
    from->made_of_copies = PETSC_TRUE;
    to->n_copies         = n_copies;
    from->n_copies       = n_copies;
    ierr = PetscInfo1(scatter,"Local scatter is made of %D copies, optimizing for it\n", n_copies);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

PetscErrorCode VecScatterCopy_PtoP_X(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata,*out_to;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_from;
  PetscErrorCode         ierr;
  PetscInt               ny,bs = in_from->bs;

  PetscFunctionBegin;
  out->ops->begin   = in->ops->begin;
  out->ops->end     = in->ops->end;
  out->ops->copy    = in->ops->copy;
  out->ops->destroy = in->ops->destroy;
  out->ops->view    = in->ops->view;

  /* allocate entire send scatter context */
  ierr = PetscNewLog(out,&out_to);CHKERRQ(ierr);
  ierr = PetscNewLog(out,&out_from);CHKERRQ(ierr);

  ny                = in_to->starts[in_to->n];
  out_to->n         = in_to->n;
  out_to->type      = in_to->type;
  out_to->sendfirst = in_to->sendfirst;

  ierr = PetscMalloc1(out_to->n,&out_to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*ny,&out_to->values,ny,&out_to->indices,out_to->n+1,&out_to->starts,out_to->n,&out_to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(in_to->n,in_from->n),&out_to->sstatus,PetscMax(in_to->n,in_from->n),&out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);

  out->todata                        = (void*)out_to;
  out_to->local.n                    = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc1(in_to->local.n,&out_to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc1(in_from->local.n,&out_from->local.vslots);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->local.vslots,in_to->local.vslots,in_to->local.n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->local.vslots,in_from->local.vslots,in_from->local.n*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    out_to->local.vslots   = 0;
    out_from->local.vslots = 0;
  }

  /* allocate entire receive context */
  out_from->type      = in_from->type;
  ny                  = in_from->starts[in_from->n];
  out_from->n         = in_from->n;
  out_from->sendfirst = in_from->sendfirst;

  ierr = PetscMalloc1(out_from->n,&out_from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(ny*bs,&out_from->values,ny,&out_from->indices,out_from->n+1,&out_from->starts,out_from->n,&out_from->procs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);

  out->fromdata                        = (void*)out_from;
  out_from->local.n                    = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;

  /*
      set up the request arrays for use with isend_init() and irecv_init()
  */
  {
    PetscMPIInt tag;
    MPI_Comm    comm;
    PetscInt    *sstarts = out_to->starts,  *rstarts = out_from->starts;
    PetscMPIInt *sprocs  = out_to->procs,   *rprocs  = out_from->procs;
    PetscInt    i;
    PetscBool   flg;
    MPI_Request *swaits   = out_to->requests,*rwaits  = out_from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = out_to->values, *Srvalues = out_from->values;

    ierr = PetscMalloc1(in_to->n,&out_to->rev_requests);CHKERRQ(ierr);
    ierr = PetscMalloc1(in_from->n,&out_from->rev_requests);CHKERRQ(ierr);

    rev_rwaits = out_to->rev_requests;
    rev_swaits = out_from->rev_requests;

    out_from->bs = out_to->bs = bs;
    tag          = ((PetscObject)out)->tag;
    ierr         = PetscObjectGetComm((PetscObject)out,&comm);CHKERRQ(ierr);

    /* Register the receives that you will use later (sends for scatter reverse) */
    for (i=0; i<out_from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
      ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rev_swaits+i);CHKERRQ(ierr);
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_rsend",&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      out_to->use_readyreceiver   = PETSC_TRUE;
      out_from->use_readyreceiver = PETSC_TRUE;
      for (i=0; i<out_to->n; i++) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      }
      if (out_from->n) {ierr = MPI_Startall_irecv(out_from->starts[out_from->n]*out_from->bs,out_from->n,out_from->requests);CHKERRQ(ierr);}
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      ierr = PetscInfo(in,"Using VecScatter ready receiver mode\n");CHKERRQ(ierr);
    } else {
      out_to->use_readyreceiver   = PETSC_FALSE;
      out_from->use_readyreceiver = PETSC_FALSE;
      flg                         = PETSC_FALSE;
      ierr                        = PetscOptionsGetBool(NULL,NULL,"-vecscatter_ssend",&flg,NULL);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscInfo(in,"Using VecScatter Ssend mode\n");CHKERRQ(ierr);
      }
      for (i=0; i<out_to->n; i++) {
        if (!flg) {
          ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
        } else {
          ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
        }
      }
    }
    /* Register receives for scatter reverse */
    for (i=0; i<out_to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,rev_rwaits+i);CHKERRQ(ierr);
    }
  }

#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  if (in_to->use_intranodeshmem) {
    /* to better understand how we copy a vecscatter context with shared memory support, please see comments
       in VecScatterCreate_PtoS. The shared memory allocation, and flag address and data address calculation
       code is almost the same.
     */
    PetscInt i;
    MPI_Info info;
    /* since the to and from data structures are not symmetric for shared memory we insure they always listed in "standard" form */
    if (!in_to->shmspace) {
      VecScatter_MPI_General *tmp;
      tmp =  in_to;   in_to =  in_from;  in_from = tmp; /* swap  in_{to,from} */
      tmp = out_to;  out_to = out_from; out_from = tmp; /* swap out_{to,from} */
    }

    /* copy the to parts for intranode shared memory communication */
    out_to->shmn    = in_to->shmn;
    out_to->shmcomm = in_to->shmcomm; /* shmcomm is destroyed only when the outer PETSc communicator is freed, so a simple copy (other than MPI_Comm_dup) is fine */

    ierr = PetscMalloc4(out_to->shmn,&out_to->shmprocs,out_to->shmn,&out_to->shmflags,out_to->shmn+1,&out_to->shmstarts,in_to->shmstarts[in_to->shmn],&out_to->shmindices);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->shmprocs,in_to->shmprocs,sizeof(PetscMPIInt)*out_to->shmn);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->shmstarts,in_to->shmstarts,sizeof(PetscInt)*(out_to->shmn+1));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->shmindices,in_to->shmindices,sizeof(PetscInt)*in_to->shmstarts[in_to->shmn]);CHKERRQ(ierr);
    /* allocate sharem memory in to, which has shmspace[] and ignores shmspaces[] */
    ierr = MPI_Info_create(&info);CHKERRQ(ierr);
    ierr = MPI_Info_set(info, "alloc_shared_noncontig", "true");CHKERRQ(ierr);
    MPI_Aint sz = bs*out_to->shmstarts[out_to->shmn]*sizeof(PetscScalar) + (out_to->shmn+1)*PETSC_LEVEL1_DCACHE_LINESIZE;
    ierr = MPI_Win_allocate_shared(sz,sizeof(PetscScalar),info,out_to->shmcomm,&out_to->shmspace,&out_to->shmwin);CHKERRQ(ierr);
    ierr = MPI_Info_free(&info);CHKERRQ(ierr);
    out_to->shmspace = (PetscScalar*)((((PETSC_UINTPTR_T)(out_to->shmspace))+(PETSC_LEVEL1_DCACHE_LINESIZE-1)) & ~(PETSC_LEVEL1_DCACHE_LINESIZE-1));
    for (i=0; i<out_to->shmn; i++) {
      out_to->shmflags[i] = (PetscInt*)((char*)out_to->shmspace + i*PETSC_LEVEL1_DCACHE_LINESIZE);
      *out_to->shmflags[i] = 0; /* init the flag to empty(0) */
    }
    out_to->shmspace = (PetscScalar*)((char*)out_to->shmspace + out_to->shmn*PETSC_LEVEL1_DCACHE_LINESIZE);

    /* copy the from parts for intranode shared memory communication */
    out_from->shmn    = in_from->shmn;
    out_from->shmcomm = in_from->shmcomm;
    out_from->shmwin  = out_to->shmwin; /* out's shared memory window is new. We can not copy the in's */

    PetscInt nn = out_from->shmn;
    ierr = PetscMalloc5(nn,&out_from->shmprocs,nn,&out_from->shmspaces,nn,&out_from->shmflags,nn+1,&out_from->shmstarts,in_from->shmstarts[in_from->shmn],&out_from->shmindices);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->shmprocs,in_from->shmprocs,sizeof(PetscMPIInt)*out_from->shmn);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->shmstarts,in_from->shmstarts,sizeof(PetscInt)*(out_from->shmn+1));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->shmindices,in_from->shmindices,sizeof(PetscInt)*in_from->shmstarts[in_from->shmn]);CHKERRQ(ierr);

    MPI_Request *reqs = NULL;
    struct {PetscInt j,m,offset;} jmo,*triples = NULL;
    ierr = PetscMalloc2(out_from->shmn,&reqs,out_from->shmn,&triples);CHKERRQ(ierr);

    for (i=0; i<out_from->shmn; i++) { ierr = MPI_Irecv(triples+i,3,MPIU_INT,out_from->shmprocs[i],0/*tag*/,out_from->shmcomm,reqs+i);CHKERRQ(ierr); }
    for (i=0; i<out_to->shmn; i++) {
      jmo.j = i;
      jmo.m = out_to->shmn;
      jmo.offset = out_to->shmstarts[i];
      ierr = MPI_Send(&jmo,3,MPIU_INT,out_to->shmprocs[i],0/*tag*/,out_to->shmcomm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(out_from->shmn,reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);

    /* figure out flag addresses and data addresses aimed to me */
    for (i=0; i<out_from->shmn; i++) {
      MPI_Aint    size;
      PetscMPIInt disp_unit;
      ierr = MPI_Win_shared_query(out_from->shmwin,out_from->shmprocs[i],&size,&disp_unit,&out_from->shmspaces[i]);CHKERRQ(ierr);
      out_from->shmspaces[i]  = (PetscScalar*)((((PETSC_UINTPTR_T)(out_from->shmspaces[i]))+(PETSC_LEVEL1_DCACHE_LINESIZE-1)) & ~(PETSC_LEVEL1_DCACHE_LINESIZE-1));
      out_from->shmflags[i]   =    (PetscInt*)((char*)out_from->shmspaces[i] + triples[i].j*PETSC_LEVEL1_DCACHE_LINESIZE); /* get address of the j-th flag */
      out_from->shmspaces[i]  = (PetscScalar*)((char*)out_from->shmspaces[i] + triples[i].m*PETSC_LEVEL1_DCACHE_LINESIZE); /* skip the flag area */
      out_from->shmspaces[i] += triples[i].offset*bs;/* and then add the offset to point to where my expected data lives */
    }

    ierr = PetscFree2(reqs,triples);CHKERRQ(ierr);
  }
#endif

  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCopy_PtoP_AllToAll(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_to,*out_from;
  PetscErrorCode         ierr;
  PetscInt               ny,bs = in_from->bs;
  PetscMPIInt            size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)in),&size);CHKERRQ(ierr);

  out->ops->begin     = in->ops->begin;
  out->ops->end       = in->ops->end;
  out->ops->copy      = in->ops->copy;
  out->ops->destroy   = in->ops->destroy;
  out->ops->view      = in->ops->view;

  /* allocate entire send scatter context */
  ierr = PetscNewLog(out,&out_to);CHKERRQ(ierr);
  ierr = PetscNewLog(out,&out_from);CHKERRQ(ierr);

  ny                = in_to->starts[in_to->n];
  out_to->n         = in_to->n;
  out_to->type      = in_to->type;
  out_to->sendfirst = in_to->sendfirst;

  ierr = PetscMalloc1(out_to->n,&out_to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*ny,&out_to->values,ny,&out_to->indices,out_to->n+1,&out_to->starts,out_to->n,&out_to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(in_to->n,in_from->n),&out_to->sstatus,PetscMax(in_to->n,in_from->n),&out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);

  out->todata                        = (void*)out_to;
  out_to->local.n                    = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc1(in_to->local.n,&out_to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc1(in_from->local.n,&out_from->local.vslots);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->local.vslots,in_to->local.vslots,in_to->local.n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->local.vslots,in_from->local.vslots,in_from->local.n*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    out_to->local.vslots   = 0;
    out_from->local.vslots = 0;
  }

  /* allocate entire receive context */
  out_from->type      = in_from->type;
  ny                  = in_from->starts[in_from->n];
  out_from->n         = in_from->n;
  out_from->sendfirst = in_from->sendfirst;

  ierr = PetscMalloc1(out_from->n,&out_from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(ny*bs,&out_from->values,ny,&out_from->indices,out_from->n+1,&out_from->starts,out_from->n,&out_from->procs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);

  out->fromdata                        = (void*)out_from;
  out_from->local.n                    = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;

  out_to->use_alltoallv = out_from->use_alltoallv = PETSC_TRUE;

  ierr = PetscMalloc2(size,&out_to->counts,size,&out_to->displs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->counts,in_to->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->displs,in_to->displs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  ierr = PetscMalloc2(size,&out_from->counts,size,&out_from->displs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->counts,in_from->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->displs,in_from->displs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------------------
    Packs and unpacks the message data into send or from receive buffers.

    These could be generated automatically.

    Fortran kernels etc. could be used.
*/
PETSC_STATIC_INLINE void Pack_1(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i;
  for (i=0; i<n; i++) y[i] = x[indicesx[i]];
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_1(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = x[i];
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] += x[i];
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = PetscMax(y[indicesy[i]],x[i]);
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_1(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = x[indicesx[i]];
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] += x[indicesx[i]];
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = PetscMax(y[indicesy[i]],x[indicesx[i]]);
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_2(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y   += 2;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_2(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      x       += 2;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      x        += 2;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      x       += 2;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_2(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_3(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y   += 3;
  }
}
PETSC_STATIC_INLINE PetscErrorCode UnPack_3(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      x       += 3;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      x        += 3;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      x       += 3;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_3(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_4(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y   += 4;
  }
}
PETSC_STATIC_INLINE PetscErrorCode UnPack_4(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      x       += 4;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      x        += 4;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      x       += 4;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_4(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_5(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y   += 5;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_5(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      x       += 5;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      x        += 5;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      x       += 5;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_5(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_6(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y   += 6;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_6(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      x       += 6;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      x        += 6;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      x       += 6;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_6(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_7(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y[6] = x[idx+6];
    y   += 7;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_7(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      y[idy+6] = x[6];
      x       += 7;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      y[idy+6] += x[6];
      x        += 7;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      y[idy+6] = PetscMax(y[idy+6],x[6]);
      x       += 7;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_7(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
      y[idy+6] = x[idx+6];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
      y[idy+6] += x[idx+6];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6] = PetscMax(y[idy+6],x[idx+6]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_8(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y[6] = x[idx+6];
    y[7] = x[idx+7];
    y   += 8;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_8(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      y[idy+6] = x[6];
      y[idy+7] = x[7];
      x       += 8;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      y[idy+6] += x[6];
      y[idy+7] += x[7];
      x        += 8;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      y[idy+6] = PetscMax(y[idy+6],x[6]);
      y[idy+7] = PetscMax(y[idy+7],x[7]);
      x       += 8;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_8(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
      y[idy+6] = x[idx+6];
      y[idy+7] = x[idx+7];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
      y[idy+6] += x[idx+6];
      y[idy+7] += x[idx+7];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6] = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7] = PetscMax(y[idy+7],x[idx+7]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_9(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y    += 9;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_9(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      x        += 9;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      x         += 9;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      x        += 9;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_9(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_10(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y    += 10;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_10(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      x        += 10;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      x         += 10;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      x        += 10;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_10(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_11(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y[10] = x[idx+10];
    y    += 11;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_11(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      y[idy+10] = x[10];
      x        += 11;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      y[idy+10] += x[10];
      x         += 11;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      y[idy+10] = PetscMax(y[idy+10],x[10]);
      x        += 11;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_11(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
      y[idy+10] = x[idx+10];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
      y[idy+10] += x[idx+10];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
      y[idy+10] = PetscMax(y[idy+10],x[idx+10]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_12(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y[10] = x[idx+10];
    y[11] = x[idx+11];
    y    += 12;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_12(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      y[idy+10] = x[10];
      y[idy+11] = x[11];
      x        += 12;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      y[idy+10] += x[10];
      y[idy+11] += x[11];
      x         += 12;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      y[idy+10] = PetscMax(y[idy+10],x[10]);
      y[idy+11] = PetscMax(y[idy+11],x[11]);
      x        += 12;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_12(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
      y[idy+10] = x[idx+10];
      y[idy+11] = x[idx+11];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
      y[idy+10] += x[idx+10];
      y[idy+11] += x[idx+11];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
      y[idy+10] = PetscMax(y[idy+10],x[idx+10]);
      y[idy+11] = PetscMax(y[idy+11],x[idx+11]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_bs(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt       i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    PetscMemcpy(y,x + idx,bs*sizeof(PetscScalar));
    y    += bs;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_bs(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy,j;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      PetscMemcpy(y + idy,x,bs*sizeof(PetscScalar));
      x        += bs;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      for (j=0; j<bs; j++) y[idy+j] += x[j];
      x         += bs;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy = *indicesy++;
      for (j=0; j<bs; j++) y[idy+j] = PetscMax(y[idy+j],x[j]);
      x  += bs;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_bs(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy,j;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      PetscMemcpy(y + idy, x + idx,bs*sizeof(PetscScalar));
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      for (j=0; j<bs; j++ )  y[idy+j] += x[idx+j];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      for (j=0; j<bs; j++ )  y[idy+j] = PetscMax(y[idy+j],x[idx+j]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* Create the VecScatterBegin/End_P for our chosen block sizes */
#define BS 1
#include <../src/vec/vec/utils/vpscat.h>
#define BS 2
#include <../src/vec/vec/utils/vpscat.h>
#define BS 3
#include <../src/vec/vec/utils/vpscat.h>
#define BS 4
#include <../src/vec/vec/utils/vpscat.h>
#define BS 5
#include <../src/vec/vec/utils/vpscat.h>
#define BS 6
#include <../src/vec/vec/utils/vpscat.h>
#define BS 7
#include <../src/vec/vec/utils/vpscat.h>
#define BS 8
#include <../src/vec/vec/utils/vpscat.h>
#define BS 9
#include <../src/vec/vec/utils/vpscat.h>
#define BS 10
#include <../src/vec/vec/utils/vpscat.h>
#define BS 11
#include <../src/vec/vec/utils/vpscat.h>
#define BS 12
#include <../src/vec/vec/utils/vpscat.h>
#define BS bs
#include <../src/vec/vec/utils/vpscat.h>

/*
   bs indicates how many elements there are in each block. Normally this would be 1.
*/
PetscErrorCode VecScatterCreateCommon_PtoS(VecScatter_MPI_General *from,VecScatter_MPI_General *to,VecScatter ctx)
{
  MPI_Comm       comm;
  PetscMPIInt    tag  = ((PetscObject)ctx)->tag, tagr;
  PetscInt       bs   = to->bs;
  PetscMPIInt    size;
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tagr);CHKERRQ(ierr);
  ctx->ops->destroy = VecScatterDestroy_PtoP;

  ctx->reproduce = PETSC_FALSE;
  to->sendfirst  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_reproduce",&ctx->reproduce,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_sendfirst",&to->sendfirst,NULL);CHKERRQ(ierr);
  from->sendfirst = to->sendfirst;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  /* check if the receives are ALL going into contiguous locations; if so can skip indexing */
  to->contiq = PETSC_FALSE;
  n = from->starts[from->n];
  from->contiq = PETSC_TRUE;
  for (i=1; i<n; i++) {
    if (from->indices[i] != from->indices[i-1] + bs) {
      from->contiq = PETSC_FALSE;
      break;
    }
  }

  to->use_alltoallv = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_alltoall",&to->use_alltoallv,NULL);CHKERRQ(ierr);
  from->use_alltoallv = to->use_alltoallv;
  if (from->use_alltoallv) PetscInfo(ctx,"Using MPI_Alltoallv() for scatter\n");
#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
  if (to->use_alltoallv) {
    to->use_alltoallw = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_nopack",&to->use_alltoallw,NULL);CHKERRQ(ierr);
  }
  from->use_alltoallw = to->use_alltoallw;
  if (from->use_alltoallw) PetscInfo(ctx,"Using MPI_Alltoallw() for scatter\n");
#endif

#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  to->use_window = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_window",&to->use_window,NULL);CHKERRQ(ierr);
  from->use_window = to->use_window;
#endif

  if (to->use_alltoallv) {
    ierr       = PetscMalloc2(size,&to->counts,size,&to->displs);CHKERRQ(ierr);
    ierr       = PetscMemzero(to->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
    for (i=0; i<to->n; i++) to->counts[to->procs[i]] = bs*(to->starts[i+1] - to->starts[i]);

    to->displs[0] = 0;
    for (i=1; i<size; i++) to->displs[i] = to->displs[i-1] + to->counts[i-1];

    ierr       = PetscMalloc2(size,&from->counts,size,&from->displs);CHKERRQ(ierr);
    ierr       = PetscMemzero(from->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
    for (i=0; i<from->n; i++) from->counts[from->procs[i]] = bs*(from->starts[i+1] - from->starts[i]);
    from->displs[0] = 0;
    for (i=1; i<size; i++) from->displs[i] = from->displs[i-1] + from->counts[i-1];

#if defined(PETSC_HAVE_MPI_ALLTOALLW) && !defined(PETSC_USE_64BIT_INDICES)
    if (to->use_alltoallw) {
      PetscMPIInt mpibs, mpilen;

      ctx->packtogether = PETSC_FALSE;
      ierr = PetscMPIIntCast(bs,&mpibs);CHKERRQ(ierr);
      ierr = PetscMalloc3(size,&to->wcounts,size,&to->wdispls,size,&to->types);CHKERRQ(ierr);
      ierr = PetscMemzero(to->wcounts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      ierr = PetscMemzero(to->wdispls,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      for (i=0; i<size; i++) to->types[i] = MPIU_SCALAR;

      for (i=0; i<to->n; i++) {
        to->wcounts[to->procs[i]] = 1;
        ierr = PetscMPIIntCast(to->starts[i+1]-to->starts[i],&mpilen);CHKERRQ(ierr);
        ierr = MPI_Type_create_indexed_block(mpilen,mpibs,to->indices+to->starts[i],MPIU_SCALAR,to->types+to->procs[i]);CHKERRQ(ierr);
        ierr = MPI_Type_commit(to->types+to->procs[i]);CHKERRQ(ierr);
      }
      ierr       = PetscMalloc3(size,&from->wcounts,size,&from->wdispls,size,&from->types);CHKERRQ(ierr);
      ierr       = PetscMemzero(from->wcounts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      ierr       = PetscMemzero(from->wdispls,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      for (i=0; i<size; i++) from->types[i] = MPIU_SCALAR;

      if (from->contiq) {
        PetscInfo(ctx,"Scattered vector entries are stored contiguously, taking advantage of this with -vecscatter_alltoall\n");
        for (i=0; i<from->n; i++) from->wcounts[from->procs[i]] = bs*(from->starts[i+1] - from->starts[i]);

        if (from->n) from->wdispls[from->procs[0]] = sizeof(PetscScalar)*from->indices[0];
        for (i=1; i<from->n; i++) from->wdispls[from->procs[i]] = from->wdispls[from->procs[i-1]] + sizeof(PetscScalar)*from->wcounts[from->procs[i-1]];

      } else {
        for (i=0; i<from->n; i++) {
          from->wcounts[from->procs[i]] = 1;
          ierr = PetscMPIIntCast(from->starts[i+1]-from->starts[i],&mpilen);CHKERRQ(ierr);
          ierr = MPI_Type_create_indexed_block(mpilen,mpibs,from->indices+from->starts[i],MPIU_SCALAR,from->types+from->procs[i]);CHKERRQ(ierr);
          ierr = MPI_Type_commit(from->types+from->procs[i]);CHKERRQ(ierr);
        }
      }
    } else ctx->ops->copy = VecScatterCopy_PtoP_AllToAll;

#else
    to->use_alltoallw   = PETSC_FALSE;
    from->use_alltoallw = PETSC_FALSE;
    ctx->ops->copy      = VecScatterCopy_PtoP_AllToAll;
#endif
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  } else if (to->use_window) {
    PetscMPIInt temptag,winsize;
    MPI_Request *request;
    MPI_Status  *status;

    ierr = PetscObjectGetNewTag((PetscObject)ctx,&temptag);CHKERRQ(ierr);
    winsize = (to->n ? to->starts[to->n] : 0)*bs*sizeof(PetscScalar);
    ierr = MPI_Win_create(to->values ? to->values : MPI_BOTTOM,winsize,sizeof(PetscScalar),MPI_INFO_NULL,comm,&to->window);CHKERRQ(ierr);
    ierr = PetscMalloc1(to->n,&to->winstarts);CHKERRQ(ierr);
    ierr = PetscMalloc2(to->n,&request,to->n,&status);CHKERRQ(ierr);
    for (i=0; i<to->n; i++) {
      ierr = MPI_Irecv(to->winstarts+i,1,MPIU_INT,to->procs[i],temptag,comm,request+i);CHKERRQ(ierr);
    }
    for (i=0; i<from->n; i++) {
      ierr = MPI_Send(from->starts+i,1,MPIU_INT,from->procs[i],temptag,comm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(to->n,request,status);CHKERRQ(ierr);
    ierr = PetscFree2(request,status);CHKERRQ(ierr);

    winsize = (from->n ? from->starts[from->n] : 0)*bs*sizeof(PetscScalar);
    ierr = MPI_Win_create(from->values ? from->values : MPI_BOTTOM,winsize,sizeof(PetscScalar),MPI_INFO_NULL,comm,&from->window);CHKERRQ(ierr);
    ierr = PetscMalloc1(from->n,&from->winstarts);CHKERRQ(ierr);
    ierr = PetscMalloc2(from->n,&request,from->n,&status);CHKERRQ(ierr);
    for (i=0; i<from->n; i++) {
      ierr = MPI_Irecv(from->winstarts+i,1,MPIU_INT,from->procs[i],temptag,comm,request+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Send(to->starts+i,1,MPIU_INT,to->procs[i],temptag,comm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(from->n,request,status);CHKERRQ(ierr);
    ierr = PetscFree2(request,status);CHKERRQ(ierr);
#endif
  } else {
    PetscBool   use_rsend = PETSC_FALSE, use_ssend = PETSC_FALSE;
    PetscInt    *sstarts  = to->starts,  *rstarts = from->starts;
    PetscMPIInt *sprocs   = to->procs,   *rprocs  = from->procs;
    MPI_Request *swaits   = to->requests,*rwaits  = from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = to->values, *Srvalues = from->values;

    /* allocate additional wait variables for the "reverse" scatter */
    ierr = PetscMalloc1(to->n,&rev_rwaits);CHKERRQ(ierr);
    ierr = PetscMalloc1(from->n,&rev_swaits);CHKERRQ(ierr);
    to->rev_requests   = rev_rwaits;
    from->rev_requests = rev_swaits;

    /* Register the receives that you will use later (sends for scatter reverse) */
    ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_rsend",&use_rsend,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_ssend",&use_ssend,NULL);CHKERRQ(ierr);
    if (use_rsend) {
      ierr = PetscInfo(ctx,"Using VecScatter ready receiver mode\n");CHKERRQ(ierr);
      to->use_readyreceiver   = PETSC_TRUE;
      from->use_readyreceiver = PETSC_TRUE;
    } else {
      to->use_readyreceiver   = PETSC_FALSE;
      from->use_readyreceiver = PETSC_FALSE;
    }
    if (use_ssend) {
      ierr = PetscInfo(ctx,"Using VecScatter Ssend mode\n");CHKERRQ(ierr);
    }

    for (i=0; i<from->n; i++) {
      if (use_rsend) {
        ierr = MPI_Rsend_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      } else if (use_ssend) {
        ierr = MPI_Ssend_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      } else {
        ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      }
    }

    for (i=0; i<to->n; i++) {
      if (use_rsend) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } else if (use_ssend) {
        ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } else {
        ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      }
    }
    /* Register receives for scatter and reverse */
    for (i=0; i<from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tagr,comm,rev_rwaits+i);CHKERRQ(ierr);
    }
    if (use_rsend) {
      if (to->n)   {ierr = MPI_Startall_irecv(to->starts[to->n]*to->bs,to->n,to->rev_requests);CHKERRQ(ierr);}
      if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    }
    ctx->ops->copy = VecScatterCopy_PtoP_X;
  }
  ierr = PetscInfo1(ctx,"Using blocksize %D scatter\n",bs);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  ierr = MPIU_Allreduce(&bs,&i,1,MPIU_INT,MPI_MIN,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&bs,&n,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
  if (bs!=i || bs!=n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Blocks size %D != %D or %D",bs,i,n);
#endif

  switch (bs) {
  case 12:
    ctx->ops->begin = VecScatterBegin_12;
    ctx->ops->end   = VecScatterEnd_12;
    break;
  case 11:
    ctx->ops->begin = VecScatterBegin_11;
    ctx->ops->end   = VecScatterEnd_11;
    break;
  case 10:
    ctx->ops->begin = VecScatterBegin_10;
    ctx->ops->end   = VecScatterEnd_10;
    break;
  case 9:
    ctx->ops->begin = VecScatterBegin_9;
    ctx->ops->end   = VecScatterEnd_9;
    break;
  case 8:
    ctx->ops->begin = VecScatterBegin_8;
    ctx->ops->end   = VecScatterEnd_8;
    break;
  case 7:
    ctx->ops->begin = VecScatterBegin_7;
    ctx->ops->end   = VecScatterEnd_7;
    break;
  case 6:
    ctx->ops->begin = VecScatterBegin_6;
    ctx->ops->end   = VecScatterEnd_6;
    break;
  case 5:
    ctx->ops->begin = VecScatterBegin_5;
    ctx->ops->end   = VecScatterEnd_5;
    break;
  case 4:
    ctx->ops->begin = VecScatterBegin_4;
    ctx->ops->end   = VecScatterEnd_4;
    break;
  case 3:
    ctx->ops->begin = VecScatterBegin_3;
    ctx->ops->end   = VecScatterEnd_3;
    break;
  case 2:
    ctx->ops->begin = VecScatterBegin_2;
    ctx->ops->end   = VecScatterEnd_2;
    break;
  case 1:
    ctx->ops->begin = VecScatterBegin_1;
    ctx->ops->end   = VecScatterEnd_1;
    break;
  default:
    ctx->ops->begin = VecScatterBegin_bs;
    ctx->ops->end   = VecScatterEnd_bs;
  }
  ctx->ops->view = VecScatterView_MPI;
  /* Check if the local scatter is made of copies; important special case */
  if (to->local.n) {
    ierr = VecScatterLocalOptimizeCopy_Private(ctx,&to->local,&from->local,bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@
     VecScatterCreateLocal - Creates a VecScatter from a list of messages it must send and receive.

     Collective on VecScatter

   Input Parameters:
+     VecScatter - obtained with VecScatterCreateEmpty()
.     nsends -
.     sendSizes -
.     sendProcs -
.     sendIdx - indices where the sent entries are obtained from (in local, on process numbering), this is one long array of size \sum_{i=0,i<nsends} sendSizes[i]
.     nrecvs - number of receives to expect
.     recvSizes -
.     recvProcs - processes who are sending to me
.     recvIdx - indices of where received entries are to be put, (in local, on process numbering), this is one long array of size \sum_{i=0,i<nrecvs} recvSizes[i]
-     bs - size of block

     Notes:  sendSizes[] and recvSizes[] cannot have any 0 entries. If you want to support having 0 entries you need
      to change the code below to "compress out" the sendProcs[] and recvProcs[] entries that have 0 entries.

       Probably does not handle sends to self properly. It should remove those from the counts that are used
      in allocating space inside of the from struct

  Level: intermediate

@*/
PetscErrorCode VecScatterCreateLocal(VecScatter ctx,PetscInt nsends,const PetscInt sendSizes[],const PetscInt sendProcs[],const PetscInt sendIdx[],PetscInt nrecvs,const PetscInt recvSizes[],const PetscInt recvProcs[],const PetscInt recvIdx[],PetscInt bs)
{
  VecScatter_MPI_General *from, *to;
  PetscInt               sendSize, recvSize;
  PetscInt               n, i;
  PetscErrorCode         ierr;

  /* allocate entire send scatter context */
  ierr  = PetscNewLog(ctx,&to);CHKERRQ(ierr);
  to->n = nsends;
  for (n = 0, sendSize = 0; n < to->n; n++) sendSize += sendSizes[n];

  ierr = PetscMalloc1(to->n,&to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*sendSize,&to->values,sendSize,&to->indices,to->n+1,&to->starts,to->n,&to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(to->n,nrecvs),&to->sstatus,PetscMax(to->n,nrecvs),&to->rstatus);CHKERRQ(ierr);

  to->starts[0] = 0;
  for (n = 0; n < to->n; n++) {
    if (sendSizes[n] <=0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"sendSizes[n=%D] = %D cannot be less than 1",n,sendSizes[n]);
    to->starts[n+1] = to->starts[n] + sendSizes[n];
    to->procs[n]    = sendProcs[n];
    for (i = to->starts[n]; i < to->starts[n]+sendSizes[n]; i++) to->indices[i] = sendIdx[i];
  }
  ctx->todata = (void*) to;

  /* allocate entire receive scatter context */
  ierr    = PetscNewLog(ctx,&from);CHKERRQ(ierr);
  from->n = nrecvs;
  for (n = 0, recvSize = 0; n < from->n; n++) recvSize += recvSizes[n];

  ierr = PetscMalloc1(from->n,&from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*recvSize,&from->values,recvSize,&from->indices,from->n+1,&from->starts,from->n,&from->procs);CHKERRQ(ierr);

  from->starts[0] = 0;
  for (n = 0; n < from->n; n++) {
    if (recvSizes[n] <=0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"recvSizes[n=%D] = %D cannot be less than 1",n,recvSizes[n]);
    from->starts[n+1] = from->starts[n] + recvSizes[n];
    from->procs[n]    = recvProcs[n];
    for (i = from->starts[n]; i < from->starts[n]+recvSizes[n]; i++) from->indices[i] = recvIdx[i];
  }
  ctx->fromdata = (void*)from;

  /* No local scatter optimization */
  from->local.n                    = 0;
  from->local.vslots               = 0;
  to->local.n                      = 0;
  to->local.vslots                 = 0;
  from->local.nonmatching_computed = PETSC_FALSE;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = PETSC_FALSE;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  from->type = VEC_SCATTER_MPI_GENERAL;
  to->type   = VEC_SCATTER_MPI_GENERAL;
  from->bs   = bs;
  to->bs     = bs;
  ierr       = VecScatterCreateCommon_PtoS(from, to, ctx);CHKERRQ(ierr);

  /* mark lengths as negative so it won't check local vector lengths */
  ctx->from_n = ctx->to_n = -1;
  PetscFunctionReturn(0);
}

/*
   create parallel to sequential scatter context.
   bs indicates how many elements there are in each block. Normally this would be 1.

   contains check that PetscMPIInt can handle the sizes needed
*/
PetscErrorCode VecScatterCreate_PtoS(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  VecScatter_MPI_General *from,*to;
  PetscMPIInt            nprocs,myrank,tag;
  PetscMPIInt            *recvfrom = NULL,*rlens = NULL,rlenlocal,rlentotal,rlenshm,nrecvs;
  PetscMPIInt            *sendto = NULL,*slens = NULL,slentotal,slenshm,nsends,nsendsshm;
  PetscInt               *range = NULL,i,j;
  PetscInt               *rstarts = NULL,count;
  PetscInt               *rindices,*sindices,*sindices2;
  MPI_Request            *sreqs = NULL,*rreqs = NULL;
  PetscErrorCode         ierr;
  PetscInt               *idxbs_sorted = NULL,*idybs_sorted = NULL;
  PetscShmcomm           pshmcomm;
  MPI_Comm               comm; /* the outer communicator */
  PetscInt               it,first,step,lblocal,ublocal;
  PetscBool              use_intranodeshmem;
#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  PetscMPIInt            jj;
  MPI_Info               info;
#endif

  PetscFunctionBegin;
  ierr  = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr  = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr  = MPI_Comm_rank(comm,&myrank);CHKERRQ(ierr);
  ierr  = MPI_Comm_size(comm,&nprocs);CHKERRQ(ierr);
  range = xin->map->range;

  use_intranodeshmem = PETSC_FALSE;
  ierr  = PetscOptionsGetBool(NULL,NULL,"-vecscatter_intranodeshmem",&use_intranodeshmem,NULL);CHKERRQ(ierr);
  if (use_intranodeshmem) { ierr = PetscShmcommGet(comm,&pshmcomm);CHKERRQ(ierr); }

  /*=========================================================================
            sort indices and locate my segment
    =========================================================================*/

  /* idxbs_sorted - [nx] inidx[]*bs and then sorted
     idybs_sorted - [ny] inidy[]*bs and then sorted
     lblocal      - lower & upper bound such that indices in ..
     ublocal      - idxbs_sorted[lblocal..ublocal) are owned by me
   */

  /* Sorted indices make code simpler, faster and also help getting rid of
     many O(P) arrays, which hurt scalability at large scale.
   */
  ierr = PetscMalloc2(nx,&idxbs_sorted,ny,&idybs_sorted);CHKERRQ(ierr);
  if (bs == 1) { /* accelerate the common case */
    ierr = PetscMemcpy(idxbs_sorted,inidx,sizeof(PetscInt)*nx);CHKERRQ(ierr);
    ierr = PetscMemcpy(idybs_sorted,inidy,sizeof(PetscInt)*ny);CHKERRQ(ierr);
  } else {
    for (i=0; i<nx; i++) idxbs_sorted[i] = inidx[i]*bs;
    for (i=0; i<ny; i++) idybs_sorted[i] = inidy[i]*bs;
  }

  ierr = PetscSortIntWithArray(nx,idxbs_sorted,idybs_sorted);CHKERRQ(ierr);

  /* search idxbs_sorted[] to locate my segment of indices. If exist, they are
     in idxbs_sorted[lblocal..ublocal), otherwise, lblocal=ublocal=0 or nx,
     depending on whether idxbs_sorted[0] > range[myrank] or not.
   */
  first = 0; count = nx; /* find first element in idxbs_sorted[] that is not less than range[myrank] */
  while (count > 0) {
    it = first; step = count/2; it += step;
    if (idxbs_sorted[it] < range[myrank]) { first  = ++it; count -= step + 1; }
    else { count = step; }
  }
  lblocal = first;

  first = 0; count = nx; /* do it again for range[myrank+1] */
  while (count > 0) {
    it = first; step = count/2; it += step;
    if (idxbs_sorted[it] < range[myrank+1]) { first  = ++it; count -= step + 1; }
    else { count = step; }
  }
  ublocal = first;

  /*=========================================================================
           collect info about messages I want to receive
    =========================================================================*/

  /* nrecvs    - number of non-empty messages, excluding the message from myself.
     recvfrom  - [nrecvs] processors I will receive messages from, excluding myself
     rindices  - [rlentotal] indices of entries I will receive
     rstarts   - [nrecvs+1] rstarts[i] is the starting index of rindices[] I expect from processor recvfrom[i]
     rlens     - [nprocs] I want to receive rlens[i] entries from processor i.
     rlentotal - total number of entries I will receive, excluding entries from myself
     rlenlocal - number of entries from myself

     Attention: rlens[] is of O(P) storage. It is the only one of this large in this function.
   */

  /* get rlens, nrecvs */
  ierr = PetscCalloc1(nprocs,&rlens);CHKERRQ(ierr);

  i = j = nrecvs = 0;
  while (i < nx) {
    if (idxbs_sorted[i] >= range[j+1]) { /* if i-th index is out of processor j's bound */
      do { j++; } while (idxbs_sorted[i] >= range[j+1] && j < nprocs); /* boost j until it falls in processor j's bound */
      if (j == nprocs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"entry %D not owned by any process, upper bound %D",idxbs_sorted[i],range[nprocs]);
    }

    if (j == myrank) { i = ublocal; } /* skip local indices, which are likely a lot */
    else { i++; if (!rlens[j]++) nrecvs++; }
  }

  rlenlocal = ublocal - lblocal;
  rlentotal = nx - rlenlocal;

  /* get rstarts, recvfrom, rindices and rreqs once we know nrecvs, rlens*/
  ierr = PetscMalloc4(rlentotal,&rindices,nrecvs+1,&rstarts,nrecvs,&recvfrom,nrecvs,&rreqs);CHKERRQ(ierr);

  j = rstarts[0] = 0;
  for (i=0; i<nprocs; i++) {
    if (rlens[i]) { recvfrom[j] = i; rstarts[j+1] = rstarts[j] + rlens[i]; j++; }
  }

  ierr = PetscMemcpy(rindices,idxbs_sorted,sizeof(PetscInt)*lblocal);CHKERRQ(ierr); /* two copies to skip indices in [lblocal,ublocal) */
  ierr = PetscMemcpy(&rindices[lblocal],idxbs_sorted+ublocal,sizeof(PetscInt)*(nx-ublocal));CHKERRQ(ierr);

  /*=========================================================================
           compute the reverse info about messages I need to send
    =========================================================================*/

  /* nsends    - number of (non-empty) messages I need to send
     sendto    - [nsends] processors I send to
     slens     - [nsends] I will send slens[i] entries to processor sendto[i]
     slentotal - sum of slens[]
     sindices  - [] store indices of entries I need to send
     sreqs     - [nsends] MPI requests
   */
  ierr = PetscGatherNumberOfMessages(comm,NULL,rlens,&nsends);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nrecvs,nsends,rlens,&sendto,&slens);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nsends,sendto,slens);CHKERRQ(ierr);
  slentotal = 0; for (i=0; i<nsends; i++) slentotal += slens[i];

  ierr = PetscFree(rlens);CHKERRQ(ierr);

  /* communicate with processors in sendto[] to populate sindices[].
     Post irecvs first and then isends. It is funny (but correct) that we
     temporarily use send stuff in MPI_Irecv and recv stuff in MPI_Isends.
   */
  ierr  = PetscMalloc2(slentotal,&sindices,nsends,&sreqs);CHKERRQ(ierr);

  count = 0;
  for (i=0; i<nsends; i++) {
    ierr   = MPI_Irecv(sindices+count,slens[i],MPIU_INT,sendto[i],tag,comm,sreqs+i);CHKERRQ(ierr);
    count += slens[i];
  }

  for (i=0; i<nrecvs; i++) { ierr = MPI_Isend(rindices+rstarts[i],rstarts[i+1]-rstarts[i],MPIU_INT,recvfrom[i],tag,comm,rreqs+i);CHKERRQ(ierr); }

  /* wait on irecvs and if supported, figure out which sendto[] processors are in the shared memory communicator
     nsendsshm - number of sendto[] processors in the shared memory communicator
     slenshm   - total number of entries sent to shared memory partners
   */
  nsendsshm = 0;
  slenshm   = 0;
#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  if (use_intranodeshmem) {
    MPI_Status send_status;
    PetscMPIInt index,n;
    for (i=0; i<nsends; i++) {
      ierr = MPI_Waitany(nsends,sreqs,&index,&send_status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&send_status,MPIU_INT,&n);CHKERRQ(ierr);
      ierr = PetscShmcommGlobalToLocal(pshmcomm,sendto[index],&jj);CHKERRQ(ierr);
      if (jj != MPI_PROC_NULL) { nsendsshm++; slenshm += n; }
    }
  } else
#endif
  { ierr = MPI_Waitall(nsends,sreqs,MPI_STATUS_IGNORE);CHKERRQ(ierr); }

  ierr = MPI_Waitall(nrecvs,rreqs,MPI_STATUS_IGNORE);CHKERRQ(ierr);

  /*=========================================================================
         allocate entire send scatter context
    =========================================================================*/
  ierr                   = PetscNewLog(ctx,&to);CHKERRQ(ierr); /* calloc to */
  ctx->todata            = (void*)to;
  to->n                  = nsends-nsendsshm;
  to->use_intranodeshmem = use_intranodeshmem;

  ierr = PetscMalloc1(to->n,&to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(to->n,nrecvs),&to->sstatus,PetscMax(to->n,nrecvs),&to->rstatus);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*(slentotal-slenshm),&to->values,slentotal-slenshm,&to->indices,to->n+1,&to->starts,to->n,&to->procs);CHKERRQ(ierr);

  sindices2     = sindices;
  to->n         = 0;
  to->starts[0] = 0;

#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  if (use_intranodeshmem) {
    to->shmn         = nsendsshm;
    ierr             = PetscMalloc4(to->shmn,&to->shmprocs,to->shmn,&to->shmflags,to->shmn+1,&to->shmstarts,slenshm,&to->shmindices);CHKERRQ(ierr);
    to->shmn         = 0;
    to->shmstarts[0] = 0;
  }
#endif

  for (i=0; i<nsends; i++) {
#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
    if (use_intranodeshmem) {
      ierr = PetscShmcommGlobalToLocal(pshmcomm,(PetscMPIInt)sendto[i],&jj);CHKERRQ(ierr);
    }

    if (use_intranodeshmem && jj != MPI_PROC_NULL) { /* sendto[i] is a shared memory partner and jj is its rank in shmcomm */
      to->shmstarts[to->shmn+1] = to->shmstarts[to->shmn] + slens[i];
      to->shmprocs[to->shmn]    = jj;
      for (j=0; j<slens[i]; j++) to->shmindices[to->shmstarts[to->shmn]+j] = sindices2[j] - range[myrank];
      to->shmn++;
    } else
#endif
    {
      to->starts[to->n+1] = to->starts[to->n] + slens[i];
      to->procs[to->n]    = sendto[i];
      for (j=0; j<slens[i]; j++) to->indices[to->starts[to->n]+j] = sindices2[j] - range[myrank];
      to->n++;
    }
    sindices2 += slens[i];
  }

  /* free send stuffs */
  ierr = PetscFree(slens);CHKERRQ(ierr);
  ierr = PetscFree(sendto);CHKERRQ(ierr);
  ierr = PetscFree2(sindices,sreqs);CHKERRQ(ierr);

#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  /* allocate the shared memory region at the sender side. The region is padded with a header
     containing sync flags. Flags are on different cache lines to avoid false sharing.
   */
  if (use_intranodeshmem) {
    ierr = PetscShmcommGetMpiShmcomm(pshmcomm,&to->shmcomm);CHKERRQ(ierr);
    ierr = MPI_Info_create(&info);CHKERRQ(ierr);
    ierr = MPI_Info_set(info, "alloc_shared_noncontig", "true");CHKERRQ(ierr);
    MPI_Aint sz = bs*to->shmstarts[to->shmn]*sizeof(PetscScalar) + (to->shmn+1)*PETSC_LEVEL1_DCACHE_LINESIZE; /* add an extra cacheline for alignment purpose */
    ierr = MPI_Win_allocate_shared(sz,sizeof(PetscScalar),info,to->shmcomm,&to->shmspace,&to->shmwin);CHKERRQ(ierr);
    ierr = MPI_Info_free(&info);CHKERRQ(ierr);

    /* Align the returned shared memory address to cacheline, where the flag area
       starts. Each flag takes one cacheline to avoid false sharing. Note we used
       alloc_shared_noncontig in shared memory allocation. The returned shared memory
       address on each process is expected to be page-aligned (and cacheline-aligned).
       However, for some unknown reason, I found it is not necessarily true on a Cray
       machine. So I allocate an extra cacheline and do the alignment myself.
     */
    to->shmspace = (PetscScalar*)((((PETSC_UINTPTR_T)(to->shmspace))+(PETSC_LEVEL1_DCACHE_LINESIZE-1)) & ~(PETSC_LEVEL1_DCACHE_LINESIZE-1));
    for (i=0; i<to->shmn; i++) {
      to->shmflags[i] = (PetscInt*)((char*)to->shmspace + i*PETSC_LEVEL1_DCACHE_LINESIZE);
      *to->shmflags[i] = 0; /* init the flag to empty (0) to say sender can write the buffer */
    }
    to->shmspace = (PetscScalar*)((char*)to->shmspace + to->shmn*PETSC_LEVEL1_DCACHE_LINESIZE); /* point the pointer to the data area */
  }
#endif

  /*=========================================================================
         allocate entire receive scatter context
    =========================================================================*/
  ierr                     = PetscNewLog(ctx,&from);CHKERRQ(ierr);
  ctx->fromdata            = (void*)from;
  from->use_intranodeshmem = use_intranodeshmem;

  /* compute rlenshm, from->n and from->shmn first to facilitate mallocs */
  rlenshm = 0;
  from->n = nrecvs;
#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  if (use_intranodeshmem) {
    from->n       = 0;
    from->shmn    = 0;
    from->shmwin  = to->shmwin;
    from->shmcomm = to->shmcomm;
    for (i=0; i<nrecvs; i++) {
      ierr = PetscShmcommGlobalToLocal(pshmcomm,recvfrom[i],&jj);CHKERRQ(ierr);
      if (jj != MPI_PROC_NULL) { from->shmn++; rlenshm += rstarts[i+1] - rstarts[i]; }
      else { from->n++; }
    }

    ierr = PetscMalloc5(from->shmn,&from->shmprocs,from->shmn,&from->shmspaces,from->shmn,&from->shmflags,from->shmn+1,&from->shmstarts,rlenshm,&from->shmindices);CHKERRQ(ierr);
  }
#endif

  ierr = PetscMalloc1(from->n,&from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*(ny-rlenshm),&from->values,ny-rlenshm,&from->indices,from->n+1,&from->starts,from->n,&from->procs);CHKERRQ(ierr);

  /* move data into receive scatter */
  from->n               = 0;
  from->starts[0]       = 0;
#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  if (use_intranodeshmem) {
    from->shmn          = 0;
    from->shmstarts[0]  = 0;
  }
#endif

  for (i=0; i<nrecvs; i++) {
    PetscInt len    = rstarts[i+1] - rstarts[i]; /* len works for both x and y */
    PetscInt ystart = rstarts[i] + (recvfrom[i] > myrank ? rlenlocal : 0); /* rstarts[] are offsets for x with locals removed. To use for y, one has to remedy them */

#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
    if (use_intranodeshmem) {
      ierr = PetscShmcommGlobalToLocal(pshmcomm,recvfrom[i],&jj);CHKERRQ(ierr);
    }

    if (use_intranodeshmem && jj != MPI_PROC_NULL) { /* recvfrom[i] is a shared memory partner and jj is its rank in shmcomm*/
      from->shmprocs[from->shmn]    = jj;
      from->shmstarts[from->shmn+1] = from->shmstarts[from->shmn] + len;
      ierr = PetscMemcpy(&from->shmindices[from->shmstarts[from->shmn]],&idybs_sorted[ystart],sizeof(PetscInt)*len);CHKERRQ(ierr);
      from->shmn++;
    } else
#endif
    {
      from->procs[from->n]    = recvfrom[i];
      from->starts[from->n+1] = from->starts[from->n] + len;
      ierr = PetscMemcpy(&from->indices[from->starts[from->n]],&idybs_sorted[ystart],sizeof(PetscInt)*len);CHKERRQ(ierr);
      from->n++;
    }
  }

  /* free recv stuffs */
  ierr = PetscFree4(rindices,rstarts,recvfrom,rreqs);CHKERRQ(ierr);

  /* query addresses of the shared memory regions of my partners. I also need
     to know offsets to those regions from where I can directly read data I need.
     We use send/recv within shmcomm to get the offsets. One could also get the
     offsets through shared memory. But since it is only used once, it is not
     worth the trouble.
   */
#if defined(PETSC_HAVE_MPI_COMM_TYPE_SHARED)
  if (use_intranodeshmem) {
    MPI_Request *reqs = NULL;
    struct {
      PetscInt j,m,offset; /* from my partner's view, I am its j-th partner out of its m partners, and I should read from this offset */
    } jmo,*triples = NULL;

    /* get the above info from my partners */
    ierr = PetscMalloc2(from->shmn,&reqs,from->shmn,&triples);CHKERRQ(ierr);
    for (i=0; i<from->shmn; i++) { ierr = MPI_Irecv(triples+i,3,MPIU_INT,from->shmprocs[i],0/*tag*/,from->shmcomm,reqs+i);CHKERRQ(ierr); }
    for (i=0; i<to->shmn; i++) {
      jmo.j = i;
      jmo.m = to->shmn;
      jmo.offset = to->shmstarts[i];
      ierr = MPI_Send(&jmo,3,MPIU_INT,to->shmprocs[i],0/*tag*/,to->shmcomm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(from->shmn,reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);

    /* figure out flag addresses and data addresses aimed to me */
    for (i=0; i<from->shmn; i++) {
      MPI_Aint    size;
      PetscMPIInt disp_unit;
      ierr = MPI_Win_shared_query(from->shmwin,from->shmprocs[i],&size,&disp_unit,&from->shmspaces[i]);CHKERRQ(ierr);
      from->shmspaces[i]  = (PetscScalar*)((((PETSC_UINTPTR_T)(from->shmspaces[i]))+(PETSC_LEVEL1_DCACHE_LINESIZE-1)) & ~(PETSC_LEVEL1_DCACHE_LINESIZE-1));
      from->shmflags[i]   =    (PetscInt*)((char*)from->shmspaces[i] + triples[i].j*PETSC_LEVEL1_DCACHE_LINESIZE); /* get address of the j-th flag */
      from->shmspaces[i]  = (PetscScalar*)((char*)from->shmspaces[i] + triples[i].m*PETSC_LEVEL1_DCACHE_LINESIZE); /* skip the flag area */
      from->shmspaces[i] += triples[i].offset*bs; /* and then add the offset to point to where my expected data lives */
    }

    ierr = PetscFree2(reqs,triples);CHKERRQ(ierr);
  }
#endif

  /*=========================================================================
       handle the scatter to myself
    =========================================================================*/
  if (rlenlocal) {
    to->local.n   = rlenlocal;
    from->local.n = rlenlocal;
    ierr = PetscMalloc1(to->local.n,&to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc1(from->local.n,&from->local.vslots);CHKERRQ(ierr);
    for (i=lblocal; i<ublocal; i++) to->local.vslots[i-lblocal] = idxbs_sorted[i] - range[myrank];
    ierr = PetscMemcpy(from->local.vslots,&idybs_sorted[lblocal],sizeof(PetscInt)*rlenlocal);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ctx,2*rlenlocal*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    to->local.n        = 0;
    to->local.vslots   = 0;
    from->local.n      = 0;
    from->local.vslots = 0;
  }

  ierr = PetscFree2(idxbs_sorted,idybs_sorted);CHKERRQ(ierr);

  from->local.nonmatching_computed = PETSC_FALSE;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = PETSC_FALSE;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  from->type = VEC_SCATTER_MPI_GENERAL;
  to->type   = VEC_SCATTER_MPI_GENERAL;
  from->bs   = bs;
  to->bs     = bs;

  ierr = VecScatterCreateCommon_PtoS(from,to,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/
/*
         Scatter from local Seq vectors to a parallel vector.
         Reverses the order of the arguments, calls VecScatterCreate_PtoS() then
         reverses the result.
*/
PetscErrorCode VecScatterCreate_StoP(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  PetscErrorCode         ierr;
  MPI_Request            *waits;
  VecScatter_MPI_General *to,*from;

  PetscFunctionBegin;
  ierr          = VecScatterCreate_PtoS(ny,inidy,nx,inidx,yin,xin,bs,ctx);CHKERRQ(ierr);
  to            = (VecScatter_MPI_General*)ctx->fromdata;
  from          = (VecScatter_MPI_General*)ctx->todata;
  ctx->todata   = (void*)to;
  ctx->fromdata = (void*)from;
  /* these two are special, they are ALWAYS stored in to struct */
  to->sstatus   = from->sstatus;
  to->rstatus   = from->rstatus;

  from->sstatus = 0;
  from->rstatus = 0;
  waits              = from->rev_requests;
  from->rev_requests = from->requests;
  from->requests     = waits;
  waits              = to->rev_requests;
  to->rev_requests   = to->requests;
  to->requests       = waits;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
PetscErrorCode VecScatterCreate_PtoP(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag,imdex,n;
  PetscInt       *owners = xin->map->range;
  PetscMPIInt    *nprocs = NULL;
  PetscInt       i,j,idx,nsends,*local_inidx = NULL,*local_inidy = NULL;
  PetscMPIInt    *owner   = NULL;
  PetscInt       *starts  = NULL,count,slen;
  PetscInt       *rvalues = NULL,*svalues = NULL,base,*values = NULL,*rsvalues,recvtotal,lastidx;
  PetscMPIInt    *onodes1,*olengths1,nrecvs;
  MPI_Comm       comm;
  MPI_Request    *send_waits = NULL,*recv_waits = NULL;
  MPI_Status     recv_status,*send_status = NULL;
  PetscBool      duplicate = PETSC_FALSE;
#if defined(PETSC_USE_DEBUG)
  PetscBool      found = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecScatterCreate_StoP(nx,inidx,ny,inidy,xin,yin,bs,ctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /*
     Each processor ships off its inidx[j] and inidy[j] to the appropriate processor
     They then call the StoPScatterCreate()
  */
  /*  first count number of contributors to each processor */
  ierr = PetscMalloc3(size,&nprocs,nx,&owner,(size+1),&starts);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  lastidx = -1;
  j       = 0;
  for (i=0; i<nx; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = bs*inidx[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++;
        owner[i] = j;
#if defined(PETSC_USE_DEBUG)
        found = PETSC_TRUE;
#endif
        break;
      }
    }
#if defined(PETSC_USE_DEBUG)
    if (!found) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D out of range",idx);
    found = PETSC_FALSE;
#endif
  }
  nsends = 0;
  for (i=0; i<size; i++) nsends += (nprocs[i] > 0);

  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,NULL,nprocs,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,nprocs,&onodes1,&olengths1);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1);CHKERRQ(ierr);
  recvtotal = 0; for (i=0; i<nrecvs; i++) recvtotal += olengths1[i];

  /* post receives:   */
  ierr = PetscMalloc5(2*recvtotal,&rvalues,2*nx,&svalues,nrecvs,&recv_waits,nsends,&send_waits,nsends,&send_status);CHKERRQ(ierr);

  count = 0;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((rvalues+2*count),2*olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i);CHKERRQ(ierr);
    count += olengths1[i];
  }
  ierr = PetscFree(onodes1);CHKERRQ(ierr);

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  starts[0]= 0;
  for (i=1; i<size; i++) starts[i] = starts[i-1] + nprocs[i-1];
  for (i=0; i<nx; i++) {
    svalues[2*starts[owner[i]]]       = bs*inidx[i];
    svalues[1 + 2*starts[owner[i]]++] = bs*inidy[i];
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) starts[i] = starts[i-1] + nprocs[i-1];
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i]) {
      ierr = MPI_Isend(svalues+2*starts[i],2*nprocs[i],MPIU_INT,i,tag,comm,send_waits+count);CHKERRQ(ierr);
      count++;
    }
  }
  ierr = PetscFree3(nprocs,owner,starts);CHKERRQ(ierr);

  /*  wait on receives */
  count = nrecvs;
  slen  = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr  = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    slen += n/2;
    count--;
  }
  if (slen != recvtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not as expected %D",slen,recvtotal);

  ierr     = PetscMalloc2(slen,&local_inidx,slen,&local_inidy);CHKERRQ(ierr);
  base     = owners[rank];
  count    = 0;
  rsvalues = rvalues;
  for (i=0; i<nrecvs; i++) {
    values    = rsvalues;
    rsvalues += 2*olengths1[i];
    for (j=0; j<olengths1[i]; j++) {
      local_inidx[count]   = values[2*j] - base;
      local_inidy[count++] = values[2*j+1];
    }
  }
  ierr = PetscFree(olengths1);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);}
  ierr = PetscFree5(rvalues,svalues,recv_waits,send_waits,send_status);CHKERRQ(ierr);

  /*
     should sort and remove duplicates from local_inidx,local_inidy
  */
#if defined(do_it_slow)
  /* sort on the from index */
  ierr  = PetscSortIntWithArray(slen,local_inidx,local_inidy);CHKERRQ(ierr);
  start = 0;
  while (start < slen) {
    count = start+1;
    last  = local_inidx[start];
    while (count < slen && last == local_inidx[count]) count++;
    if (count > start + 1) { /* found 2 or more same local_inidx[] in a row */
      /* sort on to index */
      ierr = PetscSortInt(count-start,local_inidy+start);CHKERRQ(ierr);
    }
    /* remove duplicates; not most efficient way, but probably good enough */
    i = start;
    while (i < count-1) {
      if (local_inidy[i] != local_inidy[i+1]) i++;
      else { /* found a duplicate */
        duplicate = PETSC_TRUE;
        for (j=i; j<slen-1; j++) {
          local_inidx[j] = local_inidx[j+1];
          local_inidy[j] = local_inidy[j+1];
        }
        slen--;
        count--;
      }
    }
    start = count;
  }
#endif
  if (duplicate) {
    ierr = PetscInfo(ctx,"Duplicate from to indices passed in VecScatterCreate(), they are ignored\n");CHKERRQ(ierr);
  }
  ierr = VecScatterCreate_StoP(slen,local_inidx,slen,local_inidy,xin,yin,bs,ctx);CHKERRQ(ierr);
  ierr = PetscFree2(local_inidx,local_inidy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSFCreateFromZero - Create a PetscSF that maps a Vec from sequential to distributed

  Input Parameters:
. gv - A distributed Vec

  Output Parameters:
. sf - The SF created mapping a sequential Vec to gv

  Level: developer

.seealso: DMPlexDistributedToSequential()
@*/
PetscErrorCode PetscSFCreateFromZero(MPI_Comm comm, Vec gv, PetscSF *sf)
{
  PetscSFNode   *remotenodes;
  PetscInt      *localnodes;
  PetscInt       N, n, start, numroots, l;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFCreate(comm, sf);CHKERRQ(ierr);
  ierr = VecGetSize(gv, &N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gv, &n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(gv, &start, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &localnodes);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &remotenodes);CHKERRQ(ierr);
  if (!rank) numroots = N;
  else       numroots = 0;
  for (l = 0; l < n; ++l) {
    localnodes[l]        = l;
    remotenodes[l].rank  = 0;
    remotenodes[l].index = l+start;
  }
  ierr = PetscSFSetGraph(*sf, numroots, n, localnodes, PETSC_OWN_POINTER, remotenodes, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
