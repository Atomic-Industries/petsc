#ifndef lint
static char vcid[] = "$Id: mpidense.c,v 1.62 1997/01/06 20:24:18 balay Exp bsmith $";
#endif

/*
   Basic functions for basic parallel dense matrices.
*/
    
#include "src/mat/impls/dense/mpi/mpidense.h"
#include "src/vec/vecimpl.h"

#undef __FUNC__  
#define __FUNC__ "MatSetValues_MPIDense"
static int MatSetValues_MPIDense(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIDense *A = (Mat_MPIDense *) mat->data;
  int          ierr, i, j, rstart = A->rstart, rend = A->rend, row;
  int          roworiented = A->roworiented;

  if (A->insertmode != NOT_SET_VALUES && A->insertmode != addv) {
    SETERRQ(1,0,"Cannot mix inserts and adds");
  }
  A->insertmode = addv;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,0,"Negative row");
    if (idxm[i] >= A->M) SETERRQ(1,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      if (roworiented) {
        ierr = MatSetValues(A->A,1,&row,n,idxn,v+i*n,addv); CHKERRQ(ierr);
      }
      else {
        for ( j=0; j<n; j++ ) {
          if (idxn[j] < 0) SETERRQ(1,0,"Negative column");
          if (idxn[j] >= A->N) SETERRQ(1,0,"Column too large");
          ierr = MatSetValues(A->A,1,&row,1,&idxn[j],v+i+j*m,addv); CHKERRQ(ierr);
        }
      }
    } 
    else {
      if (roworiented) {
        ierr = StashValues_Private(&A->stash,idxm[i],n,idxn,v+i*n,addv); CHKERRQ(ierr);
      }
      else { /* must stash each seperately */
        row = idxm[i];
        for ( j=0; j<n; j++ ) {
          ierr = StashValues_Private(&A->stash,row,1,&idxn[j],v+i+j*m,addv);CHKERRQ(ierr);
        }
      }
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetValues_MPIDense"
static int MatGetValues_MPIDense(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr, i, j, rstart = mdn->rstart, rend = mdn->rend, row;

  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,0,"Negative row");
    if (idxm[i] >= mdn->M) SETERRQ(1,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,0,"Negative column");
        if (idxn[j] >= mdn->N) 
          SETERRQ(1,0,"Column too large");
        ierr = MatGetValues(mdn->A,1,&row,1,&idxn[j],v+i*n+j); CHKERRQ(ierr);
      }
    } 
    else {
      SETERRQ(1,0,"Only local values currently supported");
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetArray_MPIDense"
static int MatGetArray_MPIDense(Mat A,Scalar **array)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  int ierr;

  ierr = MatGetArray(a->A,array); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreArray_MPIDense"
static int MatRestoreArray_MPIDense(Mat A,Scalar **array)
{
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyBegin_MPIDense"
static int MatAssemblyBegin_MPIDense(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  MPI_Comm     comm = mat->comm;
  int          size = mdn->size, *owners = mdn->rowners, rank = mdn->rank;
  int          *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int          tag = mat->tag, *owner,*starts,count,ierr;
  InsertMode   addv;
  MPI_Request  *send_waits,*recv_waits;
  Scalar       *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce(&mdn->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { 
    SETERRQ(1,0,"Cannot mix adds/inserts on different procs");
  }
  mdn->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (mdn->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<mdn->stash.n; i++ ) {
    idx = mdn->stash.idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce(procs,work,size,MPI_INT,MPI_SUM,comm);
  nreceives = work[rank]; 
  if (nreceives > size) SETERRQ(1,0,"Internal PETSc error");
  MPI_Allreduce(nprocs,work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simplify the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.

       This could be done better.
  */
  rvalues = (Scalar *) PetscMalloc(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv(rvalues+3*nmax*i,3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PetscMalloc( 3*(mdn->stash.n+1)*sizeof(Scalar));CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<mdn->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  mdn->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  mdn->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  mdn->stash.array[i];
  }
  PetscFree(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+3*starts[i],3*nprocs[i],MPIU_SCALAR,i,tag,comm,send_waits+count++);
    }
  }
  PetscFree(starts); PetscFree(nprocs);

  /* Free cache space */
  PLogInfo(mat,"MatAssemblyBegin_MPIDense:Number of off-processor values %d\n",mdn->stash.n);
  ierr = StashDestroy_Private(&mdn->stash); CHKERRQ(ierr);

  mdn->svalues    = svalues;    mdn->rvalues = rvalues;
  mdn->nsends     = nsends;     mdn->nrecvs = nreceives;
  mdn->send_waits = send_waits; mdn->recv_waits = recv_waits;
  mdn->rmax       = nmax;

  return 0;
}
extern int MatSetUpMultiply_MPIDense(Mat);

#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_MPIDense"
static int MatAssemblyEnd_MPIDense(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  MPI_Status   *send_status,recv_status;
  int          imdex, nrecvs=mdn->nrecvs, count=nrecvs, i, n, ierr, row, col;
  Scalar       *values,val;
  InsertMode   addv = mdn->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,mdn->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = mdn->rvalues + 3*imdex*mdn->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PetscReal(values[3*i]) - mdn->rstart;
      col = (int) PetscReal(values[3*i+1]);
      val = values[3*i+2];
      if (col >= 0 && col < mdn->N) {
        MatSetValues(mdn->A,1,&row,1,&col,&val,addv);
      } 
      else {SETERRQ(1,0,"Invalid column");}
    }
    count--;
  }
  PetscFree(mdn->recv_waits); PetscFree(mdn->rvalues);
 
  /* wait on sends */
  if (mdn->nsends) {
    send_status = (MPI_Status *) PetscMalloc(mdn->nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    MPI_Waitall(mdn->nsends,mdn->send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(mdn->send_waits); PetscFree(mdn->svalues);

  mdn->insertmode = NOT_SET_VALUES;
  ierr = MatAssemblyBegin(mdn->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mdn->A,mode); CHKERRQ(ierr);

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIDense(mat); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_MPIDense"
static int MatZeroEntries_MPIDense(Mat A)
{
  Mat_MPIDense *l = (Mat_MPIDense *) A->data;
  return MatZeroEntries(l->A);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_MPIDense"
static int MatGetBlockSize_MPIDense(Mat A,int *bs)
{
  *bs = 1;
  return 0;
}

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   mdn->A and mdn->B directly and not through the MatZeroRows() 
   routine. 
*/
#undef __FUNC__  
#define __FUNC__ "MatZeroRows_MPIDense"
static int MatZeroRows_MPIDense(Mat A,IS is,Scalar *diag)
{
  Mat_MPIDense   *l = (Mat_MPIDense *) A->data;
  int            i,ierr,N, *rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  ierr = ISGetSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc((N+1)*sizeof(int)); CHKPTRQ(owner); /* see note*/
  for ( i=0; i<N; i++ ) {
    idx = rows[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,0,"Index out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues    = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts     = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0]  = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<N; i++ ) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  ISRestoreIndices(is,&rows);

  starts[0] = 0;
  for ( i=1; i<size+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);
    }
  }
  PetscFree(starts);

  base = owners[rank];

  /*  wait on receives */
  lens   = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]  = n;
    slen += n;
    count--;
  }
  PetscFree(recv_waits); 
  
  /* move the data into the send scatter */
  lrows = (int *) PetscMalloc( (slen+1)*sizeof(int) ); CHKPTRQ(lrows);
  count = 0;
  for ( i=0; i<nrecvs; i++ ) {
    values = rvalues + i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      lrows[count++] = values[j] - base;
    }
  }
  PetscFree(rvalues); PetscFree(lens);
  PetscFree(owner); PetscFree(nprocs);
    
  /* actually zap the local rows */
  ierr = ISCreateGeneral(MPI_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PLogObjectParent(A,istmp);
  PetscFree(lrows);
  ierr = MatZeroRows(l->A,istmp,diag); CHKERRQ(ierr);
  ierr = ISDestroy(istmp); CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(send_waits); PetscFree(svalues);

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMult_MPIDense"
static int MatMult_MPIDense(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;

  ierr = VecScatterBegin(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = MatMult_SeqDense(mdn->A,mdn->lvec,yy); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_MPIDense"
static int MatMultAdd_MPIDense(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;

  ierr = VecScatterBegin(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = MatMultAdd_SeqDense(mdn->A,mdn->lvec,yy,zz); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultTrans_MPIDense"
static int MatMultTrans_MPIDense(Mat A,Vec xx,Vec yy)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  int          ierr;
  Scalar       zero = 0.0;

  ierr = VecSet(&zero,yy); CHKERRQ(ierr);
  ierr = MatMultTrans_SeqDense(a->A,xx,a->lvec); CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd_MPIDense"
static int MatMultTransAdd_MPIDense(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  int          ierr;

  ierr = VecCopy(yy,zz); CHKERRQ(ierr);
  ierr = MatMultTrans_SeqDense(a->A,xx,a->lvec); CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_MPIDense"
static int MatGetDiagonal_MPIDense(Mat A,Vec v)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  Mat_SeqDense *aloc = (Mat_SeqDense *) a->A->data;
  int          ierr, len, i, n, m = a->m, radd;
  Scalar       *x, zero = 0.0;
  
  VecSet(&zero,v);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  ierr = VecGetSize(v,&n); CHKERRQ(ierr);
  if (n != a->M) SETERRQ(1,0,"Nonconforming mat and vec");
  len = PetscMin(aloc->m,aloc->n);
  radd = a->rstart*m;
  for ( i=0; i<len; i++ ) {
    x[i] = aloc->v[radd + i*m + i];
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_MPIDense"
static int MatDestroy_MPIDense(PetscObject obj)
{
  Mat          mat = (Mat) obj;
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d",mdn->M,mdn->N);
#endif
  PetscFree(mdn->rowners); 
  ierr = MatDestroy(mdn->A); CHKERRQ(ierr);
  if (mdn->lvec)   VecDestroy(mdn->lvec);
  if (mdn->Mvctx)  VecScatterDestroy(mdn->Mvctx);
  if (mdn->factor) {
    if (mdn->factor->temp)   PetscFree(mdn->factor->temp);
    if (mdn->factor->tag)    PetscFree(mdn->factor->tag);
    if (mdn->factor->pivots) PetscFree(mdn->factor->pivots);
    PetscFree(mdn->factor);
  }
  PetscFree(mdn); 
  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping); CHKERRQ(ierr);
  }
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  return 0;
}

#include "pinclude/pviewer.h"

#undef __FUNC__  
#define __FUNC__ "MatView_MPIDense_Binary"
static int MatView_MPIDense_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;

  if (mdn->size == 1) {
    ierr = MatView(mdn->A,viewer); CHKERRQ(ierr);
  }
  else SETERRQ(1,0,"Only uniprocessor output supported");
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIDense_ASCII"
static int MatView_MPIDense_ASCII(Mat mat,Viewer viewer)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr, format;
  FILE         *fd;
  ViewerType   vtype;

  ViewerGetType(viewer,&vtype);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    int rank;
    MatInfo info;
    MPI_Comm_rank(mat->comm,&rank);
    ierr = MatGetInfo(mat,MAT_LOCAL,&info);
    PetscSequentialPhaseBegin(mat->comm,1);
      fprintf(fd,"  [%d] local rows %d nz %d nz alloced %d mem %d \n",rank,mdn->m,
         (int)info.nz_used,(int)info.nz_allocated,(int)info.memory);       
      fflush(fd);
    PetscSequentialPhaseEnd(mat->comm,1);
    ierr = VecScatterView(mdn->Mvctx,viewer); CHKERRQ(ierr);
    return 0; 
  }
  else if (format == VIEWER_FORMAT_ASCII_INFO) {
    return 0;
  }
  if (vtype == ASCII_FILE_VIEWER) {
    PetscSequentialPhaseBegin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d\n",
             mdn->rank,mdn->m,mdn->rstart,mdn->rend,mdn->n);
    ierr = MatView(mdn->A,viewer); CHKERRQ(ierr);
    fflush(fd);
    PetscSequentialPhaseEnd(mat->comm,1);
  }
  else {
    int size = mdn->size, rank = mdn->rank; 
    if (size == 1) { 
      ierr = MatView(mdn->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat          A;
      int          M = mdn->M, N = mdn->N,m,row,i, nz, *cols;
      Scalar       *vals;
      Mat_SeqDense *Amdn = (Mat_SeqDense*) mdn->A->data;

      if (!rank) {
        ierr = MatCreateMPIDense(mat->comm,M,N,M,N,PETSC_NULL,&A); CHKERRQ(ierr);
      }
      else {
        ierr = MatCreateMPIDense(mat->comm,0,N,M,N,PETSC_NULL,&A); CHKERRQ(ierr);
      }
      PLogObjectParent(mat,A);

      /* Copy the matrix ... This isn't the most efficient means,
         but it's quick for now */
      row = mdn->rstart; m = Amdn->m;
      for ( i=0; i<m; i++ ) {
        ierr = MatGetRow(mat,row,&nz,&cols,&vals); CHKERRQ(ierr);
        ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatRestoreRow(mat,row,&nz,&cols,&vals); CHKERRQ(ierr);
        row++;
      } 

      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      if (!rank) {
        ierr = MatView(((Mat_MPIDense*)(A->data))->A,viewer); CHKERRQ(ierr);
      }
      ierr = MatDestroy(A); CHKERRQ(ierr);
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIDense"
static int MatView_MPIDense(PetscObject obj,Viewer viewer)
{
  Mat          mat = (Mat) obj;
  int          ierr;
  ViewerType   vtype;
 
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER) {
    ierr = MatView_MPIDense_ASCII(mat,viewer); CHKERRQ(ierr);
  }
  else if (vtype == ASCII_FILES_VIEWER) {
    ierr = MatView_MPIDense_ASCII(mat,viewer); CHKERRQ(ierr);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_MPIDense_Binary(mat,viewer);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_MPIDense"
static int MatGetInfo_MPIDense(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  Mat          mdn = mat->A;
  int          ierr;
  double       isend[5], irecv[5];

  info->rows_global    = (double)mat->M;
  info->columns_global = (double)mat->N;
  info->rows_local     = (double)mat->m;
  info->columns_local  = (double)mat->N;
  info->block_size     = 1.0;
  ierr = MatGetInfo(mdn,MAT_LOCAL,info); CHKERRQ(ierr);
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    MPI_Allreduce(isend,irecv,3,MPI_INT,MPI_MAX,A->comm);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    MPI_Allreduce(isend,irecv,3,MPI_INT,MPI_SUM,A->comm);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  return 0;
}

/* extern int MatLUFactorSymbolic_MPIDense(Mat,IS,IS,double,Mat*);
   extern int MatLUFactorNumeric_MPIDense(Mat,Mat*);
   extern int MatLUFactor_MPIDense(Mat,IS,IS,double);
   extern int MatSolve_MPIDense(Mat,Vec,Vec);
   extern int MatSolveAdd_MPIDense(Mat,Vec,Vec,Vec);
   extern int MatSolveTrans_MPIDense(Mat,Vec,Vec);
   extern int MatSolveTransAdd_MPIDense(Mat,Vec,Vec,Vec); */

#undef __FUNC__  
#define __FUNC__ "MatSetOption_MPIDense"
static int MatSetOption_MPIDense(Mat A,MatOption op)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;

  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_COLUMNS_SORTED ||
      op == MAT_COLUMNS_UNSORTED) {
        MatSetOption(a->A,op);
  } else if (op == MAT_ROW_ORIENTED) {
        a->roworiented = 1;
        MatSetOption(a->A,op);
  } else if (op == MAT_ROWS_SORTED || 
             op == MAT_ROWS_UNSORTED ||
             op == MAT_SYMMETRIC ||
             op == MAT_STRUCTURALLY_SYMMETRIC ||
             op == MAT_YES_NEW_DIAGONALS)
    PLogInfo(A,"Info:MatSetOption_MPIDense:Option ignored\n");
  else if (op == MAT_COLUMN_ORIENTED)
    {a->roworiented = 0; MatSetOption(a->A,op);} 
  else if (op == MAT_NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,0,"MAT_NO_NEW_DIAGONALS");}
  else 
    {SETERRQ(PETSC_ERR_SUP,0,"unknown option");}
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_MPIDense"
static int MatGetSize_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  *m = mat->M; *n = mat->N;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetLocalSize_MPIDense"
static int MatGetLocalSize_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  *m = mat->m; *n = mat->N;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_MPIDense"
static int MatGetOwnershipRange_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  *m = mat->rstart; *n = mat->rend;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_MPIDense"
static int MatGetRow_MPIDense(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  int          lrow, rstart = mat->rstart, rend = mat->rend;

  if (row < rstart || row >= rend) SETERRQ(1,0,"only local rows")
  lrow = row - rstart;
  return MatGetRow(mat->A,lrow,nz,idx,v);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_MPIDense"
static int MatRestoreRow_MPIDense(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  if (idx) PetscFree(*idx);
  if (v) PetscFree(*v);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_MPIDense"
static int MatNorm_MPIDense(Mat A,NormType type,double *norm)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) A->data;
  Mat_SeqDense *mat = (Mat_SeqDense*) mdn->A->data;
  int          ierr, i, j;
  double       sum = 0.0;
  Scalar       *v = mat->v;

  if (mdn->size == 1) {
    ierr =  MatNorm(mdn->A,type,norm); CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      for (i=0; i<mat->n*mat->m; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,A->comm);
      *norm = sqrt(*norm);
      PLogFlops(2*mat->n*mat->m);
    }
    else if (type == NORM_1) { 
      double *tmp, *tmp2;
      tmp  = (double *) PetscMalloc( 2*mdn->N*sizeof(double) ); CHKPTRQ(tmp);
      tmp2 = tmp + mdn->N;
      PetscMemzero(tmp,2*mdn->N*sizeof(double));
      *norm = 0.0;
      v = mat->v;
      for ( j=0; j<mat->n; j++ ) {
        for ( i=0; i<mat->m; i++ ) {
          tmp[j] += PetscAbsScalar(*v);  v++;
        }
      }
      MPI_Allreduce(tmp,tmp2,mdn->N,MPI_DOUBLE,MPI_SUM,A->comm);
      for ( j=0; j<mdn->N; j++ ) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      PetscFree(tmp);
      PLogFlops(mat->n*mat->m);
    }
    else if (type == NORM_INFINITY) { /* max row norm */
      double ntemp;
      ierr = MatNorm(mdn->A,type,&ntemp); CHKERRQ(ierr);
      MPI_Allreduce(&ntemp,norm,1,MPI_DOUBLE,MPI_MAX,A->comm);
    }
    else {
      SETERRQ(1,0,"No support for two norm");
    }
  }
  return 0; 
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_MPIDense"
static int MatTranspose_MPIDense(Mat A,Mat *matout)
{ 
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  Mat_SeqDense *Aloc = (Mat_SeqDense *) a->A->data;
  Mat          B;
  int          M = a->M, N = a->N, m, n, *rwork, rstart = a->rstart;
  int          j, i, ierr;
  Scalar       *v;

  if (matout == PETSC_NULL && M != N) {
    SETERRQ(1,0,"Supports square matrix only in-place");
  }
  ierr = MatCreateMPIDense(A->comm,PETSC_DECIDE,PETSC_DECIDE,N,M,PETSC_NULL,&B);CHKERRQ(ierr);

  m = Aloc->m; n = Aloc->n; v = Aloc->v;
  rwork = (int *) PetscMalloc(n*sizeof(int)); CHKPTRQ(rwork);
  for ( j=0; j<n; j++ ) {
    for (i=0; i<m; i++) rwork[i] = rstart + i;
    ierr = MatSetValues(B,1,&j,m,rwork,v,INSERT_VALUES); CHKERRQ(ierr);
    v   += m;
  } 
  PetscFree(rwork);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (matout != PETSC_NULL) {
    *matout = B;
  } else {
    /* This isn't really an in-place transpose, but free data struct from a */
    PetscFree(a->rowners); 
    ierr = MatDestroy(a->A); CHKERRQ(ierr);
    if (a->lvec) VecDestroy(a->lvec);
    if (a->Mvctx) VecScatterDestroy(a->Mvctx);
    PetscFree(a); 
    PetscMemcpy(A,B,sizeof(struct _Mat)); 
    PetscHeaderDestroy(B);
  }
  return 0;
}

#include "pinclude/plapack.h"
#undef __FUNC__  
#define __FUNC__ "MatScale_MPIDense"
static int MatScale_MPIDense(Scalar *alpha,Mat inA)
{
  Mat_MPIDense *A = (Mat_MPIDense *) inA->data;
  Mat_SeqDense *a = (Mat_SeqDense *) A->A->data;
  int          one = 1, nz;

  nz = a->m*a->n;
  BLscal_( &nz, alpha, a->v, &one );
  PLogFlops(nz);
  return 0;
}

static int MatConvertSameType_MPIDense(Mat,Mat *,int);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_MPIDense,
       MatGetRow_MPIDense,MatRestoreRow_MPIDense,
       MatMult_MPIDense,MatMultAdd_MPIDense,
       MatMultTrans_MPIDense,MatMultTransAdd_MPIDense,
/*       MatSolve_MPIDense,0, */
       0,0,
       0,0,
       0,0,
/*       MatLUFactor_MPIDense,0, */
       0,MatTranspose_MPIDense,
       MatGetInfo_MPIDense,0,
       MatGetDiagonal_MPIDense,0,MatNorm_MPIDense,
       MatAssemblyBegin_MPIDense,MatAssemblyEnd_MPIDense,
       0,
       MatSetOption_MPIDense,MatZeroEntries_MPIDense,MatZeroRows_MPIDense,
       0,0,
/*       0,MatLUFactorSymbolic_MPIDense,MatLUFactorNumeric_MPIDense, */
       0,0,
       MatGetSize_MPIDense,MatGetLocalSize_MPIDense,
       MatGetOwnershipRange_MPIDense,
       0,0, MatGetArray_MPIDense, MatRestoreArray_MPIDense,
       0,MatConvertSameType_MPIDense,
       0,0,0,0,0,
       0,0,MatGetValues_MPIDense,0,0,MatScale_MPIDense,
       0,0,0,MatGetBlockSize_MPIDense};

#undef __FUNC__  
#define __FUNC__ "MatCreateMPIDense"
/*@C
   MatCreateMPIDense - Creates a sparse parallel matrix in dense format.

   Input Parameters:
.  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated 
           if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated 
           if n is given)
.  data - optional location of matrix data.  Set data=PETSC_NULL for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:
   The dense format is fully compatible with standard Fortran 77
   storage by columns.

   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=PETSC_NULL.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Currently, the only parallel dense matrix decomposition is by rows,
   so that n=N and each submatrix owns all of the global columns.

.keywords: matrix, dense, parallel

.seealso: MatCreate(), MatCreateSeqDense(), MatSetValues()
@*/
int MatCreateMPIDense(MPI_Comm comm,int m,int n,int M,int N,Scalar *data,Mat *A)
{
  Mat          mat;
  Mat_MPIDense *a;
  int          ierr, i,flg;

  /* Note:  For now, when data is specified above, this assumes the user correctly
   allocates the local dense storage space.  We should add error checking. */

  *A = 0;
  PetscHeaderCreate(mat,_Mat,MAT_COOKIE,MATMPIDENSE,comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (a = PetscNew(Mat_MPIDense)); CHKPTRQ(a);
  PetscMemcpy(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy    = MatDestroy_MPIDense;
  mat->view       = MatView_MPIDense;
  mat->factor     = 0;
  mat->mapping    = 0;

  a->factor       = 0;
  a->insertmode   = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&a->rank);
  MPI_Comm_size(comm,&a->size);

  if (M == PETSC_DECIDE) MPI_Allreduce(&m,&M,1,MPI_INT,MPI_SUM,comm);
  if (m == PETSC_DECIDE) {m = M/a->size + ((M % a->size) > a->rank);}

  /* each row stores all columns */
  if (N == PETSC_DECIDE) N = n;
  if (n == PETSC_DECIDE) {n = N/a->size + ((N % a->size) > a->rank);}
  /*  if (n != N) SETERRQ(1,0,"For now, only n=N is supported"); */
  a->N = mat->N = N;
  a->M = mat->M = M;
  a->m = mat->m = m;
  a->n = mat->n = n;

  /* build local table of row and column ownerships */
  a->rowners = (int *) PetscMalloc(2*(a->size+2)*sizeof(int)); CHKPTRQ(a->rowners);
  a->cowners = a->rowners + a->size + 1;
  PLogObjectMemory(mat,2*(a->size+2)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_MPIDense));
  MPI_Allgather(&m,1,MPI_INT,a->rowners+1,1,MPI_INT,comm);
  a->rowners[0] = 0;
  for ( i=2; i<=a->size; i++ ) {
    a->rowners[i] += a->rowners[i-1];
  }
  a->rstart = a->rowners[a->rank]; 
  a->rend   = a->rowners[a->rank+1]; 
  MPI_Allgather(&n,1,MPI_INT,a->cowners+1,1,MPI_INT,comm);
  a->cowners[0] = 0;
  for ( i=2; i<=a->size; i++ ) {
    a->cowners[i] += a->cowners[i-1];
  }

  ierr = MatCreateSeqDense(MPI_COMM_SELF,m,N,data,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&a->stash); CHKERRQ(ierr);

  /* stuff used for matrix vector multiply */
  a->lvec        = 0;
  a->Mvctx       = 0;
  a->roworiented = 1;

  *A = mat;
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatPrintHelp(mat); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatConvertSameType_MPIDense"
static int MatConvertSameType_MPIDense(Mat A,Mat *newmat,int cpvalues)
{
  Mat          mat;
  Mat_MPIDense *a,*oldmat = (Mat_MPIDense *) A->data;
  int          ierr;
  FactorCtx    *factor;

  *newmat       = 0;
  PetscHeaderCreate(mat,_Mat,MAT_COOKIE,MATMPIDENSE,A->comm);
  PLogObjectCreate(mat);
  mat->data      = (void *) (a = PetscNew(Mat_MPIDense)); CHKPTRQ(a);
  PetscMemcpy(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy   = MatDestroy_MPIDense;
  mat->view      = MatView_MPIDense;
  mat->factor    = A->factor;
  mat->assembled = PETSC_TRUE;

  a->m = mat->m = oldmat->m;
  a->n = mat->n = oldmat->n;
  a->M = mat->M = oldmat->M;
  a->N = mat->N = oldmat->N;
  if (oldmat->factor) {
    a->factor = (FactorCtx *) (factor = PetscNew(FactorCtx)); CHKPTRQ(factor);
    /* copy factor contents ... add this code! */
  } else a->factor = 0;

  a->rstart     = oldmat->rstart;
  a->rend       = oldmat->rend;
  a->size       = oldmat->size;
  a->rank       = oldmat->rank;
  a->insertmode = NOT_SET_VALUES;

  a->rowners = (int *) PetscMalloc((a->size+1)*sizeof(int)); CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,(a->size+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_MPIDense));
  PetscMemcpy(a->rowners,oldmat->rowners,(a->size+1)*sizeof(int));
  ierr = StashInitialize_Private(&a->stash); CHKERRQ(ierr);
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec); CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatConvert(oldmat->A,MATSAME,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  *newmat = mat;
  return 0;
}

#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "MatLoad_MPIDense_DenseInFile"
int MatLoad_MPIDense_DenseInFile(MPI_Comm comm,int fd,int M, int N, Mat *newmat)
{
  int        *rowners, i,size,rank,m,rstart,rend,ierr,nz,j;
  Scalar     *array,*vals,*vals_ptr;
  MPI_Status status;

  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  /* determine ownership of all rows */
  m = M/size + ((M % size) > rank);
  rowners = (int *) PetscMalloc((size+2)*sizeof(int)); CHKPTRQ(rowners);
  MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  ierr = MatCreateMPIDense(comm,m,PETSC_DECIDE,M,N,PETSC_NULL,newmat);CHKERRQ(ierr);
  ierr = MatGetArray(*newmat,&array); CHKERRQ(ierr);

  if (!rank) {
    vals = (Scalar *) PetscMalloc( m*N*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    ierr = PetscBinaryRead(fd,vals,m*N,BINARY_SCALAR); CHKERRQ(ierr);
    
    /* insert into matrix-by row (this is why cannot directly read into array */
    vals_ptr = vals;
    for ( i=0; i<m; i++ ) {
      for ( j=0; j<N; j++ ) {
        array[i + j*m] = *vals_ptr++;
      }
    }

    /* read in other processors and ship out */
    for ( i=1; i<size; i++ ) {
      nz   = (rowners[i+1] - rowners[i])*N;
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      MPI_Send(vals,nz,MPIU_SCALAR,i,(*newmat)->tag,comm);
    }
  }
  else {
    /* receive numeric values */
    vals = (Scalar*) PetscMalloc( m*N*sizeof(Scalar) ); CHKPTRQ(vals);

    /* receive message of values*/
    MPI_Recv(vals,m*N,MPIU_SCALAR,0,(*newmat)->tag,comm,&status);

    /* insert into matrix-by row (this is why cannot directly read into array */
    vals_ptr = vals;
    for ( i=0; i<m; i++ ) {
      for ( j=0; j<N; j++ ) {
        array[i + j*m] = *vals_ptr++;
      }
    }
  }
  PetscFree(rowners);
  PetscFree(vals);
  ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "MatLoad_MPIDense"
int MatLoad_MPIDense(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  int          i, nz, ierr, j,rstart, rend, fd;
  Scalar       *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0, *offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)viewer)->tag;

  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,BINARY_INT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(1,0,"not matrix object");
  }

  MPI_Bcast(header+1,3,MPI_INT,0,comm);
  M = header[1]; N = header[2]; nz = header[3];

  /*
       Handle case where matrix is stored on disk as a dense matrix 
  */
  if (nz == MATRIX_BINARY_FORMAT_DENSE) {
    return MatLoad_MPIDense_DenseInFile(comm,fd,M,N,newmat);
  }

  /* determine ownership of all rows */
  m = M/size + ((M % size) > rank);
  rowners = (int *) PetscMalloc((size+2)*sizeof(int)); CHKPTRQ(rowners);
  MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ourlens = (int*) PetscMalloc( 2*(rend-rstart)*sizeof(int) ); CHKPTRQ(ourlens);
  offlens = ourlens + (rend-rstart);
  if (!rank) {
    rowlengths = (int*) PetscMalloc( M*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);
    sndcounts = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = rowners[i+1] - rowners[i];
    MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);
    PetscFree(sndcounts);
  }
  else {
    MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT, 0,comm);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(procsnz);
    PetscMemzero(procsnz,size*sizeof(int));
    for ( i=0; i<size; i++ ) {
      for ( j=rowners[i]; j< rowners[i+1]; j++ ) {
        procsnz[i] += rowlengths[j];
      }
    }
    PetscFree(rowlengths);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for ( i=0; i<size; i++ ) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    cols = (int *) PetscMalloc( maxnz*sizeof(int) ); CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    mycols = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(mycols);
    ierr = PetscBinaryRead(fd,mycols,nz,BINARY_INT); CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for ( i=1; i<size; i++ ) {
      nz = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
      MPI_Send(cols,nz,MPI_INT,i,tag,comm);
    }
    PetscFree(cols);
  }
  else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<m; i++ ) {
      nz += ourlens[i];
    }
    mycols = (int*) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(mycols);

    /* receive message of column indices*/
    MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);
    MPI_Get_count(&status,MPI_INT,&maxnz);
    if (maxnz != nz) SETERRQ(1,0,"something is wrong with file");
  }

  /* loop over local rows, determining number of off diagonal entries */
  PetscMemzero(offlens,m*sizeof(int));
  jj = 0;
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<ourlens[i]; j++ ) {
      if (mycols[jj] < rstart || mycols[jj] >= rend) offlens[i]++;
      jj++;
    }
  }

  /* create our matrix */
  for ( i=0; i<m; i++ ) {
    ourlens[i] -= offlens[i];
  }
  ierr = MatCreateMPIDense(comm,m,PETSC_DECIDE,M,N,PETSC_NULL,newmat);CHKERRQ(ierr);
  A = *newmat;
  for ( i=0; i<m; i++ ) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    vals = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
    
    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }

    /* read in other processors and ship out */
    for ( i=1; i<size; i++ ) {
      nz = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);
    }
    PetscFree(procsnz);
  }
  else {
    /* receive numeric values */
    vals = (Scalar*) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* receive message of values*/
    MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);
    MPI_Get_count(&status,MPIU_SCALAR,&maxnz);
    if (maxnz != nz) SETERRQ(1,0,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }
  }
  PetscFree(ourlens); PetscFree(vals); PetscFree(mycols); PetscFree(rowners);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}





