#ifndef lint
static char vcid[] = "$Id: mpibdiag.c,v 1.113 1997/03/13 16:34:18 curfman Exp bsmith $";
#endif
/*
   The basic matrix operations for the Block diagonal parallel 
  matrices.
*/

#include "pinclude/pviewer.h"
#include "src/mat/impls/bdiag/mpi/mpibdiag.h"
#include "src/vec/vecimpl.h"

#undef __FUNC__  
#define __FUNC__ "MatSetValues_MPIBDiag"
int MatSetValues_MPIBDiag(Mat mat,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int          ierr, i, j, row, rstart = mbd->rstart, rend = mbd->rend;
  int          roworiented = mbd->roworiented;

  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,0,"Negative row");
    if (idxm[i] >= mbd->M) SETERRQ(1,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,0,"Negative column");
        if (idxn[j] >= mbd->N) SETERRQ(1,0,"Column too large");
        if (roworiented) {
          ierr = MatSetValues(mbd->A,1,&row,1,&idxn[j],v+i*n+j,addv); CHKERRQ(ierr);
        } else {
          ierr = MatSetValues(mbd->A,1,&row,1,&idxn[j],v+i+j*m,addv); CHKERRQ(ierr);
        }
      }
    } 
    else {
      if (roworiented) {
        ierr = StashValues_Private(&mbd->stash,idxm[i],n,idxn,v+i*n,addv); CHKERRQ(ierr);
      }
      else {
        row = idxm[i];
        for ( j=0; j<n; j++ ) {
          ierr = StashValues_Private(&mbd->stash,row,1,idxn+j,v+i+j*m,addv);CHKERRQ(ierr);
        }
      }
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetValues_MPIBDiag"
int MatGetValues_MPIBDiag(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int          ierr, i, j, row, rstart = mbd->rstart, rend = mbd->rend;

  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,0,"Negative row");
    if (idxm[i] >= mbd->M) SETERRQ(1,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,0,"Negative column");
        if (idxn[j] >= mbd->N) SETERRQ(1,0,"Column too large");
        ierr = MatGetValues(mbd->A,1,&row,1,&idxn[j],v+i*n+j); CHKERRQ(ierr);
      }
    } 
    else {
      SETERRQ(1,0,"Only local values currently supported");
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyBegin_MPIBDiag"
int MatAssemblyBegin_MPIBDiag(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  MPI_Comm     comm = mat->comm;
  int          size = mbd->size, *owners = mbd->rowners, rank = mbd->rank;
  int          *nprocs, i, j, idx, *procs, nsends, nreceives, nmax, *work;
  int          tag = mat->tag, *owner, *starts, count, ierr;
  MPI_Request  *send_waits,*recv_waits;
  InsertMode   addv;
  Scalar       *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { SETERRQ(1,0,
    "Cannot mix adds/inserts on different procs");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (mbd->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<mbd->stash.n; i++ ) {
    idx = mbd->stash.idx[i];
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
  rvalues = (Scalar *) PetscMalloc(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv(rvalues+3*nmax*i,3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PetscMalloc( 3*(mbd->stash.n+1)*sizeof(Scalar) );
  CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<mbd->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  mbd->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  mbd->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  mbd->stash.array[i];
  }
  PetscFree(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+3*starts[i],3*nprocs[i],MPIU_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PetscFree(starts); PetscFree(nprocs);

  /* Free cache space */
  PLogInfo(mat,"MatAssemblyBegin_MPIBDiag:Number of off-processor values %d\n",mbd->stash.n);
  ierr = StashDestroy_Private(&mbd->stash); CHKERRQ(ierr);

  mbd->svalues    = svalues;    mbd->rvalues = rvalues;
  mbd->nsends     = nsends;     mbd->nrecvs = nreceives;
  mbd->send_waits = send_waits; mbd->recv_waits = recv_waits;
  mbd->rmax       = nmax;

  return 0;
}
extern int MatSetUpMultiply_MPIBDiag(Mat);

#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_MPIBDiag"
int MatAssemblyEnd_MPIBDiag(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  Mat_SeqBDiag *mlocal;
  MPI_Status   *send_status, recv_status;
  int          imdex, nrecvs = mbd->nrecvs, count = nrecvs, i, n, row, col;
  int          *tmp1, *tmp2, ierr, len, ict, Mblock, Nblock;
  Scalar       *values, val;
  InsertMode   addv = mat->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,mbd->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = mbd->rvalues + 3*imdex*mbd->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PetscReal(values[3*i]) - mbd->rstart;
      col = (int) PetscReal(values[3*i+1]);
      val = values[3*i+2];
      if (col >= 0 && col < mbd->N) {
        ierr = MatSetValues(mbd->A,1,&row,1,&col,&val,addv); CHKERRQ(ierr);
      } 
      else {SETERRQ(1,0,"Invalid column");}
    }
    count--;
  }
  PetscFree(mbd->recv_waits); PetscFree(mbd->rvalues);
 
  /* wait on sends */
  if (mbd->nsends) {
    send_status = (MPI_Status *) PetscMalloc( mbd->nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(mbd->nsends,mbd->send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(mbd->send_waits); PetscFree(mbd->svalues);

  ierr = MatAssemblyBegin(mbd->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mbd->A,mode); CHKERRQ(ierr);

  /* Fix main diagonal location and determine global diagonals */
  mlocal = (Mat_SeqBDiag *) mbd->A->data;
  Mblock = mbd->M/mlocal->bs; Nblock = mbd->N/mlocal->bs;
  len    = Mblock + Nblock + 1; /* add 1 to prevent 0 malloc */
  tmp1   = (int *) PetscMalloc( 2*len*sizeof(int) ); CHKPTRQ(tmp1);
  tmp2   = tmp1 + len;
  PetscMemzero(tmp1,2*len*sizeof(int));
  mlocal->mainbd = -1; 
  for (i=0; i<mlocal->nd; i++) {
    if (mlocal->diag[i] + mbd->brstart == 0) mlocal->mainbd = i; 
    tmp1[mlocal->diag[i] + mbd->brstart + Mblock] = 1;
  }
  MPI_Allreduce(tmp1,tmp2,len,MPI_INT,MPI_SUM,mat->comm);
  ict = 0;
  for (i=0; i<len; i++) {
    if (tmp2[i]) {
      mbd->gdiag[ict] = i - Mblock;
      ict++;
    }
  }
  mbd->gnd = ict;
  PetscFree(tmp1);

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIBDiag(mat); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_MPIBDiag" /* ADIC Ignore */
int MatGetBlockSize_MPIBDiag(Mat mat,int *bs)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  Mat_SeqBDiag *dmat = (Mat_SeqBDiag *) mbd->A->data;
  *bs = dmat->bs;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_MPIBDiag"
int MatZeroEntries_MPIBDiag(Mat A)
{
  Mat_MPIBDiag *l = (Mat_MPIBDiag *) A->data;
  return MatZeroEntries(l->A);
}

/* again this uses the same basic stratagy as in the assembly and 
   scatter create routines, we should try to do it systematically 
   if we can figure out the proper level of generality. */

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   aij->A and aij->B directly and not through the MatZeroRows() 
   routine. 
*/

#undef __FUNC__  
#define __FUNC__ "MatZeroRows_MPIBDiag"
int MatZeroRows_MPIBDiag(Mat A,IS is,Scalar *diag)
{
  Mat_MPIBDiag   *l = (Mat_MPIBDiag *) A->data;
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
    if (!found) SETERRQ(1,0,"row out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) {nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce(procs,work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce(nprocs,work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
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
      MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,
                comm,send_waits+count++);
    }
  }
  PetscFree(starts);

  base = owners[rank];

  /*  wait on receives */
  lens = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count = nrecvs; slen = 0;
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
  ierr = ISCreateGeneral(MPI_COMM_SELF,slen,lrows,&istmp); CHKERRQ(ierr);  
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
#define __FUNC__ "MatMult_MPIBDiag"
int MatMult_MPIBDiag(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int          ierr;

  ierr = VecScatterBegin(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = (*mbd->A->ops.mult)(mbd->A,mbd->lvec,yy); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_MPIBDiag"
int MatMultAdd_MPIBDiag(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int          ierr;

  ierr = VecScatterBegin(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = (*mbd->A->ops.multadd)(mbd->A,mbd->lvec,yy,zz); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultTrans_MPIBDiag"
int MatMultTrans_MPIBDiag(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBDiag *a = (Mat_MPIBDiag *) A->data;
  int          ierr;
  Scalar       zero = 0.0;

  ierr = VecSet(&zero,yy); CHKERRQ(ierr);
  ierr = (*a->A->ops.multtrans)(a->A,xx,a->lvec); CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd_MPIBDiag"
int MatMultTransAdd_MPIBDiag(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBDiag *a = (Mat_MPIBDiag *) A->data;
  int          ierr;

  ierr = VecCopy(yy,zz); CHKERRQ(ierr);
  ierr = (*a->A->ops.multtrans)(a->A,xx,a->lvec); CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_MPIBDiag" /* ADIC Ignore */
int MatGetInfo_MPIBDiag(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag *) matin->data;
  Mat_SeqBDiag *dmat = (Mat_SeqBDiag *) mat->A->data;
  int          ierr;
  double       isend[5], irecv[5];

  info->rows_global    = (double)mat->M;
  info->columns_global = (double)mat->N;
  info->rows_local     = (double)mat->m;
  info->columns_local  = (double)mat->N;
  info->block_size     = (double)dmat->bs;
  ierr = MatGetInfo(mat->A,MAT_LOCAL,info);CHKERRQ(ierr);
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    MPI_Allreduce(isend,irecv,5,MPI_INT,MPI_MAX,matin->comm);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    MPI_Allreduce(isend,irecv,5,MPI_INT,MPI_SUM,matin->comm);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_MPIBDiag"
int MatGetDiagonal_MPIBDiag(Mat mat,Vec v)
{
  Mat_MPIBDiag *A = (Mat_MPIBDiag *) mat->data;
  return MatGetDiagonal(A->A,v);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_MPIBDiag" /* ADIC Ignore */
int MatDestroy_MPIBDiag(PetscObject obj)
{
  Mat          mat = (Mat) obj;
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int          ierr;

#if defined(PETSC_LOG)
  Mat_SeqBDiag *ms = (Mat_SeqBDiag *) mbd->A->data;
  PLogObjectState(obj,"Rows=%d, Cols=%d, BSize=%d, NDiag=%d",
                  mbd->M,mbd->N,ms->bs,ms->nd);
#endif
  PetscFree(mbd->rowners); 
  PetscFree(mbd->gdiag);
  ierr = MatDestroy(mbd->A); CHKERRQ(ierr);
  if (mbd->lvec) VecDestroy(mbd->lvec);
  if (mbd->Mvctx) VecScatterDestroy(mbd->Mvctx);
  PetscFree(mbd); 
  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping); CHKERRQ(ierr);
  }
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "MatView_MPIBDiag_Binary" /* ADIC Ignore */
static int MatView_MPIBDiag_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int          ierr;

  if (mbd->size == 1) {
    ierr = MatView(mbd->A,viewer); CHKERRQ(ierr);
  }
  else SETERRQ(1,0,"Only uniprocessor output supported");
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIBDiag_ASCIIorDraw" /* ADIC Ignore */
static int MatView_MPIBDiag_ASCIIorDraw(Mat mat,Viewer viewer)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  Mat_SeqBDiag *dmat = (Mat_SeqBDiag *) mbd->A->data;
  int          ierr, format, i;
  FILE         *fd;
  ViewerType   vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerGetFormat(viewer,&format);
    if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
      int nline = PetscMin(10,mbd->gnd), k, nk, np;
      PetscFPrintf(mat->comm,fd,"  block size=%d, total number of diagonals=%d\n",
                   dmat->bs,mbd->gnd);
      nk = (mbd->gnd-1)/nline + 1;
      for (k=0; k<nk; k++) {
        PetscFPrintf(mat->comm,fd,"  global diag numbers:");
        np = PetscMin(nline,mbd->gnd - nline*k);
        for (i=0; i<np; i++) 
          PetscFPrintf(mat->comm,fd,"  %d",mbd->gdiag[i+nline*k]);
        PetscFPrintf(mat->comm,fd,"\n");        
      }
      if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
        MatInfo info;
        int rank;
        MPI_Comm_rank(mat->comm,&rank);
        ierr = MatGetInfo(mat,MAT_LOCAL,&info); 
        PetscSequentialPhaseBegin(mat->comm,1);
          fprintf(fd,"[%d] local rows %d nz %d nz alloced %d mem %d \n",rank,mbd->m,
            (int)info.nz_used,(int)info.nz_allocated,(int)info.memory);       
          fflush(fd);
        PetscSequentialPhaseEnd(mat->comm,1);
        ierr = VecScatterView(mbd->Mvctx,viewer); CHKERRQ(ierr);
      }
      return 0;
    }
  }

  if (vtype == DRAW_VIEWER) {
    Draw       draw;
    PetscTruth isnull;
    ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;
  }

  if (vtype == ASCII_FILE_VIEWER) {
    PetscSequentialPhaseBegin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d\n",
             mbd->rank,mbd->m,mbd->rstart,mbd->rend,mbd->n);
    ierr = MatView(mbd->A,viewer); CHKERRQ(ierr);
    fflush(fd);
    PetscSequentialPhaseEnd(mat->comm,1);
  }
  else {
    int size = mbd->size, rank = mbd->rank; 
    if (size == 1) { 
      ierr = MatView(mbd->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat       A;
      int       M = mbd->M, N = mbd->N, m, row, nz, *cols;
      Scalar    *vals;
      Mat_SeqBDiag *Ambd = (Mat_SeqBDiag*) mbd->A->data;

      if (!rank) {
        ierr = MatCreateMPIBDiag(mat->comm,M,M,N,mbd->gnd,Ambd->bs,
               mbd->gdiag,PETSC_NULL,&A); CHKERRQ(ierr);
      }
      else {
        ierr = MatCreateMPIBDiag(mat->comm,0,M,N,0,Ambd->bs,PETSC_NULL,PETSC_NULL,&A);
               CHKERRQ(ierr);
      }
      PLogObjectParent(mat,A);

      /* Copy the matrix ... This isn't the most efficient means,
         but it's quick for now */
      row = mbd->rstart; m = Ambd->m;
      for ( i=0; i<m; i++ ) {
        ierr = MatGetRow(mat,row,&nz,&cols,&vals); CHKERRQ(ierr);
        ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatRestoreRow(mat,row,&nz,&cols,&vals); CHKERRQ(ierr);
        row++;
      } 
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      if (!rank) {
        ierr = MatView(((Mat_MPIBDiag*)(A->data))->A,viewer); CHKERRQ(ierr);
      }
      ierr = MatDestroy(A); CHKERRQ(ierr);
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIBDiag" /* ADIC Ignore */
int MatView_MPIBDiag(PetscObject obj,Viewer viewer)
{
  Mat          mat = (Mat) obj;
  int          ierr;
  ViewerType   vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER  ||  vtype == ASCII_FILES_VIEWER ||
      vtype == DRAW_VIEWER) {
    ierr = MatView_MPIBDiag_ASCIIorDraw(mat,viewer); CHKERRQ(ierr);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_MPIBDiag_Binary(mat,viewer);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_MPIBDiag" /* ADIC Ignore */
int MatSetOption_MPIBDiag(Mat A,MatOption op)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) A->data;

  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_NEW_NONZERO_LOCATION_ERROR ||
      op == MAT_NO_NEW_DIAGONALS ||
      op == MAT_YES_NEW_DIAGONALS) {
        MatSetOption(mbd->A,op);
  }
  else if (op == MAT_ROW_ORIENTED) {
    mbd->roworiented = 1;
    MatSetOption(mbd->A,op);
  }
  else if (op == MAT_COLUMN_ORIENTED) {
    mbd->roworiented = 0;
    MatSetOption(mbd->A,op);
  }
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_ROWS_UNSORTED || 
           op == MAT_COLUMNS_SORTED || 
           op == MAT_COLUMNS_UNSORTED || 
           op == MAT_SYMMETRIC ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_YES_NEW_DIAGONALS)
    PLogInfo(A,"Info:MatSetOption_MPIBDiag:Option ignored\n");
  else 
    {SETERRQ(PETSC_ERR_SUP,0,"unknown option");}
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_MPIBDiag" /* ADIC Ignore */
int MatGetSize_MPIBDiag(Mat mat,int *m,int *n)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  *m = mbd->M; *n = mbd->N;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetLocalSize_MPIBDiag" /* ADIC Ignore */
int MatGetLocalSize_MPIBDiag(Mat mat,int *m,int *n)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  *m = mbd->m; *n = mbd->N;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_MPIBDiag" /* ADIC Ignore */
int MatGetOwnershipRange_MPIBDiag(Mat matin,int *m,int *n)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag *) matin->data;
  *m = mat->rstart; *n = mat->rend;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_MPIBDiag"
int MatGetRow_MPIBDiag(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag *) matin->data;
  int          lrow;

  if (row < mat->rstart || row >= mat->rend)SETERRQ(1,0,"only for local rows")
  lrow = row - mat->rstart;
  return MatGetRow(mat->A,lrow,nz,idx,v);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_MPIBDiag" /* ADIC Ignore */
int MatRestoreRow_MPIBDiag(Mat matin,int row,int *nz,int **idx,
                                  Scalar **v)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag *) matin->data;
  int          lrow;
  lrow = row - mat->rstart;
  return MatRestoreRow(mat->A,lrow,nz,idx,v);
}


#undef __FUNC__  
#define __FUNC__ "MatNorm_MPIBDiag"
int MatNorm_MPIBDiag(Mat A,NormType type,double *norm)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) A->data;
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) mbd->A->data;
  double       sum = 0.0;
  int          ierr, d, i, nd = a->nd, bs = a->bs, len;
  Scalar       *dv;

  if (type == NORM_FROBENIUS) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      len  = a->bdlen[d]*bs*bs;
      for (i=0; i<len; i++) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(dv[i])*dv[i]);
#else
        sum += dv[i]*dv[i];
#endif
      }
    }
    MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,A->comm);
    *norm = sqrt(*norm);
    PLogFlops(2*mbd->n*mbd->m);
  }
  else if (type == NORM_1) { /* max column norm */
    double *tmp, *tmp2;
    int    j;
    tmp  = (double *) PetscMalloc( a->n*sizeof(double) ); CHKPTRQ(tmp);
    tmp2 = (double *) PetscMalloc( a->n*sizeof(double) ); CHKPTRQ(tmp2);
    ierr = MatNorm_SeqBDiag_Columns(mbd->A,tmp,a->n); CHKERRQ(ierr);
    *norm = 0.0;
    MPI_Allreduce(tmp,tmp2,a->n,MPI_DOUBLE,MPI_SUM,A->comm);
    for ( j=0; j<a->n; j++ ) {
      if (tmp2[j] > *norm) *norm = tmp2[j];
    }
    PetscFree(tmp); PetscFree(tmp2);
  }
  else if (type == NORM_INFINITY) { /* max row norm */
    double normtemp;
    ierr = MatNorm(mbd->A,type,&normtemp); CHKERRQ(ierr);
    MPI_Allreduce(&normtemp,norm,1,MPI_DOUBLE,MPI_MAX,A->comm);
  }
  return 0;
}

extern int MatPrintHelp_SeqBDiag(Mat);
#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_MPIBDiag" /* ADIC Ignore */
int MatPrintHelp_MPIBDiag(Mat A)
{
  Mat_MPIBDiag *a = (Mat_MPIBDiag*) A->data;
  if (!a->rank) return MatPrintHelp_SeqBDiag(a->A);
  else return 0;
}

extern int MatScale_SeqBDiag(Scalar*,Mat);
#undef __FUNC__  
#define __FUNC__ "MatScale_MPIBDiag"
int MatScale_MPIBDiag(Scalar *alpha,Mat A)
{
  Mat_MPIBDiag *a = (Mat_MPIBDiag*) A->data;
  return MatScale_SeqBDiag(alpha,a->A);
}

/* -------------------------------------------------------------------*/

static struct _MatOps MatOps = {MatSetValues_MPIBDiag,
       MatGetRow_MPIBDiag,MatRestoreRow_MPIBDiag,
       MatMult_MPIBDiag,MatMultAdd_MPIBDiag, 
       MatMultTrans_MPIBDiag,MatMultTransAdd_MPIBDiag, 
       0,0,0,0,
       0,0,
       0,
       0,
       MatGetInfo_MPIBDiag,0,
       MatGetDiagonal_MPIBDiag,0,MatNorm_MPIBDiag,
       MatAssemblyBegin_MPIBDiag,MatAssemblyEnd_MPIBDiag,
       0,
       MatSetOption_MPIBDiag,MatZeroEntries_MPIBDiag,MatZeroRows_MPIBDiag,
       0,0,0,0,
       MatGetSize_MPIBDiag,MatGetLocalSize_MPIBDiag,
       MatGetOwnershipRange_MPIBDiag,0,0,
       0,0,0,
       0,0,0,
       0,0,0,
       0,MatGetValues_MPIBDiag,0,
       MatPrintHelp_MPIBDiag,MatScale_MPIBDiag,
       0,0,0,MatGetBlockSize_MPIBDiag};

#undef __FUNC__  
#define __FUNC__ "MatCreateMPIBDiag"
/*@C
   MatCreateMPIBDiag - Creates a sparse parallel matrix in MPIBDiag format.

   Input Parameters:
.  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of columns (local and global)
.  nd - number of block diagonals (global) (optional)
.  bs - each element of a diagonal is an bs x bs dense matrix
.  diag - array of block diagonal numbers (length nd),
$     where for a matrix element A[i,j], 
$     where i=row and j=column, the diagonal number is
$     diag = i/bs - j/bs  (integer division)
$     Set diag=PETSC_NULL on input for PETSc to dynamically allocate
$     memory as needed.
.  diagv  - pointer to actual diagonals (in same order as diag array), 
   if allocated by user. Otherwise, set diagv=PETSC_NULL on input for PETSc
   to control memory allocation.

   Output Parameter:
.  A - the matrix 

   Notes:
   The parallel matrix is partitioned across the processors by rows, where
   each local rectangular matrix is stored in the uniprocessor block 
   diagonal format.  See the users manual for further details.

   The user MUST specify either the local or global numbers of rows
   (possibly both).

   The case bs=1 (conventional diagonal storage) is implemented as
   a special case.

   Fortran Notes:
   Fortran programmers cannot set diagv; this variable is ignored.

.keywords: matrix, block, diagonal, parallel, sparse

.seealso: MatCreate(), MatCreateSeqBDiag(), MatSetValues()
@*/
int MatCreateMPIBDiag(MPI_Comm comm,int m,int M,int N,int nd,int bs,
                     int *diag,Scalar **diagv,Mat *A)
{
  Mat          B;
  Mat_MPIBDiag *b;
  int          ierr, i, k, *ldiag, len, dset = 0, nd2,flg1,flg2;
  Scalar       **ldiagv = 0;

  *A = 0;
  if (bs == PETSC_DEFAULT) bs = 1;
  if (nd == PETSC_DEFAULT) nd = 0;
  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg1); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mat_bdiag_ndiag",&nd,&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-mat_bdiag_diags",&flg2); CHKERRQ(ierr);
  if (nd && diag == PETSC_NULL) {
    diag = (int *)PetscMalloc(nd * sizeof(int)); CHKPTRQ(diag);
    nd2 = nd; dset = 1;
    ierr = OptionsGetIntArray(PETSC_NULL,"-mat_bdiag_dvals",diag,&nd2,&flg1);CHKERRQ(ierr);
    if (nd2 != nd)
      SETERRQ(1,0,"Incompatible number of diags and diagonal vals");
  } else if (flg2) {
    SETERRQ(1,0,"Must specify number of diagonals with -mat_bdiag_ndiag");
  }

  if (bs <= 0) SETERRQ(1,0,"Blocksize must be positive");
  if ((N%bs)) SETERRQ(1,0,"Invalid block size - bad column number");
  PetscHeaderCreate(B,_Mat,MAT_COOKIE,MATMPIBDIAG,comm);
  PLogObjectCreate(B);
  B->data	= (void *) (b = PetscNew(Mat_MPIBDiag)); CHKPTRQ(b);
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy	= MatDestroy_MPIBDiag;
  B->view	= MatView_MPIBDiag;
  B->factor	= 0;
  B->mapping    = 0;

  B->insertmode = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&b->rank);
  MPI_Comm_size(comm,&b->size);

  if (M == PETSC_DECIDE) {
    if ((m%bs)) SETERRQ(1,0,"Invalid block size - bad local row number");
    MPI_Allreduce(&m,&M,1,MPI_INT,MPI_SUM,comm);
  }
  if (m == PETSC_DECIDE) {
    if ((M%bs)) SETERRQ(1,0,"Invalid block size - bad global row number");
    m = M/b->size + ((M % b->size) > b->rank);
    if ((m%bs)) SETERRQ(1,0,"Invalid block size - bad local row number");
  }
  b->M = M;    B->M = M;
  b->N = N;    B->N = N;
  b->m = m;    B->m = m;
  b->n = b->N; B->n = b->N;  /* each row stores all columns */
  b->gnd = nd;

  /* build local table of row ownerships */
  b->rowners = (int *) PetscMalloc((b->size+2)*sizeof(int)); CHKPTRQ(b->rowners);
  MPI_Allgather(&m,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);
  b->rowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart  = b->rowners[b->rank]; 
  b->rend    = b->rowners[b->rank+1]; 

  b->brstart = (b->rstart)/bs;
  b->brend   = (b->rend)/bs;

  /* Determine local diagonals; for now, assume global rows = global cols */
  /* These are sorted in MatCreateSeqBDiag */
  ldiag = (int *) PetscMalloc((nd+1)*sizeof(int)); CHKPTRQ(ldiag); 
  len = M/bs + N/bs + 1; /* add 1 to prevent 0 malloc */
  b->gdiag = (int *) PetscMalloc(len*sizeof(int)); CHKPTRQ(b->gdiag);
  k = 0;
  PLogObjectMemory(B,(nd+1)*sizeof(int) + (b->size+2)*sizeof(int)
                        + sizeof(struct _Mat) + sizeof(Mat_MPIBDiag));
  if (diagv != PETSC_NULL) {
    ldiagv = (Scalar **)PetscMalloc((nd+1)*sizeof(Scalar*)); CHKPTRQ(ldiagv); 
  }
  for (i=0; i<nd; i++) {
    b->gdiag[i] = diag[i];
    if (diag[i] > 0) { /* lower triangular */
      if (diag[i] < b->brend) {
        ldiag[k] = diag[i] - b->brstart;
        if (diagv != PETSC_NULL) ldiagv[k] = diagv[i];
        k++;
      }
    } else { /* upper triangular */
      if (b->M/bs - diag[i] > b->N/bs) {
        if (b->M/bs + diag[i] > b->brstart) {
          ldiag[k] = diag[i] - b->brstart;
          if (diagv != PETSC_NULL) ldiagv[k] = diagv[i];
          k++;
        }
      } else {
        if (b->M/bs > b->brstart) {
          ldiag[k] = diag[i] - b->brstart;
          if (diagv != PETSC_NULL) ldiagv[k] = diagv[i];
          k++;
        }
      }
    }
  }

  /* Form local matrix */
  ierr = MatCreateSeqBDiag(MPI_COMM_SELF,b->m,b->n,k,bs,ldiag,ldiagv,&b->A);CHKERRQ(ierr); 
  PLogObjectParent(B,b->A);
  PetscFree(ldiag); if (ldiagv) PetscFree(ldiagv);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&b->stash); CHKERRQ(ierr);

  /* stuff used for matrix-vector multiply */
  b->lvec        = 0;
  b->Mvctx       = 0;

  /* used for MatSetValues() input */
  b->roworiented = 1;

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1); CHKERRQ(ierr);
  if (flg1) {ierr = MatPrintHelp(B); CHKERRQ(ierr);}
  if (dset) PetscFree(diag);
  *A = B;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatBDiagGetData"
/*@C
   MatBDiagGetData - Gets the data for the block diagonal matrix format.
   For the parallel case, this returns information for the local submatrix.

   Input Parameters:
.  mat - the matrix, stored in block diagonal format.

   Output Parameters:
.  m - number of rows
.  n - number of columns
.  nd - number of block diagonals
.  bs - each element of a diagonal is an bs x bs dense matrix
.  bdlen - array of total block lengths of block diagonals
.  diag - array of block diagonal numbers,
$     where for a matrix element A[i,j], 
$     where i=row and j=column, the diagonal number is
$     diag = i/bs - j/bs  (integer division)
.  diagv - pointer to actual diagonals (in same order as diag array), 

   Notes:
   See the users manual for further details regarding this storage format.

.keywords: matrix, block, diagonal, get, data

.seealso: MatCreateSeqBDiag(), MatCreateMPIBDiag()
@*/
int MatBDiagGetData(Mat mat,int *nd,int *bs,int **diag,int **bdlen,Scalar ***diagv)
{
  Mat_MPIBDiag *pdmat;
  Mat_SeqBDiag *dmat;

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->type == MATSEQBDIAG) {
    dmat = (Mat_SeqBDiag *) mat->data;
  } else if (mat->type == MATMPIBDIAG) {
    pdmat = (Mat_MPIBDiag *) mat->data;
    dmat = (Mat_SeqBDiag *) pdmat->A->data;
  } else SETERRQ(1,0,"Valid only for MATSEQBDIAG and MATMPIBDIAG formats");
  *nd    = dmat->nd;
  *bs    = dmat->bs;
  *diag  = dmat->diag;
  *bdlen = dmat->bdlen;
  *diagv = dmat->diagv;
  return 0;
}

#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "MatLoad_MPIBDiag"
int MatLoad_MPIBDiag(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  Scalar       *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          bs, i, nz, ierr, j, rstart, rend, fd, *rowners, maxnz, *cols;
  int          header[4], rank, size, *rowlengths = 0, M, N, m,Mbs;
  int          *ourlens, *sndcounts = 0, *procsnz = 0, jj, *mycols, *smycols;
  int          tag = ((PetscObject)viewer)->tag,flg,extra_rows;

  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,BINARY_INT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(1,0,"not matrix object");
  }
  MPI_Bcast(header+1,3,MPI_INT,0,comm);
  M = header[1]; N = header[2];

  bs = 1;   /* uses a block size of 1 by default; */
  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,&flg);CHKERRQ(ierr);

  /* 
     This code adds extra rows to make sure the number of rows is 
     divisible by the blocksize
  */
  Mbs        = M/bs;
  extra_rows = bs - M + bs*(Mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  Mbs++;
  if (extra_rows && !rank) {
    PLogInfo(0,"MatLoad_MPIBDiag:Padding loaded matrix to match blocksize\n");
  }

  /* determine ownership of all rows */
  m = bs*(Mbs/size + ((Mbs % size) > rank));
  rowners = (int *) PetscMalloc((size+2)*sizeof(int)); CHKPTRQ(rowners);
  MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ourlens = (int*) PetscMalloc( (rend-rstart)*sizeof(int) ); CHKPTRQ(ourlens);
  if (!rank) {
    rowlengths = (int*) PetscMalloc( (M+extra_rows)*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);
    for ( i=0; i<extra_rows; i++ ) rowlengths[M+i] = 1;
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
      for ( j=rowners[i]; j<rowners[i+1]; j++ ) {
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
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,mycols,nz,BINARY_INT); CHKERRQ(ierr);
    if (size == 1)  for (i=0; i<extra_rows; i++) { mycols[nz+i] = M+i; }

    /* read in every one elses and ship off */
    for ( i=1; i<size-1; i++ ) {
      nz = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
      MPI_Send(cols,nz,MPI_INT,i,tag,comm);
    }
    /* read in the stuff for the last proc */
    if ( size != 1 ) {
      nz = procsnz[size-1] - extra_rows;  /* the extra rows are not on the disk */
      ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
      for ( i=0; i<extra_rows; i++ ) cols[nz+i] = M+i;
      MPI_Send(cols,nz+extra_rows,MPI_INT,size-1,tag,comm);
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

  ierr = MatCreateMPIBDiag(comm,m,M+extra_rows,N+extra_rows,PETSC_NULL,bs,PETSC_NULL,PETSC_NULL,
                           newmat); CHKERRQ(ierr);
  A = *newmat;

  if (!rank) {
    vals = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
    if (size == 1)  for (i=0; i<extra_rows; i++) { vals[nz+i] = 1.0; }   

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

    /* read in other processors (except the last one) and ship out */
    for ( i=1; i<size-1; i++ ) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);
    }
    /* the last proc */
    if ( size != 1 ){
      nz   = procsnz[i] - extra_rows;
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      for ( i=0; i<extra_rows; i++ ) vals[nz+i] = 1.0;
      MPI_Send(vals,nz+extra_rows,MPIU_SCALAR,size-1,A->tag,comm);
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







