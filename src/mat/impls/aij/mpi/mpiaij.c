
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpiaij.c,v 1.251 1998/06/19 15:56:02 bsmith Exp balay $";
#endif

#include "pinclude/pviewer.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"

/* local utility routine that creates a mapping from the global column 
number to the local number in the off-diagonal part of the local 
storage of the matrix.  This is done in a non scable way since the 
length of colmap equals the global matrix length. 
*/
#undef __FUNC__  
#define __FUNC__ "CreateColmap_MPIAIJ_Private"
int CreateColmap_MPIAIJ_Private(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *B = (Mat_SeqAIJ*) aij->B->data;
  int        n = B->n,i;

  PetscFunctionBegin;
  aij->colmap = (int *) PetscMalloc((aij->N+1)*sizeof(int));CHKPTRQ(aij->colmap);
  PLogObjectMemory(mat,aij->N*sizeof(int));
  PetscMemzero(aij->colmap,aij->N*sizeof(int));
  for ( i=0; i<n; i++ ) aij->colmap[aij->garray[i]] = i+1;
  PetscFunctionReturn(0);
}

extern int DisAssemble_MPIAIJ(Mat);

#define CHUNKSIZE   15
#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv) \
{ \
 \
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift; \
    rmax = aimax[row]; nrow = ailen[row];  \
    col1 = col - shift; \
     \
    low = 0; high = nrow; \
    while (high-low > 5) { \
      t = (low+high)/2; \
      if (rp[t] > col) high = t; \
      else             low  = t; \
    } \
      for ( _i=0; _i<nrow; _i++ ) { \
        if (rp[_i] > col1) break; \
        if (rp[_i] == col1) { \
          if (addv == ADD_VALUES) ap[_i] += value;   \
          else                  ap[_i] = value; \
          goto a_noinsert; \
        } \
      }  \
      if (nonew == 1) goto a_noinsert; \
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero into matrix"); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int    new_nz = ai[a->m] + CHUNKSIZE,len,*new_i,*new_j; \
        Scalar *new_a; \
 \
        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix"); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+sizeof(Scalar))+(a->m+1)*sizeof(int); \
        new_a   = (Scalar *) PetscMalloc( len ); CHKPTRQ(new_a); \
        new_j   = (int *) (new_a + new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = ai[ii];} \
        for ( ii=row+1; ii<a->m+1; ii++ ) {new_i[ii] = ai[ii]+CHUNKSIZE;} \
        PetscMemcpy(new_j,aj,(ai[row]+nrow+shift)*sizeof(int)); \
        len = (new_nz - CHUNKSIZE - ai[row] - nrow - shift); \
        PetscMemcpy(new_j+ai[row]+shift+nrow+CHUNKSIZE,aj+ai[row]+shift+nrow, \
                                                           len*sizeof(int)); \
        PetscMemcpy(new_a,aa,(ai[row]+nrow+shift)*sizeof(Scalar)); \
        PetscMemcpy(new_a+ai[row]+shift+nrow+CHUNKSIZE,aa+ai[row]+shift+nrow, \
                                                           len*sizeof(Scalar));  \
        /* free up old matrix storage */ \
 \
        PetscFree(a->a);  \
        if (!a->singlemalloc) {PetscFree(a->i);PetscFree(a->j);} \
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j;  \
        a->singlemalloc = 1; \
 \
        rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift; \
        rmax = aimax[row] = aimax[row] + CHUNKSIZE; \
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + sizeof(Scalar))); \
        a->maxnz += CHUNKSIZE; \
        a->reallocs++; \
      } \
      N = nrow++ - 1; a->nz++; \
      /* shift up all the later entries in this row */ \
      for ( ii=N; ii>=_i; ii-- ) { \
        rp[ii+1] = rp[ii]; \
        ap[ii+1] = ap[ii]; \
      } \
      rp[_i] = col1;  \
      ap[_i] = value;  \
      a_noinsert: ; \
      ailen[row] = nrow; \
} 

#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv) \
{ \
 \
    rp   = bj + bi[row] + shift; ap = ba + bi[row] + shift; \
    rmax = bimax[row]; nrow = bilen[row];  \
    col1 = col - shift; \
     \
    low = 0; high = nrow; \
    while (high-low > 5) { \
      t = (low+high)/2; \
      if (rp[t] > col) high = t; \
      else             low  = t; \
    } \
       for ( _i=0; _i<nrow; _i++ ) { \
        if (rp[_i] > col1) break; \
        if (rp[_i] == col1) { \
          if (addv == ADD_VALUES) ap[_i] += value;   \
          else                  ap[_i] = value; \
          goto b_noinsert; \
        } \
      }  \
      if (nonew == 1) goto b_noinsert; \
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero into matrix"); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int    new_nz = bi[b->m] + CHUNKSIZE,len,*new_i,*new_j; \
        Scalar *new_a; \
 \
        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix"); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+sizeof(Scalar))+(b->m+1)*sizeof(int); \
        new_a   = (Scalar *) PetscMalloc( len ); CHKPTRQ(new_a); \
        new_j   = (int *) (new_a + new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = bi[ii];} \
        for ( ii=row+1; ii<b->m+1; ii++ ) {new_i[ii] = bi[ii]+CHUNKSIZE;} \
        PetscMemcpy(new_j,bj,(bi[row]+nrow+shift)*sizeof(int)); \
        len = (new_nz - CHUNKSIZE - bi[row] - nrow - shift); \
        PetscMemcpy(new_j+bi[row]+shift+nrow+CHUNKSIZE,bj+bi[row]+shift+nrow, \
                                                           len*sizeof(int)); \
        PetscMemcpy(new_a,ba,(bi[row]+nrow+shift)*sizeof(Scalar)); \
        PetscMemcpy(new_a+bi[row]+shift+nrow+CHUNKSIZE,ba+bi[row]+shift+nrow, \
                                                           len*sizeof(Scalar));  \
        /* free up old matrix storage */ \
 \
        PetscFree(b->a);  \
        if (!b->singlemalloc) {PetscFree(b->i);PetscFree(b->j);} \
        ba = b->a = new_a; bi = b->i = new_i; bj = b->j = new_j;  \
        b->singlemalloc = 1; \
 \
        rp   = bj + bi[row] + shift; ap = ba + bi[row] + shift; \
        rmax = bimax[row] = bimax[row] + CHUNKSIZE; \
        PLogObjectMemory(B,CHUNKSIZE*(sizeof(int) + sizeof(Scalar))); \
        b->maxnz += CHUNKSIZE; \
        b->reallocs++; \
      } \
      N = nrow++ - 1; b->nz++; \
      /* shift up all the later entries in this row */ \
      for ( ii=N; ii>=_i; ii-- ) { \
        rp[ii+1] = rp[ii]; \
        ap[ii+1] = ap[ii]; \
      } \
      rp[_i] = col1;  \
      ap[_i] = value;  \
      b_noinsert: ; \
      bilen[row] = nrow; \
}

extern int MatSetValues_SeqAIJ(Mat,int,int*,int,int*,Scalar*,InsertMode);
#undef __FUNC__  
#define __FUNC__ "MatSetValues_MPIAIJ"
int MatSetValues_MPIAIJ(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Scalar     value;
  int        ierr,i,j, rstart = aij->rstart, rend = aij->rend;
  int        cstart = aij->cstart, cend = aij->cend,row,col;
  int        roworiented = aij->roworiented;

  /* Some Variables required in the macro */
  Mat        A = aij->A;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 
  int        *aimax = a->imax, *ai = a->i, *ailen = a->ilen,*aj = a->j;
  Scalar     *aa = a->a;

  Mat        B = aij->B;
  Mat_SeqAIJ *b = (Mat_SeqAIJ *) B->data; 
  int        *bimax = b->imax, *bi = b->i, *bilen = b->ilen,*bj = b->j;
  Scalar     *ba = b->a;

  int        *rp,ii,nrow,_i,rmax, N, col1,low,high,t; 
  int        nonew = a->nonew,shift = a->indexshift; 
  Scalar     *ap;

  PetscFunctionBegin;
  for ( i=0; i<m; i++ ) {
#if defined(USE_PETSC_BOPT_g)
    if (im[i] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (im[i] >= aij->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (in[j] >= cstart && in[j] < cend){
          col = in[j] - cstart;
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqAIJ_A_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqAIJ(aij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
#if defined(USE_PETSC_BOPT_g)
        else if (in[j] < 0) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");}
        else if (in[j] >= aij->N) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");}
#endif
        else {
          if (mat->was_assembled) {
            if (!aij->colmap) {
              ierr = CreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
            }
            col = aij->colmap[in[j]] - 1;
            if (col < 0 && !((Mat_SeqAIJ*)(aij->A->data))->nonew) {
              ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
              col =  in[j];
              /* Reinitialize the variables required by MatSetValues_SeqAIJ_B_Private() */
              B = aij->B;
              b = (Mat_SeqAIJ *) B->data; 
              bimax = b->imax; bi = b->i; bilen = b->ilen; bj = b->j;
              ba = b->a;
            }
          } else col = in[j];
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqAIJ_B_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqAIJ(aij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } 
    else {
      if (roworiented && !aij->donotstash) {
        ierr = StashValues_Private(&aij->stash,im[i],n,in,v+i*n,addv);CHKERRQ(ierr);
      }
      else {
        if (!aij->donotstash) {
          row = im[i];
          for ( j=0; j<n; j++ ) {
            ierr = StashValues_Private(&aij->stash,row,1,in+j,v+i+j*m,addv);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetValues_MPIAIJ"
int MatGetValues_MPIAIJ(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr,i,j, rstart = aij->rstart, rend = aij->rend;
  int        cstart = aij->cstart, cend = aij->cend,row,col;

  PetscFunctionBegin;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (idxm[i] >= aij->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
        if (idxn[j] >= aij->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
        if (idxn[j] >= cstart && idxn[j] < cend){
          col = idxn[j] - cstart;
          ierr = MatGetValues(aij->A,1,&row,1,&col,v+i*n+j); CHKERRQ(ierr);
        } else {
          if (!aij->colmap) {
            ierr = CreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
          }
          col = aij->colmap[idxn[j]] - 1;
          if ((col < 0) || (aij->garray[col] != idxn[j])) *(v+i*n+j) = 0.0;
          else {
            ierr = MatGetValues(aij->B,1,&row,1,&col,v+i*n+j); CHKERRQ(ierr);
          }
        }
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,0,"Only local values currently supported");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyBegin_MPIAIJ"
int MatAssemblyBegin_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         size = aij->size, *owners = aij->rowners;
  int         rank = aij->rank,tag = mat->tag, *owner,*starts,count,ierr;
  MPI_Request *send_waits,*recv_waits;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;

  PetscFunctionBegin;
  if (aij->donotstash) {
    aij->svalues    = 0; aij->rvalues    = 0;
    aij->nsends     = 0; aij->nrecvs     = 0;
    aij->send_waits = 0; aij->recv_waits = 0;
    aij->rmax       = 0;
    PetscFunctionReturn(0);
  }

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Some processors inserted others added");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (aij->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<aij->stash.n; i++ ) {
    idx = aij->stash.idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  ierr = MPI_Allreduce(procs, work,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  nreceives = work[rank]; 
  ierr = MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
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
  rvalues    = (Scalar *) PetscMalloc(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    ierr = MPI_Irecv(rvalues+3*nmax*i,3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues    = (Scalar *) PetscMalloc(3*(aij->stash.n+1)*sizeof(Scalar));CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts     = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0]  = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<aij->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  aij->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  aij->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  aij->stash.array[i];
  }
  PetscFree(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      ierr = MPI_Isend(svalues+3*starts[i],3*nprocs[i],MPIU_SCALAR,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  PetscFree(starts); 
  PetscFree(nprocs);

  /* Free cache space */
  PLogInfo(aij->A,"MatAssemblyBegin_MPIAIJ:Number of off-processor values %d\n",aij->stash.n);
  ierr = StashDestroy_Private(&aij->stash); CHKERRQ(ierr);

  aij->svalues    = svalues;    aij->rvalues    = rvalues;
  aij->nsends     = nsends;     aij->nrecvs     = nreceives;
  aij->send_waits = send_waits; aij->recv_waits = recv_waits;
  aij->rmax       = nmax;

  PetscFunctionReturn(0);
}
extern int MatSetUpMultiply_MPIAIJ(Mat);

#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_MPIAIJ"
int MatAssemblyEnd_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  MPI_Status  *send_status,recv_status;
  int         imdex,nrecvs = aij->nrecvs, count = nrecvs, i, n, ierr;
  int         row,col,other_disassembled;
  Scalar      *values,val;
  InsertMode  addv = mat->insertmode;

  PetscFunctionBegin;

  /*  wait on receives */
  while (count) {
    ierr = MPI_Waitany(nrecvs,aij->recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    values = aij->rvalues + 3*imdex*aij->rmax;
    ierr = MPI_Get_count(&recv_status,MPIU_SCALAR,&n);CHKERRQ(ierr);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PetscReal(values[3*i]) - aij->rstart;
      col = (int) PetscReal(values[3*i+1]);
      val = values[3*i+2];
      if (col >= aij->cstart && col < aij->cend) {
        col -= aij->cstart;
        ierr = MatSetValues(aij->A,1,&row,1,&col,&val,addv); CHKERRQ(ierr);
      } else {
        if (mat->was_assembled || mat->assembled) {
          if (!aij->colmap) {
            ierr = CreateColmap_MPIAIJ_Private(mat); CHKERRQ(ierr);
          }
          col = aij->colmap[col] - 1;
          if (col < 0  && !((Mat_SeqAIJ*)(aij->A->data))->nonew) {
            ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
            col = (int) PetscReal(values[3*i+1]);
          }
        }
        ierr = MatSetValues(aij->B,1,&row,1,&col,&val,addv); CHKERRQ(ierr);
      }
    }
    count--;
  }
  if (aij->recv_waits) PetscFree(aij->recv_waits); 
  if (aij->rvalues)    PetscFree(aij->rvalues);
 
  /* wait on sends */
  if (aij->nsends) {
    send_status = (MPI_Status *) PetscMalloc(aij->nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    ierr        = MPI_Waitall(aij->nsends,aij->send_waits,send_status);CHKERRQ(ierr);
    PetscFree(send_status);
  }
  if (aij->send_waits) PetscFree(aij->send_waits); 
  if (aij->svalues)    PetscFree(aij->svalues);

  ierr = MatAssemblyBegin(aij->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->A,mode); CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must 
     also disassemble ourselfs, in order that we may reassemble. */
  /*
     if nonzero structure of submatrix B cannot change then we know that
     no processor disassembled thus we can skip this stuff
  */
  if (!((Mat_SeqAIJ*) aij->B->data)->nonew)  {
    ierr = MPI_Allreduce(&mat->was_assembled,&other_disassembled,1,MPI_INT,MPI_PROD,mat->comm);CHKERRQ(ierr);
    if (mat->was_assembled && !other_disassembled) {
      ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIAIJ(mat); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(aij->B,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->B,mode); CHKERRQ(ierr);

  if (aij->rowvalues) {PetscFree(aij->rowvalues); aij->rowvalues = 0;}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_MPIAIJ"
int MatZeroEntries_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *l = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A); CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroRows_MPIAIJ"
int MatZeroRows_MPIAIJ(Mat A,IS is,Scalar *diag)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ *) A->data;
  int            i,ierr,N, *rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work,row;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values,rstart=l->rstart;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;
  PetscTruth     localdiag;

  PetscFunctionBegin;
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
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Index out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  ierr = MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  nrecvs = work[rank]; 
  ierr = MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int));CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
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
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  PetscFree(starts);

  base = owners[rank];

  /*  wait on receives */
  lens   = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
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
  ierr = ISCreateGeneral(PETSC_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PLogObjectParent(A,istmp);

  /*
        Zero the required rows. If the "diagonal block" of the matrix
     is square and the user wishes to set the diagonal we use seperate
     code so that MatSetValues() is not called for each diagonal allocating
     new memory, thus calling lots of mallocs and slowing things down.

       Contributed by: Mathew Knepley
  */
  localdiag = PETSC_FALSE;
  if (diag && (l->A->M == l->A->N)) {
    localdiag = PETSC_TRUE;
    ierr      = MatZeroRows(l->A,istmp,diag); CHKERRQ(ierr);
  } else {
    ierr = MatZeroRows(l->A,istmp,0); CHKERRQ(ierr);
  }
  ierr = MatZeroRows(l->B,istmp,0); CHKERRQ(ierr);
  ierr = ISDestroy(istmp); CHKERRQ(ierr);

  if (diag && (localdiag == PETSC_FALSE)) {
    for ( i = 0; i < slen; i++ ) {
      row = lrows[i] + rstart;
      MatSetValues(A,1,&row,1,&row,diag,INSERT_VALUES);
    }
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  }

  if (diag) {
    for ( i = 0; i < slen; i++ ) {
      row = lrows[i] + rstart;
      MatSetValues(A,1,&row,1,&row,diag,INSERT_VALUES);
    }
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  }
  PetscFree(lrows);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    ierr        = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    PetscFree(send_status);
  }
  PetscFree(send_waits); PetscFree(svalues);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMult_MPIAIJ"
int MatMult_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr,nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);  CHKERRQ(ierr);
  if (nt != a->n) {
    SETERRQ(PETSC_ERR_ARG_SIZ,0,"Incompatible partition of A and xx");
  }
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx); CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx); CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_MPIAIJ"
int MatMultAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTrans_MPIAIJ"
int MatMultTrans_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtrans)(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtrans)(a->A,xx,yy); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd_MPIAIJ"
int MatMultTransAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtrans)(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtransadd)(a->A,xx,yy,zz); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_MPIAIJ"
int MatGetDiagonal_MPIAIJ(Mat A,Vec v)
{
  int        ierr;
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;

  PetscFunctionBegin;
  if (a->M != a->N) SETERRQ(PETSC_ERR_SUP,0,"Supports only square matrix where A->A is diag block");
  if (a->rstart != a->cstart || a->rend != a->cend) {
    SETERRQ(PETSC_ERR_ARG_SIZ,0,"row partition must equal col partition");  
  }
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatScale_MPIAIJ"
int MatScale_MPIAIJ(Scalar *aa,Mat A)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatScale(aa,a->A); CHKERRQ(ierr);
  ierr = MatScale(aa,a->B); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_MPIAIJ"
int MatDestroy_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr;

  PetscFunctionBegin;
  if (--mat->refct > 0) PetscFunctionReturn(0);

  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping); CHKERRQ(ierr);
  }
  if (mat->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->bmapping); CHKERRQ(ierr);
  }
#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d",aij->M,aij->N);
#endif
  ierr = StashDestroy_Private(&aij->stash); CHKERRQ(ierr);
  PetscFree(aij->rowners); 
  ierr = MatDestroy(aij->A); CHKERRQ(ierr);
  ierr = MatDestroy(aij->B); CHKERRQ(ierr);
  if (aij->colmap) PetscFree(aij->colmap);
  if (aij->garray) PetscFree(aij->garray);
  if (aij->lvec)   VecDestroy(aij->lvec);
  if (aij->Mvctx)  VecScatterDestroy(aij->Mvctx);
  if (aij->rowvalues) PetscFree(aij->rowvalues);
  PetscFree(aij); 
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIAIJ_Binary"
extern int MatView_MPIAIJ_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  int         ierr;

  PetscFunctionBegin;
  if (aij->size == 1) {
    ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
  }
  else SETERRQ(PETSC_ERR_SUP,0,"Only uniprocessor output supported");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIAIJ_ASCIIorDraworMatlab"
extern int MatView_MPIAIJ_ASCIIorDraworMatlab(Mat mat,Viewer viewer)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ* C = (Mat_SeqAIJ*)aij->A->data;
  int         ierr, format,shift = C->indexshift,rank;
  FILE        *fd;
  ViewerType  vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILES_VIEWER || vtype == ASCII_FILE_VIEWER) { 
    ierr = ViewerGetFormat(viewer,&format);
    if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
      MatInfo info;
      int     flg;
      MPI_Comm_rank(mat->comm,&rank);
      ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info); 
      ierr = OptionsHasName(PETSC_NULL,"-mat_aij_no_inode",&flg); CHKERRQ(ierr);
      PetscSequentialPhaseBegin(mat->comm,1);
      if (flg) fprintf(fd,"[%d] Local rows %d nz %d nz alloced %d mem %d, not using I-node routines\n",
         rank,aij->m,(int)info.nz_used,(int)info.nz_allocated,(int)info.memory);
      else fprintf(fd,"[%d] Local rows %d nz %d nz alloced %d mem %d, using I-node routines\n",
         rank,aij->m,(int)info.nz_used,(int)info.nz_allocated,(int)info.memory);
      ierr = MatGetInfo(aij->A,MAT_LOCAL,&info);
      fprintf(fd,"[%d] on-diagonal part: nz %d \n",rank,(int)info.nz_used);
      ierr = MatGetInfo(aij->B,MAT_LOCAL,&info); 
      fprintf(fd,"[%d] off-diagonal part: nz %d \n",rank,(int)info.nz_used); 
      fflush(fd);
      PetscSequentialPhaseEnd(mat->comm,1);
      ierr = VecScatterView(aij->Mvctx,viewer); CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    } else if (format == VIEWER_FORMAT_ASCII_INFO) {
      PetscFunctionReturn(0);
    }
  }

  if (vtype == DRAW_VIEWER) {
    Draw       draw;
    PetscTruth isnull;
    ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (vtype == ASCII_FILE_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscSequentialPhaseBegin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
           aij->rank,aij->m,aij->rstart,aij->rend,aij->n,aij->cstart,
           aij->cend);
    ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
    ierr = MatView(aij->B,viewer); CHKERRQ(ierr);
    fflush(fd);
    PetscSequentialPhaseEnd(mat->comm,1);
  } else {
    int size = aij->size;
    rank = aij->rank;
    if (size == 1) {
      ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
    } else {
      /* assemble the entire matrix onto first processor. */
      Mat         A;
      Mat_SeqAIJ *Aloc;
      int         M = aij->M, N = aij->N,m,*ai,*aj,row,*cols,i,*ct;
      Scalar      *a;

      if (!rank) {
        ierr = MatCreateMPIAIJ(mat->comm,M,N,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);CHKERRQ(ierr);
      } else {
        ierr = MatCreateMPIAIJ(mat->comm,0,0,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);CHKERRQ(ierr);
      }
      PLogObjectParent(mat,A);

      /* copy over the A part */
      Aloc = (Mat_SeqAIJ*) aij->A->data;
      m = Aloc->m; ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = aij->rstart;
      for ( i=0; i<ai[m]+shift; i++ ) {aj[i] += aij->cstart + shift;}
      for ( i=0; i<m; i++ ) {
        ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],aj,a,INSERT_VALUES);CHKERRQ(ierr);
        row++; a += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
      } 
      aj = Aloc->j;
      for ( i=0; i<ai[m]+shift; i++ ) {aj[i] -= aij->cstart + shift;}

      /* copy over the B part */
      Aloc = (Mat_SeqAIJ*) aij->B->data;
      m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = aij->rstart;
      ct = cols = (int *) PetscMalloc( (ai[m]+1)*sizeof(int) ); CHKPTRQ(cols);
      for ( i=0; i<ai[m]+shift; i++ ) {cols[i] = aij->garray[aj[i]+shift];}
      for ( i=0; i<m; i++ ) {
        ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],cols,a,INSERT_VALUES);CHKERRQ(ierr);
        row++; a += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
      } 
      PetscFree(ct);
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      /* 
         Everyone has to call to draw the matrix since the graphics waits are
         synchronized across all processors that share the Draw object
      */
      if (!rank || vtype == DRAW_VIEWER) {
        ierr = MatView(((Mat_MPIAIJ*)(A->data))->A,viewer); CHKERRQ(ierr);
      }
      ierr = MatDestroy(A); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIAIJ"
int MatView_MPIAIJ(Mat mat,Viewer viewer)
{
  int         ierr;
  ViewerType  vtype;
 
  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER ||
      vtype == DRAW_VIEWER       || vtype == MATLAB_VIEWER) { 
    ierr = MatView_MPIAIJ_ASCIIorDraworMatlab(mat,viewer); CHKERRQ(ierr);
  } else if (vtype == BINARY_FILE_VIEWER) {
    ierr = MatView_MPIAIJ_Binary(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

/*
    This has to provide several versions.

     2) a) use only local smoothing updating outer values only once.
        b) local smoothing updating outer values each inner iteration
     3) color updating out values betwen colors.
*/
#undef __FUNC__  
#define __FUNC__ "MatRelax_MPIAIJ"
int MatRelax_MPIAIJ(Mat matin,Vec bb,double omega,MatSORType flag,
                           double fshift,int its,Vec xx)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Mat        AA = mat->A, BB = mat->B;
  Mat_SeqAIJ *A = (Mat_SeqAIJ *) AA->data, *B = (Mat_SeqAIJ *)BB->data;
  Scalar     *b,*x,*xs,*ls,d,*v,sum;
  int        ierr,*idx, *diag;
  int        n = mat->n, m = mat->m, i,shift = A->indexshift;

  PetscFunctionBegin;
  VecGetArray(xx,&x); VecGetArray(bb,&b); VecGetArray(mat->lvec,&ls);
  xs = x + shift; /* shift by one for index start of 1 */
  ls = ls + shift;
  if (!A->diag) {ierr = MatMarkDiag_SeqAIJ(AA); CHKERRQ(ierr);}
  diag = A->diag;
  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,its,xx);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr=VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    while (its--) {
      /* go down through the rows */
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + A->a[diag[i]+shift]*x[i])/d;
      }
      /* come up through the rows */
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + A->a[diag[i]+shift]*x[i])/d;
      }
    }    
  } else if (flag & SOR_LOCAL_FORWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,its,xx);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr=VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    while (its--) {
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + A->a[diag[i]+shift]*x[i])/d;
      }
    } 
  } else if (flag & SOR_LOCAL_BACKWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,its,xx);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,
                            mat->Mvctx); CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,
                            mat->Mvctx); CHKERRQ(ierr);
    while (its--) {
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + A->a[diag[i]+shift]*x[i])/d;
      }
    } 
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"Parallel SOR not supported");
  }
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_MPIAIJ"
int MatGetInfo_MPIAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Mat        A = mat->A, B = mat->B;
  int        ierr;
  double     isend[5], irecv[5];

  PetscFunctionBegin;
  info->block_size     = 1.0;
  ierr = MatGetInfo(A,MAT_LOCAL,info); CHKERRQ(ierr);
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  ierr = MatGetInfo(B,MAT_LOCAL,info); CHKERRQ(ierr);
  isend[0] += info->nz_used; isend[1] += info->nz_allocated; isend[2] += info->nz_unneeded;
  isend[3] += info->memory;  isend[4] += info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_MAX,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_SUM,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  info->rows_global       = (double)mat->M;
  info->columns_global    = (double)mat->N;
  info->rows_local        = (double)mat->m;
  info->columns_local     = (double)mat->N;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_MPIAIJ"
int MatSetOption_MPIAIJ(Mat A,MatOption op)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;

  PetscFunctionBegin;
  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_COLUMNS_UNSORTED ||
      op == MAT_COLUMNS_SORTED ||
      op == MAT_NEW_NONZERO_ALLOCATION_ERROR ||
      op == MAT_NEW_NONZERO_LOCATION_ERROR) {
        MatSetOption(a->A,op);
        MatSetOption(a->B,op);
  } else if (op == MAT_ROW_ORIENTED) {
    a->roworiented = 1; 
    MatSetOption(a->A,op);
    MatSetOption(a->B,op);
  } else if (op == MAT_ROWS_SORTED || 
             op == MAT_ROWS_UNSORTED ||
             op == MAT_SYMMETRIC ||
             op == MAT_STRUCTURALLY_SYMMETRIC ||
             op == MAT_YES_NEW_DIAGONALS)
    PLogInfo(A,"MatSetOption_MPIAIJ:Option ignored\n");
  else if (op == MAT_COLUMN_ORIENTED) {
    a->roworiented = 0;
    MatSetOption(a->A,op);
    MatSetOption(a->B,op);
  } else if (op == MAT_IGNORE_OFF_PROC_ENTRIES) {
    a->donotstash = 1;
  } else if (op == MAT_NO_NEW_DIAGONALS){
    SETERRQ(PETSC_ERR_SUP,0,"MAT_NO_NEW_DIAGONALS");
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_MPIAIJ"
int MatGetSize_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;

  PetscFunctionBegin;
  if (m) *m = mat->M;
  if (n) *n = mat->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetLocalSize_MPIAIJ"
int MatGetLocalSize_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;

  PetscFunctionBegin;
  if (m) *m = mat->m;
  if (n) *n = mat->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_MPIAIJ"
int MatGetOwnershipRange_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;

  PetscFunctionBegin;
  *m = mat->rstart; *n = mat->rend;
  PetscFunctionReturn(0);
}

extern int MatGetRow_SeqAIJ(Mat,int,int*,int**,Scalar**);
extern int MatRestoreRow_SeqAIJ(Mat,int,int*,int**,Scalar**);

#undef __FUNC__  
#define __FUNC__ "MatGetRow_MPIAIJ"
int MatGetRow_MPIAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Scalar     *vworkA, *vworkB, **pvA, **pvB,*v_p;
  int        i, ierr, *cworkA, *cworkB, **pcA, **pcB, cstart = mat->cstart;
  int        nztot, nzA, nzB, lrow, rstart = mat->rstart, rend = mat->rend;
  int        *cmap, *idx_p;

  PetscFunctionBegin;
  if (mat->getrowactive == PETSC_TRUE) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqAIJ *Aa = (Mat_SeqAIJ *) mat->A->data,*Ba = (Mat_SeqAIJ *) mat->B->data; 
    int     max = 1,m = mat->m,tmp;
    for ( i=0; i<m; i++ ) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) { max = tmp; }
    }
    mat->rowvalues = (Scalar *) PetscMalloc( max*(sizeof(int)+sizeof(Scalar))); 
    CHKPTRQ(mat->rowvalues);
    mat->rowindices = (int *) (mat->rowvalues + max);
  }

  if (row < rstart || row >= rend) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Only local rows")
  lrow = row - rstart;

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = 0; pvB = 0;}
  if (!idx) {pcA = 0; if (!v) pcB = 0;}
  ierr = (*mat->A->ops->getrow)(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = (*mat->B->ops->getrow)(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  nztot = nzA + nzB;

  cmap  = mat->garray;
  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      int imark = -1;
      if (v) {
        *v = v_p = mat->rowvalues;
        for ( i=0; i<nzB; i++ ) {
          if (cmap[cworkB[i]] < cstart)   v_p[i] = vworkB[i];
          else break;
        }
        imark = i;
        for ( i=0; i<nzA; i++ )     v_p[imark+i] = vworkA[i];
        for ( i=imark; i<nzB; i++ ) v_p[nzA+i]   = vworkB[i];
      }
      if (idx) {
        *idx = idx_p = mat->rowindices;
        if (imark > -1) {
          for ( i=0; i<imark; i++ ) {
            idx_p[i] = cmap[cworkB[i]];
          }
        } else {
          for ( i=0; i<nzB; i++ ) {
            if (cmap[cworkB[i]] < cstart)   idx_p[i] = cmap[cworkB[i]];
            else break;
          }
          imark = i;
        }
        for ( i=0; i<nzA; i++ )     idx_p[imark+i] = cstart + cworkA[i];
        for ( i=imark; i<nzB; i++ ) idx_p[nzA+i]   = cmap[cworkB[i]];
      } 
    } 
    else {
      if (idx) *idx = 0; 
      if (v)   *v   = 0;
    }
  }
  *nz = nztot;
  ierr = (*mat->A->ops->restorerow)(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = (*mat->B->ops->restorerow)(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_MPIAIJ"
int MatRestoreRow_MPIAIJ(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;

  PetscFunctionBegin;
  if (aij->getrowactive == PETSC_FALSE) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"MatGetRow not called");
  }
  aij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_MPIAIJ"
int MatNorm_MPIAIJ(Mat mat,NormType type,double *norm)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *amat = (Mat_SeqAIJ*) aij->A->data, *bmat = (Mat_SeqAIJ*) aij->B->data;
  int        ierr, i, j, cstart = aij->cstart,shift = amat->indexshift;
  double     sum = 0.0;
  Scalar     *v;

  PetscFunctionBegin;
  if (aij->size == 1) {
    ierr =  MatNorm(aij->A,type,norm); CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      v = amat->a;
      for (i=0; i<amat->nz; i++ ) {
#if defined(USE_PETSC_COMPLEX)
        sum += PetscReal(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      v = bmat->a;
      for (i=0; i<bmat->nz; i++ ) {
#if defined(USE_PETSC_COMPLEX)
        sum += PetscReal(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      ierr = MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,mat->comm);CHKERRQ(ierr);
      *norm = sqrt(*norm);
    } else if (type == NORM_1) { /* max column norm */
      double *tmp, *tmp2;
      int    *jj, *garray = aij->garray;
      tmp  = (double *) PetscMalloc( (aij->N+1)*sizeof(double) ); CHKPTRQ(tmp);
      tmp2 = (double *) PetscMalloc( (aij->N+1)*sizeof(double) ); CHKPTRQ(tmp2);
      PetscMemzero(tmp,aij->N*sizeof(double));
      *norm = 0.0;
      v = amat->a; jj = amat->j;
      for ( j=0; j<amat->nz; j++ ) {
        tmp[cstart + *jj++ + shift] += PetscAbsScalar(*v);  v++;
      }
      v = bmat->a; jj = bmat->j;
      for ( j=0; j<bmat->nz; j++ ) {
        tmp[garray[*jj++ + shift]] += PetscAbsScalar(*v); v++;
      }
      ierr = MPI_Allreduce(tmp,tmp2,aij->N,MPI_DOUBLE,MPI_SUM,mat->comm);CHKERRQ(ierr);
      for ( j=0; j<aij->N; j++ ) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      PetscFree(tmp); PetscFree(tmp2);
    } else if (type == NORM_INFINITY) { /* max row norm */
      double ntemp = 0.0;
      for ( j=0; j<amat->m; j++ ) {
        v = amat->a + amat->i[j] + shift;
        sum = 0.0;
        for ( i=0; i<amat->i[j+1]-amat->i[j]; i++ ) {
          sum += PetscAbsScalar(*v); v++;
        }
        v = bmat->a + bmat->i[j] + shift;
        for ( i=0; i<bmat->i[j+1]-bmat->i[j]; i++ ) {
          sum += PetscAbsScalar(*v); v++;
        }
        if (sum > ntemp) ntemp = sum;
      }
      ierr = MPI_Allreduce(&ntemp,norm,1,MPI_DOUBLE,MPI_MAX,mat->comm);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,0,"No support for two norm");
    }
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_MPIAIJ"
int MatTranspose_MPIAIJ(Mat A,Mat *matout)
{ 
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  Mat_SeqAIJ *Aloc = (Mat_SeqAIJ *) a->A->data;
  int        ierr,shift = Aloc->indexshift;
  int        M = a->M, N = a->N,m,*ai,*aj,row,*cols,i,*ct;
  Mat        B;
  Scalar     *array;

  PetscFunctionBegin;
  if (matout == PETSC_NULL && M != N) {
    SETERRQ(PETSC_ERR_ARG_SIZ,0,"Square matrix only for in-place");
  }

  ierr = MatCreateMPIAIJ(A->comm,a->n,a->m,N,M,0,PETSC_NULL,0,PETSC_NULL,&B);CHKERRQ(ierr);

  /* copy over the A part */
  Aloc = (Mat_SeqAIJ*) a->A->data;
  m = Aloc->m; ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  for ( i=0; i<ai[m]+shift; i++ ) {aj[i] += a->cstart + shift;}
  for ( i=0; i<m; i++ ) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],aj,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
  } 
  aj = Aloc->j;
  for ( i=0; i<ai[m]+shift; i++ ) {aj[i] -= a->cstart + shift;}

  /* copy over the B part */
  Aloc = (Mat_SeqAIJ*) a->B->data;
  m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  ct = cols = (int *) PetscMalloc( (1+ai[m]-shift)*sizeof(int) ); CHKPTRQ(cols);
  for ( i=0; i<ai[m]+shift; i++ ) {cols[i] = a->garray[aj[i]+shift];}
  for ( i=0; i<m; i++ ) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],cols,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
  } 
  PetscFree(ct);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (matout != PETSC_NULL) {
    *matout = B;
  } else {
    PetscOps       *Abops;
    struct _MatOps *Aops;

    /* This isn't really an in-place transpose .... but free data structures from a */
    PetscFree(a->rowners); 
    ierr = MatDestroy(a->A); CHKERRQ(ierr);
    ierr = MatDestroy(a->B); CHKERRQ(ierr);
    if (a->colmap) PetscFree(a->colmap);
    if (a->garray) PetscFree(a->garray);
    if (a->lvec) VecDestroy(a->lvec);
    if (a->Mvctx) VecScatterDestroy(a->Mvctx);
    PetscFree(a);

    /*
       This is horrible, horrible code. We need to keep the 
      A pointers for the bops and ops but copy everything 
      else from C.
    */
    Abops = A->bops;
    Aops  = A->ops;
    PetscMemcpy(A,B,sizeof(struct _p_Mat));
    A->bops = Abops;
    A->ops  = Aops; 
    PetscHeaderDestroy(B);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale_MPIAIJ"
int MatDiagonalScale_MPIAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat a = aij->A, b = aij->B;
  int ierr,s1,s2,s3;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(mat,&s2,&s3); CHKERRQ(ierr);
  if (rr) {
    ierr = VecGetLocalSize(rr,&s1);CHKERRQ(ierr);
    if (s1!=s3) SETERRQ(PETSC_ERR_ARG_SIZ,0,"right vector non-conforming local size");
    /* Overlap communication with computation. */
    ierr = VecScatterBegin(rr,aij->lvec,INSERT_VALUES,SCATTER_FORWARD,aij->Mvctx); CHKERRQ(ierr);
  }
  if (ll) {
    ierr = VecGetLocalSize(ll,&s1);CHKERRQ(ierr);
    if (s1!=s2) SETERRQ(PETSC_ERR_ARG_SIZ,0,"left vector non-conforming local size");
    ierr = (*b->ops->diagonalscale)(b,ll,0); CHKERRQ(ierr);
  }
  /* scale  the diagonal block */
  ierr = (*a->ops->diagonalscale)(a,ll,rr); CHKERRQ(ierr);

  if (rr) {
    /* Do a scatter end and then right scale the off-diagonal block */
    ierr = VecScatterEnd(rr,aij->lvec,INSERT_VALUES,SCATTER_FORWARD,aij->Mvctx); CHKERRQ(ierr);
    ierr = (*b->ops->diagonalscale)(b,0,aij->lvec); CHKERRQ(ierr);
  } 
  
  PetscFunctionReturn(0);
}


extern int MatPrintHelp_SeqAIJ(Mat);
#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_MPIAIJ"
int MatPrintHelp_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a   = (Mat_MPIAIJ*) A->data;
  int        ierr;

  PetscFunctionBegin;
  if (!a->rank) {
    ierr = MatPrintHelp_SeqAIJ(a->A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_MPIAIJ"
int MatGetBlockSize_MPIAIJ(Mat A,int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "MatSetUnfactored_MPIAIJ"
int MatSetUnfactored_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a   = (Mat_MPIAIJ*) A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatEqual_MPIAIJ"
int MatEqual_MPIAIJ(Mat A, Mat B, PetscTruth *flag)
{
  Mat_MPIAIJ *matB = (Mat_MPIAIJ *) B->data,*matA = (Mat_MPIAIJ *) A->data;
  Mat        a, b, c, d;
  PetscTruth flg;
  int        ierr;

  PetscFunctionBegin;
  if (B->type != MATMPIAIJ) SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Matrices must be same type");
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  ierr = MatEqual(a, c, &flg); CHKERRQ(ierr);
  if (flg == PETSC_TRUE) {
    ierr = MatEqual(b, d, &flg); CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&flg, flag, 1, MPI_INT, MPI_LAND, A->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern int MatConvertSameType_MPIAIJ(Mat,Mat *,int);
extern int MatIncreaseOverlap_MPIAIJ(Mat , int, IS *, int);
extern int MatFDColoringCreate_MPIAIJ(Mat,ISColoring,MatFDColoring);
extern int MatGetSubMatrices_MPIAIJ (Mat ,int , IS *,IS *,MatGetSubMatrixCall,Mat **);
extern int MatGetSubMatrix_MPIAIJ (Mat ,IS,IS,int,MatGetSubMatrixCall,Mat *);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_MPIAIJ,
       MatGetRow_MPIAIJ,MatRestoreRow_MPIAIJ,
       MatMult_MPIAIJ,MatMultAdd_MPIAIJ,
       MatMultTrans_MPIAIJ,MatMultTransAdd_MPIAIJ,
       0,0,
       0,0,
       0,0,
       MatRelax_MPIAIJ,
       MatTranspose_MPIAIJ,
       MatGetInfo_MPIAIJ,MatEqual_MPIAIJ,
       MatGetDiagonal_MPIAIJ,MatDiagonalScale_MPIAIJ,MatNorm_MPIAIJ,
       MatAssemblyBegin_MPIAIJ,MatAssemblyEnd_MPIAIJ,
       0,
       MatSetOption_MPIAIJ,MatZeroEntries_MPIAIJ,MatZeroRows_MPIAIJ,
       0,0,0,0,
       MatGetSize_MPIAIJ,MatGetLocalSize_MPIAIJ,MatGetOwnershipRange_MPIAIJ,
       0,0,
       0,0,MatConvertSameType_MPIAIJ,0,0,
       0,0,0,
       MatGetSubMatrices_MPIAIJ,MatIncreaseOverlap_MPIAIJ,MatGetValues_MPIAIJ,0,
       MatPrintHelp_MPIAIJ,
       MatScale_MPIAIJ,0,0,0,
       MatGetBlockSize_MPIAIJ,0,0,0,0, 
       MatFDColoringCreate_MPIAIJ,0,MatSetUnfactored_MPIAIJ,
       0,0,MatGetSubMatrix_MPIAIJ};


#undef __FUNC__  
#define __FUNC__ "MatCreateMPIAIJ"
/*@C
   MatCreateMPIAIJ - Creates a sparse parallel matrix in AIJ format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the 
           y vector for the matrix-vector product y = Ax.
.  n - This value should be the same as the local size used in creating the 
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if N is given) For square matrices n is almost always m.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.  d_nz - number of nonzeros per row in diagonal portion of local submatrix
           (same for all local rows)
.  d_nzz - array containing the number of nonzeros in the various rows of the 
           diagonal portion of the local submatrix (possibly different for each row)
           or PETSC_NULL. You must leave room for the diagonal entry even if
           it is zero.
.  o_nz - number of nonzeros per row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nzz - array containing the number of nonzeros in the various rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each row) or PETSC_NULL.

   Output Parameter:
.  A - the matrix 

   Notes:
   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one processor
   than it must be used on all processors that share the object for that argument.

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users manual for details.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   By default, this format uses inodes (identical nodes) when possible.
   We search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
+  -mat_aij_no_inode  - Do not use inodes
.  -mat_aij_inode_limit <limit> - Sets inode limit (max limit=5)
-  -mat_aij_oneindex - Internally use indexing starting at 1
        rather than 0.  Note that when calling MatSetValues(),
        the user still MUST index entries starting at 0!

   Storage Information:
   For a square global matrix we define each processor's diagonal portion 
   to be its local rows and the corresponding columns (a square submatrix);  
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix). 

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both).  Set 
   d_nz=PETSC_DEFAULT and d_nnz=PETSC_NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

.vb
             0 1 2 3 4 5 6 7 8 9 10 11
            -------------------
     row 3  |  o o o d d d o o o o o o
     row 4  |  o o o d d d o o o o o o
     row 5  |  o o o d d d o o o o o o
            -------------------
.ve 
 
   Thus, any entries in the d locations are stored in the d (diagonal) 
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQAIJ format for compressed row storage.

   Now d_nz should indicate the number of nonzeros per row in the d matrix,
   and o_nz should indicate the number of nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz. For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues()
@*/
int MatCreateMPIAIJ(MPI_Comm comm,int m,int n,int M,int N,int d_nz,int *d_nnz,int o_nz,int *o_nnz,Mat *A)
{
  Mat          B;
  Mat_MPIAIJ   *b;
  int          ierr, i,sum[2],work[2],size;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size);
  if (size == 1) {
    if (M == PETSC_DECIDE) M = m;
    if (N == PETSC_DECIDE) N = n;
    ierr = MatCreateSeqAIJ(comm,M,N,d_nz,d_nnz,A); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  *A = 0;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIAIJ,comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data       = (void *) (b = PetscNew(Mat_MPIAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_MPIAIJ));
  PetscMemcpy(B->ops,&MatOps,sizeof(struct _MatOps));
  B->ops->destroy    = MatDestroy_MPIAIJ;
  B->ops->view       = MatView_MPIAIJ;
  B->factor     = 0;
  B->assembled  = PETSC_FALSE;
  B->mapping    = 0;

  B->insertmode = NOT_SET_VALUES;
  b->size       = size;
  MPI_Comm_rank(comm,&b->rank);

  if (m == PETSC_DECIDE && (d_nnz != PETSC_NULL || o_nnz != PETSC_NULL)) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Cannot have PETSC_DECIDE rows but set d_nnz or o_nnz");
  }

  if (M == PETSC_DECIDE || N == PETSC_DECIDE) {
    work[0] = m; work[1] = n;
    ierr = MPI_Allreduce( work, sum,2,MPI_INT,MPI_SUM,comm );CHKERRQ(ierr);
    if (M == PETSC_DECIDE) M = sum[0];
    if (N == PETSC_DECIDE) N = sum[1];
  }
  if (m == PETSC_DECIDE) {m = M/b->size + ((M % b->size) > b->rank);}
  if (n == PETSC_DECIDE) {n = N/b->size + ((N % b->size) > b->rank);}
  b->m = m; B->m = m;
  b->n = n; B->n = n;
  b->N = N; B->N = N;
  b->M = M; B->M = M;

  /* build local table of row and column ownerships */
  b->rowners = (int *) PetscMalloc(2*(b->size+2)*sizeof(int)); CHKPTRQ(b->rowners);
  PLogObjectMemory(B,2*(b->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIAIJ));
  b->cowners = b->rowners + b->size + 2;
  ierr = MPI_Allgather(&m,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  b->rowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart = b->rowners[b->rank]; 
  b->rend   = b->rowners[b->rank+1]; 
  ierr = MPI_Allgather(&n,1,MPI_INT,b->cowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  b->cowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->cowners[i] += b->cowners[i-1];
  }
  b->cstart = b->cowners[b->rank]; 
  b->cend   = b->cowners[b->rank+1]; 

  if (d_nz == PETSC_DEFAULT) d_nz = 5;
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m,n,d_nz,d_nnz,&b->A); CHKERRQ(ierr);
  PLogObjectParent(B,b->A);
  if (o_nz == PETSC_DEFAULT) o_nz = 0;
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m,N,o_nz,o_nnz,&b->B); CHKERRQ(ierr);
  PLogObjectParent(B,b->B);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&b->stash); CHKERRQ(ierr);
  b->donotstash  = 0;
  b->colmap      = 0;
  b->garray      = 0;
  b->roworiented = 1;

  /* stuff used for matrix vector multiply */
  b->lvec      = 0;
  b->Mvctx     = 0;

  /* stuff for MatGetRow() */
  b->rowindices   = 0;
  b->rowvalues    = 0;
  b->getrowactive = PETSC_FALSE;

  *A = B;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatConvertSameType_MPIAIJ"
int MatConvertSameType_MPIAIJ(Mat matin,Mat *newmat,int cpvalues)
{
  Mat        mat;
  Mat_MPIAIJ *a,*oldmat = (Mat_MPIAIJ *) matin->data;
  int        ierr, len=0, flg;

  PetscFunctionBegin;
  *newmat       = 0;
  PetscHeaderCreate(mat,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIAIJ,matin->comm,MatDestroy,MatView);
  PLogObjectCreate(mat);
  mat->data       = (void *) (a = PetscNew(Mat_MPIAIJ)); CHKPTRQ(a);
  PetscMemcpy(mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->ops->destroy    = MatDestroy_MPIAIJ;
  mat->ops->view       = MatView_MPIAIJ;
  mat->factor     = matin->factor;
  mat->assembled  = PETSC_TRUE;

  a->m = mat->m   = oldmat->m;
  a->n = mat->n   = oldmat->n;
  a->M = mat->M   = oldmat->M;
  a->N = mat->N   = oldmat->N;

  a->rstart       = oldmat->rstart;
  a->rend         = oldmat->rend;
  a->cstart       = oldmat->cstart;
  a->cend         = oldmat->cend;
  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  mat->insertmode = NOT_SET_VALUES;
  a->rowvalues    = 0;
  a->getrowactive = PETSC_FALSE;

  a->rowners = (int *) PetscMalloc(2*(a->size+2)*sizeof(int)); CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,2*(a->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIAIJ));
  a->cowners = a->rowners + a->size + 2;
  PetscMemcpy(a->rowners,oldmat->rowners,2*(a->size+2)*sizeof(int));
  ierr = StashInitialize_Private(&a->stash); CHKERRQ(ierr);
  if (oldmat->colmap) {
    a->colmap = (int *) PetscMalloc((a->N)*sizeof(int));CHKPTRQ(a->colmap);
    PLogObjectMemory(mat,(a->N)*sizeof(int));
    PetscMemcpy(a->colmap,oldmat->colmap,(a->N)*sizeof(int));
  } else a->colmap = 0;
  if (oldmat->garray) {
    len = ((Mat_SeqAIJ *) (oldmat->B->data))->n;
    a->garray = (int *) PetscMalloc((len+1)*sizeof(int)); CHKPTRQ(a->garray);
    PLogObjectMemory(mat,len*sizeof(int));
    if (len) PetscMemcpy(a->garray,oldmat->garray,len*sizeof(int));
  } else a->garray = 0;
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec); CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatConvert(oldmat->A,MATSAME,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  ierr =  MatConvert(oldmat->B,MATSAME,&a->B); CHKERRQ(ierr);
  PLogObjectParent(mat,a->B);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatPrintHelp(mat); CHKERRQ(ierr);
  }
  *newmat = mat;
  PetscFunctionReturn(0);
}

#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "MatLoad_MPIAIJ"
int MatLoad_MPIAIJ(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  Scalar       *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          i, nz, ierr, j,rstart, rend, fd;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0, *offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)viewer)->tag;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,1,"Matrix in special format on disk, cannot load as MPIAIJ");
    }
  }

  ierr = MPI_Bcast(header+1,3,MPI_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];
  /* determine ownership of all rows */
  m = M/size + ((M % size) > rank);
  rowners = (int *) PetscMalloc((size+2)*sizeof(int)); CHKPTRQ(rowners);
  ierr = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
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
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT); CHKERRQ(ierr);
    sndcounts = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = rowners[i+1] - rowners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
    PetscFree(sndcounts);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT, 0,comm);CHKERRQ(ierr);
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
    nz     = procsnz[0];
    mycols = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(mycols);
    ierr   = PetscBinaryRead(fd,mycols,nz,PETSC_INT); CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for ( i=1; i<size; i++ ) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT); CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPI_INT,i,tag,comm);CHKERRQ(ierr);
    }
    PetscFree(cols);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<m; i++ ) {
      nz += ourlens[i];
    }
    mycols = (int*) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(mycols);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPI_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"something is wrong with file");
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
  ierr = MatCreateMPIAIJ(comm,m,PETSC_DECIDE,M,N,0,ourlens,0,offlens,newmat);CHKERRQ(ierr);
  A = *newmat;
  MatSetOption(A,MAT_COLUMNS_SORTED); 
  for ( i=0; i<m; i++ ) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    vals = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR); CHKERRQ(ierr);
    
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
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR); CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);CHKERRQ(ierr);
    }
    PetscFree(procsnz);
  } else {
    /* receive numeric values */
    vals = (Scalar*) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for ( i=0; i<m; i++ ) {
      ierr     = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }
  }
  PetscFree(ourlens); PetscFree(vals); PetscFree(mycols); PetscFree(rowners);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_MPIAIJ"
/*
    Not great since it makes two copies of the submatrix, first an SeqAIJ 
  in local and then by concatenating the local matrices the end result.
  Writing it directly would be much like MatGetSubMatrices_MPIAIJ()
*/
int MatGetSubMatrix_MPIAIJ(Mat mat,IS isrow,IS iscol,int csize,MatGetSubMatrixCall call,Mat *newmat)
{
  int        ierr, i, m,n,rstart,row,rend,nz,*cwork,size,rank,j;
  Mat        *local,M, Mreuse;
  Scalar     *vwork,*aa;
  MPI_Comm   comm = mat->comm;
  Mat_SeqAIJ *aij;
  int        *ii, *jj,nlocal,*dlens,*olens,dlen,olen,jend;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  if (call ==  MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"SubMatrix",(PetscObject *)&Mreuse);CHKERRQ(ierr);
    if (!Mreuse) SETERRQ(1,1,"Submatrix passed in was not used before, cannot reuse");
    local = &Mreuse;
    ierr  = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_REUSE_MATRIX,&local);CHKERRQ(ierr);
  } else {
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&local);CHKERRQ(ierr);
    Mreuse = *local;
    PetscFree(local);
  }

  /* 
      m - number of local rows
      n - number of columns (same on all processors)
      rstart - first row in new global matrix generated
  */
  ierr = MatGetSize(Mreuse,&m,&n);CHKERRQ(ierr);
  if (call == MAT_INITIAL_MATRIX) {
    aij = (Mat_SeqAIJ *) (Mreuse)->data;
    if (aij->indexshift) SETERRQ(PETSC_ERR_SUP,1,"No support for index shifted matrix");
    ii  = aij->i;
    jj  = aij->j;

    /*
        Determine the number of non-zeros in the diagonal and off-diagonal 
        portions of the matrix in order to do correct preallocation
    */

    /* first get start and end of "diagonal" columns */
    if (csize == PETSC_DECIDE) {
      nlocal = n/size + ((n % size) > rank);
    } else {
      nlocal = csize;
    }
    ierr   = MPI_Scan(&nlocal,&rend,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
    rstart = rend - nlocal;
    if (rank == size - 1 && rend != n) {
      SETERRQ(1,1,"Local column sizes do not add up to total number of columns");
    }

    /* next, compute all the lengths */
    dlens = (int *) PetscMalloc( (2*m+1)*sizeof(int) );CHKPTRQ(dlens);
    olens = dlens + m;
    for ( i=0; i<m; i++ ) {
      jend = ii[i+1] - ii[i];
      olen = 0;
      dlen = 0;
      for ( j=0; j<jend; j++ ) {
        if ( *jj < rstart || *jj >= rend) olen++;
        else dlen++;
        jj++;
      }
      olens[i] = olen;
      dlens[i] = dlen;
    }
    ierr = MatCreateMPIAIJ(comm,m,PETSC_DECIDE,PETSC_DECIDE,n,0,dlens,0,olens,&M);CHKERRQ(ierr);
    PetscFree(dlens);
  } else {
    int ml,nl;

    M = *newmat;
    ierr = MatGetLocalSize(M,&ml,&nl);CHKERRQ(ierr);
    if (ml != m) SETERRQ(PETSC_ERR_ARG_SIZ,1,"Previous matrix must be same size/layout as request");
    ierr = MatZeroEntries(M);CHKERRQ(ierr);
    /*
         The next two lines are needed so we may call MatSetValues_MPIAIJ() below directly,
       rather than the slower MatSetValues().
    */
    M->was_assembled = PETSC_TRUE; 
    M->assembled     = PETSC_FALSE;
  }
  ierr = MatGetOwnershipRange(M,&rstart,&rend); CHKERRQ(ierr);
  aij = (Mat_SeqAIJ *) (Mreuse)->data;
  if (aij->indexshift) SETERRQ(PETSC_ERR_SUP,1,"No support for index shifted matrix");
  ii  = aij->i;
  jj  = aij->j;
  aa  = aij->a;
  for (i=0; i<m; i++) {
    row   = rstart + i;
    nz    = ii[i+1] - ii[i];
    cwork = jj;     jj += nz;
    vwork = aa;     aa += nz;
    ierr = MatSetValues_MPIAIJ(M,1,&row,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  *newmat = M;

  /* save submatrix used in processor for next request */
  if (call ==  MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)M,"SubMatrix",(PetscObject)Mreuse);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)Mreuse);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
