#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vecio.c,v 1.31 1997/07/09 20:49:32 balay Exp bsmith $";
#endif

/* 
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's 
   VecView (with viewer types BINARY_FILE_VIEWER)
 */

#include "petsc.h"
#include "src/vec/impls/mpi/pvecimpl.h"
#include "sys.h"
#include "pinclude/pviewer.h"

#undef __FUNC__  
#define __FUNC__ "VecLoad"
/*@C 
  VecLoad - Loads a vector that has been stored in binary format
  with VecView().

  Input Parameters:
. comm - MPI communicator
. viewer - binary file viewer, obtained from ViewerFileOpenBinary()

  Output Parameter:
. newvec - the newly loaded vector

  Notes:
  The input file must contain the full global vector, as
  written by the routine VecView().

   Notes for advanced users:
   Most users should not need to know the details of the binary storage
   format, since VecLoad() and VecView() completely hide these details.
   But for anyone who's interested, the standard binary matrix storage
   format is

$    int    VEC_COOKIE
$    int    number of rows
$    Scalar *values of all nonzeros

.keywords: vector, load, binary, input

.seealso: ViewerFileOpenBinary(), VecView(), MatLoad() 
@*/  
int VecLoad(Viewer viewer,Vec *newvec)
{
  int         i, rows, ierr, type, fd,rank,size,n;
  Vec         vec;
  Vec_MPI     *v;
  Scalar      *avec;
  MPI_Comm    comm;
  MPI_Request *requests;
  MPI_Status  status,*statuses;
  ViewerType  vtype;

  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype != BINARY_FILE_VIEWER) SETERRQ(1,0,"Must be binary viewer");
  PLogEventBegin(VEC_Load,viewer,0,0,0);
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  PetscObjectGetComm((PetscObject)viewer,&comm);
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  if (!rank) {
    /* Read vector header. */
    ierr = PetscBinaryRead(fd,&type,1,BINARY_INT); CHKERRQ(ierr);
    if ((VecType)type != VEC_COOKIE) SETERRQ(1,0,"Non-vector object");
    ierr = PetscBinaryRead(fd,&rows,1,BINARY_INT); CHKERRQ(ierr);
    MPI_Bcast(&rows,1,MPI_INT,0,comm);
    ierr = VecCreate(comm,rows,&vec); CHKERRQ(ierr);
    v = (Vec_MPI*) vec->data;
    ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,avec,v->n,BINARY_SCALAR);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);

    if (size > 1) {
      /* read in other chuncks and send to other processors */
      /* determine maximum chunck owned by other */
      n = 1;
      for ( i=1; i<size; i++ ) {
        n = PetscMax(n,v->ownership[i] - v->ownership[i-1]);
      }
      avec = (Scalar *) PetscMalloc( n*sizeof(Scalar) ); CHKPTRQ(avec);
      requests = (MPI_Request *) PetscMalloc((size-1)*sizeof(MPI_Request));CHKPTRQ(requests);
      statuses = (MPI_Status *) PetscMalloc((size-1)*sizeof(MPI_Status));CHKPTRQ(statuses);
      for ( i=1; i<size; i++ ) {
        n = v->ownership[i+1]-v->ownership[i];
        ierr = PetscBinaryRead(fd,avec,n,BINARY_SCALAR);CHKERRQ(ierr);
        MPI_Isend(avec,n,MPIU_SCALAR,i,vec->tag,vec->comm,requests+i-1);
      }
      MPI_Waitall(size-1,requests,statuses);
      PetscFree(avec); PetscFree(requests); PetscFree(statuses);
    }
  } else {
    MPI_Bcast(&rows,1,MPI_INT,0,comm);
    ierr = VecCreate(comm,rows,&vec); CHKERRQ(ierr);
    VecGetLocalSize(vec,&n);CHKERRQ(ierr); 
    VecGetArray(vec,&avec); CHKERRQ(ierr);
    MPI_Recv(avec,n,MPIU_SCALAR,0,vec->tag,vec->comm,&status);
    ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);
  }
  *newvec = vec;
  VecAssemblyBegin(vec);
  VecAssemblyEnd(vec);
  PLogEventEnd(VEC_Load,viewer,0,0,0);
  return 0;
}


