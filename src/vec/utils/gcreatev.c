#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.14 1995/05/03 13:15:19 bsmith Exp curfman $";
#endif


#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "is.h"
#include "vec.h"

/*@C
    VecCreate - Creates a vector, where the vector type is determined 
    from the options database.  Generates a parallel MPI vector if the 
    communicator has more than one processor.

    Input Parameters:
.   comm - MPI communicator
.   n - global vector length
 
    Output Parameter:
.   V - location to stash resulting vector

    Options Database Key:
$   -mpi_objects : use MPI vectors, even for the uniprocessor case

    Notes:
    Use VecDuplicate() or VecGetVecs() to form additional vectors
    of the same type as an existing vector.

.keywords: vector, create, initial

.seealso: VecCreateSequential(), VecCreateMPI(), VecDuplicate(), VecGetVecs()
@*/
int VecCreate(MPI_Comm comm,int n,Vec *V)
{
  int numtid;
  MPI_Comm_size(comm,&numtid);
  if (numtid > 1 || OptionsHasName(0,0,"-mpi_objects")) {
    return VecCreateMPI(comm,PETSC_DECIDE,n,V);
  }
  return VecCreateSequential(comm,n,V);
}
 
