#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex11.c,v 1.3 1998/12/03 03:57:16 bsmith Exp bsmith $";
#endif

/* Program usage:  mpirun ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates VecStrideNorm().\n\n";

/*T
   Concepts: Vectors^Norms of sub-vectors;
   Routines: VecCreate(); VecSetFromOptions(); VecSet(); VecSetBlockSize(); VecStrideNorm(); VecNorm(); 
   Processors: n
T*/

/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   is.h     - index sets
     sys.h    - system routines       viewer.h - viewers
*/

#include "vec.h"

int main(int argc,char **argv)
{
  Vec      x;               /* vectors */
  double   norm;
  int      n = 20, ierr, flg;
  Scalar   one = 1.0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* 
     Create a vector, specifying only its global dimension.
     When using VecCreate() and VecSetFromOptions(), the vector format (currently parallel,
     shared, or sequential) is determined at runtime.  Also, the parallel
     partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
        VecCreateShared() - parallel vector that uses shared memory
                            (available only on the SGI); otherwise,
                            is the same as VecCreateMPI()

     With VecCreate() and VecSetFromOptions() the option -vec_type mpi or -vec_type shared causes the 
     particular type of vector to be formed.

  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x); CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);

  /*
     Set the vectors to entries to a constant value.
  */
  ierr = VecSet(&one,x); CHKERRA(ierr);

  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Norm of entire vector %g\n",norm);

  ierr = VecSetBlockSize(x,2);CHKERRA(ierr);
  ierr = VecStrideNorm(x,0,NORM_2,&norm); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Norm of sub-vector %g\n",norm);

  ierr = VecStrideNorm(x,1,NORM_2,&norm); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Norm of sub-vector %g\n",norm);

  ierr = VecStrideNorm(x,1,NORM_1,&norm); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Norm of sub-vector %g\n",norm);

  ierr = VecStrideNorm(x,1,NORM_INFINITY,&norm); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Norm of sub-vector %g\n",norm);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
