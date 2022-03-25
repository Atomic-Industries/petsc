static const char help[] = "Test PetscSF with MPI large count (more than 2 billion elements in messages)\n\n";

#include <petscsys.h>
#include <petscsf.h>

int main(int argc,char **argv)
{
  PetscSF           sf;
  PetscInt          i,nroots,nleaves;
  PetscInt          n = (1ULL<<31) + 1024; /* a little over 2G elements */
  PetscSFNode       *iremote = NULL;
  PetscMPIInt       rank,size;
  char              *rootdata=NULL,*leafdata=NULL;
  Vec               x,y;
  VecScatter        vscat;
  PetscInt          rstart,rend;
  IS                ix;
  const PetscScalar *xv;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"The test can only run with two MPI ranks");

  /* Test PetscSF */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetFromOptions(sf));

  if (!rank) {
    nroots  = n;
    nleaves = 0;
  } else {
    nroots  = 0;
    nleaves = n;
    CHKERRQ(PetscMalloc1(nleaves,&iremote));
    for (i=0; i<nleaves; i++) {
      iremote[i].rank  = 0;
      iremote[i].index = i;
    }
  }
  CHKERRQ(PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES));
  CHKERRQ(PetscMalloc2(nroots,&rootdata,nleaves,&leafdata));
  if (!rank) {
    memset(rootdata,11,nroots);
    rootdata[nroots-1] = 12; /* Use a different value at the end */
  }

  CHKERRQ(PetscSFBcastBegin(sf,MPI_SIGNED_CHAR,rootdata,leafdata,MPI_REPLACE)); /* rank 0->1, bcast rootdata to leafdata */
  CHKERRQ(PetscSFBcastEnd(sf,MPI_SIGNED_CHAR,rootdata,leafdata,MPI_REPLACE));
  CHKERRQ(PetscSFReduceBegin(sf,MPI_SIGNED_CHAR,leafdata,rootdata,MPI_SUM)); /* rank 1->0, add leafdata to rootdata */
  CHKERRQ(PetscSFReduceEnd(sf,MPI_SIGNED_CHAR,leafdata,rootdata,MPI_SUM));
  if (!rank) {
    PetscCheckFalse(rootdata[0] != 22 || rootdata[nroots-1] != 24,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF: wrong results");
  }

  CHKERRQ(PetscFree2(rootdata,leafdata));
  CHKERRQ(PetscFree(iremote));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test VecScatter */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(x,rank==0? n : 64,PETSC_DECIDE));
  CHKERRQ(VecSetSizes(y,rank==0? 64 : n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetFromOptions(y));

  CHKERRQ(VecGetOwnershipRange(x,&rstart,&rend));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,rend-rstart,rstart,1,&ix));
  CHKERRQ(VecScatterCreate(x,ix,y,ix,&vscat));

  CHKERRQ(VecSet(x,3.0));
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecScatterBegin(vscat,y,x,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(vscat,y,x,ADD_VALUES,SCATTER_REVERSE));

  CHKERRQ(VecGetArrayRead(x,&xv));
  PetscCheckFalse(xv[0] != 6.0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"VecScatter: wrong results");
  CHKERRQ(VecRestoreArrayRead(x,&xv));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecScatterDestroy(&vscat));
  CHKERRQ(ISDestroy(&ix));

  CHKERRQ(PetscFinalize());
  return 0;
}

/**TEST
   test:
     requires: defined(PETSC_HAVE_MPI_LARGE_COUNT) defined(PETSC_USE_64BIT_INDICES)
     TODO: need a machine with big memory (~150GB) to run the test
     nsize: 2
     args: -sf_type {{basic neighbor}}

TEST**/
