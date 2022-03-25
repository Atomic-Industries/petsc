
static char help[] = "Tests DMDA with variable multiple degrees of freedom per node.\n\n";

/*
   This code only compiles with gcc, since it is not ANSI C
*/

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode doit(DM da,Vec global)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,M,N,dof;

  CHKERRQ(DMDAGetInfo(da,0,&M,&N,0,0,0,0,&dof,0,0,0,0,0));
  {
    struct {PetscScalar inside[dof];} **mystruct;
    CHKERRQ(DMDAVecGetArrayRead(da,global,(void*) &mystruct));
    for (i=0; i<N; i++) {
      for (j=0; j<M; j++) {
        for (k=0; k<dof; k++) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%D %D %g\n",i,j,(double)mystruct[i][j].inside[0]));

          mystruct[i][j].inside[1] = 2.1;
        }
      }
    }
    CHKERRQ(DMDAVecRestoreArrayRead(da,global,(void*) &mystruct));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt       dof = 2,M = 3,N = 3,m = PETSC_DECIDE,n = PETSC_DECIDE;
  PetscErrorCode ierr;
  DM             da;
  Vec            global,local;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-dof",&dof,0));
  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,dof,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));

  CHKERRQ(doit(da,global));

  CHKERRQ(VecView(global,0));

  /* Free memory */
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}
