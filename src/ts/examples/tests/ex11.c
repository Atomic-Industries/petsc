static char help[] = "Tests TSTrajectoryGetVecs. \n\n";
/*
  This example tests TSTrajectory and the ability of TSTrajectoryGetVecs
  to reconstructs states and derivatives via interpolation (if necessary).
  It also tests TSTrajectory{Get|Restore}UpdatedHistoryVecs
*/
#include <petscts.h>

PetscScalar func(PetscInt p, PetscReal t)  { return p ? t*func(p-1,t) : 1.0; }
PetscScalar dfunc(PetscInt p, PetscReal t)  { return p > 0 ? (PetscReal)p*func(p-1,t) : 0.0; }

int main(int argc,char **argv)
{
  TS             ts;
  Vec            W,W2,Wdot;
  TSTrajectory   tj;
  PetscReal      times[10];
  PetscReal      TT[10] = { 0.2, 0.9, 0.1, 0.3, 0.6, 0.7, 0.5, 1.0, 0.4, 0.8 };
  PetscInt       i, p = 1, Nt = 10;
  PetscInt       II[10] = { 1, 4, 9, 2, 3, 6, 5, 8, 0, 7 };
  PetscBool      sort;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = VecCreate(PETSC_COMM_WORLD,&W);CHKERRQ(ierr);
  ierr = VecSetSizes(W,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetUp(W);CHKERRQ(ierr);
  ierr = VecDuplicate(W,&Wdot);CHKERRQ(ierr);
  ierr = VecDuplicate(W,&W2);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,W2);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,10);CHKERRQ(ierr);
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
  ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(tj,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(tj,ts);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(tj,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(tj,ts);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-interptimes",times,&Nt,NULL);CHKERRQ(ierr);
  sort = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-sorttimes",&sort,NULL);CHKERRQ(ierr);
  if (sort) {
    ierr = PetscSortReal(10,TT);CHKERRQ(ierr);
  }
  sort = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-sortkeys",&sort,NULL);CHKERRQ(ierr);
  if (sort) {
    ierr = PetscSortInt(10,II);CHKERRQ(ierr);
  }
  p = PetscMax(p,-p);

  /* populate trajectory */
  for (i=0; i < 10; i++) {
    ierr = VecSet(W,func(p,TT[i]));CHKERRQ(ierr);
    ierr = TSSetStepNumber(ts,II[i]);CHKERRQ(ierr);
    ierr = TSTrajectorySet(tj,ts,II[i],TT[i],W);CHKERRQ(ierr);
  }
  for (i = 0; i < Nt; i++) {
    PetscReal testtime = times[i];
    const PetscScalar *aW,*aWdot;

    ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,W,Wdot);CHKERRQ(ierr);
    ierr = VecGetArrayRead(W,&aW);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Wdot,&aWdot);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0]));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0]));
    ierr = VecRestoreArrayRead(W,&aW);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Wdot,&aWdot);CHKERRQ(ierr);
  }
  for (i = Nt-1; i >= 0; i--) {
    PetscReal         testtime = times[i];
    const PetscScalar *aW;

    ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,W,NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(W,&aW);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0]));
    ierr = VecRestoreArrayRead(W,&aW);CHKERRQ(ierr);
  }
  for (i = Nt-1; i >= 0; i--) {
    PetscReal         testtime = times[i];
    const PetscScalar *aWdot;

    ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,NULL,Wdot);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Wdot,&aWdot);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0]));
    ierr = VecRestoreArrayRead(Wdot,&aWdot);CHKERRQ(ierr);
  }
  for (i = 0; i < Nt; i++) {
    PetscReal         testtime = times[i];
    const PetscScalar *aW,*aWdot;
    Vec               hW,hWdot;

    ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,ts,testtime,&hW,&hWdot);CHKERRQ(ierr);
    ierr = VecGetArrayRead(hW,&aW);CHKERRQ(ierr);
    ierr = VecGetArrayRead(hWdot,&aWdot);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0]));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0]));
    ierr = VecRestoreArrayRead(hW,&aW);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(hWdot,&aWdot);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&hW,&hWdot);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&W);CHKERRQ(ierr);
  ierr = VecDestroy(&W2);CHKERRQ(ierr);
  ierr = VecDestroy(&Wdot);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
