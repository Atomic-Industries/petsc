#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fmg.c,v 1.11 1997/10/19 03:24:36 bsmith Exp bsmith $";
#endif
/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include "src/sles/pc/impls/mg/mgimpl.h"

extern int MGMCycle_Private(MG *);

/*
       MGFCycle_Private - Given an MG structure created with MGCreate() runs 
               full multigrid. 

    Iput Parameters:
.   mg - structure created with MGCreate().

    Note: This may not be what others call full multigrid. What we
          do is restrict the rhs to all levels, then starting 
          on the coarsest level work our way up generating 
          initial guess for the next level. This provides an
          improved preconditioner but not a great improvement.
*/
#undef __FUNC__  
#define __FUNC__ "MGFCycle_Private"
int MGFCycle_Private(MG *mg)
{
  int    i, l = mg[0]->levels,ierr;
  Scalar zero = 0.0;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for ( i=l-1; i>0; i-- ){
    ierr = MatMult(mg[i]->restrct, mg[i]->b, mg[i-1]->b ); CHKERRQ(ierr);
  }
  
  /* work our way up through the levels */
  ierr = VecSet(&zero, mg[0]->x );  CHKERRQ(ierr);
  for ( i=0; i<l-1; i++ ) {
    ierr = MGMCycle_Private(&mg[i]);  CHKERRQ(ierr);
    ierr = VecSet(&zero, mg[i+1]->x ); CHKERRQ(ierr); 
    ierr = MatMultTransAdd(mg[i+1]->interpolate,mg[i]->x,mg[i+1]->x,mg[i+1]->x);CHKERRQ(ierr); 
  }
  ierr = MGMCycle_Private(&mg[l-1]);  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       MGKCycle_Private - Given an MG structure created with MGCreate() runs 
               full Kascade MG solve.

    Iput Parameters:
.   mg - structure created with MGCreate().

    Note: This may not be what others call Kascadic MG.
*/
#undef __FUNC__  
#define __FUNC__ "MGKCycle_Private"
int MGKCycle_Private(MG *mg)
{
  int    i, l = mg[0]->levels,its,ierr;
  Scalar zero = 0.0;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for ( i=l-1; i>0; i-- ){
    ierr = MatMult(mg[i]->restrct,mg[i]->b, mg[i-1]->b); CHKERRQ(ierr); 
  }
  
  /* work our way up through the levels */
  ierr = VecSet(&zero, mg[0]->x ); 
  for ( i=0; i<l-1; i++ ) {
    ierr = SLESSolve(mg[i]->smoothd,mg[i]->b,mg[i]->x,&its); CHKERRQ(ierr);
    ierr = VecSet(&zero, mg[i+1]->x );  CHKERRQ(ierr);
    ierr = MatMultTransAdd(mg[i+1]->interpolate,mg[i]->x,mg[i+1]->x,mg[i+1]->x);CHKERRQ(ierr);
  }
  ierr = SLESSolve(mg[l-1]->smoothd,mg[l-1]->b,mg[l-1]->x,&its); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


