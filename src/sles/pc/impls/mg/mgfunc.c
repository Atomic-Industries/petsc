#ifndef lint
static char vcid[] = "$Id: mgfunc.c,v 1.16 1996/11/07 15:08:56 bsmith Exp bsmith $";
#endif

#include "src/pc/impls/mg/mgimpl.h"       /*I "sles.h" I*/
                          /*I "mg.h"   I*/

/*@C
   MGGetCoarseSolve - Gets the solver context to be used on the coarse grid.

   Input Parameter:
.  pc - the multigrid context 

   Output Parameter:
.  sles - the coarse grid solver context 

.keywords: MG, multigrid, get, coarse grid
@*/ 
int MGGetCoarseSolve(PC pc,SLES *sles)  
{ 
  MG *mg = (MG*) pc->data;
  *sles =  mg[0]->smoothd;
  return 0;
}

/*@
   MGDefaultResidual - Default routine to calculate the residual.

   Input Parameters:
.  mat - the matrix
.  b   - the right-hand-side
.  x   - the approximate solution
 
   Output Parameter:
.  r - location to store the residual

.keywords: MG, default, multigrid, residual

.seealso: MGSetResidual()
@*/
int MGDefaultResidual(Mat mat,Vec b,Vec x,Vec r)
{
  int    ierr;
  Scalar mone = -1.0;
  ierr = MatMult(mat,x,r); CHKERRQ(ierr);
  ierr = VecAYPX(&mone,b,r); CHKERRQ(ierr);
  return 0;
}

/*@
   MGSetResidual - Sets the function to be used to calculate the residual 
   on the lth level. 

   Input Paramters:
.  pc       - the multigrid context
.  l        - the level to supply
.  residual - function used to form residual (usually MGDefaultResidual)
.  mat      - matrix associated with residual

.keywords:  MG, set, multigrid, residual, level

.seealso: MGDefaultResidual()
@*/
int MGSetResidual(PC pc,int l,int (*residual)(Mat,Vec,Vec,Vec),Mat mat) 
{
  MG *mg = (MG*) pc->data;
  mg[l]->residual = residual;  
  mg[l]->A        = mat;
  return 0;
}

/*@
   MGSetInterpolate - Sets the function to be used to calculate the 
   interpolation on the lth level. 

   Input Parameters:
.  pc  - the multigrid context
.  mat - the interpolation operator
.  l   - the level to supply

.keywords:  multigrid, set, interpolate, level

.seealso: MGSetRestriction()
@*/
int MGSetInterpolate(PC pc,int l,Mat mat)
{ 
  MG *mg = (MG*) pc->data;
  mg[l]->interpolate = mat;  
  return 0;
}

/*@
    MGSetRestriction - Sets the function to be used to restrict vector
    from level l to l-1. 

   Input Parameters:
.  pc - the multigrid context 
.  mat - the restriction matrix
.  l - the level to supply

.keywords: MG, set, multigrid, restriction, level

.seealso: MGSetInterpolate()
@*/
int MGSetRestriction(PC pc,int l,Mat mat)  
{
  MG *mg = (MG*) pc->data;
  mg[l]->restrct  = mat;  
  return 0;
}

/*@C
   MGGetSmoother - Gets the SLES context to be used as smoother for 
   both pre- and post-smoothing.  Call both MGGetSmootherUp() and 
   MGGetSmootherDown() to use different functions for pre- and 
   post-smoothing.

   Input Parameters:
.  pc - the multigrid context 
.  l - the level to supply

   Ouput Parameters:
.  sles - the smoother

.keywords: MG, get, multigrid, level, smoother, pre-smoother, post-smoother

.seealso: MGGetSmootherUp(), MGGetSmootherDown()
@*/
int MGGetSmoother(PC pc,int l,SLES *sles)
{
  MG *mg = (MG*) pc->data;
  *sles = mg[l]->smoothd;  
  return 0;
}

/*@C
   MGGetSmootherUp - Gets the SLES context to be used as smoother after 
   coarse grid correction (post-smoother). 

   Input Parameters:
.  pc - the multigrid context 
.  l  - the level to supply

   Ouput Parameters:
.  sles - the smoother

.keywords: MG, multigrid, get, smoother, up, post-smoother, level

.seealso: MGGetSmootherUp(), MGGetSmootherDown()
@*/
int MGGetSmootherUp(PC pc,int l,SLES *sles)
{
  MG   *mg = (MG*) pc->data;
  int  ierr;
  char *prefix;

  /*
     This is called only if user wants a different pre-smoother from post.
     Thus we check if a different one has already been allocated, 
     if not we allocate it.
  */
  ierr = PCGetOptionsPrefix(pc,&prefix); CHKERRQ(ierr);

  if (mg[l]->smoothu == mg[l]->smoothd) {
    ierr = SLESCreate(pc->comm,&mg[l]->smoothu); CHKERRQ(ierr);
    ierr = SLESSetOptionsPrefix( mg[l]->smoothu,prefix); CHKERRQ(ierr);
    PLogObjectParent(pc,mg[l]->smoothu);
  }
  *sles = mg[l]->smoothu;
  return 0;
}

/*@C
   MGGetSmootherDown - Gets the SLES context to be used as smoother before 
   coarse grid correction (pre-smoother). 

   Input Parameters:
.  pc - the multigrid context 
.  l  - the level to supply

   Ouput Parameters:
.  sles - the smoother

.keywords: MG, multigrid, get, smoother, down, pre-smoother, level

.seealso: MGGetSmootherUp(), MGGetSmoother()
@*/
int MGGetSmootherDown(PC pc,int l,SLES *sles)
{
  MG *mg = (MG*) pc->data;
  *sles = mg[l]->smoothd;  
  return 0;
}

/*@
   MGSetCyclesOnLevel - Sets the number of cycles to run on this level. 

   Input Parameters:
.  pc - the multigrid context 
.  l  - the level this is to be used for
.  n  - the number of cycles

.keywords: MG, multigrid, set, cycles, V-cycle, W-cycle, level

.seealso: MGSetCycles()
@*/
int MGSetCyclesOnLevel(PC pc,int l,int c) 
{
  MG *mg = (MG*) pc->data;
  mg[l]->cycles  = c;
  return 0;
}

/*@
   MGSetRhs - Sets the vector space to be used to store the right-hand side
   on a particular level.  The user should free this space at the conclusion 
   of multigrid use. 

   Input Parameters:
.  pc - the multigrid context 
.  l  - the level this is to be used for
.  c  - the space

.keywords: MG, multigrid, set, right-hand-side, rhs, level

.seealso: MGSetX(), MGSetR()
@*/
int MGSetRhs(PC pc,int l,Vec c)  
{ 
  MG *mg = (MG*) pc->data;
  mg[l]->b  = c;
  return 0;
}

/*@
   MGSetX - Sets the vector space to be used to store the solution on a 
   particular level.  The user should free this space at the conclusion 
   of multigrid use.

   Input Parameters:
.  pc - the multigrid context 
.  l - the level this is to be used for
.  c - the space

.keywords: MG, multigrid, set, solution, level

.seealso: MGSetRhs(), MGSetR()
@*/
int MGSetX(PC pc,int l,Vec c)  
{ 
  MG *mg = (MG*) pc->data;
  mg[l]->x  = c;
  return 0;
}

/*@
   MGSetR - Sets the vector space to be used to store the residual on a
   particular level.  The user should free this space at the conclusion of
   multigrid use.

   Input Parameters:
.  pc - the multigrid context 
.  l - the level this is to be used for
.  c - the space

.keywords: MG, multigrid, set, residual, level
@*/
int MGSetR(PC pc,int l,Vec c)
{ 
  MG *mg = (MG*) pc->data;
  mg[l]->r  = c;
  return 0;
}






