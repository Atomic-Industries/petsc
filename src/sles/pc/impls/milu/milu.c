#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sda2.c,v 1.11 1998/03/20 22:53:26 bsmith Exp $";
#endif

/*
    Contributed by  Victor Eijkhout <eijkhout@cs.utk.edu>, September 1998
*/
/*
    One may access this routine by either 
  1) run your program with the option -pc_type PCCreate_mILU
  2) in your program after creating a SLES object with SLESCreate() and extracting
     the PC Object with SLESGetPC() call
     PCSetType(pc,"PCCreate_mILU");
*/

#include <stdlib.h>
#include "src/pc/pcimpl.h"

/*
  Manteuffel variant of ILU
  @article{Ma:incompletefactorization,
  author = {T.A. Manteuffel},
  title = {An incomplete factorization technique for positive definite
      linear systems},
  journal = {Math. Comp.},
  volume = {34},
  year = {1980},
  pages = {473--497},
  abstract = {Extension of Meyerink/vdVorst to H-matrices;
      shifted ICCG: if $A=D-B$ (diagonal) then
      $A(\alpha)=D-{1\over 1+\alpha}B$; for $\alpha\geq\alpha_n>0$
      all pivots will be positive; find $\alpha_n$ by trial and error.},
  keywords = {incomplete factorization, positive definite matrices,
      H-matrices}
  }
*/

/****************************************************************
  User interface routines
****************************************************************/
#undef __FUNC__
#define __FUNC__ "PCmILUSetLevels"
int PCmILUSetLevels(PC pc,int levels)
{
  PC base_pc = (PC) pc->data;
  int ierr;

  PetscFunctionBegin;
  ierr = PCILUSetLevels(base_pc,levels); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "PCmILUSetBaseType"
int PCmILUSetBaseType(PC pc,PCType type)
{
  PC base_pc = (PC) pc->data;
  int ierr;

  PetscFunctionBegin;
  ierr = PCSetType(base_pc,type); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/****************************************************************
  Implementation
****************************************************************/

#undef __FUNC__
#define __FUNC__ "PCSetup_mILU"
static int PCSetup_mILU(PC pc)
{
  PC base_pc = (PC) pc->data;
  Mat omat = pc->pmat,pmat;
  Vec diag;
  Scalar *dia;
  double *mprop;
  int lsize,first,last,ierr;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(omat,&first,&last); CHKERRQ(ierr);
  lsize = last-first;
  mprop = (Scalar *) PetscMalloc((lsize+1)*sizeof(double)); CHKPTRQ(mprop);
  {
    int irow;
    for (irow=first; irow<last; irow++) {
      int icol,ncols,*cols; Scalar *vals; double mp=0.;
      ierr = MatGetRow(omat,irow,&ncols,&cols,&vals); CHKERRQ(ierr);
      for (icol=0; icol<ncols; icol++) {
	if (cols[icol]==irow) {
	  mp += PetscAbsScalar(vals[icol]);
	} else {
	  mp -= PetscAbsScalar(vals[icol]);
	}
      }
      ierr = MatRestoreRow(omat,irow,&ncols,&cols,&vals); CHKERRQ(ierr);
      mprop[irow-first] = -PetscMin(0,mp);
    }
  }
  ierr = MatConvert(omat,MATSAME,&pmat); CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,lsize,&diag); CHKERRQ(ierr);
  ierr = MatGetDiagonal(omat,diag); CHKERRQ(ierr);
  ierr = VecGetArray(diag,&dia); CHKERRQ(ierr);
  ierr = PCSetOperators(base_pc,pc->mat,pmat,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = PCSetVector(base_pc,pc->vec); CHKERRQ(ierr);

#define ATTEMPTS 5
  {
    Mat lu; Vec piv;
    Scalar *elt;
    int irow,bd,t,try = 0;
    ierr = VecDuplicate(diag,&piv); CHKERRQ(ierr);
    do {
      ierr = PCSetUp(base_pc); CHKERRQ(ierr);
      ierr = PCGetFactoredMatrix(base_pc,&lu);
      ierr = MatGetDiagonal(lu,piv); CHKERRA(ierr);
      ierr = VecGetArray(piv,&elt); CHKERRA(ierr);
      bd = 0; for (t=0; t<lsize; t++) if (elt[t]<.0) bd++;
      ierr = VecRestoreArray(piv,&elt); CHKERRA(ierr);
      if (bd>0) {
	/*printf("negative pivots %d\n",bd);*/
	try++;
	for (t=0; t<lsize; t++) {
	  Scalar v = dia[t]+(mprop[t]*try)/ATTEMPTS;
	  int row  = first+t;
	  ierr = MatSetValues(pmat,1,&row,1,&row,&v,INSERT_VALUES);CHKERRQ(ierr);
	}
	ierr = MatAssemblyBegin(pmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(pmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = PCSetOperators(base_pc,pc->mat,pmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      } /*else printf("mILU factorisation succeeded on %d\n",try);*/
    } while (bd>0);
    ierr = VecDestroy(piv); CHKERRQ(ierr);
  }
  
  ierr = VecRestoreArray(diag,&dia); CHKERRQ(ierr);
  ierr = VecDestroy(diag); CHKERRQ(ierr);
  PetscFree(mprop);

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "PCApply_mILU"
static int PCApply_mILU(PC pc,Vec x,Vec y)
{
  PC base_pc = (PC) pc->data;
  int ierr;
  
  PetscFunctionBegin;
  ierr = PCApply(base_pc,x,y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "PCDestroy_mILU"
static int PCDestroy_mILU(PC pc)
{
  PC base_pc = (PC) pc->data;
  int ierr;
  
  PetscFunctionBegin;
  ierr = MatDestroy(base_pc->pmat);CHKERRQ(ierr);
  ierr = PCDestroy(base_pc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_mILU"
static int PCView_mILU(PC pc,Viewer viewer)
{
  PC base_pc = (PC) pc->data;
  FILE       *fd;
  int        ierr;
  ViewerType vtype;
 
  PetscFunctionBegin;
  ViewerGetType(viewer,&vtype);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(pc->comm,fd,"    modified ILU preconditioner\n");
    PetscFPrintf(pc->comm,fd,"    see src/contrib/pc/milu/milu.c\n");
    PetscFPrintf(pc->comm,fd,"    base PC used by mILU next\n");
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  ierr = PCView(base_pc,viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "PCCreate_mILU"
int PCCreate_mILU(PC pc)
{
  PC base_pc;
  int ierr;

  PetscFunctionBegin;
  pc->apply            = PCApply_mILU;
  pc->applyrich        = 0;
  pc->destroy          = PCDestroy_mILU;
  pc->setfromoptions   = 0;
  pc->printhelp        = 0;
  pc->setup            = PCSetup_mILU;
  pc->view             = PCView_mILU;

  ierr = PCCreate(pc->comm,&base_pc); CHKERRQ(ierr);
  ierr = PCSetType(base_pc,PCILU); CHKERRQ(ierr);
  pc->data = (void *) base_pc;

  PetscFunctionReturn(0);
}


