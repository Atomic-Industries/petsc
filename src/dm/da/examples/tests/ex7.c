#ifndef lint
static char vcid[] = "$Id: ex7.c,v 1.5 1996/03/19 21:29:46 bsmith Exp bsmith $";
#endif

static char help[] = "Tests DALocalToLocal().\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  int            rank,M=8,ierr,dof=1,stencil_width=1,flg=0,i,start,end,P=5;
  int            flg2,flg3,N = 6,m=PETSC_DECIDE,n=PETSC_DECIDE,p=PETSC_DECIDE;
  DAPeriodicType periodic;
  DAStencilType  stencil_type;
  DA             da;
  Vec            local,global,local_copy;
  Scalar         value,mone = -1.0;
  double         norm,work;
  Viewer         viewer;
  char           filename[64];
  FILE           *file;


  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-dof",&dof,&flg);  CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-stencil_width",&stencil_width,&flg);  CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-periodic",(int*)&periodic,&flg);  CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-stencil_type",(int*)&stencil_type,&flg);  CHKERRA(ierr); 

  ierr = OptionsHasName(PETSC_NULL,"-2d",&flg2); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-3d",&flg3); CHKERRA(ierr);
  if (flg2) {
    ierr = DACreate2d(MPI_COMM_WORLD,periodic,stencil_type,M,N,m,n,dof,stencil_width,&da);
    CHKERRA(ierr);
  } else if (flg3) {
    ierr = DACreate3d(MPI_COMM_WORLD,periodic,stencil_type,M,N,P,m,n,p,dof,stencil_width,&da);
    CHKERRA(ierr);
  }
  else {
    ierr = DACreate1d(MPI_COMM_WORLD,periodic,M,dof,stencil_width,PETSC_DECIDE,&da);CHKERRA(ierr);
  }

  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);
  ierr = VecDuplicate(local,&local_copy); CHKERRA(ierr);

  
  /* zero out vectors so that ghostpoints are zero */
  value = 0;
  ierr = VecSet(&value,local); CHKERRA(ierr);
  ierr = VecSet(&value,local_copy); CHKERRA(ierr);

  ierr = VecGetOwnershipRange(global,&start,&end); CHKERRA(ierr);
  for ( i=start; i<end; i++ ) {
    value = i + 1;
    VecSetValues(global,1,&i,&value,INSERT_VALUES); 
  }
  ierr = VecAssemblyBegin(global); CHKERRA(ierr);
  ierr = VecAssemblyEnd(global); CHKERRA(ierr);

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);


  ierr = DALocalToLocalBegin(da,local,INSERT_VALUES,local_copy); CHKERRA(ierr);
  ierr = DALocalToLocalEnd(da,local,INSERT_VALUES,local_copy); CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-save",&flg); CHKERRA(ierr);
  if (flg) {
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    sprintf(filename,"local.%d",rank);
    ierr = ViewerFileOpenASCII(MPI_COMM_SELF,filename,&viewer);CHKERRA(ierr);
    ierr = ViewerASCIIGetPointer(viewer,&file); CHKERRA(ierr);
    ierr = VecView(local,viewer); CHKERRA(ierr);
    fprintf(file,"Vector with correct ghost points\n");
    ierr = VecView(local_copy,viewer); CHKERRA(ierr);
    ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  }

  ierr = VecAXPY(&mone,local,local_copy); CHKERRA(ierr);
  ierr = VecNorm(local_copy,NORM_MAX,&work); CHKERRA(ierr);
  MPI_Allreduce( &work, &norm,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD );
  PetscPrintf(MPI_COMM_WORLD,"Norm of difference %g should be zero\n",norm);
   

  ierr = DADestroy(da); CHKERRA(ierr);
  ierr = VecDestroy(local_copy); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
