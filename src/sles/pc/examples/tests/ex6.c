#ifndef lint
static char vcid[] = "$Id: ex39.c,v 1.13 1996/08/01 14:33:53 balay Exp $";
#endif

static char help[] = "Creates a matrix using 9 pt stensil, and uses it to \n\
test  MatIncreaseOverlap (needed for aditive schwarts preconditioner \n\
  -m <size>       : problem size\n\
  -x1, -x2 <size> : no of subdomains in x and y directions\n\n";
#include "sles.h"
#include  <stdio.h>

int FormElementStiffness(double H,Scalar *Ke)
{
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  return 0;
}
int FormElementRhs(double x, double y, double H,Scalar *r)
{
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0; 
  return 0;
}

int main(int argc,char **args)
{
  Mat         C; 
  int         i, m = 2,  N,M, ierr,idx[4], flg, Nsub1, Nsub2, ol=1, x1, x2;
  Scalar      Ke[16];
  double      x,y,h;
  IS          *is1, *is2;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
/*  OptionsGetInt(PETSC_NULL,"-ol",&ol,&flg);*/
  N = (m+1)*(m+1); /* dimension of matrix */
  M = m*m; /* number of elements */
  h = 1.0/m;       /* mesh width */
  x1= (m+1)/2;
  x2= x1;
  ierr = OptionsGetInt(PETSC_NULL,"-x1",&x1,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-x2",&x2,&flg); CHKERRA(ierr);
  /* create stiffness matrix */
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,N,N,9,PETSC_NULL,&C); CHKERRA(ierr);

  /* forms the element stiffness for the Laplacian */
  ierr = FormElementStiffness(h*h,Ke); CHKERRA(ierr);
  for ( i=0; i<M; i++ ) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + ( i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES); CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);


  for (ol=0; ol<m+2; ++ol) {

    ierr = PCASMCreateSubdomains2D( m+1, m+1 , x1, x2, 1, 0 , &Nsub1, &is1); CHKERRA(ierr);
    ierr = MatIncreaseOverlap(C, Nsub1, is1, ol);                    CHKERRA(ierr);
    ierr = PCASMCreateSubdomains2D( m+1, m+1,x1, x2, 1, ol, &Nsub2, &is2); CHKERRA(ierr);
    
    PetscPrintf(MPI_COMM_SELF,"flg == 1 => both index sets are same\n");
    if( Nsub1 != Nsub2){
      PetscPrintf(MPI_COMM_SELF,"Error: No of indes sets don't match\n");
    }
    
    for (i=0; i<Nsub1; ++i) {
      ISEqual(is1[i], is2[i], (PetscTruth*)&flg);
      PetscPrintf(MPI_COMM_SELF,"i =  %d, flg = %d \n",i, flg);
      
    }
    for (i=0; i<Nsub1; ++i) ISDestroy(is1[i]);     
    for (i=0; i<Nsub2; ++i) ISDestroy(is2[i]);     
  

    ierr = PetscFree(is1); CHKERRA(ierr);
    ierr = PetscFree(is2); CHKERRA(ierr);
  }
    ierr = MatDestroy(C);  CHKERRA(ierr);  
    PetscFinalize();
return 0;
}

