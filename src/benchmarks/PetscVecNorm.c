#ifndef lint
static char vcid[] = "$Id: PetscMemcpy.c,v 1.7 1997/03/09 18:00:35 bsmith Exp $";
#endif

#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  PLogDouble x, y, z;
  int        i;
  Scalar     A[10000], B[10000];

  PetscInitialize(&argc, &argv,0,0);
  for (i=0; i<10000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  x = PetscGetTime();

  x = PetscGetTime();
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  y = PetscGetTime();
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  z = PetscGetTime();

  fprintf(stderr,"%s : \n","PetscMemcpy");
  fprintf(stderr,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stderr,"    %-11s : %e sec\n","Per Scalar",(2*y-x-z)/100000.0);

  PetscFinalize();
  return 0;
}
