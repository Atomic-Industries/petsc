#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.6 1997/11/28 16:22:40 bsmith Exp $";
#endif

static char help[] = "Tests AOData \n\n";

#include "petsc.h"
#include "ao.h"
#include <math.h>

int main(int argc,char **argv)
{
  int         n,nglobal, bs = 2, *keys, *data,ierr,flg,rank,size,i,start;
  double      *gd;
  AOData      aodata;
  Viewer      binary;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); n = rank + 2;
  MPI_Allreduce(&n,&nglobal,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  /*
       Create a database with two sets of keys 
  */
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata);CHKERRA(ierr);

  /*
       Put two segments in the first key and one in the second
  */
  ierr = AODataKeyAdd(aodata,"key1",PETSC_DECIDE,nglobal); CHKERRA(ierr);
  ierr = AODataKeyAdd(aodata,"key2",PETSC_DECIDE,nglobal); CHKERRA(ierr);

  /* allocate space for the keys each processor will provide */
  keys = (int *) PetscMalloc( n*sizeof(int) );CHKPTRA(keys);

  /*
     We assign the first set of keys (0 to 2) to processor 0, etc.
     This computes the first local key on each processor
  */
  MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);
  start -= n;

  for ( i=0; i<n; i++ ) {
    keys[i]     = start + i;
  }

  /* 
      Allocate data for the first key and first segment 
  */
  data = (int *) PetscMalloc( bs*n*sizeof(int) );CHKPTRA(data);
  for ( i=0; i<n; i++ ) {
    data[2*i]   = -(start + i);
    data[2*i+1] = -(start + i) - 10000;
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg1",bs,n,keys,data,PETSC_INT);CHKERRA(ierr); 
  PetscFree(data);

  /*
      Allocate data for first key and second segment 
  */
  bs   = 3;
  gd   = (double *) PetscMalloc( bs*n*sizeof(double) );CHKPTRA(gd);
  for ( i=0; i<n; i++ ) {
    gd[3*i]   = -(start + i);
    gd[3*i+1] = -(start + i) - 10000;
    gd[3*i+2] = -(start + i) - 100000;
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg2",bs,n,keys,gd,PETSC_DOUBLE);CHKERRA(ierr); 

  /*
       Use same data for second key and first segment 
  */
  ierr = AODataSegmentAdd(aodata,"key2","seg1",bs,n,keys,gd,PETSC_DOUBLE);CHKERRA(ierr); 
  PetscFree(gd);
  PetscFree(keys);

  /*
        Save the database to a file
  */
  ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,"dataoutput",BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(aodata,binary);CHKERRA(ierr);
  ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);
 
  ierr = AODataDestroy(aodata); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 


