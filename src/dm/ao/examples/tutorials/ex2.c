#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.5 1997/11/03 04:47:01 bsmith Exp $";
#endif

static char help[] = 
"Reads a PETSc matrix and vector from a file and reorders it.\n\
This version first preloads and solves a small system, then loads \n\
another (larger) system and reorders it.  This example illustrates\n\
preloading of instructions with the smaller system so that more accurate\n\
performance monitoring can be done with the larger one (that actually\n\
is the system of interest).  See the 'Performance Hints' chapter of the\n\
users manual for a discussion of preloading.  Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n";

/*T
   Concepts: Mat^Reordering a matrix - loading a binary matrix and vector;
   Concepts: PLog^Profiling multiple stages of code;
   Routines: MatGetReordering();
   Routines: PLogStageRegister(); PLogStagePush(); PLogStagePop(); PLogFlops();
   Routines: PetscBarrier(); PetscGetTime();
   Routines: MatGetTypeFromOptions(); MatLoad(); VecLoad();
   Routines: ViewerFileOpenBinary(); ViewerDestroy();
   Processors: n
T*/

/* 
  Include "mat.h" so that we can use matrices.
  automatically includes:
     petsc.h  - base PETSc routines   vec.h    - vectors
     sys.h    - system routines       mat.h    - matrices
     is.h     - index sets            viewer.h - viewers               
*/
#include "mat.h"

int main(int argc,char **args)
{
  MatType           mtype;            /* matrix format */
  Mat               A;                /* matrix */
  Viewer            fd;               /* viewer */
  char              file[2][128];     /* input file name */
  char              stagename[4][16]; /* names of profiling stages */
  IS                isrow,iscol;      /* row and column permutations */
  int               ierr, flg, i,loops  = 2;
  MatReorderingType rtype = ORDER_RCM;
  PetscTruth        set;

  PetscInitialize(&argc,&args,(char *)0,help);


  /* 
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = OptionsGetString(PETSC_NULL,"-f0",file[0],127,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate binary file with the -f0 option");
  ierr = OptionsGetString(PETSC_NULL,"-f1",file[1],127,&flg); CHKERRA(ierr);
  if (!flg) {loops = 1;} /* don't bother with second system */

  /* -----------------------------------------------------------
                  Beginning of loop
     ----------------------------------------------------------- */
  /* 
     Loop through the reordering 2 times.  
      - The intention here is to preload and solve a small system;
        then load another (larger) system and solve it as well.
        This process preloads the instructions with the smaller
        system so that more accurate performance monitoring (via
        -log_summary) can be done with the larger one (that actually
        is the system of interest). 
  */
  for ( i=0; i<loops; i++ ) {

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Load system i
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Begin profiling next stage
    */
    PLogStagePush(2*i);
    sprintf(stagename[2*i],"Load System %d",i);
    PLogStageRegister(2*i,stagename[2*i]);

    /* 
       Open binary file.  Note that we use BINARY_RDONLY to indicate
       reading from this file.
    */
    ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,file[i],BINARY_RDONLY,&fd);
           CHKERRA(ierr);

    /* 
       Determine matrix format to be used (specified at runtime).
       See the manpage for MatLoad() for available formats.
    */
    ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,0,&mtype,&set); CHKERRQ(ierr);

    /*
       Load the matrix; then destroy the viewer.
    */
    ierr = MatLoad(fd,mtype,&A); CHKERRA(ierr);
    ierr = ViewerDestroy(fd); CHKERRA(ierr);


    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                      Setup loop i
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Conclude profiling last stage; begin profiling next stage.
    */
    /*    PLogStagePop(); */
    PetscBarrier(A);
    PLogStagePush(2*i+1);
    sprintf(stagename[2*i+1],"SLESSetUp %d",i);
    PLogStageRegister(2*i+1,stagename[2*i+1]);

    ierr = MatGetReordering(A,rtype,&isrow,&iscol); CHKERRQ(ierr);

    /*
       Conclude profiling this stage
    */
    PLogStagePop();

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Solve system i
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


    /* 
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
    */
    ierr = MatDestroy(A); CHKERRA(ierr);
    ierr = ISDestroy(isrow);  CHKERRA(ierr);
    ierr = ISDestroy(iscol);  CHKERRA(ierr);
  }
  /* -----------------------------------------------------------
                      End of reordering loop
     ----------------------------------------------------------- */

  PetscFinalize();
  return 0;
}

