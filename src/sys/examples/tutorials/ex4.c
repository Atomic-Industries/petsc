#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.1 1999/01/22 22:23:37 bsmith Exp bsmith $";
#endif

static char help[] = "Prints loadable objects from dynamic library.\n\n";

/*T
   Concepts: Dynamic libraries;
   Routines: PetscInitialize(); PetscPrintf(); PetscFinalize();
   Processors: n
T*/
 
#include "petsc.h"
int main(int argc,char **argv)
{
  int  ierr,flag;
  char *string,filename[256];
  void *handle;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints the various options that can be applied at 
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help); CHKERRA(ierr);

  ierr = OptionsGetString(PETSC_NULL,"-library",filename,256,&flag);CHKERRA(ierr);
  if (!flag) {
    SETERRA(1,1,"Must indicate library name with -library");
  }

  ierr = DLLibraryOpen(PETSC_COMM_WORLD,filename,&handle);CHKERRA(ierr);

  ierr = DLLibraryGetInfo(handle,"Contents",&string);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Contents:%s\n",string);CHKERRA(ierr);
  ierr = DLLibraryGetInfo(handle,"Authors",&string);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Authors:%s\n",string);CHKERRA(ierr);
  ierr = DLLibraryGetInfo(handle,"Version",&string);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Version:%s\n",string);CHKERRA(ierr);

  ierr = PetscFinalize(); CHKERRA(ierr);
  return 0;
}
