#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pinit.c,v 1.7 1998/12/04 23:25:28 bsmith Exp bsmith $";
#endif
/*

   This file defines the initialization of PETSc, including PetscInitialize()

*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "sys.h"
#include "src/sys/nreg.h"

/* -----------------------------------------------------------------------------------------*/


extern int PetscInitialize_DynamicLibraries();
extern int PetscFinalize_DynamicLibraries();
extern int FListDestroyAll();

#include "snes.h" /* so that cookies are defined */

/*
       Checks the options database for initializations related to the 
    PETSc components
*/
#undef __FUNC__  
#define __FUNC__ "OptionsCheckInitial_Components"
int OptionsCheckInitial_Components(void)
{
  MPI_Comm comm = PETSC_COMM_WORLD;
  int      flg1,ierr;
  char     mname[256];

  PetscFunctionBegin;

  ierr = OptionsGetString(PETSC_NULL,"-log_info_exclude",mname,256, &flg1);CHKERRQ(ierr);
  if (flg1) {
    if (PetscStrstr(mname,"null")) {
      PLogInfoDeactivateClass(PETSC_NULL);
    }
    if (PetscStrstr(mname,"vec")) {
      PLogInfoDeactivateClass(VEC_COOKIE);
    }
    if (PetscStrstr(mname,"mat")) {
      PLogInfoDeactivateClass(MAT_COOKIE);
    }
    if (PetscStrstr(mname,"sles")) {
      PLogInfoDeactivateClass(SLES_COOKIE);
    }
    if (PetscStrstr(mname,"snes")) {
      PLogInfoDeactivateClass(SNES_COOKIE);
    }
  }
  ierr = OptionsGetString(PETSC_NULL,"-log_summary_exclude",mname,256, &flg1);CHKERRQ(ierr);
  if (flg1) {
    if (PetscStrstr(mname,"vec")) {
      PLogEventDeactivateClass(VEC_COOKIE);
    }
    if (PetscStrstr(mname,"mat")) {
      PLogEventDeactivateClass(MAT_COOKIE);
    }
    if (PetscStrstr(mname,"sles")) {
      PLogEventDeactivateClass(SLES_COOKIE);
    }
    if (PetscStrstr(mname,"snes")) {
      PLogEventDeactivateClass(SNES_COOKIE);
    }
  }
    
  ierr = OptionsHasName(PETSC_NULL,"-log_sync",&flg1);CHKERRQ(ierr);
  if (flg1) {
    PLogEventActivate(VEC_ScatterBarrier);
    PLogEventActivate(VEC_NormBarrier);
    PLogEventActivate(VEC_NormComm);
    PLogEventActivate(VEC_DotBarrier);
    PLogEventActivate(VEC_DotComm);
    PLogEventActivate(VEC_MDotBarrier);
    PLogEventActivate(VEC_MDotComm);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help", &flg1); CHKERRQ(ierr);
  if (flg1) {
#if defined (USE_PETSC_LOG)
    (*PetscHelpPrintf)(comm,"------Additional PETSc component options--------\n");
    (*PetscHelpPrintf)(comm," -log_summary_exclude: <vec,mat,sles,snes>\n");
    (*PetscHelpPrintf)(comm," -log_info_exclude: <null,vec,mat,sles,snes,ts>\n");
    (*PetscHelpPrintf)(comm,"-----------------------------------------------\n");
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscInitializeNoArguments"
/*@C
      PetscInitializeNoArguments - Calls PetscInitialize() from C/C++ without
        the command line arguments.

.seealso: PetscInitialize(), PetscInitializeFortran()
@*/
int PetscInitializeNoArguments(void)
{
  int ierr;

  PetscFunctionBegin;
  ierr = AliceInitializeNoArguments();
  PetscFunctionReturn(ierr);
}

#undef __FUNC__  
#define __FUNC__ "PetscInitialize"
/*@C
   PetscInitialize - Initializes the PETSc database and MPI. 
   PetscInitialize() calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of 
   your program -- usually the very first line! 

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use PETSC_NULL for default)
-  help - [optional] Help message to print, use PETSC_NULL for no message

   Options Database Keys:
+  -start_in_debugger [noxterm,dbx,xdb,gdb,...] - Starts program in debugger
+  -on_error_attach_debugger [noxterm,dbx,xdb,gdb,...] - Starts debugger when error detected
.  -debugger_nodes [node1,node2,...] - Indicates nodes to start in debugger
.  -debugger_pause [sleeptime] (in seconds) - Pauses debugger
.  -trmalloc - Indicates use of PETSc error-checking malloc
.  -trmalloc_off - Indicates not to use error-checking malloc
.  -fp_trap - Stops on floating point exceptions (Note that on the
              IBM RS6000 this slows code by at least a factor of 10.)
-  -no_signal_handler - Indicates not to trap error signals

   Options Database Keys for Profiling:
   See the 'Profiling' chapter of the users manual for details.
+  -log_trace [filename] - Print traces of all PETSc calls
        to the screen (useful to determine where a program
        hangs without running in the debugger).  See PLogTraceBegin().
.  -log_info <optional filename> - Prints verbose information to the screen
-  -log_info_exclude <null,vec,mat,sles,snes,ts> - Excludes some of the verbose messages

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   Fortran Version:
   In Fortran this routine has the format
$       call PetscInitialize(file,ierr)

+   ierr - error return code
-   file - [optional] PETSc database file name, defaults to 
           ~username/.petscrc (use PETSC_NULL_CHARACTER for default)
           
   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use PETSC_NULL as 
   in the C version.  See the users manual for details.


.keywords: initialize, options, database, startup

.seealso: PetscFinalize(), PetscInitializeFortran()
@*/
int PetscInitialize(int *argc,char ***args,char file[],const char help[])
{
  int        ierr;

  PetscFunctionBegin;
  ierr = AliceInitialize(argc,args,file,help);CHKERRQ(ierr);
  ierr = OptionsCheckInitial_Components();CHKERRQ(ierr);

  /*
      Initialize the default dynamic libraries
  */
  PetscFunctionReturn(ierr);
}

#undef __FUNC__  
#define __FUNC__ "PetscFinalize"
/*@C 
   PetscFinalize - Checks for options to be called at the conclusion
   of the program and calls MPI_Finalize().

   Collective on PETSC_COMM_WORLD

   Options Database Keys:
+  -optionstable - Calls OptionsPrint()
.  -optionsleft - Prints unused options that remain in the database
.  -mpidump - Calls PetscMPIDump()
.  -trdump - Calls PetscTrDump()
.  -trinfo - Prints total memory usage
.  -trdebug - Calls malloc_debug(2) to activate memory
        allocation diagnostics (used by PETSC_ARCH=sun4, 
        BOPT=[g,g_c++,g_complex] only!)
-  -trmalloc_log - Prints summary of memory usage

   Options Database Keys for Profiling:
   See the 'Profiling' chapter of the users manual for details.
+  -log_summary [filename] - Prints summary of flop and timing
        information to screen. If the filename is specified the
        summary is written to the file. (for code compiled with 
        USE_PETSC_LOG).  See PLogPrintSummary().
.  -log_all [filename] - Logs extensive profiling information
        (for code compiled with USE_PETSC_LOG). See PLogDump(). 
.  -log [filename] - Logs basic profiline information (for
        code compiled with USE_PETSC_LOG).  See PLogDump().
.  -log_sync - Log the synchronization in scatters, inner products
        and norms
-  -log_mpe [filename] - Creates a logfile viewable by the 
      utility Upshot/Nupshot (in MPICH distribution)

   Note:
   See PetscInitialize() for more general runtime options.

.keywords: finalize, exit, end

.seealso: PetscInitialize(), OptionsPrint(), PetscTrDump(), PetscMPIDump()
@*/
int PetscFinalize(void)
{
  int ierr;
  
  PetscFunctionBegin;
  /*
     Destroy all the function registration lists created
  */
  ierr = NRDestroyAll(); CHKERRQ(ierr);
  ierr = FListDestroyAll(); CHKERRQ(ierr); 
  ierr = PetscFinalize_DynamicLibraries();CHKERRQ(ierr);
  ierr = AliceFinalize();
  PetscFunctionReturn(ierr);
}
