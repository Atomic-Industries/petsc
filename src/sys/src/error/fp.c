#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fp.c,v 1.48 1998/04/26 03:01:29 bsmith Exp balay $";
#endif
/*
*	IEEE error handler for all machines. Since each machine has 
*   enough slight differences we have completely separate codes for each one.
*
*/
#include <signal.h>
#include "petsc.h"           /*I  "petsc.h"  I*/
#include "sys.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"


/*----------------sun4 ---------------------------------------------------*/
#if defined(PARCH_sun4) 
#include <floatingpoint.h>
#if defined(__cplusplus)
extern "C" {
#endif
int ieee_flags(char*,char*,char*,char**);
int ieee_handler(char *,char *,sigfpe_handler_type(int,int,struct sigcontext*,char *));
#if defined(__cplusplus)
}
#endif
struct { int code_no; char *name; } error_codes[] = {
           { FPE_INTDIV_TRAP	, "integer divide" } ,
	   { FPE_FLTOPERR_TRAP	, "IEEE operand error" } ,
	   { FPE_FLTOVF_TRAP	, "floating point overflow" } ,
	   { FPE_FLTUND_TRAP	, "floating point underflow" } ,
	   { FPE_FLTDIV_TRAP	, "floating pointing divide" } ,
	   { FPE_FLTINEX_TRAP	, "inexact floating point result" } ,
	   { 0			, "unknown error" } 
} ;
#define SIGPC(scp) (scp->sc_pc)

#undef __FUNC__  
#define __FUNC__ "PetscDefaultFPTrap" 
sigfpe_handler_type PetscDefaultFPTrap(int sig,int code,struct sigcontext *scp,char *addr)
{
  int err_ind = -1, j,ierr;

  PetscFunctionBegin;
  for ( j = 0 ; error_codes[j].code_no ; j++ ) {
    if ( error_codes[j].code_no == code ) err_ind = j ;
  }

  if ( err_ind >= 0 ) {
    (*PetscErrorPrintf)("*** %s occurred at pc=%X ***\n",error_codes[err_ind].name, SIGPC(scp));
  } else {
    (*PetscErrorPrintf)("*** floating point error 0x%x occurred at pc=%X ***\n",code, SIGPC(scp));
  }
  ierr = PetscError(PETSC_ERR_FP,"unknownfunction","Unknown file",0,1,0,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSetFPTrap"
/*@
   PetscSetFPTrap - Enables traps/exceptions on common floating point errors.
                    This option may not work on certain machines.

   Input Parameters:
.  flag - PETSC_FP_TRAP_ON, PETSC_FP_TRAP_OFF.

   Not Collective

   Options Database Keys:
$  -fp_trap - turns on floating point trapping

   Description:
   On systems that support it, this routine causes floating point
   overflow, divide-by-zero, and invalid-operand (e.g., a NaN) to
   cause a message to be printed and the program to exit.

   Caution:
   On certain machines, in particular the IBM rs6000, floating point 
   trapping is VERY slow.


.keywords: floating point trap, overflow, divide-by-zero, invalid-operand
@*/
int PetscSetFPTrap(int flag)
{
  char *out; 

  PetscFunctionBegin;
  /* Clear accumulated exceptions.  Used to try and suppress
	   meaningless messages from f77 programs */
  (void) ieee_flags( "clear", "exception", "all", &out );
  if (flag == PETSC_FP_TRAP_ON) {
    if (ieee_handler("set","common",PetscDefaultFPTrap)) {
      /*
        To trap more fp exceptions, including undrflow, change the above line to
        if (ieee_handler("set","all",PetscDefaultFPTrap)) {
      */
      (*PetscErrorPrintf)( "Can't set floatingpoint handler\n");
    }
  } else {
    if (ieee_handler("clear","common",PetscDefaultFPTrap)) {
      (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
    }
  }
  PetscFunctionReturn(0);
}

/* ------------------------ SOLARIS --------------------------------------*/
#elif defined(PARCH_solaris) && defined(HAVE_SUNMATH_H)
#include <sunmath.h>
#include <floatingpoint.h>
#include <siginfo.h>
#include <ucontext.h>

struct { int code_no; char *name; } error_codes[] = {
  {  FPE_FLTINV, "invalid floating point operand"},
  {  FPE_FLTRES, "inexact floating point result"},
  {  FPE_FLTDIV, "division-by-zero"},
  {  FPE_FLTUND, "floating point underflow"},
  {  FPE_FLTOVF, "floating point overflow"},
  {  0,          "unknown error"}
};
#define SIGPC(scp) (scp->si_addr)

#undef __FUNC__  
#define __FUNC__ "PetscDefaultFPTrap"
void PetscDefaultFPTrap(int sig, siginfo_t *scp,ucontext_t *uap)
{
  int err_ind, j,ierr,code = scp->si_code;

  PetscFunctionBegin;
  err_ind = -1 ;
  for ( j = 0 ; error_codes[j].code_no ; j++ ) {
    if ( error_codes[j].code_no == code ) err_ind = j ;
  }

  if ( err_ind >= 0 ) {
    (*PetscErrorPrintf)( "*** %s occurred at pc=%X ***\n",error_codes[err_ind].name, SIGPC(scp));
  } else {
    (*PetscErrorPrintf)("*** floating point error 0x%x occurred at pc=%X ***\n",code, SIGPC(scp));
  }
  ierr = PetscError(PETSC_ERR_FP,"unknownfunction","Unknown file",0,1,0,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}

int PetscSetFPTrap(int flag)
{
  char *out; 

  PetscFunctionBegin;
  (void) ieee_flags( "clear", "exception", "all", &out );
  if (flag == PETSC_FP_TRAP_ON) {
    if (ieee_handler("set","common",(sigfpe_handler_type)PetscDefaultFPTrap)) {
      (*PetscErrorPrintf)( "Can't set floating point handler\n");
    }
  
    /*
    sigfpe(FPE_FLTINV,PetscDefaultFPTrap);
    sigfpe(FPE_FLTRES,PetscDefaultFPTrap);
    sigfpe(FPE_FLTDIV,PetscDefaultFPTrap);
    sigfpe(FPE_FLTUND,PetscDefaultFPTrap);
    sigfpe(FPE_FLTOVF,PetscDefaultFPTrap); 
    */
  } else {
    if (ieee_handler("clear","common",(sigfpe_handler_type)PetscDefaultFPTrap)) {
     (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
    }
  }
  PetscFunctionReturn(0);
}

/* ------------------------ IRIX64 --------------------------------------*/
/*
   64 bit machine does not have fp handling!!!!
*/
#elif defined(PARCH_IRIX64) || defined (PARCH_IRIX)
#undef __FUNC__  
#define __FUNC__ "PetscSetFPTrap"
int PetscSetFPTrap(int flag)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* ------------------------ IRIX --------------------------------------*/

#elif defined (PARCH_IRIX5)
#include <sigfpe.h>
struct { int code_no; char *name; } error_codes[] = {
       { _INVALID   , "IEEE operand error" } ,
       { _OVERFL    , "floating point overflow" } ,
       { _UNDERFL   , "floating point underflow" } ,
       { _DIVZERO   , "floating point divide" } ,
       { 0          , "unknown error" }
} ;
#undef __FUNC__  
#define __FUNC__ "PetscDefaultFPTrap" 
void PetscDefaultFPTrap( unsigned exception[],int val[] )
{
  int err_ind, j, code,ierr;

  PetscFunctionBegin;
  code = exception[0];
  err_ind = -1 ;
  for ( j = 0 ; error_codes[j].code_no ; j++ ){
    if ( error_codes[j].code_no == code ) err_ind = j ;
  }
  if ( err_ind >= 0 ){
    (*PetscErrorPrintf)( "*** %s occurred ***\n",error_codes[err_ind].name );
  } else{
    (*PetscErrorPrintf)("*** floating point error 0x%x occurred ***\n",code);  
  }
  ierr = PetscError(PETSC_ERR_FP,"unknownfunction","Unknown file",0,1,0,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSetFPTrap" 
int PetscSetFPTrap(int flag)
{
  PetscFunctionBegin;
  if (flag == PETSC_FP_TRAP_ON) {
#if !defined(__cplusplus)
    sigfpe_[_EN_OVERFL].abort = 1;
    sigfpe_[_EN_DIVZERO].abort = 1;
    sigfpe_[_EN_INVALID].abort = 1;
#endif
    handle_sigfpes(_ON,_EN_OVERFL|_EN_DIVZERO|_EN_INVALID,PetscDefaultFPTrap,_ABORT_ON_ERROR,0);
  } else {
    handle_sigfpes(_OFF,_EN_OVERFL|_EN_DIVZERO|_EN_INVALID,0,_ABORT_ON_ERROR,0);
  }
  PetscFunctionReturn(0);
}
/* ------------------------Paragon-------------------------------------*/
#elif defined(PARCH_paragon)
/* 
    You have to compile YOUR code with -Knoieee to catch divide-by-zero 
*/

#include <ieeefp.h>
struct { int code_no; char *name; } error_codes[] = {
       { FP_X_OFL    , "floating point overflow" } ,
       { FP_X_DZ     , "floating point divide" } ,
       { FP_X_INV    , "invalid floating point operand" } ,
       { 0           , "unknown error" }
} ;

#undef __FUNC__  
#define __FUNC__ "PetscDefaultFPTrap"
void PetscDefaultFPTrap(int sig)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscError(PETSC_ERR_FP,"unknownfunction","Unknown file",0,1,0,"floating point error");
}

#undef __FUNC__  
#define __FUNC__ "PetscSetFPTrap"
int PetscSetFPTrap(int on)
{
  int flag;

  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    fpsetmask( FP_X_OFL | FP_X_DZ | FP_X_INV );
    flag = (int) 	signal(SIGFPE,PetscDefaultFPTrap);
    if (flag == -1) (*PetscErrorPrintf)( "Can't set floatingpoint handler\n");
  } else {
    fpsetmask(0);
    flag = (int)  signal(SIGFPE,PetscDefaultFPTrap);
    if (flag == -1) (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
  }
  PetscFunctionReturn(0);
}
/*--------------- IBM RS6000.----------------------------------------------*/
/* In "fast" mode, floating point traps are imprecise and ignored.
   This is the reason for the fptrap( FP_TRAP_SYNC ) call */
#elif defined(PARCH_rs6000) 
struct sigcontext;
#include <fpxcp.h>
#include <fptrap.h>
#include <stdlib.h>
#define FPE_FLTOPERR_TRAP (fptrap_t)(0x20000000)
#define FPE_FLTOVF_TRAP   (fptrap_t)(0x10000000)
#define FPE_FLTUND_TRAP   (fptrap_t)(0x08000000)
#define FPE_FLTDIV_TRAP   (fptrap_t)(0x04000000)
#define FPE_FLTINEX_TRAP  (fptrap_t)(0x02000000)

struct { int code_no; char *name; } error_codes[] = {
           {FPE_FLTOPERR_TRAP	, "IEEE operand error" } ,
	   { FPE_FLTOVF_TRAP	, "floating point overflow" } ,
	   { FPE_FLTUND_TRAP	, "floating point underflow" } ,
	   { FPE_FLTDIV_TRAP	, "floating point divide" } ,
	   { FPE_FLTINEX_TRAP	, "inexact floating point result" } ,
	   { 0			, "unknown error" } 
} ;
#define SIGPC(scp) (0) /* Info MIGHT be in scp->sc_jmpbuf.jmp_context.iar */
/* 
   For some reason, scp->sc_jmpbuf doesn't work on the RS6000, even though
   it looks like it should from the include definitions.  It is probably
   some strange interaction with the "POSIX_SOURCE" that we require.
*/

#undef __FUNC__  
#define __FUNC__ "PetscDefaultFPTrap"
void PetscDefaultFPTrap(int sig,int code,struct sigcontext *scp )
{
  int      ierr,err_ind, j;
  fp_ctx_t flt_context;

  PetscFunctionBegin;
  fp_sh_trap_info( scp, &flt_context );
    
  err_ind = -1 ;
  for ( j = 0 ; error_codes[j].code_no ; j++ ) {
    if ( error_codes[j].code_no == flt_context.trap ) err_ind = j ;
  }

  if ( err_ind >= 0 ){
    (*PetscErrorPrintf)( "*** %s occurred ***\n",error_codes[err_ind].name );
  } else{
    (*PetscErrorPrintf)("*** floating point error 0x%x occurred ***\n", flt_context.trap );
  }
  ierr = PetscError(PETSC_ERR_FP,"unknownfunction","Unknown file",0,1,0,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSetFPTrap"
int PetscSetFPTrap(int on)
{
  int flag;

  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    signal( SIGFPE, (void (*)(int))PetscDefaultFPTrap );
    fp_trap( FP_TRAP_SYNC );
    fp_enable( TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW );
    /* fp_enable(mask) for individual traps.  Values are:
       TRP_INVALID
       TRP_DIV_BY_ZERO
       TRP_OVERFLOW
       TRP_UNDERFLOW
       TRP_INEXACT
       Can OR then together.
       fp_enable_all(); for all traps.
    */
  } else {
    signal(SIGFPE,SIG_DFL);
    fp_disable( TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW );
    fp_trap( FP_TRAP_OFF );
  }
  PetscFunctionReturn(0);
}

/* -------------------------Default -----------------------------------*/
#else 
struct { int code_no; char *name; } error_codes[] = {
	   { 0		, "unknown error" } 
} ;
#undef __FUNC__  
#define __FUNC__ "PetscDefaultFPTrap"
void PetscDefaultFPTrap(int sig)
{
  int ierr;

  PetscFunctionBegin;
  (*PetscErrorPrintf)( "*** floating point error occurred ***\n" );
  ierr = PetscError(PETSC_ERR_FP,"unknownfunction","Unknown file",0,1,0,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}
#undef __FUNC__  
#define __FUNC__ "PetscSetFPTrap"
int PetscSetFPTrap(int on)
{
  int flag;

  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    flag = (int) signal(SIGFPE,PetscDefaultFPTrap);
    if (flag == -1) (*PetscErrorPrintf)( "Can't set floatingpoint handler\n");
  } else {
    flag = (int) signal(SIGFPE,SIG_DFL);
    if (flag == -1) (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
  }
  PetscFunctionReturn(0);
}
#endif

/***************************************************************************
  This code enables and clears "benign" underflows.  This is for code that
  does NOT care how underflows are handled, just that they be handled "fast".
  Note that for general-purpose code, this can be very, very dangerous.
  But for many PDE calculations, it may even be provable that underflows
  are benign.

  Not all machines need worry or care about this, but for those that do,
  we provide routines to change to/from a benign mode.
 ***************************************************************************/
#if defined(PARCH_paragon)
#undef __FUNC__  
#define __FUNC__ "PetscSetBenignUnderflows"
int PetscSetBenignUnderflows(void)
{

  PetscFunctionBegin;
  /* This needs the following assembly-language program:
  .globl     _set_fsr
  .globl     _set_fsr_

  _set_fsr:
  _set_fsr_:
      st.c   r16,fsr
      bri    r1
      nop
 */

  /*         set_fsr( 0x21 );  */
  PetscFunctionReturn(0);
}
#elif defined(PARCH_rs6000)
#undef __FUNC__  
#define __FUNC__ "PetscSetBenignUnderflows"
int PetscSetBenignUnderflows(void)
{
  PetscFunctionBegin;
  /* abrupt_underflow seems to have disappeared! */
  /* abrupt_underflow(); */
  PetscFunctionReturn(0);
}
#else
#undef __FUNC__  
#define __FUNC__ "PetscSetBenignUnderflows"
int PetscSetBenignUnderflows(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif




