#ifndef lint
static char vcid[] = "$Id: xmon.c,v 1.23 1997/01/06 20:22:18 balay Exp bsmith $";
#endif

#include "petsc.h"
#include "src/draw/drawimpl.h"  /*I  "draw.h"  I*/
#include "src/ksp/kspimpl.h"              /*I  "ksp.h"   I*/
#include <math.h>


#undef __FUNC__  
#define __FUNC__ "KSPLGMonitorCreate" /* ADIC Ignore */
/*@C
   KSPLGMonitorCreate - Creates a line graph context for use with 
   KSP to monitor convergence of preconditioned residual norms.

   Input Parameters:
.  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
.  m, n - the screen width and height in pixels

   Output Parameter:
.  draw - the drawing context

   Options Database Key:
$    -ksp_xmonitor : automatically sets line graph monitor

   Notes: 
   Use KSPLGMonitorDestroy() to destroy this line graph, not DrawLGDestroy().

.keywords: KSP, monitor, line graph, residual, create

.seealso: KSPLGMonitorDestroy(), KSPSetMonitor(), KSPLGTrueMonitorCreate()
@*/
int KSPLGMonitorCreate(char *host,char *label,int x,int y,int m,
                       int n, DrawLG *draw)
{
  Draw win;
  int  ierr;
  ierr = DrawOpenX(MPI_COMM_SELF,host,label,x,y,m,n,&win); CHKERRQ(ierr);
  ierr = DrawLGCreate(win,1,draw); CHKERRQ(ierr);
  PLogObjectParent(*draw,win);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPLGMonitor" /* ADIC Ignore */
int KSPLGMonitor(KSP ksp,int n,double rnorm,void *monctx)
{
  DrawLG lg = (DrawLG) monctx;
  double    x, y;

  if (!n) DrawLGReset(lg);
  x = (double) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  DrawLGAddPoint(lg,&x,&y);
  if (n < 20 || (n % 5)) {
    DrawLGDraw(lg);
  }
  return 0;
} 
 
#undef __FUNC__  
#define __FUNC__ "KSPLGMonitorDestroy" /* ADIC Ignore */
/*@C
   KSPLGMonitorDestroy - Destroys a line graph context that was created 
   with KSPLGMonitorCreate().

   Input Parameter:
.  draw - the drawing context

.keywords: KSP, monitor, line graph, destroy

.seealso: KSPLGMonitorCreate(), KSPLGTrueMonitorDestroy(), KSPSetMonitor()
@*/
int KSPLGMonitorDestroy(DrawLG drawlg)
{
  Draw draw;
  DrawLGGetDraw(drawlg,&draw);
  DrawDestroy(draw);
  DrawLGDestroy(drawlg);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPLGTrueMonitorCreate" /* ADIC Ignore */
/*@C
   KSPLGTrueMonitorCreate - Creates a line graph context for use with 
   KSP to monitor convergence of true residual norms (as opposed to
   preconditioned residual norms).

   Input Parameters:
.  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
.  m, n - the screen width and height in pixels

   Output Parameter:
.  draw - the drawing context

   Options Database Key:
$    -ksp_xtruemonitor : automatically sets true line graph monitor

   Notes: 
   Use KSPLGTrueMonitorDestroy() to destroy this line graph, not
   DrawLGDestroy().

.keywords: KSP, monitor, line graph, residual, create, true

.seealso: KSPLGMonitorDestroy(), KSPSetMonitor(), KSPDefaultMonitor()
@*/
int KSPLGTrueMonitorCreate(MPI_Comm comm,char *host,char *label,int x,int y,int m,
                       int n, DrawLG *draw)
{
  Draw win;
  int  ierr,rank;

  MPI_Comm_rank(comm,&rank);
  if (rank) { *draw = 0; return 0;}

  ierr = DrawOpenX(MPI_COMM_SELF,host,label,x,y,m,n,&win); CHKERRQ(ierr);
  ierr = DrawLGCreate(win,2,draw); CHKERRQ(ierr);
  PLogObjectParent(*draw,win);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPLGTrueMonitor" /* ADIC Ignore */
int KSPLGTrueMonitor(KSP ksp,int n,double rnorm,void *monctx)
{
  DrawLG    lg = (DrawLG) monctx;
  double    x[2], y[2],scnorm;
  int       ierr,rank;
  Vec       resid,work;

  MPI_Comm_rank(ksp->comm,&rank);
  if (!rank) { 
    if (!n) DrawLGReset(lg);
    x[0] = x[1] = (double) n;
    if (rnorm > 0.0) y[0] = log10(rnorm); else y[0] = -15.0;
  }

  ierr = VecDuplicate(ksp->vec_rhs,&work); CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid); CHKERRQ(ierr);
  ierr = VecNorm(resid,NORM_2,&scnorm); CHKERRQ(ierr);
  VecDestroy(work);

  if (!rank) {
    if (scnorm > 0.0) y[1] = log10(scnorm); else y[1] = -15.0;
    DrawLGAddPoint(lg,x,y);
    if (n <= 20 || (n % 3)) {
      DrawLGDraw(lg);
    }
  }
  return 0;
} 
 
#undef __FUNC__  
#define __FUNC__ "KSPLGTrueMonitorDestroy" /* ADIC Ignore */
/*@C
   KSPLGTrueMonitorDestroy - Destroys a line graph context that was created 
   with KSPLGTrueMonitorCreate().

   Input Parameter:
.  draw - the drawing context

.keywords: KSP, monitor, line graph, destroy, true

.seealso: KSPLGTrueMonitorCreate(), KSPSetMonitor()
@*/
int KSPLGTrueMonitorDestroy(DrawLG drawlg)
{
  Draw draw;
  DrawLGGetDraw(drawlg,&draw);
  DrawDestroy(draw);
  DrawLGDestroy(drawlg);
  return 0;
}


