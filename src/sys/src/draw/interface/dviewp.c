#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dviewp.c,v 1.17 1998/03/12 23:20:42 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/
#include <math.h>

#undef __FUNC__  
#define __FUNC__ "DrawSetViewPort" 
/*@
   DrawSetViewPort - Sets the portion of the window (page) to which draw
   routines will write.

   Input Parameters:
.  xl,yl,xr,yr - upper right and lower left corners of subwindow
                 These numbers must always be between 0.0 and 1.0.
                 Lower left corner is (0,0).
.  draw - the drawing context

   Collective on Draw

.keywords:  draw, set, view, port
@*/
int DrawSetViewPort(Draw draw,double xl,double yl,double xr,double yr)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  if (xl < 0.0 || xr > 1.0 || yl < 0.0 || yr > 1.0 || xr <= xl || yr <= yl) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"ViewPort values must be >= 0 and <= 1"); 
  }
  draw->port_xl = xl; draw->port_yl = yl;
  draw->port_xr = xr; draw->port_yr = yr;
  if (draw->ops->setviewport) {
    ierr = (*draw->ops->setviewport)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawSplitViewPort" 
/*@
   DrawSplitViewPort - Splits a window shared by several processes into smaller
        view ports. One for each process. 

   Input Parameters:
.  draw - the drawing context

   Collective on Draw

.keywords:  draw, set, view, port, split
@*/
int DrawSplitViewPort(Draw draw)
{
  int    rank,size,n,ierr;
  double xl,xr,yl,yr,h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);

  MPI_Comm_rank(draw->comm,&rank);
  MPI_Comm_size(draw->comm,&size);

  n = (int) (.1 + sqrt((double) size));
  while ( n*n < size) {n++;}

  h  = 1.0/n;
  xl = (rank % n)*h;
  xr = xl + h;
  yl = (rank/n)*h;
  yr = yl + h;

  ierr = DrawLine(draw,xl,yl,xl,yr,DRAW_BLACK); CHKERRQ(ierr);
  ierr = DrawLine(draw,xl,yr,xr,yr,DRAW_BLACK); CHKERRQ(ierr);
  ierr = DrawLine(draw,xr,yr,xr,yl,DRAW_BLACK); CHKERRQ(ierr);
  ierr = DrawLine(draw,xr,yl,xl,yl,DRAW_BLACK); CHKERRQ(ierr);
  DrawSynchronizedFlush(draw);

  draw->port_xl = xl;
  draw->port_xr = xr;
  draw->port_yl = yl;
  draw->port_yr = yr;

  if (draw->ops->setviewport) {
    ierr =  (*draw->ops->setviewport)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
