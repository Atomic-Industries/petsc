#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dgcoor.c,v 1.15 1998/04/27 14:23:00 curfman Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawGetCoordinates" 
/*@
   DrawGetCoordinates - Gets the application coordinates of the corners of
   the window (or page).

   Not Collective

   Input Parameter:
.  draw - the drawing object

   Ouput Parameters:
.  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.

.keywords:  draw, get, coordinates
@*/
int DrawGetCoordinates(Draw draw,double *xl,double *yl,double *xr,double *yr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidPointer(xl);
  PetscValidPointer(yl);
  PetscValidPointer(xr);
  PetscValidPointer(yr);
  *xl = draw->coor_xl; *yl = draw->coor_yl;
  *xr = draw->coor_xr; *yr = draw->coor_yr;
  PetscFunctionReturn(0);
}
