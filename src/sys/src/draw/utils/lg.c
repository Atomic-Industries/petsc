#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: lg.c,v 1.44 1997/06/05 12:56:08 bsmith Exp balay $";
#endif
/*
       Contains the data structure for plotting several line
    graphs in a window with an axis. This is intended for line 
    graphs that change dynamically by adding more points onto 
    the end of the X axis.
*/

#include "petsc.h"         /*I "petsc.h" I*/

struct _p_DrawLG {
  PETSCHEADER 
  int         len,loc;
  Draw        win;
  DrawAxis    axis;
  double      xmin, xmax, ymin, ymax, *x, *y;
  int         nopts, dim;
  int         use_dots;
};

#define CHUNCKSIZE 100

#undef __FUNC__  
#define __FUNC__ "DrawLGCreate" /* ADIC Ignore */
/*@C
    DrawLGCreate - Creates a line graph data structure.

    Input Parameters:
.   win - the window where the graph will be made.
.   dim - the number of line cures which will be drawn

    Output Parameters:
.   outctx - the line graph context

.keywords:  draw, line, graph, create

.seealso:  DrawLGDestroy()
@*/
int DrawLGCreate(Draw win,int dim,DrawLG *outctx)
{
  int         ierr;
  PetscObject vobj = (PetscObject) win;
  DrawLG      lg;

  if (vobj->cookie == DRAW_COOKIE && vobj->type == DRAW_NULLWINDOW) {
    ierr = DrawOpenNull(vobj->comm,(Draw*)outctx); CHKERRQ(ierr);
    (*outctx)->win = win;
    return 0;
  }
  PetscHeaderCreate(lg,_p_DrawLG,DRAWLG_COOKIE,0,vobj->comm);
  lg->view    = 0;
  lg->destroy = 0;
  lg->nopts   = 0;
  lg->win     = win;
  lg->dim     = dim;
  lg->xmin    = 1.e20;
  lg->ymin    = 1.e20;
  lg->xmax    = -1.e20;
  lg->ymax    = -1.e20;
  lg->x       = (double *)PetscMalloc(2*dim*CHUNCKSIZE*sizeof(double));CHKPTRQ(lg->x);
  PLogObjectMemory(lg,2*dim*CHUNCKSIZE*sizeof(double));
  lg->y       = lg->x + dim*CHUNCKSIZE;
  lg->len     = dim*CHUNCKSIZE;
  lg->loc     = 0;
  lg->use_dots= 0;
  ierr = DrawAxisCreate(win,&lg->axis); CHKERRQ(ierr);
  PLogObjectParent(lg,lg->axis);
  *outctx = lg;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawLGSetDimension" /* ADIC Ignore */
/*@
   DrawLGSetDimension - Change the number of lines that are to be drawn.

   Input Parameter:
.  lg - the line graph context.
.  dim - the number of curves.

.keywords:  draw, line, graph, reset
@*/
int DrawLGSetDimension(DrawLG lg,int dim)
{
  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  if (lg->dim == dim) return 0;

  PetscFree(lg->x);
  lg->dim = dim;
  lg->x       = (double *)PetscMalloc(2*dim*CHUNCKSIZE*sizeof(double));CHKPTRQ(lg->x);
  PLogObjectMemory(lg,2*dim*CHUNCKSIZE*sizeof(double));
  lg->y       = lg->x + dim*CHUNCKSIZE;
  lg->len     = dim*CHUNCKSIZE;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawLGReset" /* ADIC Ignore */
/*@
   DrawLGReset - Clears line graph to allow for reuse with new data.

   Input Parameter:
.  lg - the line graph context.

.keywords:  draw, line, graph, reset
@*/
int DrawLGReset(DrawLG lg)
{
  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
  lg->loc   = 0;
  lg->nopts = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawLGDestroy" /* ADIC Ignore */
/*@C
   DrawLGDestroy - Frees all space taken up by line graph data structure.

   Input Parameter:
.  lg - the line graph context

.keywords:  draw, line, graph, destroy

.seealso:  DrawLGCreate()
@*/
int DrawLGDestroy(DrawLG lg)
{
  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {
    return PetscObjectDestroy((PetscObject) lg);
  }
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  DrawAxisDestroy(lg->axis);
  PetscFree(lg->x);
  PLogObjectDestroy(lg);
  PetscHeaderDestroy(lg);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawLGAddPoint" /* ADIC Ignore */
/*@
   DrawLGAddPoint - Adds another point to each of the line graphs. 
   The new point must have an X coordinate larger than the old points.

   Input Parameters:
.  lg - the LineGraph data structure
.  x, y - the points to two vectors containing the new x and y 
          point for each curve.

.keywords:  draw, line, graph, add, point

.seealso: DrawLGAddPoints()
@*/
int DrawLGAddPoint(DrawLG lg,double *x,double *y)
{
  int i;
  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {return 0;}

  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  if (lg->loc+lg->dim >= lg->len) { /* allocate more space */
    double *tmpx,*tmpy;
    tmpx = (double *) PetscMalloc((2*lg->len+2*lg->dim*CHUNCKSIZE)*sizeof(double));CHKPTRQ(tmpx);
    PLogObjectMemory(lg,2*lg->dim*CHUNCKSIZE*sizeof(double));
    tmpy = tmpx + lg->len + lg->dim*CHUNCKSIZE;
    PetscMemcpy(tmpx,lg->x,lg->len*sizeof(double));
    PetscMemcpy(tmpy,lg->y,lg->len*sizeof(double));
    PetscFree(lg->x);
    lg->x = tmpx; lg->y = tmpy;
    lg->len += lg->dim*CHUNCKSIZE;
  }
  for (i=0; i<lg->dim; i++) {
    if (x[i] > lg->xmax) lg->xmax = x[i]; 
    if (x[i] < lg->xmin) lg->xmin = x[i];
    if (y[i] > lg->ymax) lg->ymax = y[i]; 
    if (y[i] < lg->ymin) lg->ymin = y[i];

    lg->x[lg->loc]   = x[i];
    lg->y[lg->loc++] = y[i];
  }
  lg->nopts++;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawLGIndicateDataPoints" /* ADIC Ignore */
/*@
   DrawLGIndicateDataPoints - Causes LG to draw a big dot for each data-point.

   Input Parameters:
.  lg - the linegraph context

.keywords:  draw, line, graph, indicate, data, points
@*/
int DrawLGIndicateDataPoints(DrawLG lg)
{
  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {return 0;}

  lg->use_dots = 1;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawLGAddPoints" /* ADIC Ignore */
/*@C
   DrawLGAddPoints - Adds several points to each of the line graphs.
   The new points must have an X coordinate larger than the old points.

   Input Parameters:
.  lg - the LineGraph data structure
.  xx,yy - points to two arrays of pointers that point to arrays 
           containing the new x and y points for each curve.
.  n - number of points being added

.keywords:  draw, line, graph, add, points

.seealso: DrawLGAddPoint()
@*/
int DrawLGAddPoints(DrawLG lg,int n,double **xx,double **yy)
{
  int    i, j, k;
  double *x,*y;

  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  if (lg->loc+n*lg->dim >= lg->len) { /* allocate more space */
    double *tmpx,*tmpy;
    int    chunk = CHUNCKSIZE;
    if (n > chunk) chunk = n;
    tmpx = (double *) PetscMalloc((2*lg->len+2*lg->dim*chunk)*sizeof(double));CHKPTRQ(tmpx);
    PLogObjectMemory(lg,2*lg->dim*chunk*sizeof(double));
    tmpy = tmpx + lg->len + lg->dim*chunk;
    PetscMemcpy(tmpx,lg->x,lg->len*sizeof(double));
    PetscMemcpy(tmpy,lg->y,lg->len*sizeof(double));
    PetscFree(lg->x);
    lg->x    = tmpx; lg->y = tmpy;
    lg->len += lg->dim*chunk;
  }
  for (j=0; j<lg->dim; j++) {
    x = xx[j]; y = yy[j];
    k = lg->loc + j;
    for ( i=0; i<n; i++ ) {
      if (x[i] > lg->xmax) lg->xmax = x[i]; 
      if (x[i] < lg->xmin) lg->xmin = x[i];
      if (y[i] > lg->ymax) lg->ymax = y[i]; 
      if (y[i] < lg->ymin) lg->ymin = y[i];

      lg->x[k]   = x[i];
      lg->y[k] = y[i];
      k += lg->dim;
    }
  }
  lg->loc   += n*lg->dim;
  lg->nopts += n;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawLGDraw" /* ADIC Ignore */
/*@
   DrawLGDraw - Redraws a line graph.

   Input Parameter:
.  lg - the line graph context

.keywords:  draw, line, graph
@*/
int DrawLGDraw(DrawLG lg)
{
  double   xmin=lg->xmin, xmax=lg->xmax, ymin=lg->ymin, ymax=lg->ymax;
  int      i, j, dim = lg->dim,nopts = lg->nopts;
  Draw     win = lg->win;
  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);

  if (nopts < 2) return 0;
  if (xmin > xmax || ymin > ymax) return 0;
  DrawClear(win);
  DrawAxisSetLimits(lg->axis, xmin, xmax, ymin, ymax);
  DrawAxisDraw(lg->axis);
  for ( i=0; i<dim; i++ ) {
    for ( j=1; j<nopts; j++ ) {
      DrawLine(win,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],
                   lg->x[j*dim+i],lg->y[j*dim+i],DRAW_BLACK+i);
      if (lg->use_dots) {
        DrawString(win,lg->x[j*dim+i],lg->y[j*dim+i],DRAW_RED,"x");
      }
    }
  }
  DrawSyncFlush(lg->win);
  DrawPause(lg->win);
  return 0;
} 
 
#undef __FUNC__  
#define __FUNC__ "DrawLGSetLimits" /* ADIC Ignore */
/*@
   DrawLGSetLimits - Sets the axis limits for a line graph. If more
   points are added after this call, the limits will be adjusted to
   include those additional points.

   Input Parameters:
.  xlg - the line graph context
.  x_min,x_max,y_min,y_max - the limits

.keywords:  draw, line, graph, set limits
@*/
int DrawLGSetLimits( DrawLG lg,double x_min,double x_max,double y_min,
                                  double y_max) 
{
  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  (lg)->xmin = x_min; 
  (lg)->xmax = x_max; 
  (lg)->ymin = y_min; 
  (lg)->ymax = y_max;
  return 0;
}
 
#undef __FUNC__  
#define __FUNC__ "DrawLGGetAxis" /* ADIC Ignore */
/*@C
   DrawLGGetAxis - Gets the axis context associated with a line graph.
   This is useful if one wants to change some axis property, such as
   labels, color, etc. The axis context should not be destroyed by the
   application code.

   Input Parameter:
.  lg - the line graph context

   Output Parameter:
.  axis - the axis context

.keywords: draw, line, graph, get, axis
@*/
int DrawLGGetAxis(DrawLG lg,DrawAxis *axis)
{
  if (lg && lg->cookie == DRAW_COOKIE && lg->type == DRAW_NULLWINDOW) {
    *axis = 0;
    return 0;
  }
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  *axis = lg->axis;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawLGGetDraw" /* ADIC Ignore */
/*@C
    DrawLGGetDraw - Gets the draw context associated with a line graph.

   Input Parameter:
.  lg - the line graph context

   Output Parameter:
.  win - the draw context

.keywords: draw, line, graph, get, context
@*/
int DrawLGGetDraw(DrawLG lg,Draw *win)
{
  if (!lg || lg->cookie != DRAW_COOKIE || lg->type != DRAW_NULLWINDOW) {
    PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  }
  *win = lg->win;
  return 0;
}
