#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: cmesh.c,v 1.51 1998/04/13 17:26:10 bsmith Exp curfman $";
#endif

#include "src/draw/drawimpl.h"   /*I "draw.h" I*/
#include "vec.h"        /*I "vec.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawScalePopup"
int DrawScalePopup(Draw popup,double min,double max)
{
  double   xl = 0.0, yl = 0.0, xr = 1.0, yr = 1.0,value;
  int      i,c = DRAW_BASIC_COLORS,rank;
  char     string[32];
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject) popup,&comm);
  MPI_Comm_rank(comm,&rank);
  if (rank) PetscFunctionReturn(0);

  for ( i=0; i<10; i++ ) {
    DrawRectangle(popup,xl,yl,xr,yr,c,c,c,c);
    yl += .1; yr += .1; c = (int) ((double) c + (245.-DRAW_BASIC_COLORS)/9.);
  }
  for ( i=0; i<10; i++ ) {
    value = min + i*(max-min)/9.0;
    /* look for a value that should be zero, but is not due to round-off */
    if (PetscAbsDouble(value) < 1.e-10 && max-min > 1.e-6) value = 0.0;
    sprintf(string,"%g",value);
    DrawString(popup,.2,.02 + i/10.0,DRAW_BLACK,string);
  }
  DrawSetTitle(popup,"Contour Scale");
  DrawFlush(popup);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "DrawTensorContour"
/*@C
   DrawTensorContour - Draws a contour plot for a two-dimensional array
   that is stored as a PETSc vector.

   Collective on Draw and Vec

   Input Parameters:
+   win - the window to draw in
.   m,n - the global number of mesh points in the x and y directions
.   x,y - the locations of the global mesh points (optional, use PETSC_NULL
          to indicate uniform spacing on [0,1])
-   V - the vector

   Options Database Keys:
+  -draw_x_private_colormap - Indicates use of private colormap
-  -draw_contour_grid - Draws grid contour

    Note: 
    This may be a basic enough function to be a graphics primative
    but at this time it uses DrawTriangle().

.keywords: Draw, tensor, contour, vector
@*/
int DrawTensorContour(Draw win,int m,int n,double *x,double *y,Vec V)
{
  int           N, rank, ierr;
  Vec           W;
  IS            from, to;
  VecScatter    ctx;
  PetscObject   vobj = (PetscObject) win;
  Draw          popup;
#if !defined(USE_PETSC_COMPLEX)
  int          xin=1,yin=1,i,pause,showgrid;
  double        h, *v, min, max, scale = 1.0;
#endif

  PetscFunctionBegin;
  if (vobj->cookie == DRAW_COOKIE && vobj->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  MPI_Comm_rank(win->comm,&rank);

  /* move entire vector to first processor */
  if (rank == 0) {
    ierr = VecGetSize(V,&N); CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&W); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&from); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&to); CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF,0,&W); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&from); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&to); CHKERRQ(ierr);
  }
  PLogObjectParent(win,W); PLogObjectParent(win,from); PLogObjectParent(win,to);
  ierr = VecScatterCreate(V,from,W,to,&ctx); CHKERRQ(ierr);
  PLogObjectParent(win,ctx);
  ierr = VecScatterBegin(V,W,INSERT_VALUES,SCATTER_FORWARD,ctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(V,W,INSERT_VALUES,SCATTER_FORWARD,ctx); CHKERRQ(ierr);
  ISDestroy(from); ISDestroy(to); VecScatterDestroy(ctx);

  /* create scale window */
  ierr = DrawCheckResizedWindow(win); CHKERRQ(ierr);

  ierr = DrawGetPopup(win,&popup); CHKERRQ(ierr);

  if (rank == 0) {
#if !defined(USE_PETSC_COMPLEX)
    double  xl = 0.0, yl = 0.0, xr = 1.0, yr = .1;

    ierr = VecGetArray(W,&v); CHKERRQ(ierr);

    ierr = VecMax(W,0,&max); CHKERRQ(ierr); ierr = VecMin(W,0,&min); CHKERRQ(ierr);
    if (max - min < 1.e-7) {min -= 5.e-8; max += 5.e-8;}

    /* Draw the scale window */
    ierr = DrawScalePopup(popup,min,max); CHKERRQ(ierr);

    ierr = OptionsHasName(PETSC_NULL,"-draw_contour_grid",&showgrid);CHKERRQ(ierr);

    /* fill up x and y coordinates */
    if (!x) {
      xin = 0; 
      x = (double *) PetscMalloc( m*sizeof(double) ); CHKPTRQ(x);
      h = 1.0/(m-1);
      x[0] = 0.0;
      for ( i=1; i<m; i++ ) x[i] = x[i-1] + h;
    }
    if (!y) {
      yin = 0; 
      y = (double *) PetscMalloc( n*sizeof(double) ); CHKPTRQ(y);
      h = 1.0/(n-1);
      y[0] = 0.0;
      for ( i=1; i<n; i++ ) y[i] = y[i-1] + h;
    }

    /* Draw the contour plot */
    ierr = DrawGetPause(win,&pause); CHKERRA(ierr);
    while (1) {
      ierr = DrawTensorContourPatch(win,m,n,x,y,max,min,v);CHKERRQ(ierr);

      if (showgrid) {
        for ( i=0; i<m; i++ ) {
          DrawLine(win,x[i],y[0],x[i],y[n-1],DRAW_BLACK);
        }
        for ( i=0; i<n; i++ ) {
          DrawLine(win,x[0],y[i],x[m-1],y[i],DRAW_BLACK);
        }
      }
      ierr = DrawFlush(win); CHKERRQ(ierr);

      if (pause == -1) {
        DrawButton  button;
        double      xc,yc;
    
        ierr = DrawGetCoordinates(win,&xl,&yl,&xr,&yr); CHKERRQ(ierr);

        ierr = DrawGetMouseButton(win,&button,&xc,&yc,0,0); CHKERRQ(ierr);
        if  (button == BUTTON_RIGHT) break;
        if (button == BUTTON_LEFT) scale = .5;
        else if (button == BUTTON_CENTER) scale = 2.;
         xl = scale*(xl  - xc) + xc;
         xr = scale*(xr  - xc) + xc;
         yl = scale*(yl  - yc) + yc;
         yr = scale*(yr  - yc) + yc;
         ierr = DrawSetCoordinates(win,xl,yl,xr,yr); CHKERRQ(ierr);
      } else {break;}
     
      /* cannot check resize because requires synchronization accross draw */
      /* ierr = DrawCheckResizedWindow(win); CHKERRQ(ierr); */
      if (!rank) {ierr = DrawClear(win); CHKERRQ(ierr);}
      ierr = DrawFlush(win); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(W,&v); CHKERRQ(ierr);
    
    if (!xin) PetscFree(x); 
    if (!yin) PetscFree(y);
#endif
  }
  ierr = VecDestroy(W); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecContourScale"
/*@
    VecContourScale - Prepares a vector of values to be plotted using 
    the DrawTriangle() contour plotter.

    Collective on Vec

    Input Parameters:
+   v - the vector of values
.   vmin - minimum value (for lowest color)
-   vmax - maximum value (for highest color)

@*/
int VecContourScale(Vec v,double vmin,double vmax)
{
  Scalar *values;
  int    ierr,n,i;
  double scale;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);

  if (PetscAbsDouble(vmax - vmin) < 1.e-50) {
     scale = 1.0;
  } else {
    scale = (245.0 - DRAW_BASIC_COLORS)/(vmax - vmin); 
  }

  ierr = VecGetArray(v,&values);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    values[i] = DRAW_BASIC_COLORS + scale*(values[i] - vmin);
  }
  ierr = VecRestoreArray(v,&values);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
