

#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: xops.c,v 1.103 1997/12/31 18:25:51 bsmith Exp bsmith $";
#endif
/*
    Defines the operations for the X Draw implementation.
*/

#include "src/draw/impls/x/ximpl.h"         /*I  "petsc.h" I*/

#if defined(HAVE_X11)

/*
     These macros transform from the users coordinates to the 
   X-window pixel coordinates.
*/
#define XTRANS(win,xwin,x) \
   (int)(((xwin)->w)*((win)->port_xl + (((x - (win)->coor_xl)*\
                                   ((win)->port_xr - (win)->port_xl))/\
                                   ((win)->coor_xr - (win)->coor_xl))))
#define YTRANS(win,xwin,y) \
   (int)(((xwin)->h)*(1.0-(win)->port_yl - (((y - (win)->coor_yl)*\
                                       ((win)->port_yr - (win)->port_yl))/\
                                       ((win)->coor_yr - (win)->coor_yl))))

#undef __FUNC__  
#define __FUNC__ "DrawLine_X" 
int DrawLine_X(Draw Win, double xl, double yl, double xr, double yr,int cl)
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  int     x1,y1,x2,y2;

  PetscFunctionBegin;
  XiSetColor( XiWin, cl );
  x1 = XTRANS(Win,XiWin,xl);   x2  = XTRANS(Win,XiWin,xr); 
  y1 = YTRANS(Win,XiWin,yl);   y2  = YTRANS(Win,XiWin,yr); 
  XDrawLine( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set, x1, y1, x2, y2);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawPoint_X" 
static int DrawPoint_X(Draw Win,double x,double  y,int c)
{
  int     xx,yy;
  Draw_X* XiWin = (Draw_X*) Win->data;

  PetscFunctionBegin;
  xx = XTRANS(Win,XiWin,x);  yy = YTRANS(Win,XiWin,y);
  XiSetColor( XiWin, c );
  XDrawPoint( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,xx, yy);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawRectangle_X" 
static int DrawRectangle_X(Draw Win, double xl, double yl, double xr, double yr,
                           int c1, int c2,int c3,int c4)
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  int     x1,y1,w,h, c = (c1 + c2 + c3 + c4)/4;

  PetscFunctionBegin;
  XiSetColor( XiWin, c );
  x1 = XTRANS(Win,XiWin,xl);   w  = XTRANS(Win,XiWin,xr) - x1; 
  y1 = YTRANS(Win,XiWin,yr);   h  = YTRANS(Win,XiWin,yl) - y1;
  if (w <= 0) w = 1; if (h <= 0) h = 1;
  XFillRectangle( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set, x1, y1, w, h);
  PetscFunctionReturn(0);
}

extern int XiDrawInterpolatedTriangle(Draw_X*,int,int,int,int,int,int,int,int,int);

#undef __FUNC__  
#define __FUNC__ "DrawTriangle_X" 
static int DrawTriangle_X(Draw Win, double X1, double Y1, double X2, 
                          double Y2,double X3,double Y3, int c1, int c2,int c3)
{
  Draw_X* XiWin = (Draw_X*) Win->data;

  PetscFunctionBegin;
  if (c1 == c2 && c2 == c3) {
    XPoint pt[3];
    XiSetColor( XiWin, c1 );
    pt[0].x = XTRANS(Win,XiWin,X1);
    pt[0].y = YTRANS(Win,XiWin,Y1); 
    pt[1].x = XTRANS(Win,XiWin,X2);
    pt[1].y = YTRANS(Win,XiWin,Y2); 
    pt[2].x = XTRANS(Win,XiWin,X3);
    pt[2].y = YTRANS(Win,XiWin,Y3); 
    XFillPolygon(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set,pt,3,Convex,
                 CoordModeOrigin);
  } else {
    int x1,y1,x2,y2,x3,y3;
    x1 = XTRANS(Win,XiWin,X1);
    y1 = YTRANS(Win,XiWin,Y1); 
    x2 = XTRANS(Win,XiWin,X2);
    y2 = YTRANS(Win,XiWin,Y2); 
    x3 = XTRANS(Win,XiWin,X3);
    y3 = YTRANS(Win,XiWin,Y3); 
    XiDrawInterpolatedTriangle(XiWin,x1,y1,c1,x2,y2,c2,x3,y3,c3);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawString_X" 
static int DrawString_X(Draw Win,double x,double  y,int c,char *chrs )
{
  int     xx,yy;
  Draw_X* XiWin = (Draw_X*) Win->data;
  char*   substr;

  PetscFunctionBegin;
  xx = XTRANS(Win,XiWin,x);  yy = YTRANS(Win,XiWin,y);
  XiSetColor( XiWin, c );
  
  substr = PetscStrtok(chrs,"\n");
  XDrawString( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,
               xx, yy - XiWin->font->font_descent, substr, PetscStrlen(substr) );
  substr = PetscStrtok(0,"\n");
  while (substr) {
    yy += 4*XiWin->font->font_descent;
    XDrawString( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,
                 xx, yy - XiWin->font->font_descent, substr, PetscStrlen(substr) );
    substr = PetscStrtok(0,"\n");
  }

  PetscFunctionReturn(0);
}

int XiFontFixed( Draw_X*,int, int,XiFont **);

#undef __FUNC__  
#define __FUNC__ "DrawStringSetSize_X" 
static int DrawStringSetSize_X(Draw Win,double x,double  y)
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  int     ierr,w,h;

  PetscFunctionBegin;
  w = (int)((XiWin->w)*x*(Win->port_xr - Win->port_xl)/(Win->coor_xr - Win->coor_xl));
  h = (int)((XiWin->h)*y*(Win->port_yr - Win->port_yl)/(Win->coor_yr - Win->coor_yl));
  PetscFree(XiWin->font);
  ierr = XiFontFixed( XiWin,w, h, &XiWin->font);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawStringGetSize_X" 
int DrawStringGetSize_X(Draw Win,double *x,double  *y)
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  double  w,h;

  PetscFunctionBegin;
  w = XiWin->font->font_w; h = XiWin->font->font_h;
  *x = w*(Win->coor_xr - Win->coor_xl)/(XiWin->w)*(Win->port_xr - Win->port_xl);
  *y = h*(Win->coor_yr - Win->coor_yl)/(XiWin->h)*(Win->port_yr - Win->port_yl);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawStringVertical_X" 
int DrawStringVertical_X(Draw Win,double x,double  y,int c,char *chrs )
{
  int     xx,yy,n = PetscStrlen(chrs),i;
  Draw_X* XiWin = (Draw_X*) Win->data;
  char    tmp[2];
  double  tw,th;
  
  PetscFunctionBegin;
  tmp[1] = 0;
  XiSetColor( XiWin, c );
  DrawStringGetSize_X(Win,&tw,&th);
  xx = XTRANS(Win,XiWin,x);
  for ( i=0; i<n; i++ ) {
    tmp[0] = chrs[i];
    yy = YTRANS(Win,XiWin,y-th*i);
    XDrawString( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,
                xx, yy - XiWin->font->font_descent, tmp, 1 );
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawFlush_X" 
static int DrawFlush_X(Draw Win )
{
  Draw_X* XiWin = (Draw_X*) Win->data;

  PetscFunctionBegin;
  if (XiWin->drw) {
    XCopyArea( XiWin->disp,XiWin->drw,XiWin->win,XiWin->gc.set,0,0,XiWin->w,XiWin->h,0,0);
  }
  XFlush( XiWin->disp ); XSync(XiWin->disp,False);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawSynchronizedFlush_X" 
static int DrawSynchronizedFlush_X(Draw Win )
{
  int     rank,ierr;
  Draw_X* XiWin = (Draw_X*) Win->data;

  PetscFunctionBegin;
  XFlush( XiWin->disp );
  if (XiWin->drw) {
    MPI_Comm_rank(Win->comm,&rank);
    /* make sure data has actually arrived at server */
    XSync(XiWin->disp,False);
    ierr = MPI_Barrier(Win->comm);CHKERRQ(ierr);
    if (!rank) {
      XCopyArea(XiWin->disp,XiWin->drw,XiWin->win,XiWin->gc.set,0,0,XiWin->w,XiWin->h,0,0);
      XFlush( XiWin->disp );
    }
    XSync(XiWin->disp,False);
    ierr = MPI_Barrier(Win->comm);CHKERRQ(ierr);
  } else {
    ierr = MPI_Barrier(Win->comm);CHKERRQ(ierr);
    XSync(XiWin->disp,False);
    ierr = MPI_Barrier(Win->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawSetViewport_X" 
static int DrawSetViewport_X(Draw Win,double xl,double yl,double xr,double yr)
{
  Draw_X*    XiWin = (Draw_X*) Win->data;
  XRectangle box;

  PetscFunctionBegin;
  box.x = (int) (xl*XiWin->w);   box.y = (int) ((1.0-yr)*XiWin->h);
  box.width = (int) ((xr-xl)*XiWin->w);box.height = (int) ((yr-yl)*XiWin->h);
  XSetClipRectangles(XiWin->disp,XiWin->gc.set,0,0,&box,1,Unsorted);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawClear_X" 
static int DrawClear_X(Draw Win)
{
  Draw_X*  XiWin = (Draw_X*) Win->data;
  int      x,  y,  w,  h;

  PetscFunctionBegin;
  x = (int) (Win->port_xl*XiWin->w);
  w = (int) ((Win->port_xr - Win->port_xl)*XiWin->w);
  y = (int) ((1.0-Win->port_yr)*XiWin->h);
  h = (int) ((Win->port_yr - Win->port_yl)*XiWin->h);
  XiSetPixVal(XiWin, XiWin->background );
  XFillRectangle(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set, x, y, w, h);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawSynchronizedClear_X" 
static int DrawSynchronizedClear_X(Draw Win)
{
  int     rank,ierr;
  Draw_X* XiWin = (Draw_X*) Win->data;

  PetscFunctionBegin;
  ierr = MPI_Barrier(Win->comm);CHKERRQ(ierr);
  MPI_Comm_rank(Win->comm,&rank);
  if (!rank) {
    DrawClear_X(Win);
  }
  XFlush( XiWin->disp );
  ierr = MPI_Barrier(Win->comm);CHKERRQ(ierr);
  XSync(XiWin->disp,False);
  ierr = MPI_Barrier(Win->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawSetDoubleBuffer_X" 
static int DrawSetDoubleBuffer_X(Draw Win)
{
  Draw_X*  win = (Draw_X*) Win->data;
  int      rank,ierr;

  PetscFunctionBegin;
  if (win->drw) PetscFunctionReturn(0);

  MPI_Comm_rank(Win->comm,&rank);
  if (!rank) {
    win->drw = XCreatePixmap(win->disp,win->win,win->w,win->h,win->depth);
  }
  /* try to make sure it is actually done before passing info to all */
  XSync(win->disp,False);
  ierr = MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,Win->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <X11/cursorfont.h>

#undef __FUNC__  
#define __FUNC__ "DrawGetMouseButton_X" 
static int DrawGetMouseButton_X(Draw draw,DrawButton *button,double* x_user,
                                double *y_user,double *x_phys,double *y_phys)
{
  XEvent       report;
  Draw_X*      win = (Draw_X*) draw->data;
  Window       root, child;
  int          root_x, root_y,px,py;
  unsigned int keys_button;
  Cursor       cursor = 0;

  PetscFunctionBegin;
  /* 
         Try to change the border color to red to indicate requesting input.
       Does not appear to work.
  XSetWindowBorder(win->disp, win->win,win->cmapping[DRAW_RED]);
  XFlush( win->disp ); XSync(win->disp,False);
  */
  if (!cursor) {
    cursor = XCreateFontCursor(win->disp,XC_hand2); 
    if (!cursor) SETERRQ(PETSC_ERR_LIB,1,"Unable to create X cursor");
  }
  XDefineCursor(win->disp, win->win, cursor);

  XSelectInput( win->disp, win->win, ButtonPressMask | ButtonReleaseMask );

  while (XCheckTypedEvent( win->disp, ButtonPress, &report ));
  XMaskEvent( win->disp, ButtonReleaseMask, &report );
  switch (report.xbutton.button) {
    case Button1: *button = BUTTON_LEFT;   break;
    case Button2: *button = BUTTON_CENTER; break;
    case Button3: *button = BUTTON_RIGHT;  break;
  }
  XQueryPointer(win->disp, report.xmotion.window,&root,&child,&root_x,&root_y,
                &px,&py,&keys_button);

  if (x_phys) *x_phys = ((double) px)/((double) win->w);
  if (y_phys) *y_phys = 1.0 - ((double) py)/((double) win->h);

  if (x_user) *x_user = draw->coor_xl + ((((double) px)/((double) win->w)-draw->port_xl))*
                        (draw->coor_xr - draw->coor_xl)/(draw->port_xr - draw->port_xl);
  if (y_user) *y_user = draw->coor_yl + 
                        ((1.0 - ((double) py)/((double) win->h)-draw->port_yl))*
                        (draw->coor_yr - draw->coor_yl)/(draw->port_yr - draw->port_yl);

  XUndefineCursor(win->disp, win->win);
  XFlush( win->disp ); XSync(win->disp,False);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawPause_X" 
static int DrawPause_X(Draw draw)
{
  int ierr;

  PetscFunctionBegin;
  if (draw->pause > 0) PetscSleep(draw->pause);
  else if (draw->pause < 0) {
    DrawButton button;
    int        rank;
    MPI_Comm_rank(draw->comm,&rank);
    if (!rank) {
      ierr = DrawGetMouseButton(draw,&button,0,0,0,0); CHKERRQ(ierr);
      if (button == BUTTON_CENTER) draw->pause = 0;
    }
    ierr = MPI_Bcast(&draw->pause,1,MPI_INT,0,draw->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawGetPopup_X" 
static int DrawGetPopup_X(Draw draw,Draw *popup)
{
  int     ierr;
  Draw_X* win = (Draw_X*) draw->data;

  PetscFunctionBegin;
  ierr = DrawOpenX(draw->comm,PETSC_NULL,PETSC_NULL,win->x,win->y+win->h+25,150,220,popup);CHKERRQ(ierr);
  draw->popup = *popup;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawSetTitle_X" 
static int DrawSetTitle_X(Draw draw,char *title)
{
  Draw_X        *win = (Draw_X *) draw->data;
  XTextProperty prop;

  PetscFunctionBegin;
  XGetWMName(win->disp,win->win,&prop);
  prop.value  = (unsigned char *)title; 
  prop.nitems = (long) PetscStrlen(title);
  XSetWMName(win->disp,win->win,&prop); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawResizeWindow_X"
static int DrawResizeWindow_X(Draw draw,int w,int h)
{
  Draw_X       *win = (Draw_X *) draw->data;
  unsigned int ww, hh, border, depth;
  int          x,y;
  int          ierr;
  Window       root;

  PetscFunctionBegin;
  XResizeWindow(win->disp,win->win,w,h);
  XGetGeometry(win->disp,win->win,&root,&x,&y,&ww,&hh,&border,&depth);
  ierr = DrawCheckResizedWindow(draw); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawCheckResizedWindow_X" 
static int DrawCheckResizedWindow_X(Draw draw)
{
  Draw_X       *win = (Draw_X *) draw->data;
  int          x,y,rank,ierr;
  Window       root;
  unsigned int w, h, border, depth,geo[2];
  double       xl,xr,yl,yr;
  XRectangle   box;

  PetscFunctionBegin;
  MPI_Comm_rank(draw->comm,&rank);
  if (!rank) {
    XSync(win->disp,False);
    XGetGeometry(win->disp,win->win,&root,&x,&y,geo,geo+1,&border,&depth);
  }
  ierr = MPI_Bcast(geo,2,MPI_INT,0,draw->comm);CHKERRQ(ierr);
  w = geo[0]; 
  h = geo[1];
  if (w == win->w && h == win->h) PetscFunctionReturn(0);

  /* record new window sizes */

  win->h = h; win->w = w;

  /* Free buffer space and create new version (only first processor does this) */
  if (win->drw) {
    win->drw = XCreatePixmap(win->disp,win->win,win->w,win->h,win->depth);
  }
  /* reset the clipping */
  xl = draw->port_xl; yl = draw->port_yl;
  xr = draw->port_xr; yr = draw->port_yr;
  box.x     = (int) (xl*win->w);     box.y      = (int) ((1.0-yr)*win->h);
  box.width = (int) ((xr-xl)*win->w);box.height = (int) ((yr-yl)*win->h);
  XSetClipRectangles(win->disp,win->gc.set,0,0,&box,1,Unsorted);

  /* try to make sure it is actually done before passing info to all */
  XSync(win->disp,False);
  ierr = MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,draw->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _DrawOps DvOps = { DrawSetDoubleBuffer_X,
                                 DrawFlush_X,DrawLine_X,0,0,DrawPoint_X,0,
                                 DrawString_X,DrawStringVertical_X,
                                 DrawStringSetSize_X,DrawStringGetSize_X,
                                 DrawSetViewport_X,DrawClear_X,
                                 DrawSynchronizedFlush_X,
                                 DrawRectangle_X,
                                 DrawTriangle_X,
                                 DrawGetMouseButton_X,
                                 DrawPause_X,
                                 DrawSynchronizedClear_X, 
				 0, 0,
                                 DrawGetPopup_X,
                                 DrawSetTitle_X,
                                 DrawCheckResizedWindow_X,
                                 DrawResizeWindow_X };

#undef __FUNC__  
#define __FUNC__ "DrawDestroy_X" 
int DrawDestroy_X(PetscObject obj)
{
  Draw   ctx = (Draw) obj;
  Draw_X *win = (Draw_X *) ctx->data;
  int    ierr;

  PetscFunctionBegin;
  if (ctx->popup) {ierr = DrawDestroy(ctx->popup); CHKERRQ(ierr);}
  if (ctx->title) PetscFree(ctx->title);
  PetscFree(win->font);
  PetscFree(win);
  PLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  PetscFunctionReturn(0);
}

extern int XiQuickWindow(Draw_X*,char*,char*,int,int,int,int,int);
extern int XiQuickWindowFromWindow(Draw_X*,char*,Window,int);

#undef __FUNC__  
#define __FUNC__ "DrawXGetDisplaySize_Private" 
int DrawXGetDisplaySize_Private(char *name,int *width,int *height)
{
  Display *display;

  PetscFunctionBegin;
  display = XOpenDisplay( name );
  if (!display) {
    *width  = 0; 
    *height = 0; 
    (*PetscErrorPrintf)("Unable to open display on %s\n",name);
    SETERRQ(PETSC_ERR_LIB,0,"Could not open display: make sure your DISPLAY variable\n\
    is set, or you use the -display name option and xhost + has been\n\
    run on your displaying machine.\n" );
  }

  *width  = DisplayWidth(display,0);
  *height = DisplayHeight(display,0);

  XCloseDisplay(display);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawOpenX" 
/*@C
   DrawOpenX - Opens an X-window for use with the Draw routines.

   Input Parameters:
.  comm - the communicator that will share X-window
.  display - the X display on which to open, or null for the local machine
.  title - the title to put in the title bar, or null for no title
.  x, y - the screen coordinates of the upper left corner of window
.         may use PETSC_DECIDE for these two arguments, then PETSc places the 
.         window
.  w, h - the screen width and height in pixels

   Output Parameters:
.  ctx - the drawing context.

   Options Database Keys:
$  -nox : disables all x-windows output
$  -display <name> : name of machine for the X display
$  -draw_pause <pause> : sets time (in seconds) that the
$     program pauses after DrawPause() has been called
$     (0 is default, -1 implies until user input).
$  -draw_x_shared_colormap: causes PETSc to use a shared
$     colormap. By default PETSc creates a seperate color 
$     for its windows, you must put the mouse into the graphics 
$     window to see  the correct colors. This options forces
$     PETSc to use the default colormap which will usually result
$     in bad contour plots.
$  -draw_double_buffer: uses double buffering for smooth animation.
$  -geometry: location and size of window

   Note:
   When finished with the drawing context, it should be destroyed
   with DrawDestroy().

   Note for Fortran Programmers:
   Whenever indicating null character data in a Fortran code,
   PETSC_NULL_CHARACTER must be employed; using PETSC_NULL is not
   correct for character data!  Thus, PETSC_NULL_CHARACTER can be
   used for the display and title input parameters.

.keywords: draw, open, x

.seealso: DrawSynchronizedFlush(), DrawDestroy()
@*/
int DrawOpenX(MPI_Comm comm,char* display,char *title,int x,int y,int w,int h,Draw* inctx)
{
  Draw       ctx;
  Draw_X     *Xwin;
  int        ierr,size,rank,flg,xywh[4],osize = 4;
  char       string[128];
  static int xavailable = 0,yavailable = 0,xmax = 0,ymax = 0, ybottom = 0;

  PetscFunctionBegin;
  ierr = OptionsHasName(PETSC_NULL,"-nox",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = DrawOpenNull(comm,inctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* allow user to set location and size of window */
  if (VIEWER_DRAWX_SELF_PRIVATE) PetscFunctionReturn(0);
  xywh[0] = x; xywh[1] = y; xywh[2] = w; xywh[3] = h;
  ierr = OptionsGetIntArray(PETSC_NULL,"-geometry",xywh,&osize,&flg);CHKERRQ(ierr);
  x = xywh[0]; y = xywh[1]; w = xywh[2]; h = xywh[3];

  if (!display) {
    PetscGetDisplay(string,128);
    display = string;
  }

  /*
      Initialize the display size
  */
  if (xmax == 0) {
    ierr = DrawXGetDisplaySize_Private(display,&xmax,&ymax); CHKERRQ(ierr);
  }

  if (x == PETSC_DECIDE || y == PETSC_DECIDE) {
    /*
       PETSc tries to place windows starting in the upper left corner and 
       moving across to the right. 
    
              --------------------------------------------
              |  Region used so far +xavailable,yavailable |
              |                     +                      |
              |                     +                      |
              |++++++++++++++++++++++ybottom               |
              |                                            |
              |                                            |
              |--------------------------------------------|
    */
    /*  First: can we add it to the right? */
    if (xavailable+w+10 <= xmax) {
      x       = xavailable;
      y       = yavailable;
      ybottom = PetscMax(ybottom,y + h + 30);
    } else {
      /* No, so add it below on the left */
      xavailable = 0;
      x          = 0;
      yavailable = ybottom;    
      y          = ybottom;
      ybottom    = ybottom + h + 30;
    }
  }
  /* update available region */
  xavailable = PetscMax(xavailable,x + w + 10);
  if (xavailable >= xmax) {
    xavailable = 0;
    yavailable = yavailable + h + 30;
    ybottom    = yavailable;
  }

  *inctx = 0;
  PetscHeaderCreate(ctx,_p_Draw,struct _DrawOps,DRAW_COOKIE,DRAW_XWINDOW,comm,DrawDestroy,0);
  PLogObjectCreate(ctx);
  PetscMemcpy(ctx->ops,&DvOps,sizeof(DvOps));
  ctx->destroy = DrawDestroy_X;
  ctx->view    = 0;
  ctx->pause   = 0;
  ctx->coor_xl = 0.0;  ctx->coor_xr = 1.0;
  ctx->coor_yl = 0.0;  ctx->coor_yr = 1.0;
  ctx->port_xl = 0.0;  ctx->port_xr = 1.0;
  ctx->port_yl = 0.0;  ctx->port_yr = 1.0;
  ctx->popup   = 0;

  if (title) {
    int len    = PetscStrlen(title);
    ctx->title = (char *) PetscMalloc((len+1)*sizeof(char*));CHKPTRQ(ctx->title);
    PLogObjectMemory(ctx,(len+1)*sizeof(char*));
    PetscStrcpy(ctx->title,title);
  } else {
    ctx->title = 0;
  }

  ierr = OptionsGetInt(PETSC_NULL,"-draw_pause",&ctx->pause,&flg);CHKERRQ(ierr);

  /* actually create and open the window */
  Xwin         = (Draw_X *) PetscMalloc( sizeof(Draw_X) ); CHKPTRQ(Xwin);
  PLogObjectMemory(ctx,sizeof(Draw_X)+sizeof(struct _p_Draw));
  PetscMemzero(Xwin,sizeof(Draw_X));
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  if (rank == 0) {
    if (x < 0 || y < 0)   SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative corner of window");
    if (w <= 0 || h <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative window width or height");
    ierr = XiQuickWindow(Xwin,display,title,x,y,w,h,256); CHKERRQ(ierr);
    ierr = MPI_Bcast(&Xwin->win,1,MPI_UNSIGNED_LONG,0,comm);CHKERRQ(ierr);
  } else {
    unsigned long win;
    ierr = MPI_Bcast(&win,1,MPI_UNSIGNED_LONG,0,comm);CHKERRQ(ierr);
    ierr = XiQuickWindowFromWindow( Xwin,display, win,256 ); CHKERRQ(ierr);
  }

  Xwin->x      = x;
  Xwin->y      = y;
  Xwin->w      = w;
  Xwin->h      = h;
  ctx->data    = (void *) Xwin;

  /*
      Need barrier here so processor 0 doesn't destroy the window before other 
    processors have completed XiQuickWindow()
  */
  ierr = DrawClear(ctx);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(ctx);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-draw_double_buffer",&flg);CHKERRQ(ierr);
  if (flg) {
     ierr = DrawSetDoubleBuffer(ctx); CHKERRQ(ierr);
  } 
  *inctx       = ctx;

  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ "DrawOpenX" 
int DrawOpenX(MPI_Comm comm,char* disp,char *ttl,int x,int y,int w,int h,Draw* ctx)
{
  int rank,flag,ierr;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  OptionsHasName(PETSC_NULL,"-nox",&flag);
  if (!flag && !rank) {
    (*PetscErrorPrintf)("PETSc installed without X windows on this machine\n");
    (*PetscErrorPrintf)("proceeding without graphics\n");
  }
  ierr = DrawOpenNull(comm,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif


#undef __FUNC__  
#define __FUNC__ "ViewerDrawOpenX" 
/*@C
   ViewerDrawOpenX - Opens an X window for use as a viewer. If you want to 
   do graphics in this window, you must call ViewerDrawGetDraw() and
   perform the graphics on the Draw object.

   Input Parameters:
.  comm - communicator that will share window
.  display - the X display on which to open, or null for the local machine
.  title - the title to put in the title bar, or null for no title
.  x, y - the screen coordinates of the upper left corner of window
.  w, h - the screen width and height in pixels

   Output Parameters:
.  viewer - the viewer

   Format Options:
.   VIEWER_FORMAT_DRAW_BASIC
.   VIEWER_FORMAT_DRAW_LG     - displays using a line graph

   Options Database Keys:
   ViewerDrawOpenX() calls DrawOpenX(), so see the man page for
   DrawOpenX() for runtime options, including
$  -nox : disable all x-windows output
$  -display <name> : name of machine for the X display
$  -draw_pause <pause> : sets time (in seconds) that the
$     program pauses after DrawPause() has been called
$     (0 is default, -1 implies until user input).

   Note for Fortran Programmers:
   Whenever indicating null character data in a Fortran code,
   PETSC_NULL_CHARACTER must be employed; using PETSC_NULL is not
   correct for character data!  Thus, PETSC_NULL_CHARACTER can be
   used for the display and title input parameters.

.keywords: draw, open, x, viewer

.seealso: DrawOpenX()
@*/
int ViewerDrawOpenX(MPI_Comm comm,char* display,char *title,int x,int y,
                    int w,int h,Viewer *viewer)
{
  int    ierr;
  Viewer ctx;

  *viewer = 0;
  PetscHeaderCreate(ctx,_p_Viewer,int,VIEWER_COOKIE,DRAW_VIEWER,comm,ViewerDestroy,0);
  PLogObjectCreate(ctx);
  ierr = DrawOpenX(comm,display,title,x,y,w,h,&ctx->draw);CHKERRQ(ierr);
  PLogObjectParent(ctx,ctx->draw);

  ctx->flush   = ViewerFlush_Draw;
  ctx->destroy = ViewerDestroy_Draw;
  ctx->format  = 0;


  /* these are created on the fly if requested */
  ctx->drawlg   = 0; 
  ctx->drawaxis = 0;
  *viewer       = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawClear" 
/*@
    ViewerDrawClear - Clears a Draw graphic associated with a viewer.

  Input Parameter:
.  viewer - the viewer 

@*/
int ViewerDrawClear(Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  if (viewer->type != DRAW_VIEWER) return 0;
  ierr = DrawClear(viewer->draw); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewersDrawOpenX" 
int ViewersDrawOpenX(MPI_Comm comm,char* display,char **titles,int n,int w,int h,Viewer **viewer)
{
  int i,ierr;

  PetscFunctionBegin;
  *viewer = (Viewer * ) PetscMalloc((n+1)*sizeof(Viewer *));CHKPTRQ(*viewer);
  for ( i=0; i<n; i++ ) {
    ierr = ViewerDrawOpenX(comm,display,titles[i],PETSC_DECIDE,PETSC_DECIDE,w,h,(*viewer+i));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewersDestroy" 
int ViewersDestroy(int n,Viewer *viewer)
{
  int i,ierr;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    ierr = ViewerDestroy(viewer[i]);CHKERRQ(ierr);
  }
  PetscFree(viewer);
  PetscFunctionReturn(0); 
}
 
/* -------------------------------------------------------------------*/
/* 
     Default X window viewers, may be used at any time.
*/

Viewer VIEWER_DRAWX_SELF_PRIVATE = 0, VIEWER_DRAWX_WORLD_PRIVATE_0 = 0,
       VIEWER_DRAWX_WORLD_PRIVATE_1 = 0, VIEWER_DRAWX_WORLD_PRIVATE_2 = 0;

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeDrawXSelf_Private" 
int ViewerInitializeDrawXSelf_Private()
{
  int ierr,xywh[4],size = 4,flg;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_SELF_PRIVATE) PetscFunctionReturn(0);
  xywh[0] = PETSC_DECIDE; xywh[1] = PETSC_DECIDE; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_self_geometry",xywh,&size,&flg);CHKERRQ(ierr);
  ierr = ViewerDrawOpenX(PETSC_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_SELF_PRIVATE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeDrawXWorld_Private_0" 
int ViewerInitializeDrawXWorld_Private_0()
{
  int ierr,xywh[4],size = 4,flg;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_WORLD_PRIVATE_0) PetscFunctionReturn(0);
  xywh[0] = PETSC_DECIDE; xywh[1] = PETSC_DECIDE; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_world_geometry",xywh,&size,&flg);CHKERRQ(ierr);
  ierr = ViewerDrawOpenX(PETSC_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_WORLD_PRIVATE_0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeDrawXWorld_Private_1" 
int ViewerInitializeDrawXWorld_Private_1()
{
  int ierr,xywh[4],size = 4,flg;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_WORLD_PRIVATE_1) PetscFunctionReturn(0);
  xywh[0] = PETSC_DECIDE; xywh[1] = PETSC_DECIDE; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_world_geometry",xywh,&size,&flg);CHKERRQ(ierr);
  ierr = ViewerDrawOpenX(PETSC_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_WORLD_PRIVATE_1); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeDrawXWorld_Private_2" 
int ViewerInitializeDrawXWorld_Private_2()
{
  int ierr,xywh[4],size = 4,flg;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_WORLD_PRIVATE_2) PetscFunctionReturn(0);
  xywh[0] = PETSC_DECIDE; xywh[1] = PETSC_DECIDE; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_world_geometry",xywh,&size,&flg);CHKERRQ(ierr);
  ierr = ViewerDrawOpenX(PETSC_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_WORLD_PRIVATE_2); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "ViewerDestroyDrawX_Private" 
int ViewerDestroyDrawX_Private()
{
  int ierr;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_WORLD_PRIVATE_0) {
    ierr = ViewerDestroy(VIEWER_DRAWX_WORLD_PRIVATE_0); CHKERRQ(ierr);
  }
  if (VIEWER_DRAWX_WORLD_PRIVATE_1) {
    ierr = ViewerDestroy(VIEWER_DRAWX_WORLD_PRIVATE_1); CHKERRQ(ierr);
  }
  if (VIEWER_DRAWX_WORLD_PRIVATE_2) {
    ierr = ViewerDestroy(VIEWER_DRAWX_WORLD_PRIVATE_2); CHKERRQ(ierr);
  }
  if (VIEWER_DRAWX_SELF_PRIVATE) {
    ierr = ViewerDestroy(VIEWER_DRAWX_SELF_PRIVATE); CHKERRQ(ierr);
  }
  /*
      Free any viewers created with the VIEWER_DRAWX_(MPI_Comm comm) trick.
  */
  ierr = VIEWER_DRAWX_Destroy(PETSC_COMM_WORLD); CHKERRQ(ierr);
  ierr = VIEWER_DRAWX_Destroy(PETSC_COMM_SELF); CHKERRQ(ierr);
  ierr = VIEWER_DRAWX_Destroy(MPI_COMM_WORLD); CHKERRQ(ierr);
  ierr = VIEWER_DRAWX_Destroy(MPI_COMM_SELF); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Drawx_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Drawx_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "VIEWER_DRAWX_" 
/*@C
     VIEWER_DRAWX_ - Creates a window viewer shared by all processors 
                     in a communicator.

  Input Parameters:
.  comm - the MPI communicator to share the window viewer

  Note: Unlike almost all other PETSc routines this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,VIEWER_DRAWX_(comm));

.seealso: VIEWER_DRAWX_WORLD, VIEWER_DRAWX_SELF, ViewerDrawOpenX(), 
C@*/
Viewer VIEWER_DRAWX_(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Drawx_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Drawx_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_DRAWX_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Drawx_keyval, (void **)&viewer, &flag );
  if (ierr) {PetscError(__LINE__,"VIEWER_DRAWX_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */
    ierr = ViewerDrawOpenX(comm,0,0,PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer); 
    if (ierr) {PetscError(__LINE__,"VIEWER_DRAWX_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put( comm, Petsc_Viewer_Drawx_keyval, (void *) viewer );
    if (ierr) {PetscError(__LINE__,"VIEWER_DRAWX_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a Viewer associated with this communicator it is destroyed.
*/
int VIEWER_DRAWX_Destroy(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Drawx_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Drawx_keyval, (void **)&viewer, &flag );CHKERRQ(ierr);
  if (flag) { 
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Drawx_keyval);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}



