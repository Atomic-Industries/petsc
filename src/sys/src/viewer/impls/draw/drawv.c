#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: drawv.c,v 1.28 1998/12/17 22:12:13 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "src/viewer/impls/draw/vdraw.h" /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_Draw" 
int ViewerDestroy_Draw(Viewer v)
{
  int         ierr,i;
  Viewer_Draw *vdraw = (Viewer_Draw*) v->data;

  PetscFunctionBegin;
  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    if (vdraw->drawaxis[i]) {ierr = DrawAxisDestroy(vdraw->drawaxis[i]); CHKERRQ(ierr);}
    if (vdraw->drawlg[i])   {ierr = DrawLGDestroy(vdraw->drawlg[i]); CHKERRQ(ierr);}
    if (vdraw->draw[i])     {ierr = DrawDestroy(vdraw->draw[i]); CHKERRQ(ierr);}
  }
  PetscFree(vdraw);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerFlush_Draw" 
int ViewerFlush_Draw(Viewer v)
{
  int         ierr,i;
  Viewer_Draw *vdraw = (Viewer_Draw*) v->data;

  PetscFunctionBegin;
  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    if (vdraw->draw[i]) {ierr = DrawSynchronizedFlush(vdraw->draw[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawGetDraw" 
/*@C
    ViewerDrawGetDraw - Returns Draw object from Viewer object.
    This Draw object may then be used to perform graphics using 
    DrawXXX() commands.

    Not collective (but Draw returned will be parallel object if Viewer is)

    Input Parameter:
+   viewer - the viewer (created with ViewerDrawOpen()
-   windownumber - indicates which subwindow (usually 0)

    Ouput Parameter:
.   draw - the draw object

.keywords: viewer, draw, get

.seealso: ViewerDrawGetLG(), ViewerDrawGetAxis()
@*/
int ViewerDrawGetDraw(Viewer v, int windownumber, Draw *draw)
{
  Viewer_Draw *vdraw;
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (PetscStrcmp(v->type_name,DRAW_VIEWER)) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  if (windownumber < 0 || windownumber >= VIEWER_DRAW_MAX) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Window number out of range");
  }

  vdraw = (Viewer_Draw *) v->data;
  if (!vdraw->draw[windownumber]) {
    ierr = DrawCreate(v->comm,vdraw->display,0,PETSC_DECIDE,PETSC_DECIDE,vdraw->w,vdraw->h,
                     &vdraw->draw[windownumber]);CHKERRQ(ierr);
    ierr = DrawSetFromOptions(vdraw->draw[windownumber]);CHKERRQ(ierr);
  }
  *draw = vdraw->draw[windownumber];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawGetDrawLG" 
/*@C
    ViewerDrawGetDrawLG - Returns DrawLG object from Viewer object.
    This DrawLG object may then be used to perform graphics using 
    DrawLGXXX() commands.

    Not Collective (but DrawLG object will be parallel if Viewer is)

    Input Parameter:
+   viewer - the viewer (created with ViewerDrawOpen())
-   windownumber - indicates which subwindow (usually 0)

    Ouput Parameter:
.   draw - the draw line graph object

.keywords: viewer, draw, get, line graph

.seealso: ViewerDrawGetDraw(), ViewerDrawGetAxis()
@*/
int ViewerDrawGetDrawLG(Viewer v, int windownumber,DrawLG *drawlg)
{
  int         ierr;
  Viewer_Draw *vdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (PetscStrcmp(v->type_name,DRAW_VIEWER)) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  if (windownumber < 0 || windownumber >= VIEWER_DRAW_MAX) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Window number out of range");
  }
  vdraw = (Viewer_Draw *) v->data;
  if (!vdraw->draw[windownumber]) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"No window with that number");
  }

  if (!vdraw->drawlg[windownumber]) {
    ierr = DrawLGCreate(vdraw->draw[windownumber],1,&vdraw->drawlg[windownumber]);CHKERRQ(ierr);
    PLogObjectParent(v,vdraw->drawlg[windownumber]);
  }
  *drawlg = vdraw->drawlg[windownumber];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawGetDrawAxis" 
/*@C
    ViewerDrawGetDrawAxis - Returns DrawAxis object from Viewer object.
    This DrawAxis object may then be used to perform graphics using 
    DrawAxisXXX() commands.

    Not Collective (but DrawAxis object will be parallel if Viewer is)

    Input Parameter:
+   viewer - the viewer (created with ViewerDrawOpen()
-   windownumber - indicates which subwindow (usually 0)

    Ouput Parameter:
.   drawaxis - the draw axis object

.keywords: viewer, draw, get, line graph

.seealso: ViewerDrawGetDraw(), ViewerDrawGetLG()
@*/
int ViewerDrawGetDrawAxis(Viewer v, int windownumber, DrawAxis *drawaxis)
{
  int         ierr;
  Viewer_Draw *vdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (PetscStrcmp(v->type_name,DRAW_VIEWER)) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  if (windownumber < 0 || windownumber >= VIEWER_DRAW_MAX) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Window number out of range");
  }
  vdraw = (Viewer_Draw *) v->data;
  if (!vdraw->draw[windownumber]) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"No window with that number");
  }

  if (!vdraw->drawaxis[windownumber]) {
    ierr = DrawAxisCreate(vdraw->draw[windownumber],&vdraw->drawaxis[windownumber]);CHKERRQ(ierr);
    PLogObjectParent(v,vdraw->drawaxis[windownumber]);
  }
  *drawaxis = vdraw->drawaxis[windownumber];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawOpen" 
/*@C
   ViewerDrawOpen - Opens an X window for use as a viewer. If you want to 
   do graphics in this window, you must call ViewerDrawGetDraw() and
   perform the graphics on the Draw object.

   Collective on MPI_Comm

   Input Parameters:
+  comm - communicator that will share window
.  display - the X display on which to open, or null for the local machine
.  title - the title to put in the title bar, or null for no title
.  x, y - the screen coordinates of the upper left corner of window
-  w, h - the screen width and height in pixels

   Output Parameters:
.  viewer - the viewer

   Format Options:
+  VIEWER_FORMAT_DRAW_BASIC - displays with basic format
-  VIEWER_FORMAT_DRAW_LG    - displays using a line graph

   Options Database Keys:
   ViewerDrawOpen() calls DrawOpen(), so see the manual page for
   DrawOpen() for runtime options, including
+  -draw_type x or null
.  -nox - Disables all x-windows output
.  -display <name> - Specifies name of machine for the X display
-  -draw_pause <pause> - Sets time (in seconds) that the
     program pauses after DrawPause() has been called
     (0 is default, -1 implies until user input).

   Note for Fortran Programmers:
   Whenever indicating null character data in a Fortran code,
   PETSC_NULL_CHARACTER must be employed; using PETSC_NULL is not
   correct for character data!  Thus, PETSC_NULL_CHARACTER can be
   used for the display and title input parameters.

.keywords: draw, open, x, viewer

.seealso: DrawOpen()
@*/
int ViewerDrawOpen(MPI_Comm comm,const char display[],const char title[],int x,int y,
                    int w,int h,Viewer *viewer)
{
  int         ierr,i;
  Viewer      ctx;
  Viewer_Draw *vdraw;

  *viewer = 0;
  PetscHeaderCreate(ctx,_p_Viewer,struct _ViewerOps,VIEWER_COOKIE,0,"Viewer",comm,ViewerDestroy,0);
  PLogObjectCreate(ctx);
  vdraw     = PetscNew(Viewer_Draw);CHKPTRQ(vdraw);
  ctx->data = (void *) vdraw;

  ctx->ops->flush   = ViewerFlush_Draw;
  ctx->ops->destroy = ViewerDestroy_Draw;
  ctx->format       = 0;

  ctx->type_name = (char *)PetscMalloc((1+PetscStrlen(DRAW_VIEWER))*sizeof(char));CHKPTRQ(ctx->type_name);
  PetscStrcpy(ctx->type_name,DRAW_VIEWER);

  /* these are created on the fly if requested */
  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    vdraw->draw[i]     = 0; 
    vdraw->drawlg[i]   = 0; 
    vdraw->drawaxis[i] = 0;
  }
  vdraw->h  = h;
  vdraw->w  = w;
  if (display) {
    vdraw->display = (char *) PetscMalloc((1+PetscStrlen(display))*sizeof(char));CHKPTRQ(vdraw->display);
    PetscStrcpy(vdraw->display,display);
  } else {
    vdraw->display = 0;
  } 
  ierr      = DrawCreate(comm,display,title,x,y,w,h,&vdraw->draw[0]);CHKERRQ(ierr);
  ierr      = DrawSetFromOptions(vdraw->draw[0]);CHKERRQ(ierr);
  PLogObjectParent(ctx,vdraw->draw[0]);
  *viewer         = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawClear" 
/*@
    ViewerDrawClear - Clears a Draw graphic associated with a viewer.

    Not Collective

    Input Parameter:
.   viewer - the viewer 

@*/
int ViewerDrawClear(Viewer viewer)
{
  int         ierr,i;
  Viewer_Draw *vdraw;

  PetscFunctionBegin;
  if (PetscTypeCompare(viewer->type_name,DRAW_VIEWER)) PetscFunctionReturn(0);
  vdraw = (Viewer_Draw *) viewer->data;
  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    if (vdraw->draw[i]) {ierr = DrawClear(vdraw->draw[i]); CHKERRQ(ierr);}
  }
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
int ViewerInitializeDrawXSelf_Private(void)
{
  int ierr,xywh[4],size = 4,flg;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_SELF_PRIVATE) PetscFunctionReturn(0);
  xywh[0] = PETSC_DECIDE; xywh[1] = PETSC_DECIDE; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_self_geometry",xywh,&size,&flg);CHKERRQ(ierr);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_SELF_PRIVATE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeDrawXWorld_Private_0" 
int ViewerInitializeDrawXWorld_Private_0(void)
{
  int ierr,xywh[4],size = 4,flg;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_WORLD_PRIVATE_0) PetscFunctionReturn(0);
  xywh[0] = PETSC_DECIDE; xywh[1] = PETSC_DECIDE; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_world_geometry",xywh,&size,&flg);CHKERRQ(ierr);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_WORLD_PRIVATE_0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeDrawXWorld_Private_1" 
int ViewerInitializeDrawXWorld_Private_1(void)
{
  int ierr,xywh[4],size = 4,flg;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_WORLD_PRIVATE_1) PetscFunctionReturn(0);
  xywh[0] = PETSC_DECIDE; xywh[1] = PETSC_DECIDE; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_world_geometry",xywh,&size,&flg);CHKERRQ(ierr);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_WORLD_PRIVATE_1); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeDrawXWorld_Private_2" 
int ViewerInitializeDrawXWorld_Private_2(void)
{
  int ierr,xywh[4],size = 4,flg;

  PetscFunctionBegin;
  if (VIEWER_DRAWX_WORLD_PRIVATE_2) PetscFunctionReturn(0);
  xywh[0] = PETSC_DECIDE; xywh[1] = PETSC_DECIDE; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_world_geometry",xywh,&size,&flg);CHKERRQ(ierr);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_WORLD_PRIVATE_2); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "ViewerDestroyDrawX_Private" 
int ViewerDestroyDrawX_Private(void)
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

     Collective on MPI_Comm

     Input Parameter:
.    comm - the MPI communicator to share the window viewer

     Notes:
     Unlike almost all other PETSc routines, VIEWER_DRAWX_ does not return 
     an error code.  The window viewer is usually used in the form
$       XXXView(XXX object,VIEWER_DRAWX_(comm));

.seealso: VIEWER_DRAWX_WORLD, VIEWER_DRAWX_SELF, ViewerDrawOpen(), 
@*/
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
    ierr = ViewerDrawOpen(comm,0,0,PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer); 
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

/* ---------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "ViewerDrawOpenVRML" 
/*@C
   ViewerDrawOpenVRML - Opens an VRML file for use as a viewer. If you want to 
   do graphics in this window, you must call ViewerDrawGetDraw() and
   perform the graphics on the Draw object.

   Input Parameters:
+  comm - communicator that will share window
.  fname - the filename on which to open, or null for stdout
.  title - the title to put in the title bar
.  x, y - the screen coordinates of the upper left corner of window
-  w, h - the screen width and height in pixels

   Output Parameters:
.  viewer - the viewer

   Options Database Keys:
.  -nox - Disables all x-windows output
.  -display <name> - Specifies name of machine for the X display

.keywords: draw, open, VRML, viewer

.seealso: DrawOpenVRML()
@*/
int ViewerDrawOpenVRML(MPI_Comm comm,const char fname[],const char title[],Viewer *viewer)
{
  int         ierr;
  Viewer      ctx;
  Viewer_Draw *vdraw;

  *viewer = 0;
  PetscHeaderCreate(ctx,_p_Viewer,struct _ViewerOps,VIEWER_COOKIE,0,"Viewer",comm,ViewerDestroy,0);
  vdraw     = PetscNew(Viewer_Draw);CHKPTRQ(vdraw);
  ctx->data = (void *) vdraw;
  PLogObjectCreate(ctx);
  ierr = DrawOpenVRML(comm,fname,title,&vdraw->draw[0]); CHKERRQ(ierr);
  PLogObjectParent(ctx,vdraw->draw[0]);

  ctx->ops->flush   = ViewerFlush_Draw;
  ctx->ops->destroy = ViewerDestroy_Draw;
  ctx->type_name = (char *)PetscMalloc((1+PetscStrlen(DRAW_VIEWER))*sizeof(char));CHKPTRQ(ctx->type_name);
  PetscStrcpy(ctx->type_name,DRAW_VIEWER);

  *viewer           = ctx;
  PetscFunctionReturn(0);
}
