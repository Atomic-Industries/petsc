/* $Id: drawimpl.h,v 1.24 1998/03/06 00:16:45 bsmith Exp bsmith $ */
/*
       Abstract data structure and functions for graphics.
*/

#if !defined(_DRAWIMPL_H)
#define _DRAWIMPL_H

#include "petsc.h"

struct _DrawOps {
  int (*setdoublebuffer)(Draw);
  int (*flush)(Draw);
  int (*line)(Draw,double,double,double,double,int);
  int (*linesetwidth)(Draw,double);
  int (*linegetwidth)(Draw,double*);
  int (*point)(Draw,double,double,int);
  int (*pointsetsize)(Draw,double);
  int (*text)(Draw,double,double,int,char*);
  int (*textvertical)(Draw,double,double,int,char*);
  int (*textsetsize)(Draw,double,double);
  int (*textgetsize)(Draw,double*,double*);
  int (*setviewport)(Draw,double,double,double,double);
  int (*clear)(Draw);
  int (*syncflush)(Draw);
  int (*rectangle)(Draw,double,double,double,double,int,int,int,int);
  int (*triangle)(Draw,double,double,double,double,double,double,int,int,int);
  int (*getmousebutton)(Draw,DrawButton*,double *,double *,double*,double*);
  int (*pause)(Draw);
  int (*syncclear)(Draw);
  int (*beginpage)(Draw);
  int (*endpage)(Draw);
  int (*getpopup)(Draw,Draw*);
  int (*settitle)(Draw,char *);
  int (*checkresizedwindow)(Draw);
  int (*resizewindow)(Draw,int,int);
};

struct _p_Draw {
  PETSCHEADER(struct _DrawOps)
  int             pause;       /* sleep time after a sync flush */
  double          port_xl,port_yl,port_xr,port_yr;
  double          coor_xl,coor_yl,coor_xr,coor_yr;
  char            *title;
  Draw            popup;
  void            *data;
};

/*
     This is for the Draw version of the viewer
*/
#include "pinclude/pviewer.h"
struct _p_Viewer {
  VIEWERHEADER
  Draw         draw;
  DrawLG       drawlg;
  DrawAxis     drawaxis;
};

extern int ViewerDestroy_Draw(PetscObject);
extern int ViewerFlush_Draw(Viewer);

#endif
