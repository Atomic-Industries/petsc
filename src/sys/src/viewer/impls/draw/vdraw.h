/* $Id: vdraw.h,v 1.1 1998/11/30 16:59:52 bsmith Exp bsmith $ */
/*
     Data structure for the Draw version of the viewer
*/

#if !defined(__VDRAW_H)
#define __VDRAWL_H

#define VIEWER_DRAW_MAX 5

#include "src/sys/src/viewer/viewerimpl.h"
typedef struct {
  Draw         draw[VIEWER_DRAW_MAX];
  DrawLG       drawlg[VIEWER_DRAW_MAX];
  DrawAxis     drawaxis[VIEWER_DRAW_MAX];
  int          w,h;                          /* These are saved in case additional windows are opened */
  char         *display;
} Viewer_Draw;

#endif
