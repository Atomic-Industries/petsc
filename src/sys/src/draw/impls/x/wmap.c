#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: wmap.c,v 1.13 1997/02/22 02:27:19 bsmith Exp balay $";
#endif

#if defined(HAVE_X11)
#include "src/draw/impls/x/ximpl.h"

/*
    This routine waits until the window is actually created or destroyed
    Returns 0 if window is mapped; 1 if window is destroyed.
 */
#undef __FUNC__  
#define __FUNC__ "Xi_wait_map" /* ADIC Ignore */
int Xi_wait_map( Draw_X *XiWin)
{
  XEvent  event;
  int     w, h;

  /*
   This is a bug.  XSelectInput should be set BEFORE the window is mapped
  */
  /*
  XSelectInput( XiWin->disp, XiWin->win,
	        ExposureMask | StructureNotifyMask );
  */
  while (1) {
    XMaskEvent( XiWin->disp, ExposureMask | StructureNotifyMask, &event );
    if (event.xany.window != XiWin->win) {
      break;
      /* Bug for now */
    }
    else {
      switch (event.type) {
        case ConfigureNotify:
        /* window has been moved or resized */
        w       = event.xconfigure.width  - 2 * event.xconfigure.border_width;
        h       = event.xconfigure.height - 2 * event.xconfigure.border_width;
        XiWin->w  = w;
        XiWin->h  = h;
        break;
      case DestroyNotify:
        return 1;
      case Expose:
        return 0;
      /* else ignore event */
      }
    }
  }
  return 0;
}

#endif
