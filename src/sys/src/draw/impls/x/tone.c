
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tone.c,v 1.18 1997/12/04 19:37:13 bsmith Exp bsmith $";
#endif

/* Include petsc in case it is including petscconf.h */
#include "petsc.h"

/*
    Code for drawing color interpolated triangles using X-windows.
*/
#if defined(HAVE_X11)
#include "src/draw/impls/x/ximpl.h"

#define SHIFT_VAL 6

#undef __FUNC__  
#define __FUNC__ "XiDrawInterpolatedTriangle"
int XiDrawInterpolatedTriangle(Draw_X* win, int x1, int y1, int t1, 
                                int x2,int y2,int t2,int x3,int y3,int t3)
{
  double rfrac, lfrac;
  int    lc, rc = 0, lx, rx = 0, xx, y, c;
  int    rc_lc, rx_lx, t2_t1, x2_x1, t3_t1, x3_x1, t3_t2, x3_x2;
  double R_y2_y1, R_y3_y1, R_y3_y2;

  PetscFunctionBegin;
  t1 = t1 << SHIFT_VAL;
  t2 = t2 << SHIFT_VAL;
  t3 = t3 << SHIFT_VAL;

  /* Sort the vertices */
#define SWAP(a,b) {int _a; _a=a; a=b; b=_a;}
  if (y1 > y2) {
    SWAP(y1,y2);SWAP(t1,t2); SWAP(x1,x2);
  }
  if (y1 > y3) {
    SWAP(y1,y3);SWAP(t1,t3); SWAP(x1,x3);
  }
  if (y2 > y3) {
    SWAP(y2,y3);SWAP(t2,t3); SWAP(x2,x3);
  }
  /* This code is decidely non-optimal; it is intended to be a start at
   an implementation */

  if (y2 != y1) R_y2_y1 = 1.0/((double)(y2-y1)); else R_y2_y1 = 0.0; 
  if (y3 != y1) R_y3_y1 = 1.0/((double)(y3-y1)); else R_y3_y1 = 0.0;
  t2_t1   = t2 - t1;
  x2_x1   = x2 - x1;
  t3_t1   = t3 - t1;
  x3_x1   = x3 - x1;
  for (y=y1; y<=y2; y++) {
    /* Draw a line with the correct color from t1-t2 to t1-t3 */
    /* Left color is (y-y1)/(y2-y1) * (t2-t1) + t1 */
    lfrac = ((double)(y-y1)) * R_y2_y1; 
    lc    = (int)(lfrac * (t2_t1) + t1);
    lx    = (int)(lfrac * (x2_x1) + x1);
    /* Right color is (y-y1)/(y3-y1) * (t3-t1) + t1 */
    rfrac = ((double)(y - y1)) * R_y3_y1; 
    rc    = (int)(rfrac * (t3_t1) + t1);
    rx    = (int)(rfrac * (x3_x1) + x1);
    /* Draw the line */
    rc_lc = rc - lc; 
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx=lx; xx<=rx; xx++) {
        c = (((xx-lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        XiSetColor( win, c );
        XDrawPoint(win->disp,XiDrawable(win),win->gc.set,xx,y);
      }
    } else if (rx < lx) {
      for (xx=lx; xx>=rx; xx--) {
        c = (((xx-lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        XiSetColor( win, c );
        XDrawPoint(win->disp,XiDrawable(win),win->gc.set,xx,y);
      }
    } else {
      c = lc >> SHIFT_VAL;
      XiSetColor( win, c );
      XDrawPoint(win->disp,XiDrawable(win),win->gc.set,lx,y);
    }
  }

  /* For simplicity, "move" t1 to the intersection of t1-t3 with the line y=y2.
     We take advantage of the previous iteration. */
  if (y2 >= y3) PetscFunctionReturn(0);
  if (y1 < y2) {
    t1 = rc;
    y1 = y2;
    x1 = rx;

    t3_t1   = t3 - t1;
    x3_x1   = x3 - x1;    
  }
  t3_t2 = t3 - t2;
  x3_x2 = x3 - x2;
  if (y3 != y2) R_y3_y2 = 1.0/((double)(y3-y2)); else R_y3_y2 = 0.0;
  if (y3 != y1) R_y3_y1 = 1.0/((double)(y3-y1)); else R_y3_y1 = 0.0;
  for (y=y2; y<=y3; y++) {
    /* Draw a line with the correct color from t2-t3 to t1-t3 */
    /* Left color is (y-y1)/(y2-y1) * (t2-t1) + t1 */
    lfrac = ((double)(y-y2)) * R_y3_y2; 
    lc    = (int)(lfrac * (t3_t2) + t2);
    lx    = (int)(lfrac * (x3_x2) + x2);
    /* Right color is (y-y1)/(y3-y1) * (t3-t1) + t1 */
    rfrac = ((double)(y - y1)) * R_y3_y1 ; 
    rc    = (int)(rfrac * (t3_t1) + t1);
    rx    = (int)(rfrac * (x3_x1) + x1);
    /* Draw the line */
    rc_lc = rc - lc; 
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx=lx; xx<=rx; xx++) {
        c = (((xx-lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        XiSetColor( win, c );
        XDrawPoint(win->disp,XiDrawable(win),win->gc.set,xx,y);
      }
    }
    else if (rx < lx) {
      for (xx=lx; xx>=rx; xx--) {
        c = (((xx-lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        XiSetColor( win, c );
        XDrawPoint(win->disp,XiDrawable(win),win->gc.set,xx,y);
      }
    }
    else {
      c = lc >> SHIFT_VAL;
      XiSetColor( win, c );
      XDrawPoint(win->disp,XiDrawable(win),win->gc.set,lx,y);
    }
  }
  PetscFunctionReturn(0);
}
#else
int dummy_tone()
{
  PetscFunctionReturn(0);
}
#endif
