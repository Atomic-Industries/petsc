#ifndef lint
static char vcid[] = "$Id: axis.c,v 1.34 1996/12/16 22:14:34 balay Exp balay $";
#endif
/*
   This file contains a simple routine for generating a 2-d axis.
*/

#include "petsc.h"
#include "draw.h"              /*I "draw.h" I*/
#include <math.h>

struct _DrawAxis {
    PETSCHEADER
    double  xlow, ylow, xhigh, yhigh;     /* User - coord limits */
    char    *(*ylabelstr)(double,double), /* routines to generate labels */ 
            *(*xlabelstr)(double,double);
    int     (*xlabels)(), (*ylabels)()  , /* location of labels */
            (*xticks)(double,double,int,int*,double*,int),
            (*yticks)(double,double,int,int*,double*,int);  
                                          /* location and size of ticks */
    Draw win;
    int     ac,tc,cc;                     /* axis, tick, charactor color */
    char    *xlabel,*ylabel,*toplabel;
};

#define MAXSEGS 20

static int    PetscADefTicks(double,double,int,int*,double*,int);
static char   *PetscADefLabel(double,double);
static double PetscAGetNice(double,double,int );
static int    PetscAGetBase(double,double,int,double*,int*);

#undef __FUNC__  
#define __FUNC__ "PetscRint"
static double PetscRint(double x )
{
  if (x > 0) return floor( x + 0.5 );
  return floor( x - 0.5 );
}

#undef __FUNC__  
#define __FUNC__ "DrawAxisCreate"
/*@C
   DrawAxisCreate - Generate the axis data structure.

   Input Parameters:

   Ouput Parameters:
.   axis - the axis datastructure

@*/
int DrawAxisCreate(Draw win,DrawAxis *ctx)
{
  DrawAxis ad;
  PetscObject vobj = (PetscObject) win;

  if (vobj->cookie == DRAW_COOKIE && vobj->type == DRAW_NULLWINDOW) {
     return DrawOpenNull(vobj->comm,(Draw*)ctx);
  }
  PetscHeaderCreate(ad,_DrawAxis,DRAWAXIS_COOKIE,0,vobj->comm);
  PLogObjectCreate(ad);
  PLogObjectParent(win,ad);
  ad->xticks    = PetscADefTicks;
  ad->yticks    = PetscADefTicks;
  ad->xlabelstr = PetscADefLabel;
  ad->ylabelstr = PetscADefLabel;
  ad->win       = win;
  ad->ac        = DRAW_BLACK;
  ad->tc        = DRAW_BLACK;
  ad->cc        = DRAW_BLACK;
  ad->xlabel    = 0;
  ad->ylabel    = 0;
  ad->toplabel  = 0;

  *ctx = ad;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawAxisDestroy"
/*@C
      DrawAxisDestroy - Frees the space used by an axis structure.

  Input Parameters:
.   axis - the axis context
@*/
int DrawAxisDestroy(DrawAxis ad)
{
  if (!ad) return 0;
  PLogObjectDestroy(ad);
  PetscHeaderDestroy(ad);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawAxisSetColors"
/*@
    DrawAxisSetColors -  Sets the colors to be used for the axis,       
                         tickmarks, and text.

   Input Parameters:
.   axis - the axis
.   ac - the color of the axis lines
.   tc - the color of the tick marks
.   cc - the color of the text strings
@*/
int DrawAxisSetColors(DrawAxis ad,int ac,int tc,int cc)
{
  if (!ad) return 0;
  ad->ac = ac; ad->tc = tc; ad->cc = cc;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawAxisSetLabels"
/*@C
    DrawAxisSetLabels -  Sets the x and y axis labels.


   Input Parameters:
.   axis - the axis
.   top - the label at the top of the image
.   xlabel,ylabel - the labes for the x and y axis
@*/
int DrawAxisSetLabels(DrawAxis ad,char* top,char *xlabel,char *ylabel)
{
  if (!ad) return 0;
  ad->xlabel   = xlabel;
  ad->ylabel   = ylabel;
  ad->toplabel = top;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawAxisSetLimits"
/*@
    DrawAxisSetLimits -  Sets the limits (in user coords) of the axis
    
    Input parameters:
.   ad - Axis structure
.   xmin,xmax - limits in x
.   ymin,ymax - limits in y
@*/
int DrawAxisSetLimits(DrawAxis ad,double xmin,double xmax,double ymin,double ymax)
{
  if (!ad) return 0;
  ad->xlow = xmin;
  ad->xhigh= xmax;
  ad->ylow = ymin;
  ad->yhigh= ymax;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawAxisDraw"
/*@
    DrawAxisDraw - Draws an axis.

    Input Parameter:
.   ad - Axis structure

    Note:
    This draws the actual axis.  The limits etc have already been set.
    By picking special routines for the ticks and labels, special
    effects may be generated.  These routines are part of the Axis
    structure (ad).
@*/
int DrawAxisDraw(DrawAxis ad)
{
  int       i,  ntick, numx, numy, ac = ad->ac, tc = ad->tc,cc = ad->cc,rank;
  double    tickloc[MAXSEGS], sep;
  char      *p;
  Draw      awin = ad->win;
  double    h,w,tw,th,xl,xr,yl,yr;
 
  if (!ad) return 0;
   MPI_Comm_rank(ad->comm,&rank); if (rank) return 0;

  if (ad->xlow == ad->xhigh) {ad->xlow -= .5; ad->xhigh += .5;}
  if (ad->ylow == ad->yhigh) {ad->ylow -= .5; ad->yhigh += .5;}
  xl = ad->xlow; xr = ad->xhigh; yl = ad->ylow; yr = ad->yhigh;
  DrawSetCoordinates(awin,xl,yl,xr,yr);
  DrawTextGetSize(awin,&tw,&th);
  numx = (int) (.15*(xr-xl)/tw); if (numx > 6) numx = 6; if (numx< 2) numx = 2;
  numy = (int) (.5*(yr-yl)/th); if (numy > 6) numy = 6; if (numy< 2) numy = 2;
  xl -= 8*tw; xr += 2*tw; yl -= 2.5*th; yr += 2*th;
  if (ad->xlabel) yl -= 2*th;
  if (ad->ylabel) xl -= 2*tw;
  DrawSetCoordinates(awin,xl,yl,xr,yr);
  DrawTextGetSize(awin,&tw,&th);

  DrawLine( awin, ad->xlow,ad->ylow,ad->xhigh,ad->ylow,ac);
  DrawLine( awin, ad->xlow,ad->ylow,ad->xlow,ad->yhigh,ac);

  if (ad->toplabel) {
    w = xl + .5*(xr - xl) - .5*((int)PetscStrlen(ad->toplabel))*tw;
    h = ad->yhigh;
    DrawText(awin,w,h,cc,ad->toplabel); 
  }

  /* Draw the ticks and labels */
  if (ad->xticks) {
    (*ad->xticks)( ad->xlow, ad->xhigh, numx, &ntick, tickloc, MAXSEGS );
    /* Draw in tick marks */
    for (i=0; i<ntick; i++ ) {
      DrawLine(awin,tickloc[i],ad->ylow-.5*th,tickloc[i],ad->ylow+.5*th,
               tc);
    }
    /* label ticks */
    for (i=0; i<ntick; i++) {
	if (ad->xlabelstr) {
	    if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
	    else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
	    else               sep = 0.0;
	    p = (*ad->xlabelstr)( tickloc[i], sep );
	    w = .5*((int)PetscStrlen(p)) * tw;
	    DrawText( awin, tickloc[i]-w,ad->ylow-1.2*th,cc,p); 
        }
    }
  }
  if (ad->xlabel) {
    w = xl + .5*(xr - xl) - .5*((int)PetscStrlen(ad->xlabel))*tw;
    h = ad->ylow - 2.5*th;
    DrawText(awin,w,h,cc,ad->xlabel); 
  }
  if (ad->yticks) {
    (*ad->yticks)( ad->ylow, ad->yhigh, numy, &ntick, tickloc, MAXSEGS );
    /* Draw in tick marks */
    for (i=0; i<ntick; i++ ) {
      DrawLine(awin,ad->xlow -.5*tw,tickloc[i],ad->xlow+.5*tw,tickloc[i],
               tc);
    }
    /* label ticks */
    for (i=0; i<ntick; i++) {
	if (ad->ylabelstr) {
	    if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
	    else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
	    else               sep = 0.0;
	    p = (*ad->xlabelstr)( tickloc[i], sep );
	    w = ad->xlow - ((int)PetscStrlen(p)) * tw - 1.2*tw;
	    DrawText( awin, w,tickloc[i]-.5*th,cc,p); 
        }
    }
  }
  if (ad->ylabel) {
    h = yl + .5*(yr - yl) + .5*((int)PetscStrlen(ad->ylabel))*th;
    w = xl + .5*tw;
    DrawTextVertical(awin,w,h,cc,ad->ylabel); 
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStripAllZeros"
/*
    Removes all zeros but one from .0000 
*/
static int PetscStripAllZeros(char *buf)
{
  int i,n = (int) PetscStrlen(buf);
  if (buf[0] != '.') return 0;
  for ( i=1; i<n; i++ ) {
    if (buf[i] != '0') return 0;
  }
  buf[0] = '0';
  buf[1] = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStripTrailingZeros"
/*
    Removes trailing zeros
*/
static int PetscStripTrailingZeros(char *buf)
{
  int i,n = (int) PetscStrlen(buf),m = -1;

  /* locate decimal point */
  for ( i=0; i<n; i++ ) {
    if (buf[i] == '.') {m = i; break;}
  }
  /* if not decimal point then no zeros to remove */
  if (m == -1) return 0;
  /* start at right end of string removing 0s */
  for ( i=n-1; i>m; i++ ) {
    if (buf[i] != '0') return 0;
    buf[i] = 0;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStripInitialZero"
/*
    Removes leading 0 from 0.22 or -0.22
*/
static int PetscStripInitialZero(char *buf)
{
  int i,n = (int) PetscStrlen(buf); 
  if (buf[0] == '0') {
    for ( i=0; i<n; i++ ) {
      buf[i] = buf[i+1];
    }
  } else if (buf[0] == '-' && buf[1] == '0') {
    for ( i=1; i<n; i++ ) {
      buf[i] = buf[i+1];
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStripZeros"
/*
     Removes the extraneous zeros in numbers like 1.10000e6
*/
static int PetscStripZeros(char *buf)
{
  int i,j,n = (int) PetscStrlen(buf);
  if (n<5) return 0;
  for ( i=1; i<n-1; i++ ) {
    if (buf[i] == 'e' && buf[i-1] == '0') {
      for ( j=i; j<n+1; j++ ) buf[j-1] = buf[j];
      PetscStripZeros(buf);
      return 0;
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStripZerosPlus"
/*
      Removes the plus in something like 1.1e+2
*/
static int PetscStripZerosPlus(char *buf)
{
  int i,j,n = (int) PetscStrlen(buf);
  if (n<5) return 0;
  for ( i=1; i<n-2; i++ ) {
    if (buf[i] == '+') {
      if (buf[i+1] == '0') {
        for ( j=i+1; j<n+1; j++ ) buf[j-1] = buf[j+1];
        return 0;
      }
      else {
        for ( j=i+1; j<n+1; j++ ) buf[j] = buf[j+1];
        return 0;  
      }
    } else if (buf[i] == '-') {
      if (buf[i+1] == '0') {
        for ( j=i+1; j<n+1; j++ ) buf[j] = buf[j+1];
        return 0;
      }
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscADefLabel"
/*
   val is the label value.  sep is the separation to the next (or previous)
   label; this is useful in determining how many significant figures to   
   keep.
 */
static char *PetscADefLabel(double val,double sep )
{
  static char buf[40];
  char   fmat[10];
  int    w, d;

  /* Find the string */
  if (PetscAbsDouble(val)/sep <  1.e-6) {
    buf[0] = '0'; buf[1] = 0;
  } else if (PetscAbsDouble(val) < 1.0e6 && PetscAbsDouble(val) > 1.e-4) {
    /* Compute the number of digits */
    w = 0;
    d = 0;
    if (sep > 0.0) {
	d = (int) ceil( - log10 ( sep ) );
	if (d < 0) d = 0;
	if (PetscAbsDouble(val) < 1.0e-6*sep) {
	    /* This is the case where we are near zero and less than a small
	       fraction of the sep.  In this case, we use 0 as the value */
	    val = 0.0;
	    w   = d;
        }
	else if (val == 0.0) w   = d;
	else w = (int) (ceil( log10( PetscAbsDouble( val ) ) ) + d);
	if (w < 1)   w ++;
	if (val < 0) w ++;
    }

    if (PetscRint(val) == val) {
	if (w > 0) sprintf( fmat, "%%%dd", w );
	else PetscStrcpy( fmat, "%d" );
	sprintf( buf, fmat, (int)val );
        PetscStripInitialZero(buf);
        PetscStripAllZeros(buf);
        PetscStripTrailingZeros(buf);
    } else {
	/* The code used here is inappropriate for a val of 0, which
	   tends to print with an excessive numer of digits.  In this
	   case, we should look at the next/previous values and 
	   use those widths */
	if (w > 0) sprintf( fmat, "%%%d.%dlf", w + 1, d );
	else PetscStrcpy( fmat, "%lf" );
	sprintf( buf, fmat, val );
        PetscStripInitialZero(buf);
        PetscStripAllZeros(buf);
        PetscStripTrailingZeros(buf);
    }
  } else {
    sprintf( buf, "%e", val );
    /* remove the extraneous 0's before the e */
    PetscStripZeros(buf);
    PetscStripZerosPlus(buf);
    PetscStripInitialZero(buf);
    PetscStripAllZeros(buf);
    PetscStripTrailingZeros(buf);
  }
  return buf;
}

#undef __FUNC__  
#define __FUNC__ "PetscADefTicks"
/* Finds "nice" locations for the ticks */
static int PetscADefTicks( double low, double high, int num, int *ntick,
                           double * tickloc,int  maxtick )
{
  int    i;
  double x, base;
  int    power;

  /* patch if low == high */
  if (PetscAbsDouble(low-high) < 1.e-8) {
    low  -= .01;
    high += .01;
  }

  PetscAGetBase( low, high, num, &base, &power );
  x = PetscAGetNice( low, base, -1 );

  /* Values are of the form j * base */
  /* Find the starting value */
  if (x < low) x += base;

  i = 0;
  while (i < maxtick && x <= high) {
    tickloc[i++] = x;
    x += base;
  }
  *ntick = i;

  if (i < 2 && num < 10) {
    PetscADefTicks( low, high, num+1, ntick, tickloc, maxtick );
  }
  return 0;
}

#define EPS 1.e-6

#undef __FUNC__  
#define __FUNC__ "PetscExp10"
static double PetscExp10(double d )
{
  return pow( 10.0, d );
}

#undef __FUNC__  
#define __FUNC__ "PetscMod"
static double PetscMod(double x,double y )
{
  int     i;
  i   = ((int) x ) / ( (int) y );
  x   = x - i * y;
  while (x > y) x -= y;
  return x;
}

#undef __FUNC__  
#define __FUNC__ "PetscCopysign"
static double PetscCopysign(double a,double b )
{
  if (b >= 0) return a;
  return -a;
}

#undef __FUNC__  
#define __FUNC__ "PetscAGetNice"
/*
    Given a value "in" and a "base", return a nice value.
    based on "sgn", extend up (+1) or down (-1)
 */
static double PetscAGetNice(double in,double base,int sgn )
{
  double  etmp;

  etmp    = in / base + 0.5 + PetscCopysign ( 0.5, (double) sgn );
  etmp    = etmp - 0.5 + PetscCopysign( 0.5, etmp ) -
		       PetscCopysign ( EPS * etmp, (double) sgn );
  return base * ( etmp - PetscMod( etmp, 1.0 ) );
}

#undef __FUNC__  
#define __FUNC__ "PetscAGetBase"
static int PetscAGetBase(double vmin,double vmax,int num,double*Base,int*power)
{
  double  base, ftemp;
  static double base_try[5] = {10.0, 5.0, 2.0, 1.0, 0.5};
  int     i;

  /* labels of the form n * BASE */
  /* get an approximate value for BASE */
  base    = ( vmax - vmin ) / (double) (num + 1);

  /* make it of form   m x 10^power,   m in [1.0, 10) */
  if (base <= 0.0) {
    base    = PetscAbsDouble( vmin );
    if (base < 1.0) base = 1.0;
  }
  ftemp   = log10( ( 1.0 + EPS ) * base );
  if (ftemp < 0.0)  ftemp   -= 1.0;
  *power  = (int) ftemp;
  base    = base * PetscExp10( (double) - *power );
  if (base < 1.0) base    = 1.0;
  /* now reduce it to one of 1, 2, or 5 */
  for (i=1; i<5; i++) {
    if (base >= base_try[i]) {
	base            = base_try[i-1] * PetscExp10( (double) *power );
	if (i == 1) *power    = *power + 1;
	break;
    }
  }
  *Base   = base;
  return 0;
}

