#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: text.c,v 1.27 1998/03/24 21:00:11 balay Exp bsmith $";
#endif

/* Include petsc in case it is including petscconf.h */
#include "petsc.h"

#if defined(HAVE_X11)
/*
   This file contains simple code to manage access to fonts, insuring that
   library routines access/load fonts only once
 */

#include "src/sys/src/draw/impls/x/ximpl.h"


int XiInitFonts(Draw_X *);
int XiMatchFontSize(XiFont*,int,int);
int XiLoadFont(Draw_X*,XiFont*);
/*
    XiFontFixed - Return a pointer to the selected font.

    Warning: Loads a new font for each window. This should be 
   ok because there will never be many windows and the graphics
   are not intended to be high performance.
*/
#undef __FUNC__  
#define __FUNC__ "XiFontFixed"
int XiFontFixed( Draw_X *XBWin,int w, int h,XiFont **outfont )
{
  static XiFont *curfont = 0,*font;

  PetscFunctionBegin;
  if (!curfont) { XiInitFonts( XBWin );}
  font = (XiFont*) PetscMalloc(sizeof(XiFont)); CHKPTRQ(font);
  XiMatchFontSize( font, w, h );
  XiLoadFont( XBWin, font );
  curfont = font;
  *outfont = curfont;
  PetscFunctionReturn(0);
}

/* this is set by XListFonts at startup */
#define NFONTS 20
static struct {
    int w, h, descent;
} nfonts[NFONTS];
static int act_nfonts = 0;

/*
  These routines determine the font to be used based on the requested size,
  and load it if necessary
*/

#undef __FUNC__  
#define __FUNC__ "XiLoadFont"
int XiLoadFont( Draw_X *XBWin, XiFont *font )
{
  char        font_name[100];
  XFontStruct *FontInfo;
  XGCValues   values ;

  PetscFunctionBegin;
  (void) sprintf(font_name, "%dx%d", font->font_w, font->font_h );
  font->fnt  = XLoadFont( XBWin->disp, font_name );

  /* The font->descent may not have been set correctly; get it now that
      the font has been loaded */
  FontInfo   = XQueryFont( XBWin->disp, font->fnt );
  font->font_descent   = FontInfo->descent;

  /* Storage leak; should probably just free FontInfo? */
  /* XFreeFontInfo( FontInfo ); */

  /* Set the current font in the CG */
  values.font = font->fnt ;
  XChangeGC( XBWin->disp, XBWin->gc.set, GCFont, &values ) ; 
  PetscFunctionReturn(0);
}

/* Code to find fonts and their characteristics */
#undef __FUNC__  
#define __FUNC__ "XiInitFonts" 
int XiInitFonts( Draw_X *XBWin )
{
  char         **names;
  int          cnt, i, j;
  XFontStruct  *info;

  PetscFunctionBegin;
  /* This just gets the most basic fixed-width fonts */
  names   = XListFontsWithInfo( XBWin->disp, "?x??", NFONTS, &cnt, &info );
  j       = 0;
  for (i=0; i < cnt; i++) {
    names[i][1]         = '\0';
    nfonts[j].w         = info[i].max_bounds.width ;
    nfonts[j].h         = info[i].ascent + info[i].descent;
    nfonts[j].descent   = info[i].descent;
    if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
    j++;
    if (j >= NFONTS) break;
  }
  act_nfonts    = j;
  if (cnt > 0)  {
    XFreeFontInfo( names, info, cnt );
  }
  /* If the above fails, try this: */
  if (act_nfonts == 0) {
    /* This just gets the most basic fixed-width fonts */
    names   = XListFontsWithInfo( XBWin->disp, "?x", NFONTS, &cnt, &info );
    j       = 0;
    for (i=0; i < cnt; i++) {
        if (PetscStrlen(names[i]) != 2) continue;
        names[i][1]         = '\0';
	nfonts[j].w         = info[i].max_bounds.width ;
        /* nfonts[j].w         = info[i].max_bounds.lbearing +
                                    info[i].max_bounds.rbearing; */
        nfonts[j].h         = info[i].ascent + info[i].descent;
        nfonts[j].descent   = info[i].descent;
        if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
        j++;
	if (j >= NFONTS) break;
    }
    act_nfonts    = j;
    XFreeFontInfo( names, info, cnt );
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "XiMatchFontSize" 
int XiMatchFontSize( XiFont *font, int w, int h )
{
  int i,max,imax,tmp;

  PetscFunctionBegin;
  for (i=0; i<act_nfonts; i++) {
    if (nfonts[i].w == w && nfonts[i].h == h) {
        font->font_w        = w;
        font->font_h        = h;
        font->font_descent  = nfonts[i].descent;
        PetscFunctionReturn(0);
    }
  }

  /* determine closest fit, per max. norm */
  imax = 0;
  max  = PetscMax(PetscAbsInt(nfonts[0].w - w),PetscAbsInt(nfonts[0].h - h));
  for (i=1; i<act_nfonts; i++) {
    tmp = PetscMax(PetscAbsInt(nfonts[i].w - w),PetscAbsInt(nfonts[i].h - h));
    if (tmp < max) {max = tmp; imax = i;}
  }

  /* should use font with closest match */
  font->font_w        = nfonts[imax].w;
  font->font_h        = nfonts[imax].h;
  font->font_descent  = nfonts[imax].descent;
  PetscFunctionReturn(0);
}
#else
int dummy_text(void)
{
  PetscFunctionReturn(0);
}
#endif
