#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: plogmpe.c,v 1.26 1998/03/16 18:24:44 balay Exp bsmith $";
#endif
/*
      PETSc code to log PETSc events using MPE
*/
#if defined(USE_PETSC_LOG) && defined (HAVE_MPE)
#include "petsc.h"        /*I    "petsc.h"   I*/
#include "sys.h"
#include "mpe.h"

/* 
   Make sure that all events used by PETSc have the
   corresponding flags set here: 
     1 - activated for MPE logging
     0 - not activated for MPE logging
 */
int PLogEventMPEFlags[] = {  1,1,1,1,1,  /* 0 - 24*/
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        0,1,1,1,1,  /* 25 -49 */
                        1,1,1,1,1,
                        1,1,0,0,0,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 50 - 74 */
                        1,1,1,1,1,
                        1,1,1,1,0,
                        0,0,0,0,0,
                        1,1,1,0,1,
                        1,1,1,1,1, /* 75 - 99 */
                        1,1,1,1,1,
                        1,1,0,0,0,
                        1,1,0,0,0,
                        0,0,0,0,0,
                        1,0,0,0,0, /* 100 - 124 */ 
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0, /* 125 - 149 */
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0, /* 150 - 174 */
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0, /* 175 - 199 */
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0};

/* For Colors, check out the file  /usr/local/X11/lib/rgb.txt */

char *(PLogEventColor[]) = {"AliceBlue:      ",
                            "BlueViolet:     ",
                            "CadetBlue:      ",
                            "CornflowerBlue: ",
                            "DarkGoldenrod:  ",
                            "DarkGreen:      ",
                            "DarkKhaki:      ",
                            "DarkOliveGreen: ",
                            "DarkOrange:     ",
                            "DarkOrchid:     ",
                            "DarkSeaGreen:   ",
                            "DarkSlateGray:  ",
                            "DarkTurquoise:  ",
                            "DeepPink:       ",
                            "DeepSkyBlue:    ",
                            "DimGray:        ", 
                            "DodgerBlue:     ",
                            "GreenYellow:    ",
                            "HotPink:        ",
                            "IndianRed:      ",
                            "LavenderBlush:  ",
                            "LawnGreen:      ",
                            "LemonChiffon:   ", 
                            "LightCoral:     ",
                            "LightCyan:      ",
                            "LightPink:      ",
                            "LightSalmon:    ",
                            "LightSlateGray: ",
                            "LightYellow:    ",
                            "LimeGreen:      ",
                            "MediumPurple:   ",
                            "MediumSeaGreen: ",
                            "MediumSlateBlue:",
                            "MidnightBlue:   ",
                            "MintCream:      ",
                            "MistyRose:      ",
                            "NavajoWhite:    ",
                            "NavyBlue:       ",
                            "OliveDrab:      ",
                            "OrangeRed:      ",
                            "PaleGoldenrod:  ",
                            "PaleVioletRed:  ",
                            "PapayaWhip:     ",
                            "PeachPuff:      ",
                            "RosyBrown:      ",
                            "SaddleBrown:    ",
                            "SpringGreen:    ",
                            "SteelBlue:      ",
                            "VioletRed:      ",
                            "beige:          ",
                            "chocolate:      ",
                            "coral:          ",
                            "gold:           ",
                            "magenta:        ",
                            "maroon:         ",
                            "orchid:         ",
                            "pink:           ",
                            "plum:           ",
                            "red:            ",
                            "tan:            ",
                            "tomato:         ",
                            "violet:         ",
                            "wheat:          ",
                            "yellow:         ",
                            "AliceBlue:      ",
                            "BlueViolet:     ",
                            "CadetBlue:      ",
                            "CornflowerBlue: ",
                            "DarkGoldenrod:  ",
                            "DarkGreen:      ",
                            "DarkKhaki:      ",
                            "DarkOliveGreen: ",
                            "DarkOrange:     ",
                            "DarkOrchid:     ",
                            "DarkSeaGreen:   ",
                            "DarkSlateGray:  ",
                            "DarkTurquoise:  ",
                            "DeepPink:       ",
                            "DeepSkyBlue:    ",
                            "DimGray:        ", 
                            "DodgerBlue:     ",
                            "GreenYellow:    ",
                            "HotPink:        ",
                            "IndianRed:      ",
                            "LavenderBlush:  ",
                            "LawnGreen:      ",
                            "LemonChiffon:   ", 
                            "LightCoral:     ",
                            "LightCyan:      ",
                            "LightPink:      ",
                            "LightSalmon:    ",
                            "LightSlateGray: ",
                            "LightYellow:    ",
                            "LimeGreen:      ",
                            "MediumPurple:   ",
                            "MediumSeaGreen: ",
                            "MediumSlateBlue:",
                            "MidnightBlue:   ",
                            "MintCream:      ",
                            "MistyRose:      ",
                            "NavajoWhite:    ",
                            "NavyBlue:       ",
                            "OliveDrab:      ",
                            "OrangeRed:      ",
                            "PaleGoldenrod:  ",
                            "PaleVioletRed:  ",
                            "PapayaWhip:     ",
                            "PeachPuff:      ",
                            "RosyBrown:      ",
                            "SaddleBrown:    ",
                            "SpringGreen:    ",
                            "SteelBlue:      ",
                            "VioletRed:      ",
                            "beige:          ",
                            "chocolate:      ",
                            "coral:          ",
                            "gold:           ",
                            "magenta:        ",
                            "maroon:         ",
                            "orchid:         ",
                            "pink:           ",
                            "AliceBlue:      ",
                            "BlueViolet:     ",
                            "CadetBlue:      ",
                            "CornflowerBlue: ",
                            "DarkGoldenrod:  ",
                            "DarkGreen:      ",
                            "DarkKhaki:      ",
                            "DarkOliveGreen: ",
                            "DarkOrange:     ",
                            "DarkOrchid:     ",
                            "DarkSeaGreen:   ",
                            "DarkSlateGray:  ",
                            "DarkTurquoise:  ",
                            "DeepPink:       ",
                            "DeepSkyBlue:    ",
                            "DimGray:        ", 
                            "DodgerBlue:     ",
                            "GreenYellow:    ",
                            "HotPink:        ",
                            "IndianRed:      ",
                            "LavenderBlush:  ",
                            "LawnGreen:      ",
                            "LemonChiffon:   ", 
                            "LightCoral:     ",
                            "LightCyan:      ",
                            "LightPink:      ",
                            "LightSalmon:    ",
                            "LightSlateGray: ",
                            "LightYellow:    ",
                            "LimeGreen:      ",
                            "MediumPurple:   ",
                            "MediumSeaGreen: ",
                            "MediumSlateBlue:",
                            "MidnightBlue:   ",
                            "MintCream:      ",
                            "MistyRose:      ",
                            "NavajoWhite:    ",
                            "NavyBlue:       ",
                            "OliveDrab:      ",
                            "OrangeRed:      ",
                            "PaleGoldenrod:  ",
                            "PaleVioletRed:  ",
                            "PapayaWhip:     ",
                            "PeachPuff:      ",
                            "RosyBrown:      ",
                            "SaddleBrown:    ",
                            "SpringGreen:    ",
                            "SteelBlue:      ",
                            "VioletRed:      ",
                            "beige:          ",
                            "chocolate:      ",
                            "coral:          ",
                            "gold:           ",
                            "magenta:        ",
                            "maroon:         ",
                            "orchid:         ",
                            "pink:           ",
                            "plum:           ",
                            "red:            ",
                            "tan:            ",
                            "tomato:         ",
                            "violet:         ",
                            "wheat:          ",
                            "yellow:         ",
                            "AliceBlue:      ",
                            "BlueViolet:     ",
                            "CadetBlue:      ",
                            "CornflowerBlue: ",
                            "DarkGoldenrod:  ",
                            "DarkGreen:      ",
                            "DarkKhaki:      ",
                            "DarkOliveGreen: ",
                            "DarkOrange:     ",
                            "DarkOrchid:     ",
                            "DarkSeaGreen:   ",
                            "DarkSlateGray:  ",
                            "DarkTurquoise:  ",
                            "DeepPink:       ",
                            "DeepSkyBlue:    "};

/*
    Indicates if a color was malloced for each event, or if it is
  the default color. Used to ensure malloced space is properly freed.
*/
int PLogEventColorMalloced[] = {0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0};

int UseMPE = 0;
extern char *PLogEventName[];

#undef __FUNC__  
#define __FUNC__ "PLogMPEBegin"
/*@C
   PLogMPEBegin - Turns on MPE logging of events. This creates large log files 
     and slows the program down.

   Options Database Keys:
$  -log_mpe : Prints extensive log information (for code compiled
$      with USE_PETSC_LOG)

   Notes:
   A related routine is PLogBegin (with the options key -log), which is 
   intended for production runs since it logs only flop rates and object
   creation (and shouldn't significantly slow the programs).

.keywords: log, all, begin

.seealso: PLogDump(), PLogBegin(), PLogAllBegin(), PLogEventActivate(),
          PLogEventDeactivate()
@*/
int PLogMPEBegin()
{
  int i, rank;
    
  PetscFunctionBegin;
  /* Do MPE initialization */
  MPE_Init_log();
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (!rank) {
    for ( i=0; i < PLOG_USER_EVENT_HIGH; i++) {
      if (PLogEventMPEFlags[i]) {
        MPE_Describe_state(MPEBEGIN+2*i,MPEBEGIN+2*i+1,PLogEventName[i],PLogEventColor[i]);
      }
    }
  }
  UseMPE = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PLogEventMPEDeactivate"
/*@
    PLogEventMPEDeactivate - Indicates that a particular event should not be
       logged using MPE. Note: the event may be either a pre-defined
       PETSc event (found in include/petsclog.h) or an event number obtained
       with PLogEventRegister().

  Input Parameter:
.   event - integer indicating event

   Example of Usage:
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     PLogEventMPEDeactivate(VEC_SetValues);
$      code where you do not want to log VecSetValues() 
$     PLogEventMPEActivate(VEC_SetValues);
$      code where you do want to log VecSetValues() 
$     .......
$     PetscFinalize();
$

.seealso: PLogEventMPEActivate(),PlogEventActivate(),PlogEventDeactivate()
@*/
int PLogEventMPEDeactivate(int event)
{
  PetscFunctionBegin;
  PLogEventMPEFlags[event] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PLogEventMPEActivate"
/*@
    PLogEventMPEActivate - Indicates that a particular event should be
       logged using MPE. Note: the event may be either a pre-defined
       PETSc event (found in include/petsclog.h) or an event number obtained
       with PLogEventRegister().

  Input Parameter:
.   event - integer indicating event

   Example of Usage:
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     PLogEventMPEDeactivate(VEC_SetValues);
$      code where you do not want to log VecSetValues() 
$     PLogEventMPEActivate(VEC_SetValues);
$      code where you do want to log VecSetValues() 
$     .......
$     PetscFinalize();
$

.seealso: PLogEventMPEDeactivate(),PLogEventActivate(),PLogEventDeactivate()
@*/
int PLogEventMPEActivate(int event)
{
  PetscFunctionBegin;
  PLogEventMPEFlags[event] = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PLogMPEDump"
/*@C
   PLogMPEDump - Dumps the MPE logging info to file for later use with Upshot.


.keywords: log, destroy

.seealso: PLogDump(), PLogAllBegin(), PLogMPEBegin()
@*/
int PLogMPEDump(char* sname)
{
  PetscFunctionBegin;
  if (!sname) sname = "mpe.log";
  MPE_Finish_log(sname); 
  PetscFunctionReturn(0);
}

#else

/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int PETScMPEDummy(int dummy)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif

