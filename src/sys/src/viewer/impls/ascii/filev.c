#ifndef lint
static char vcid[] = "$Id: filev.c,v 1.49 1996/12/17 18:04:20 balay Exp balay $";
#endif

#include "petsc.h"
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
#include <stdarg.h>

struct _Viewer {
  VIEWERHEADER
  FILE          *fd;
  char          *outputname,*outputnames[10];
};

Viewer VIEWER_STDOUT_SELF, VIEWER_STDERR_SELF, VIEWER_STDOUT_WORLD;

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerInitialize_Private"
int ViewerInitialize_Private()
{
  ViewerFileOpenASCII(MPI_COMM_SELF,"stderr",&VIEWER_STDERR_SELF);
  ViewerFileOpenASCII(MPI_COMM_SELF,"stdout",&VIEWER_STDOUT_SELF);
  ViewerFileOpenASCII(PETSC_COMM_WORLD,"stdout",&VIEWER_STDOUT_WORLD);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerDestroy_File"
static int ViewerDestroy_File(PetscObject obj)
{
  Viewer v = (Viewer) obj;
  int    rank = 0;
  if (v->type == ASCII_FILES_VIEWER) {MPI_Comm_rank(v->comm,&rank);} 
  if (!rank && v->fd != stderr && v->fd != stdout) fclose(v->fd);
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerDestroy_Private"
int ViewerDestroy_Private()
{
  ViewerDestroy_File((PetscObject)VIEWER_STDERR_SELF);
  ViewerDestroy_File((PetscObject)VIEWER_STDOUT_SELF);
  ViewerDestroy_File((PetscObject)VIEWER_STDOUT_WORLD);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerFlush_File"
int ViewerFlush_File(Viewer v)
{
  int rank;
  MPI_Comm_rank(v->comm,&rank);
  if (rank) return 0;
  fflush(v->fd);
  return 0;  
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerASCIIGetPointer"
/*@C
    ViewerASCIIGetPointer - Extracts the file pointer from an ASCII viewer.

.   viewer - viewer context, obtained from ViewerFileOpenASCII()
.   fd - file pointer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, pointer

.seealso: ViewerFileOpenASCII()
@*/
int ViewerASCIIGetPointer(Viewer viewer, FILE **fd)
{
  *fd = viewer->fd;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerFileGetOutputname_Private"
int ViewerFileGetOutputname_Private(Viewer viewer, char **name)
{
  *name = viewer->outputname;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerGetFormat"
int ViewerGetFormat(Viewer viewer,int *format)
{
  *format =  viewer->format;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerFileOpenASCII"
/*@C
   ViewerFileOpenASCII - Opens an ASCII file as a viewer.

   Input Parameters:
.  comm - the communicator
.  name - the file name

   Output Parameter:
.  lab - the viewer to use with the specified file

   Notes:
   If a multiprocessor communicator is used (such as MPI_COMM_WORLD), 
   then only the first processor in the group opens the file.  All other 
   processors send their data to the first processor to print. 

   Each processor can instead write its own independent output by
   specifying the communicator MPI_COMM_SELF.

   As shown below, ViewerFileOpenASCII() is useful in conjunction with 
   MatView() and VecView()
$
$    ViewerFileOpenASCII(MPI_COMM_WORLD,"mat.output",&viewer);
$    MatView(matrix,viewer);

   This viewer can be destroyed with ViewerDestroy().

.keywords: Viewer, file, open

.seealso: MatView(), VecView(), ViewerDestroy(), ViewerFileOpenBinary(),
          ViewerASCIIGetPointer()
@*/
int ViewerFileOpenASCII(MPI_Comm comm,char *name,Viewer *lab)
{
  Viewer v;
  if (comm == MPI_COMM_SELF) {
    PetscHeaderCreate(v,_Viewer,VIEWER_COOKIE,ASCII_FILE_VIEWER,comm);
  } else {
    PetscHeaderCreate(v,_Viewer,VIEWER_COOKIE,ASCII_FILES_VIEWER,comm);
  }
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_File;
  v->flush       = ViewerFlush_File;

  if (!PetscStrcmp(name,"stderr")) v->fd = stderr;
  else if (!PetscStrcmp(name,"stdout")) v->fd = stdout;
  else {
    v->fd        = fopen(name,"w"); 
    if (!v->fd) SETERRQ(1,"Cannot open file");
  }
  v->format        = VIEWER_FORMAT_ASCII_DEFAULT;
  v->iformat       = 0;
  v->outputname    = 0;
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  *lab           = v;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerSetFormat"
/*@C
   ViewerSetFormat - Sets the format for file viewers.

   Input Parameters:
.  v - the viewer
.  format - the format
.  char - optional object name

   Notes:
   Available formats include
$    VIEWER_FORMAT_ASCII_DEFAULT - default
$    VIEWER_FORMAT_ASCII_MATLAB - Matlab format
$    VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
$      (which is in many cases the same as the default)
$    VIEWER_FORMAT_ASCII_INFO - basic information about object
$    VIEWER_FORMAT_ASCII_INFO_LONG - more detailed info
$       about object
$    VIEWER_FORMAT_ASCII_COMMON - identical output format for
$       all objects of a particular type
$    VIEWER_FORMAT_BINARY_NATIVE - store the object to the binary
$      file in its native format (for example, dense
$       matrices are stored as dense)

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

.keywords: Viewer, file, set, format

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerPushFormat(), ViewerPopFormat()
@*/
int ViewerSetFormat(Viewer v,int format,char *name)
{
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->format     = format;
    v->outputname = name;
  } else {
    v->format     = format;
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerPushFormat"
/*@C
   ViewerPushFormat - Sets the format for file viewers.

   Input Parameters:
.  v - the viewer
.  format - the format
.  char - optional object name

   Notes:
   Available formats include
$    VIEWER_FORMAT_ASCII_DEFAULT - default
$    VIEWER_FORMAT_ASCII_MATLAB - Matlab format
$    VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
$      (which is in many cases the same as the default)
$    VIEWER_FORMAT_ASCII_INFO - basic information about object
$    VIEWER_FORMAT_ASCII_INFO_LONG - more detailed info
$       about object
$    VIEWER_FORMAT_ASCII_COMMON - identical output format for
$       all objects of a particular type
$    VIEWER_FORMAT_BINARY_NATIVE - store the object to the binary
$      file in its native format (for example, dense
$       matrices are stored as dense)

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

.keywords: Viewer, file, set, format

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPopFormat()
@*/
int ViewerPushFormat(Viewer v,int format,char *name)
{
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->iformat > 9) SETERRQ(1,"Too many pushes");

  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->formats[v->iformat]       = v->format;
    v->outputnames[v->iformat++] = v->outputname;
    v->format                    = format;
    v->outputname                = name;
  } else if (v->type == BINARY_FILE_VIEWER) {
    v->formats[v->iformat++]     = v->format;
    v->format                    = format;
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerPopFormat"
/*@C
   ViewerPopFormat - Resets the format for file viewers.

   Input Parameters:
.  v - the viewer

.keywords: Viewer, file, set, format, push, pop

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPushFormat()
@*/
int ViewerPopFormat(Viewer v)
{
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->iformat <= 0) return 0;

  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->format     = v->formats[--v->iformat];
    v->outputname = v->outputnames[v->iformat];
  } else if (v->type == BINARY_FILE_VIEWER) {
    v->format     = v->formats[--v->iformat];
  }
  return 0;
}




