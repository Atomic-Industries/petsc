#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fpath.c,v 1.13 1997/08/22 15:11:48 bsmith Exp bsmith $";
#endif
/*
      Code for opening and closing files.
*/
#include "src/sys/src/files.h"

#if defined(HAVE_PWD_H)

#undef __FUNC__  
#define __FUNC__ "PetscGetFullPath"
/*@C
   PetscGetFullPath - Given a filename, returns the fully qualified file name.

   Input Parameters:
.  path     - pathname to qualify
.  fullpath - pointer to buffer to hold full pathname
.  flen     - size of fullpath

.keywords: system, get, full, path

.seealso: PetscGetRelativePath()
@*/
int PetscGetFullPath( char *path, char *fullpath, int flen )
{
  struct passwd *pwde;
  int           ln;

  PetscFunctionBegin;
  if (path[0] == '/') {
    if (PetscStrncmp("/tmp_mnt/",path,9) == 0) PetscStrncpy(fullpath, path + 8, flen);
    else PetscStrncpy( fullpath, path, flen); 
    PetscFunctionReturn(0);
  }
  PetscGetWorkingDirectory( fullpath, flen );
  PetscStrncat( fullpath,"/",flen - PetscStrlen(fullpath) );
  if ( path[0] == '.' && path[1] == '/' ) {
    PetscStrncat( fullpath, path+2, flen - PetscStrlen(fullpath) - 1 );
  } else {
    PetscStrncat( fullpath, path, flen - PetscStrlen(fullpath) - 1 );
  }

  /* Remove the various "special" forms (~username/ and ~/) */
  if (fullpath[0] == '~') {
    char tmppath[MAXPATHLEN];
    if (fullpath[1] == '/') {
	pwde = getpwuid( geteuid() );
	if (!pwde) PetscFunctionReturn(0);
	PetscStrcpy( tmppath, pwde->pw_dir );
	ln = PetscStrlen( tmppath );
	if (tmppath[ln-1] != '/') PetscStrcat( tmppath+ln-1, "/" );
	PetscStrcat( tmppath, fullpath + 2 );
	PetscStrncpy( fullpath, tmppath, flen );
    } else {
	char *p, *name;

	/* Find username */
	name = fullpath + 1;
	p    = name;
#if defined( PARCH_nt_gnu)
	while (*p && isalnum((int)(*p))) p++;
#else
	while (*p && isalnum(*p)) p++;
#endif
	*p = 0; p++;
	pwde = getpwnam( name );
	if (!pwde) PetscFunctionReturn(0);
	
	PetscStrcpy( tmppath, pwde->pw_dir );
	ln = PetscStrlen( tmppath );
	if (tmppath[ln-1] != '/') PetscStrcat( tmppath+ln-1, "/" );
	PetscStrcat( tmppath, p );
	PetscStrncpy( fullpath, tmppath, flen );
    }
  }
  /* Remove the automounter part of the path */
  if (PetscStrncmp( fullpath, "/tmp_mnt/", 9 ) == 0) {
    char tmppath[MAXPATHLEN];
    PetscStrcpy( tmppath, fullpath + 8 );
    PetscStrcpy( fullpath, tmppath );
  }
  /* We could try to handle things like the removal of .. etc */
  PetscFunctionReturn(0);
}
#elif defined (PARCH_nt)
#undef __FUNC__  
#define __FUNC__ "PetscGetFullPath"
int PetscGetFullPath( char *path, char *fullpath, int flen )
{
  PetscFunctionBegin;
  _fullpath(fullpath,path,flen);
  PetscFunctionReturn(0);
}
#else
#undef __FUNC__  
#define __FUNC__ "PetscGetFullPath"
int PetscGetFullPath( char *path, char *fullpath, int flen )
{
  PetscFunctionBegin;
  PetscStrcpy( fullpath, path );
  PetscFunctionReturn(0);
}	
#endif
