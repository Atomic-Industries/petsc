/*
      Routines for opening dynamic link libraries (DLLs), keeping a searchable
   path of DLLs, obtaining remote DLLs via a URL and opening them locally.
*/

#include <petsc/private/petscimpl.h>

/* ------------------------------------------------------------------------------*/
/*
      Code to maintain a list of opened dynamic libraries and load symbols
*/
struct _n_PetscDLLibrary {
  PetscDLLibrary next;
  PetscDLHandle  handle;
  char           libname[PETSC_MAX_PATH_LEN];
};

PetscErrorCode  PetscDLLibraryPrintPath(PetscDLLibrary libs)
{
  PetscFunctionBegin;
  while (libs) {
    PetscErrorPrintf("  %s\n",libs->libname);
    libs = libs->next;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscDLLibraryRetrieve - Copies a PETSc dynamic library from a remote location
     (if it is remote), indicates if it exits and its local name.

     Collective

   Input Parameters:
+   comm - processors that are opening the library
-   libname - name of the library, can be relative or absolute

   Output Parameters:
+   name - actual name of file on local filesystem if found
.   llen - length of the name buffer
-   found - true if the file exists

   Level: developer

   Notes:
   [[<http,ftp>://hostname]/directoryname/]filename[.so.1.0]

   ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, or ${any environmental variable}
   occurring in directoryname and filename will be replaced with appropriate values.
@*/
PetscErrorCode  PetscDLLibraryRetrieve(MPI_Comm comm,const char libname[],char *lname,size_t llen,PetscBool  *found)
{
  char           *buf,*par2,suffix[16],*gz,*so;
  size_t         len;

  PetscFunctionBegin;
  /*
     make copy of library name and replace $PETSC_ARCH etc
     so we can add to the end of it to look for something like .so.1.0 etc.
  */
  CHKERRQ(PetscStrlen(libname,&len));
  len  = PetscMax(4*len,PETSC_MAX_PATH_LEN);
  CHKERRQ(PetscMalloc1(len,&buf));
  par2 = buf;
  CHKERRQ(PetscStrreplace(comm,libname,par2,len));

  /* temporarily remove .gz if it ends library name */
  CHKERRQ(PetscStrrstr(par2,".gz",&gz));
  if (gz) {
    CHKERRQ(PetscStrlen(gz,&len));
    if (len != 3) gz  = NULL; /* do not end (exactly) with .gz */
    else          *gz = 0;    /* ends with .gz, so remove it   */
  }
  /* strip out .a from it if user put it in by mistake */
  CHKERRQ(PetscStrlen(par2,&len));
  if (par2[len-1] == 'a' && par2[len-2] == '.') par2[len-2] = 0;

  CHKERRQ(PetscFileRetrieve(comm,par2,lname,llen,found));
  if (!(*found)) {
    /* see if library name does already not have suffix attached */
    CHKERRQ(PetscStrncpy(suffix,".",sizeof(suffix)));
    CHKERRQ(PetscStrlcat(suffix,PETSC_SLSUFFIX,sizeof(suffix)));
    CHKERRQ(PetscStrrstr(par2,suffix,&so));
    /* and attach the suffix if it is not there */
    if (!so) CHKERRQ(PetscStrcat(par2,suffix));

    /* restore the .gz suffix if it was there */
    if (gz) CHKERRQ(PetscStrcat(par2,".gz"));

    /* and finally retrieve the file */
    CHKERRQ(PetscFileRetrieve(comm,par2,lname,llen,found));
  }

  CHKERRQ(PetscFree(buf));
  PetscFunctionReturn(0);
}

/*@C
   PetscDLLibraryOpen - Opens a PETSc dynamic link library

     Collective

   Input Parameters:
+   comm - processors that are opening the library
-   path - name of the library, can be relative or absolute

   Output Parameter:
.   entry - a PETSc dynamic link library entry

   Level: developer

   Notes:
   [[<http,ftp>://hostname]/directoryname/]libbasename[.so.1.0]

   If the library has the symbol PetscDLLibraryRegister_basename() in it then that function is automatically run
   when the library is opened.

   ${PETSC_ARCH} occurring in directoryname and filename
   will be replaced with the appropriate value.

.seealso: PetscLoadDynamicLibrary(), PetscDLLibraryAppend()
@*/
PetscErrorCode  PetscDLLibraryOpen(MPI_Comm comm,const char path[],PetscDLLibrary *entry)
{
  PetscBool      foundlibrary,match;
  char           libname[PETSC_MAX_PATH_LEN],par2[PETSC_MAX_PATH_LEN],suffix[16],*s;
  char           *basename,registername[128];
  PetscDLHandle  handle;
  PetscErrorCode (*func)(void) = NULL;

  PetscFunctionBegin;
  PetscValidCharPointer(path,2);
  PetscValidPointer(entry,3);

  *entry = NULL;

  /* retrieve the library */
  CHKERRQ(PetscInfo(NULL,"Retrieving %s\n",path));
  CHKERRQ(PetscDLLibraryRetrieve(comm,path,par2,PETSC_MAX_PATH_LEN,&foundlibrary));
  PetscCheck(foundlibrary,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate dynamic library:\n  %s",path);
  /* Eventually ./configure should determine if the system needs an executable dynamic library */
#define PETSC_USE_NONEXECUTABLE_SO
#if !defined(PETSC_USE_NONEXECUTABLE_SO)
  CHKERRQ(PetscTestFile(par2,'x',&foundlibrary));
  PetscCheck(foundlibrary,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Dynamic library is not executable:\n  %s\n  %s",path,par2);
#endif

  /* copy path and setup shared library suffix  */
  CHKERRQ(PetscStrncpy(libname,path,PETSC_MAX_PATH_LEN));
  CHKERRQ(PetscStrncpy(suffix,".",sizeof(suffix)));
  CHKERRQ(PetscStrlcat(suffix,PETSC_SLSUFFIX,sizeof(suffix)));
  /* remove wrong suffixes from libname */
  CHKERRQ(PetscStrrstr(libname,".gz",&s));
  if (s && s[3] == 0) s[0] = 0;
  CHKERRQ(PetscStrrstr(libname,".a",&s));
  if (s && s[2] == 0) s[0] = 0;
  /* remove shared suffix from libname */
  CHKERRQ(PetscStrrstr(libname,suffix,&s));
  if (s) s[0] = 0;

  /* open the dynamic library */
  CHKERRQ(PetscInfo(NULL,"Opening dynamic library %s\n",libname));
  CHKERRQ(PetscDLOpen(par2,PETSC_DL_DECIDE,&handle));

  /* look for [path/]libXXXXX.YYY and extract out the XXXXXX */
  CHKERRQ(PetscStrrchr(libname,'/',&basename)); /* XXX Windows ??? */
  if (!basename) basename = libname;
  CHKERRQ(PetscStrncmp(basename,"lib",3,&match));
  if (match) basename = basename + 3;
  else {
    CHKERRQ(PetscInfo(NULL,"Dynamic library %s does not have lib prefix\n",libname));
  }
  for (s=basename; *s; s++) if (*s == '-') *s = '_';
  CHKERRQ(PetscStrncpy(registername,"PetscDLLibraryRegister_",sizeof(registername)));
  CHKERRQ(PetscStrlcat(registername,basename,sizeof(registername)));
  CHKERRQ(PetscDLSym(handle,registername,(void**)&func));
  if (func) {
    CHKERRQ(PetscInfo(NULL,"Loading registered routines from %s\n",libname));
    CHKERRQ((*func)());
  } else {
    CHKERRQ(PetscInfo(NULL,"Dynamic library %s does not have symbol %s\n",libname,registername));
  }

  CHKERRQ(PetscNew(entry));
  (*entry)->next   = NULL;
  (*entry)->handle = handle;
  CHKERRQ(PetscStrcpy((*entry)->libname,libname));
  PetscFunctionReturn(0);
}

/*@C
   PetscDLLibrarySym - Load a symbol from the dynamic link libraries.

   Collective

   Input Parameters:
+  comm - communicator that will open the library
.  outlist - list of already open libraries that may contain symbol (can be NULL and only the executable is searched for the function)
.  path     - optional complete library name (if provided checks here before checking outlist)
-  insymbol - name of symbol

   Output Parameter:
.  value - if symbol not found then this value is set to NULL

   Level: developer

   Notes:
    Symbol can be of the form
        [/path/libname[.so.1.0]:]functionname[()] where items in [] denote optional

        Will attempt to (retrieve and) open the library if it is not yet been opened.

@*/
PetscErrorCode  PetscDLLibrarySym(MPI_Comm comm,PetscDLLibrary *outlist,const char path[],const char insymbol[],void **value)
{
  char           libname[PETSC_MAX_PATH_LEN],suffix[16],*symbol,*s;
  PetscDLLibrary nlist,prev,list = NULL;

  PetscFunctionBegin;
  if (outlist) PetscValidPointer(outlist,2);
  if (path) PetscValidCharPointer(path,3);
  PetscValidCharPointer(insymbol,4);
  PetscValidPointer(value,5);

  if (outlist) list = *outlist;
  *value = NULL;

  CHKERRQ(PetscStrchr(insymbol,'(',&s));
  if (s) {
    /* make copy of symbol so we can edit it in place */
    CHKERRQ(PetscStrallocpy(insymbol,&symbol));
    /* If symbol contains () then replace with a NULL, to support functionname() */
    CHKERRQ(PetscStrchr(symbol,'(',&s));
    s[0] = 0;
  } else symbol = (char*)insymbol;

  /*
       Function name does include library
       -------------------------------------
  */
  if (path && path[0] != '\0') {
    /* copy path and remove suffix from libname */
    CHKERRQ(PetscStrncpy(libname,path,PETSC_MAX_PATH_LEN));
    CHKERRQ(PetscStrncpy(suffix,".",sizeof(suffix)));
    CHKERRQ(PetscStrlcat(suffix,PETSC_SLSUFFIX,sizeof(suffix)));
    CHKERRQ(PetscStrrstr(libname,suffix,&s));
    if (s) s[0] = 0;
    /* Look if library is already opened and in path */
    prev  = NULL;
    nlist = list;
    while (nlist) {
      PetscBool match;
      CHKERRQ(PetscStrcmp(nlist->libname,libname,&match));
      if (match) goto done;
      prev  = nlist;
      nlist = nlist->next;
    }
    /* open the library and append it to path */
    CHKERRQ(PetscDLLibraryOpen(comm,path,&nlist));
    CHKERRQ(PetscInfo(NULL,"Appending %s to dynamic library search path\n",path));
    if (prev) prev->next = nlist;
    else {if (outlist) *outlist   = nlist;}

done:;
    CHKERRQ(PetscDLSym(nlist->handle,symbol,value));
    if (*value) {
      CHKERRQ(PetscInfo(NULL,"Loading function %s from dynamic library %s\n",insymbol,path));
    }

    /*
         Function name does not include library so search path
         -----------------------------------------------------
    */
  } else {
    while (list) {
      CHKERRQ(PetscDLSym(list->handle,symbol,value));
      if (*value) {
        CHKERRQ(PetscInfo(NULL,"Loading symbol %s from dynamic library %s\n",symbol,list->libname));
        break;
      }
      list = list->next;
    }
    if (!*value) {
      CHKERRQ(PetscDLSym(NULL,symbol,value));
      if (*value) {
        CHKERRQ(PetscInfo(NULL,"Loading symbol %s from object code\n",symbol));
      }
    }
  }

  if (symbol != insymbol) {
    CHKERRQ(PetscFree(symbol));
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscDLLibraryAppend - Appends another dynamic link library to the seach list, to the end
                of the search path.

     Collective

     Input Parameters:
+     comm - MPI communicator
-     path - name of the library

     Output Parameter:
.     outlist - list of libraries

     Level: developer

     Notes:
    if library is already in path will not add it.

  If the library has the symbol PetscDLLibraryRegister_basename() in it then that function is automatically run
      when the library is opened.

.seealso: PetscDLLibraryOpen()
@*/
PetscErrorCode  PetscDLLibraryAppend(MPI_Comm comm,PetscDLLibrary *outlist,const char path[])
{
  PetscDLLibrary list,prev;
  size_t         len;
  PetscBool      match,dir;
  char           program[PETSC_MAX_PATH_LEN],found[8*PETSC_MAX_PATH_LEN];
  char           *libname,suffix[16],*s;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidPointer(outlist,2);

  /* is path a directory? */
  CHKERRQ(PetscTestDirectory(path,'r',&dir));
  if (dir) {
    CHKERRQ(PetscInfo(NULL,"Checking directory %s for dynamic libraries\n",path));
    CHKERRQ(PetscStrncpy(program,path,sizeof(program)));
    CHKERRQ(PetscStrlen(program,&len));
    if (program[len-1] == '/') {
      CHKERRQ(PetscStrlcat(program,"*.",sizeof(program)));
    } else {
      CHKERRQ(PetscStrlcat(program,"/*.",sizeof(program)));
    }
    CHKERRQ(PetscStrlcat(program,PETSC_SLSUFFIX,sizeof(program)));

    CHKERRQ(PetscLs(comm,program,found,8*PETSC_MAX_PATH_LEN,&dir));
    if (!dir) PetscFunctionReturn(0);
  } else {
    CHKERRQ(PetscStrncpy(found,path,PETSC_MAX_PATH_LEN));
  }
  CHKERRQ(PetscStrncpy(suffix,".",sizeof(suffix)));
  CHKERRQ(PetscStrlcat(suffix,PETSC_SLSUFFIX,sizeof(suffix)));

  CHKERRQ(PetscTokenCreate(found,'\n',&token));
  CHKERRQ(PetscTokenFind(token,&libname));
  while (libname) {
    /* remove suffix from libname */
    CHKERRQ(PetscStrrstr(libname,suffix,&s));
    if (s) s[0] = 0;
    /* see if library was already open then we are done */
    list  = prev = *outlist;
    match = PETSC_FALSE;
    while (list) {
      CHKERRQ(PetscStrcmp(list->libname,libname,&match));
      if (match) break;
      prev = list;
      list = list->next;
    }
    /* restore suffix from libname */
    if (s) s[0] = '.';
    if (!match) {
      /* open the library and add to end of list */
      CHKERRQ(PetscDLLibraryOpen(comm,libname,&list));
      CHKERRQ(PetscInfo(NULL,"Appending %s to dynamic library search path\n",libname));
      if (!*outlist) *outlist   = list;
      else           prev->next = list;
    }
    CHKERRQ(PetscTokenFind(token,&libname));
  }
  CHKERRQ(PetscTokenDestroy(&token));
  PetscFunctionReturn(0);
}

/*@C
     PetscDLLibraryPrepend - Add another dynamic library to search for symbols to the beginning of
                 the search path.

     Collective

     Input Parameters:
+     comm - MPI communicator
-     path - name of the library

     Output Parameter:
.     outlist - list of libraries

     Level: developer

     Notes:
    If library is already in path will remove old reference.

@*/
PetscErrorCode  PetscDLLibraryPrepend(MPI_Comm comm,PetscDLLibrary *outlist,const char path[])
{
  PetscDLLibrary list,prev;
  size_t         len;
  PetscBool      match,dir;
  char           program[PETSC_MAX_PATH_LEN],found[8*PETSC_MAX_PATH_LEN];
  char           *libname,suffix[16],*s;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidPointer(outlist,2);

  /* is path a directory? */
  CHKERRQ(PetscTestDirectory(path,'r',&dir));
  if (dir) {
    CHKERRQ(PetscInfo(NULL,"Checking directory %s for dynamic libraries\n",path));
    CHKERRQ(PetscStrncpy(program,path,sizeof(program)));
    CHKERRQ(PetscStrlen(program,&len));
    if (program[len-1] == '/') {
      CHKERRQ(PetscStrlcat(program,"*.",sizeof(program)));
    } else {
      CHKERRQ(PetscStrlcat(program,"/*.",sizeof(program)));
    }
    CHKERRQ(PetscStrlcat(program,PETSC_SLSUFFIX,sizeof(program)));

    CHKERRQ(PetscLs(comm,program,found,8*PETSC_MAX_PATH_LEN,&dir));
    if (!dir) PetscFunctionReturn(0);
  } else {
    CHKERRQ(PetscStrncpy(found,path,PETSC_MAX_PATH_LEN));
  }

  CHKERRQ(PetscStrncpy(suffix,".",sizeof(suffix)));
  CHKERRQ(PetscStrlcat(suffix,PETSC_SLSUFFIX,sizeof(suffix)));

  CHKERRQ(PetscTokenCreate(found,'\n',&token));
  CHKERRQ(PetscTokenFind(token,&libname));
  while (libname) {
    /* remove suffix from libname */
    CHKERRQ(PetscStrstr(libname,suffix,&s));
    if (s) s[0] = 0;
    /* see if library was already open and move it to the front */
    prev  = NULL;
    list  = *outlist;
    match = PETSC_FALSE;
    while (list) {
      CHKERRQ(PetscStrcmp(list->libname,libname,&match));
      if (match) {
        CHKERRQ(PetscInfo(NULL,"Moving %s to begin of dynamic library search path\n",libname));
        if (prev) prev->next = list->next;
        if (prev) list->next = *outlist;
        *outlist = list;
        break;
      }
      prev = list;
      list = list->next;
    }
    /* restore suffix from libname */
    if (s) s[0] = '.';
    if (!match) {
      /* open the library and add to front of list */
      CHKERRQ(PetscDLLibraryOpen(comm,libname,&list));
      CHKERRQ(PetscInfo(NULL,"Prepending %s to dynamic library search path\n",libname));
      list->next = *outlist;
      *outlist   = list;
    }
    CHKERRQ(PetscTokenFind(token,&libname));
  }
  CHKERRQ(PetscTokenDestroy(&token));
  PetscFunctionReturn(0);
}

/*@C
     PetscDLLibraryClose - Destroys the search path of dynamic libraries and closes the libraries.

    Collective on PetscDLLibrary

    Input Parameter:
.     head - library list

     Level: developer

@*/
PetscErrorCode  PetscDLLibraryClose(PetscDLLibrary list)
{
  PetscBool      done = PETSC_FALSE;
  PetscDLLibrary prev,tail;

  PetscFunctionBegin;
  if (!list) PetscFunctionReturn(0);
  /* traverse the list in reverse order */
  while (!done) {
    if (!list->next) done = PETSC_TRUE;
    prev = tail = list;
    while (tail->next) {
      prev = tail;
      tail = tail->next;
    }
    prev->next = NULL;
    /* close the dynamic library and free the space in entry data-structure*/
    CHKERRQ(PetscInfo(NULL,"Closing dynamic library %s\n",tail->libname));
    CHKERRQ(PetscDLClose(&tail->handle));
    CHKERRQ(PetscFree(tail));
  }
  PetscFunctionReturn(0);
}
