
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: reg.c,v 1.28 1999/01/05 18:18:16 bsmith Exp bsmith $";
#endif
/*
    Provides a general mechanism to allow one to register new routines in
    dynamic libraries for many of the PETSc objects (including, e.g., KSP and PC).
*/
#include "petsc.h"
#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "FListGetPathAndFunction"
int FListGetPathAndFunction(const char name[],char *path[],char *function[])
{
  char work[256],*lfunction;

  PetscFunctionBegin;
  PetscStrncpy(work,name,256);
  lfunction = PetscStrrchr(work,':');
  if (lfunction != work) {
    lfunction[-1] = 0;
    *path = (char *) PetscMalloc( (PetscStrlen(work) + 1)*sizeof(char));CHKPTRQ(*path);
    PetscStrcpy(*path,work);
  } else {
    *path = 0;
  }
  *function = (char *) PetscMalloc((PetscStrlen(lfunction)+1)*sizeof(char));CHKPTRQ(*function);
  PetscStrcpy(*function,lfunction);
  PetscFunctionReturn(0);
}

#if defined(USE_DYNAMIC_LIBRARIES)

/*
    This is the list used by the DLRegister routines
*/
DLLibraryList DLLibrariesLoaded = 0;

#undef __FUNC__  
#define __FUNC__ "PetscInitialize_DynamicLibraries"
/*
    PetscInitialize_DynamicLibraries - Adds the default dynamic link libraries to the 
    search path.
*/ 
int PetscInitialize_DynamicLibraries(void)
{
  char *libname[32],libs[256];
  int  nmax,i,ierr,flg;

  PetscFunctionBegin;

  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_prepend",libname,&nmax,&flg);CHKERRQ(ierr);
  for ( i=0; i<nmax; i++ ) {
    ierr = DLLibraryPrepend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    PetscFree(libname[i]);
  }

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetsc"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscvec"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscmat"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscsles"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscsnes"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscts"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_append",libname,&nmax,&flg);CHKERRQ(ierr);
  for ( i=0; i<nmax; i++ ) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    PetscFree(libname[i]);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscFinalize_DynamicLibraries"
/*
     PetscFinalize_DynamicLibraries - Closes the opened dynamic libraries.
*/ 
int PetscFinalize_DynamicLibraries(void)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DLLibraryClose(DLLibrariesLoaded);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ "PetscInitalize_DynamicLibraries"
int PetscInitialize_DynamicLibraries(void)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "PetscFinalize_DynamicLibraries"
int PetscFinalize_DynamicLibraries(void)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
#endif

/* ------------------------------------------------------------------------------*/
struct _FList {
  int    (*routine)(void *); /* the routine */
  char   *path;              /* path of link library containing routine */
  char   *name;              /* string to identify routine */
  char   *rname;             /* routine name in dynamic library */
  FList next;               /* next pointer */
  FList next_list;          /* used to maintain list of all lists for freeing */
};

/*
     Keep a linked list of FLists so that we can destroy all the left-over ones.
*/
static FList   dlallhead = 0;


/*
   FListAdd - Given a routine and a string id, saves that routine in the
   specified registry.

   Synopsis:
   int FListAdd(FList *fl, char *name, char *rname,int (*fnc)(void *))

   Input Parameters:
+  fl    - pointer registry
.  name  - string to identify routine
.  rname - routine name in dynamic library
-  fnc   - function pointer (optional if using dynamic libraries)

   Notes:
   Users who wish to register new methods for use by a particular PETSc
   component (e.g., SNES) should generally call the registration routine
   for that particular component (e.g., SNESRegister()) instead of
   calling FListAdd() directly.

.seealso: FListDestroy(), SNESRegister(), KSPRegister(),
          PCRegister(), TSRegister()
*/

#undef __FUNC__  
#define __FUNC__ "FListAdd_Private"
int FListAdd_Private( FList *fl,const char name[],const char rname[],int (*fnc)(void *))
{
  FList   entry,ne;
  int      ierr;
  char     *fpath,*fname;

  PetscFunctionBegin;

  if (!*fl) {
    entry          = (FList) PetscMalloc(sizeof(struct _FList));CHKPTRQ(entry);
    entry->name    = (char *)PetscMalloc( PetscStrlen(name) + 1 ); CHKPTRQ(entry->name);
    PetscStrcpy( entry->name, name );
    ierr = FListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
    entry->path    = fpath;
    entry->rname   = fname;
    entry->routine = fnc;
    entry->next    = 0;
    *fl = entry;

    /* add this new list to list of all lists */
    if (!dlallhead) {
      dlallhead        = *fl;
      (*fl)->next_list = 0;
    } else {
      ne               = dlallhead;
      dlallhead        = *fl;
      (*fl)->next_list = ne;
    }
  } else {
    /* search list to see if it is already there */
    ne = *fl;
    while (ne) {
      if (!PetscStrcmp(ne->name,name)) { /* found duplicate */
        ierr = FListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
        if (ne->path) PetscFree(ne->path);
        if (ne->rname) PetscFree(ne->rname);
        ne->path    = fpath;
        ne->rname   = fname;
        ne->routine = fnc;
        PetscFunctionReturn(0);
      }
      if (ne->next) ne = ne->next; else break;
    }
    /* create new entry and add to end of list */
    entry          = (FList) PetscMalloc(sizeof(struct _FList));CHKPTRQ(entry);
    entry->name    = (char *)PetscMalloc( PetscStrlen(name) + 1 ); CHKPTRQ(entry->name);
    PetscStrcpy( entry->name, name );
    ierr = FListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
    entry->path    = fpath;
    entry->rname   = fname;
    entry->routine = fnc;
    entry->next    = 0;
    ne->next = entry;
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListDestroy"
/*
    FListDestroy - Destroys a list of registered routines.

    Input Parameter:
.   fl  - pointer to list

.seealso: FListAdd()
*/
int FListDestroy(FList fl)
{
  FList   next,entry,tmp = dlallhead;

  PetscFunctionBegin;
  if (!fl) PetscFunctionReturn(0);

  if (!dlallhead) {
    SETERRQ(1,1,"Internal PETSc error, function registration corrupted");
  }

  /*
       Remove this entry from the master DL list 
  */
  if (dlallhead == fl) {
    if (dlallhead->next_list) {
      dlallhead = dlallhead->next_list;
    } else {
      dlallhead = 0;
    }
  } else {
    while (tmp->next_list != fl) {
      tmp = tmp->next_list;
      if (!tmp->next_list) SETERRQ(1,1,"Internal PETSc error, function registration corrupted");
    }
    tmp->next_list = tmp->next_list->next_list;
  }

  /* free this list */
  entry = fl;
  while (entry) {
    next = entry->next;
    if (entry->path) PetscFree(entry->path);
    PetscFree( entry->name );
    PetscFree( entry->rname );
    PetscFree( entry );
    entry = next;
  }

 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListDestroyAll"
int FListDestroyAll(void)
{
  FList tmp2,tmp1 = dlallhead;
  int    ierr;

  PetscFunctionBegin;
  while (tmp1) {
    tmp2 = tmp1->next_list;
    ierr = FListDestroy(tmp1);CHKERRQ(ierr);
    tmp1 = tmp2;
  }
  dlallhead = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListFind"
/*
    FListFind - Given a name, finds the matching routine.

    Input Parameters:
+   comm - processors looking for routine
.   fl   - pointer to list
-   name - name string

    Output Parameters:
.   r - the routine

    Notes:
    The routine's id or name MUST have been registered with the FList via
    FListAdd() before FListFind() can be called.

.seealso: FListAdd()
*/
int FListFind(MPI_Comm comm,FList fl,const char name[], int (**r)(void *))
{
  FList        entry = fl;
  char          *function, *path;
  int           ierr;
 
  PetscFunctionBegin;
  *r = 0;
  ierr = FListGetPathAndFunction(name,&path,&function);CHKERRQ(ierr);

  /*
        If path then append it to search libraries
  */
#if defined(USE_DYNAMIC_LIBRARIES)
  if (path) {
    ierr = DLLibraryAppend(comm,&DLLibrariesLoaded,path); CHKERRQ(ierr);
  }
#endif

  while (entry) {
    if ((path && entry->path && !PetscStrcmp(path,entry->path) && !PetscStrcmp(function,entry->rname)) ||
        (path && entry->path && !PetscStrcmp(path,entry->path) && !PetscStrcmp(function,entry->name)) ||
        (!path &&  !PetscStrcmp(function,entry->name)) || 
        (!path &&  !PetscStrcmp(function,entry->rname))) {

      if (entry->routine) {
        *r = entry->routine; 
         if (path) PetscFree(path);
         PetscFree(function);
         PetscFunctionReturn(0);
      }

      /* it is not yet in memory so load from dynamic library */
#if defined(USE_DYNAMIC_LIBRARIES)
      ierr = DLLibrarySym(comm,&DLLibrariesLoaded,path,entry->rname,(void **)r);CHKERRQ(ierr);
      if (*r) {
        entry->routine = *r;
        if (path) PetscFree(path);
        PetscFree(function);
        PetscFunctionReturn(0);
      } else {
        PetscErrorPrintf("Registered function name: %s\n",entry->rname);
        ierr = DLLibraryPrintPath();CHKERRQ(ierr);
        SETERRQ(1,1,"Unable to find function: either it is mis-spelled or dynamic library is not in path");
      }
#endif
    }
    entry = entry->next;
  }

#if defined(USE_DYNAMIC_LIBRARIES)
  /* Function never registered; try for it anyway */
  ierr = DLLibrarySym(comm,&DLLibrariesLoaded,path,function,(void **)r);CHKERRQ(ierr);
  if (path) PetscFree(path);
  if (*r) {
    ierr = FListAdd(&fl,name,name,r); CHKERRQ(ierr);
    PetscFree(function);
    PetscFunctionReturn(0);
  }
#endif

  /*
       Don't generate error, just end
  PetscErrorPrintf("Function name: %s\n",function);
  ierr = DLLibraryPrintPath();CHKERRQ(ierr);
  SETERRQ(1,1,"Unable to find function: either it is mis-spelled or dynamic library is not in path");
  */

  PetscFree(function);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListView"
/*
   FListView - prints out contents of an FList

   Collective over MPI_Comm

   Input Parameters:
+  flist - the list of functions
-  viewer - currently ignored

.seealso: FListAdd(), FListPrintTypes()
*/
int FListView(FList list,Viewer viewer)
{
  int        ierr;
  ViewerType vtype;

  PetscFunctionBegin;
  if (!viewer) viewer = VIEWER_STDOUT_SELF;

  ierr = ViewerGetType(viewer,&vtype);CHKERRQ(ierr);
  if (!PetscTypeCompare(vtype,ASCII_VIEWER)) SETERRQ(1,1,"Only ASCII viewer supported");

  while (list) {
    if (list->path) {
      ViewerASCIIPrintf(viewer," %s %s %s\n",list->path,list->name,list->rname);
    } else {
      ViewerASCIIPrintf(viewer," %s %s\n",list->name,list->rname);
    }
    list = list->next;
  }
  ViewerASCIIPrintf(viewer,"\n");
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "FListPrintTypes"
/*
   FListPrintTypes - Prints the methods available.

   Collective over MPI_Comm

   Input Parameters:
+  comm   - the communicator (usually MPI_COMM_WORLD)
.  fd     - file to print to, usually stdout
.  prefix - prefix to prepend to name (optional)
.  name   - option string
-  list   - list of types

.seealso: FListAdd()
*/
int FListPrintTypes(MPI_Comm comm,FILE *fd,const char prefix[],const char name[],FList list)
{
  int      count = 0;
  char     p[64];

  PetscFunctionBegin;
  if (!fd) fd = stdout;

  PetscStrcpy(p,"-");
  if (prefix) PetscStrcat(p,prefix);
  PetscFPrintf(comm,fd,"  %s%s (one of)",p,name);

  while (list) {
    PetscFPrintf(comm,fd," %s",list->name);
    list = list->next;
    count++;
    if (count == 8) PetscFPrintf(comm,fd,"\n     ");
  }
  PetscFPrintf(comm,fd,"\n");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListDuplicate"
/*
    FListDuplicate - Creates a new list from a give object list.

    Input Parameters:
.   fl   - pointer to list

    Output Parameters:
.   nl - the new list (should point to 0 to start, otherwise appends)


*/
int FListDuplicate(FList fl, FList *nl)
{
  int  ierr;
  char path[1024];

  PetscFunctionBegin;
  while (fl) {
    /* this is silly, rebuild the complete pathname */
    if (fl->path) {
      PetscStrcpy(path,fl->path);
      PetscStrcat(path,":");
      PetscStrcat(path,fl->name);
    } else {
      PetscStrcpy(path,fl->name);
    }       
    ierr = FListAdd(nl,path,fl->rname,fl->routine); CHKERRQ(ierr);
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}





