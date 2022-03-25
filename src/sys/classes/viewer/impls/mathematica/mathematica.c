
#include <petsc/private/viewerimpl.h>   /* "petscsys.h" */
#include <petsc/private/pcimpl.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <mathematica.h>

#if defined(PETSC_HAVE__SNPRINTF) && !defined(PETSC_HAVE_SNPRINTF)
#define snprintf _snprintf
#endif

PetscViewer PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE = NULL;
static void *mathematicaEnv                        = NULL;

static PetscBool PetscViewerMathematicaPackageInitialized = PETSC_FALSE;
/*@C
  PetscViewerMathematicaFinalizePackage - This function destroys everything in the Petsc interface to Mathematica. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscViewerMathematicaFinalizePackage(void)
{
  PetscFunctionBegin;
  if (mathematicaEnv) MLDeinitialize((MLEnvironment) mathematicaEnv);
  PetscViewerMathematicaPackageInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerMathematicaInitializePackage - This function initializes everything in the Petsc interface to Mathematica. It is
  called from PetscViewerInitializePackage().

  Level: developer

.seealso: PetscSysInitializePackage(), PetscInitialize()
@*/
PetscErrorCode  PetscViewerMathematicaInitializePackage(void)
{
  PetscError ierr;

  PetscFunctionBegin;
  if (PetscViewerMathematicaPackageInitialized) PetscFunctionReturn(0);
  PetscViewerMathematicaPackageInitialized = PETSC_TRUE;

  mathematicaEnv = (void*) MLInitialize(0);

  CHKERRQ(PetscRegisterFinalize(PetscViewerMathematicaFinalizePackage));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerInitializeMathematicaWorld_Private()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE) PetscFunctionReturn(0);
  CHKERRQ(PetscViewerMathematicaOpen(PETSC_COMM_WORLD, PETSC_DECIDE, NULL, NULL, &PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerDestroy_Mathematica(PetscViewer viewer)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  MLClose(vmath->link);
  CHKERRQ(PetscFree(vmath->linkname));
  CHKERRQ(PetscFree(vmath->linkhost));
  CHKERRQ(PetscFree(vmath));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerDestroyMathematica_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE) {
    CHKERRQ(PetscViewerDestroy(PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerMathematicaSetupConnection_Private(PetscViewer v)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) v->data;
#if defined(MATHEMATICA_3_0)
  int                     argc = 6;
  char                    *argv[6];
#else
  int                     argc = 5;
  char                    *argv[5];
#endif
  char                    hostname[256];
  long                    lerr;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  /* Link name */
  argv[0] = "-linkname";
  if (!vmath->linkname) argv[1] = "math -mathlink";
  else                  argv[1] = vmath->linkname;

  /* Link host */
  argv[2] = "-linkhost";
  if (!vmath->linkhost) {
    CHKERRQ(PetscGetHostName(hostname, sizeof(hostname)));
    argv[3] = hostname;
  } else argv[3] = vmath->linkhost;

  /* Link mode */
#if defined(MATHEMATICA_3_0)
  argv[4] = "-linkmode";
  switch (vmath->linkmode) {
  case MATHEMATICA_LINK_CREATE:
    argv[5] = "Create";
    break;
  case MATHEMATICA_LINK_CONNECT:
    argv[5] = "Connect";
    break;
  case MATHEMATICA_LINK_LAUNCH:
    argv[5] = "Launch";
    break;
  }
#else
  switch (vmath->linkmode) {
  case MATHEMATICA_LINK_CREATE:
    argv[4] = "-linkcreate";
    break;
  case MATHEMATICA_LINK_CONNECT:
    argv[4] = "-linkconnect";
    break;
  case MATHEMATICA_LINK_LAUNCH:
    argv[4] = "-linklaunch";
    break;
  }
#endif
  vmath->link = MLOpenInEnv(mathematicaEnv, argc, argv, &lerr);
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscViewerCreate_Mathematica(PetscViewer v)
{
  PetscViewer_Mathematica *vmath;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerMathematicaInitializePackage());

  CHKERRQ(PetscNewLog(v,&vmath));
  v->data         = (void*) vmath;
  v->ops->destroy = PetscViewerDestroy_Mathematica;
  v->ops->flush   = 0;
  CHKERRQ(PetscStrallocpy(PETSC_VIEWER_MATHEMATICA, &((PetscObject)v)->type_name));

  vmath->linkname     = NULL;
  vmath->linkhost     = NULL;
  vmath->linkmode     = MATHEMATICA_LINK_CONNECT;
  vmath->graphicsType = GRAPHICS_MOTIF;
  vmath->plotType     = MATHEMATICA_TRIANGULATION_PLOT;
  vmath->objName      = NULL;

  CHKERRQ(PetscViewerMathematicaSetFromOptions(v));
  CHKERRQ(PetscViewerMathematicaSetupConnection_Private(v));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerMathematicaParseLinkMode(char *modename, LinkMode *mode)
{
  PetscBool      isCreate, isConnect, isLaunch;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscStrcasecmp(modename, "Create",  &isCreate));
  CHKERRQ(PetscStrcasecmp(modename, "Connect", &isConnect));
  CHKERRQ(PetscStrcasecmp(modename, "Launch",  &isLaunch));
  if (isCreate)       *mode = MATHEMATICA_LINK_CREATE;
  else if (isConnect) *mode = MATHEMATICA_LINK_CONNECT;
  else if (isLaunch)  *mode = MATHEMATICA_LINK_LAUNCH;
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid Mathematica link mode: %s", modename);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerMathematicaSetFromOptions(PetscViewer v)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) v->data;
  char                    linkname[256];
  char                    modename[256];
  char                    hostname[256];
  char                    type[256];
  PetscInt                numPorts;
  PetscInt                *ports;
  PetscInt                numHosts;
  int                     h;
  char                    **hosts;
  PetscMPIInt             size, rank;
  PetscBool               opt;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v), &size));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)v), &rank));

  /* Get link name */
  CHKERRQ(PetscOptionsGetString("viewer_", "-math_linkname", linkname, sizeof(linkname), &opt));
  if (opt) {
    CHKERRQ(PetscViewerMathematicaSetLinkName(v, linkname));
  }
  /* Get link port */
  numPorts = size;
  CHKERRQ(PetscMalloc1(size, &ports));
  CHKERRQ(PetscOptionsGetIntArray("viewer_", "-math_linkport", ports, &numPorts, &opt));
  if (opt) {
    if (numPorts > rank) snprintf(linkname, sizeof(linkname), "%6d", ports[rank]);
    else                 snprintf(linkname, sizeof(linkname), "%6d", ports[0]);
    CHKERRQ(PetscViewerMathematicaSetLinkName(v, linkname));
  }
  CHKERRQ(PetscFree(ports));
  /* Get link host */
  numHosts = size;
  CHKERRQ(PetscMalloc1(size, &hosts));
  CHKERRQ(PetscOptionsGetStringArray("viewer_", "-math_linkhost", hosts, &numHosts, &opt));
  if (opt) {
    if (numHosts > rank) {
      CHKERRQ(PetscStrncpy(hostname, hosts[rank], sizeof(hostname)));
    } else {
      CHKERRQ(PetscStrncpy(hostname, hosts[0], sizeof(hostname)));
    }
    CHKERRQ(PetscViewerMathematicaSetLinkHost(v, hostname));
  }
  for (h = 0; h < numHosts; h++) {
    CHKERRQ(PetscFree(hosts[h]));
  }
  CHKERRQ(PetscFree(hosts));
  /* Get link mode */
  CHKERRQ(PetscOptionsGetString("viewer_", "-math_linkmode", modename, sizeof(modename), &opt));
  if (opt) {
    LinkMode mode;

    CHKERRQ(PetscViewerMathematicaParseLinkMode(modename, &mode));
    CHKERRQ(PetscViewerMathematicaSetLinkMode(v, mode));
  }
  /* Get graphics type */
  CHKERRQ(PetscOptionsGetString("viewer_", "-math_graphics", type, sizeof(type), &opt));
  if (opt) {
    PetscBool isMotif, isPS, isPSFile;

    CHKERRQ(PetscStrcasecmp(type, "Motif",  &isMotif));
    CHKERRQ(PetscStrcasecmp(type, "PS",     &isPS));
    CHKERRQ(PetscStrcasecmp(type, "PSFile", &isPSFile));
    if (isMotif)       vmath->graphicsType = GRAPHICS_MOTIF;
    else if (isPS)     vmath->graphicsType = GRAPHICS_PS_STDOUT;
    else if (isPSFile) vmath->graphicsType = GRAPHICS_PS_FILE;
  }
  /* Get plot type */
  CHKERRQ(PetscOptionsGetString("viewer_", "-math_type", type, sizeof(type), &opt));
  if (opt) {
    PetscBool isTri, isVecTri, isVec, isSurface;

    CHKERRQ(PetscStrcasecmp(type, "Triangulation",       &isTri));
    CHKERRQ(PetscStrcasecmp(type, "VectorTriangulation", &isVecTri));
    CHKERRQ(PetscStrcasecmp(type, "Vector",              &isVec));
    CHKERRQ(PetscStrcasecmp(type, "Surface",             &isSurface));
    if (isTri)          vmath->plotType = MATHEMATICA_TRIANGULATION_PLOT;
    else if (isVecTri)  vmath->plotType = MATHEMATICA_VECTOR_TRIANGULATION_PLOT;
    else if (isVec)     vmath->plotType = MATHEMATICA_VECTOR_PLOT;
    else if (isSurface) vmath->plotType = MATHEMATICA_SURFACE_PLOT;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerMathematicaSetLinkName(PetscViewer v, const char *name)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) v->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(name,2);
  CHKERRQ(PetscStrallocpy(name, &vmath->linkname));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerMathematicaSetLinkPort(PetscViewer v, int port)
{
  char           name[16];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snprintf(name, 16, "%6d", port);
  CHKERRQ(PetscViewerMathematicaSetLinkName(v, name));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerMathematicaSetLinkHost(PetscViewer v, const char *host)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) v->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(host,2);
  CHKERRQ(PetscStrallocpy(host, &vmath->linkhost));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerMathematicaSetLinkMode(PetscViewer v, LinkMode mode)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) v->data;

  PetscFunctionBegin;
  vmath->linkmode = mode;
  PetscFunctionReturn(0);
}

/*----------------------------------------- Public Functions --------------------------------------------------------*/
/*@C
  PetscViewerMathematicaOpen - Communicates with Mathemtica using MathLink.

  Collective

  Input Parameters:
+ comm    - The MPI communicator
. port    - [optional] The port to connect on, or PETSC_DECIDE
. machine - [optional] The machine to run Mathematica on, or NULL
- mode    - [optional] The connection mode, or NULL

  Output Parameter:
. viewer  - The Mathematica viewer

  Level: intermediate

  Notes:
  Most users should employ the following commands to access the
  Mathematica viewers
$
$    PetscViewerMathematicaOpen(MPI_Comm comm, int port, char *machine, char *mode, PetscViewer &viewer)
$    MatView(Mat matrix, PetscViewer viewer)
$
$                or
$
$    PetscViewerMathematicaOpen(MPI_Comm comm, int port, char *machine, char *mode, PetscViewer &viewer)
$    VecView(Vec vector, PetscViewer viewer)

   Options Database Keys:
+    -viewer_math_linkhost <machine> - The host machine for the kernel
.    -viewer_math_linkname <name>    - The full link name for the connection
.    -viewer_math_linkport <port>    - The port for the connection
.    -viewer_math_mode <mode>        - The mode, e.g. Launch, Connect
.    -viewer_math_type <type>        - The plot type, e.g. Triangulation, Vector
-    -viewer_math_graphics <output>  - The output type, e.g. Motif, PS, PSFile

.seealso: MatView(), VecView()
@*/
PetscErrorCode  PetscViewerMathematicaOpen(MPI_Comm comm, int port, const char machine[], const char mode[], PetscViewer *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerCreate(comm, v));
#if 0
  LinkMode linkmode;
  CHKERRQ(PetscViewerMathematicaSetLinkPort(*v, port));
  CHKERRQ(PetscViewerMathematicaSetLinkHost(*v, machine));
  CHKERRQ(PetscViewerMathematicaParseLinkMode(mode, &linkmode));
  CHKERRQ(PetscViewerMathematicaSetLinkMode(*v, linkmode));
#endif
  CHKERRQ(PetscViewerSetType(*v, PETSC_VIEWER_MATHEMATICA));
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerMathematicaGetLink - Returns the link to Mathematica

  Input Parameters:
+ viewer - The Mathematica viewer
- link   - The link to Mathematica

  Level: intermediate

.keywords PetscViewer, Mathematica, link
.seealso PetscViewerMathematicaOpen()
@*/
PetscErrorCode  PetscViewerMathematicaGetLink(PetscViewer viewer, MLINK *link)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID,1);
  *link = vmath->link;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerMathematicaSkipPackets - Discard packets sent by Mathematica until a certain packet type is received

  Input Parameters:
+ viewer - The Mathematica viewer
- type   - The packet type to search for, e.g RETURNPKT

  Level: advanced

.keywords PetscViewer, Mathematica, packets
.seealso PetscViewerMathematicaSetName(), PetscViewerMathematicaGetVector()
@*/
PetscErrorCode  PetscViewerMathematicaSkipPackets(PetscViewer viewer, int type)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;
  MLINK                   link   = vmath->link; /* The link to Mathematica */
  int                     pkt;                 /* The packet type */

  PetscFunctionBegin;
  while ((pkt = MLNextPacket(link)) && (pkt != type)) MLNewPacket(link);
  if (!pkt) {
    MLClearError(link);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, (char*) MLErrorMessage(link));
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerMathematicaGetName - Retrieve the default name for objects communicated to Mathematica

  Input Parameter:
. viewer - The Mathematica viewer

  Output Parameter:
. name   - The name for new objects created in Mathematica

  Level: intermediate

.keywords PetscViewer, Mathematica, name
.seealso PetscViewerMathematicaSetName(), PetscViewerMathematicaClearName()
@*/
PetscErrorCode  PetscViewerMathematicaGetName(PetscViewer viewer, const char **name)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(name,2);
  *name = vmath->objName;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerMathematicaSetName - Override the default name for objects communicated to Mathematica

  Input Parameters:
+ viewer - The Mathematica viewer
- name   - The name for new objects created in Mathematica

  Level: intermediate

.keywords PetscViewer, Mathematica, name
.seealso PetscViewerMathematicaSetName(), PetscViewerMathematicaClearName()
@*/
PetscErrorCode  PetscViewerMathematicaSetName(PetscViewer viewer, const char name[])
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(name,2);
  vmath->objName = name;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerMathematicaClearName - Use the default name for objects communicated to Mathematica

  Input Parameter:
. viewer - The Mathematica viewer

  Level: intermediate

.keywords PetscViewer, Mathematica, name
.seealso PetscViewerMathematicaGetName(), PetscViewerMathematicaSetName()
@*/
PetscErrorCode  PetscViewerMathematicaClearName(PetscViewer viewer)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID,1);
  vmath->objName = NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerMathematicaGetVector - Retrieve a vector from Mathematica

  Input Parameter:
. viewer - The Mathematica viewer

  Output Parameter:
. v      - The vector

  Level: intermediate

.keywords PetscViewer, Mathematica, vector
.seealso VecView(), PetscViewerMathematicaPutVector()
@*/
PetscErrorCode  PetscViewerMathematicaGetVector(PetscViewer viewer, Vec v)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;
  MLINK                   link;   /* The link to Mathematica */
  char                    *name;
  PetscScalar             *mArray,*array;
  long                    mSize;
  int                     n;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID,1);
  PetscValidHeaderSpecific(v,      VEC_CLASSID,2);

  /* Determine the object name */
  if (!vmath->objName) name = "vec";
  else                 name = (char*) vmath->objName;

  link = vmath->link;
  CHKERRQ(VecGetLocalSize(v, &n));
  CHKERRQ(VecGetArray(v, &array));
  MLPutFunction(link, "EvaluatePacket", 1);
  MLPutSymbol(link, name);
  MLEndPacket(link);
  CHKERRQ(PetscViewerMathematicaSkipPackets(viewer, RETURNPKT));
  MLGetRealList(link, &mArray, &mSize);
  PetscCheckFalse(n != mSize,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Incompatible vector sizes %d %d",n,mSize);
  CHKERRQ(PetscArraycpy(array, mArray, mSize));
  MLDisownRealList(link, mArray, mSize);
  CHKERRQ(VecRestoreArray(v, &array));
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerMathematicaPutVector - Send a vector to Mathematica

  Input Parameters:
+ viewer - The Mathematica viewer
- v      - The vector

  Level: intermediate

.keywords PetscViewer, Mathematica, vector
.seealso VecView(), PetscViewerMathematicaGetVector()
@*/
PetscErrorCode  PetscViewerMathematicaPutVector(PetscViewer viewer, Vec v)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;
  MLINK                   link   = vmath->link; /* The link to Mathematica */
  char                    *name;
  PetscScalar             *array;
  int                     n;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (!vmath->objName) name = "vec";
  else                 name = (char*) vmath->objName;

  CHKERRQ(VecGetLocalSize(v, &n));
  CHKERRQ(VecGetArray(v, &array));

  /* Send the Vector object */
  MLPutFunction(link, "EvaluatePacket", 1);
  MLPutFunction(link, "Set", 2);
  MLPutSymbol(link, name);
  MLPutRealList(link, array, n);
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  CHKERRQ(PetscViewerMathematicaSkipPackets(viewer, RETURNPKT));
  /* Skip ReturnPacket */
  MLNewPacket(link);

  CHKERRQ(VecRestoreArray(v, &array));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerMathematicaPutMatrix(PetscViewer viewer, int m, int n, PetscReal *a)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;
  MLINK                   link   = vmath->link; /* The link to Mathematica */
  char                    *name;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (!vmath->objName) name = "mat";
  else                 name = (char*) vmath->objName;

  /* Send the dense matrix object */
  MLPutFunction(link, "EvaluatePacket", 1);
  MLPutFunction(link, "Set", 2);
  MLPutSymbol(link, name);
  MLPutFunction(link, "Transpose", 1);
  MLPutFunction(link, "Partition", 2);
  MLPutRealList(link, a, m*n);
  MLPutInteger(link, m);
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  CHKERRQ(PetscViewerMathematicaSkipPackets(viewer, RETURNPKT));
  /* Skip ReturnPacket */
  MLNewPacket(link);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerMathematicaPutCSRMatrix(PetscViewer viewer, int m, int n, int *i, int *j, PetscReal *a)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica*) viewer->data;
  MLINK                   link   = vmath->link; /* The link to Mathematica */
  const char              *symbol;
  char                    *name;
  PetscBool               match;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (!vmath->objName) name = "mat";
  else                 name = (char*) vmath->objName;

  /* Make sure Mathematica recognizes sparse matrices */
  MLPutFunction(link, "EvaluatePacket", 1);
  MLPutFunction(link, "Needs", 1);
  MLPutString(link, "LinearAlgebra`CSRMatrix`");
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  CHKERRQ(PetscViewerMathematicaSkipPackets(viewer, RETURNPKT));
  /* Skip ReturnPacket */
  MLNewPacket(link);

  /* Send the CSRMatrix object */
  MLPutFunction(link, "EvaluatePacket", 1);
  MLPutFunction(link, "Set", 2);
  MLPutSymbol(link, name);
  MLPutFunction(link, "CSRMatrix", 5);
  MLPutInteger(link, m);
  MLPutInteger(link, n);
  MLPutFunction(link, "Plus", 2);
  MLPutIntegerList(link, i, m+1);
  MLPutInteger(link, 1);
  MLPutFunction(link, "Plus", 2);
  MLPutIntegerList(link, j, i[m]);
  MLPutInteger(link, 1);
  MLPutRealList(link, a, i[m]);
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  CHKERRQ(PetscViewerMathematicaSkipPackets(viewer, RETURNPKT));
  /* Skip ReturnPacket */
  MLNewPacket(link);

  /* Check that matrix is valid */
  MLPutFunction(link, "EvaluatePacket", 1);
  MLPutFunction(link, "ValidQ", 1);
  MLPutSymbol(link, name);
  MLEndPacket(link);
  CHKERRQ(PetscViewerMathematicaSkipPackets(viewer, RETURNPKT));
  MLGetSymbol(link, &symbol);
  CHKERRQ(PetscStrcmp("True", (char*) symbol, &match));
  if (!match) {
    MLDisownSymbol(link, symbol);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid CSR matrix in Mathematica");
  }
  MLDisownSymbol(link, symbol);
  /* Skip ReturnPacket */
  MLNewPacket(link);
  PetscFunctionReturn(0);
}
