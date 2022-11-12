/* Network Riemann Solver implementation */

#if !defined(__NETRSIMPL_H)
#define __NETRSIMPL_H

#include <petscriemannsolver.h>
#include <petscnetrs.h>
#include <petsc/private/petscimpl.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscnetrp.h>

PETSC_EXTERN PetscBool NetRSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode NetRSRegisterAll(void);

/* HMap from NetRP -> PetscInt, allows for the easy interface for adding NetRP to vertice.s 
of the DMNetwork, i.e input interface is in in terms of NetRP, and the exact index of that NetRP 
in the DMLabel and other arrays isn't important */

#define PetscHashNetRPKeyHash(key) PetscHashPointer((key)) 
#define PetscHashNetRPKeyEqual(k1, k2) (k1==k2) 

PETSC_HASH_MAP(HMapNetRPI, NetRP, PetscInt, PetscHashNetRPKeyHash, PetscHashNetRPKeyEqual, -1)


typedef struct _NetRSOps *NetRSOps;
struct _NetRSOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,NetRS);
  PetscErrorCode (*setup)(NetRS);
  PetscErrorCode (*view)(NetRS,PetscViewer);
  PetscErrorCode (*destroy)(NetRS);
  PetscErrorCode (*reset)(NetRS);
  PetscErrorCode (*setupvecspace)(NetRS); 
  PetscErrorCode (*resetvecspace)(NetRS); 
  PetscErrorCode (*solve)(NetRS,Vec,Vec); 
};

struct _p_NetRS {
  PETSCHEADER(struct _NetRSOps);
  PetscBool      setupcalled, setupvectorspace; 
  void           *data; /* implementation object */
  void           *user; /* user context */

  DM             network; /* internal DMNetwork for storing data about the topology of the Riemann Problem */  

  /* TODO: Make PetscFlux class */
  RiemannSolver  rs; /* For holding physics information, a hack for now to be replaced by FluxFunction */

  /* For setting up the preallocation of objects and constructing the sub NetRS problems */

  DMLabel        subgraphs; /* TODO : Name better. Each stratum corresponds to the set of vertices associated with a 
  specific NetRS solver  */

  PetscHMapNetRPI netrphmap; /* map from netrp to index index into arrays/value in the DMLabel. */
  NetRP          *netrp; /* arrray of local riemann problem/solver, one for each label */
  Vec            totalU, totalFlux; /* total vectors for the input U and output Flux, includes all NetRP problems added to the NetRS */ 



  /* DMNetwork Graph stuff, should be moved to DMNetwork itself */
  PetscBool      vertexdeg_shared_cached; 
  /* This label should also be a disjoint label if that implementation exists */
  DMLabel        VertexDeg_shared; /*TODO: Name Better. Stores the vertex degrees for all 
  share vertices . 
  
  Values correspond to the following the vertex deg. 

  A vertex v has value n if deg(v) = n in the full graph and v is shared among processors. 

  Needed as only local graph edge connectivity is stored when v is shared among processors. 

  this assumes that there is no edge overlap in the distributed graph. Would need a rework 
  in that case */
 
  PetscHMapI     vertex_shared_offset; /* if vertex v is shared among processors we need 
  an offset of the local connected edges returned from DMNetworkGetSupportingEdges(), in the 
  (if it existed) globally connected edges. That is I need a local number of the supported edges 
  for a vertex. 

  This gives that map. Needed in NetRS as local flux and input data for the riemann problem 
  generates a vector space like the following 

  V_v  = \union_{e \in E(v)} V_{e,v} 

  where E(v) are the connected edges in the global graph. And V_{e,v} = \R^m 
  for the Riemann problem, m being the number of fields in the flux function. 

  But each processor only see the e that belong to it, so need a way to extract
  local V_v, that is V_{v,rank}. This offset does that. 
  */

  
  PetscHSetI     vertexdegrees_total; /* set of all vertex degrees in the full local network */
  PetscHSetI     *vertexdegrees; /* set of all vertex degrees for each subgraph induced by the DMLabel */
/* End of DMNetwork Graph stuff */

 };
#endif