/* $Id: f90_t3e.h,v 1.2 1998/04/06 22:51:43 balay Exp $ */

#define F90_INT_ID     33570816
#define F90_DOUBLE_ID  50348032
#define F90_COMPLEX_ID 67141632
#define F90_COOKIE     -1744830464

#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif

typedef struct {
  long lower;   /* starting index of the fortran array */
  long extent;  /* length of the array */
  long mult;    /* no of datatype units */
} tripple;
 
#define f90_header() \
void* addr;        /* Pointer to the data/array */  \
long  sd;          /* sizeof(DataType) */          \
short cookie;      /* cookie*/                     \
short ndim;        /* No of dimentions */          \
int   id;          /* Integer? double? */          \
int   a,b;


typedef struct {
  f90_header()
  tripple dim[1];
}array1d;

typedef struct {
  f90_header()
  tripple dim[2];
}array2d;

typedef struct {
  f90_header()
  tripple dim[3];
}array3d;

typedef struct {
  f90_header()
  tripple dim[4];
}array4d;

