PETSCFLAGS = -DPETSC_DEBUG  -DPETSC_LOG -DPETSC_BOPT_g -Dlint
COPTFLAGS  = -g -Wall -Wshadow
#
# To prohibit Fortran implicit typing, add -u in FOPTFLAGS definition
#
#FOPTFLAGS  = -g -dalign
FOPTFLAGS  = -g 
SYS_LIB    = /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o

