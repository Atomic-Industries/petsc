/* $Id: bitarray.h,v 1.8 1997/12/04 19:32:12 bsmith Exp bsmith $ */

/*    

          BT - Bit array objects: used to compactly store logical arrays of variables.

     BTCreate(m,bt)        - creates a bit array with enough room to hold m values
     BTDestroy(bt)         - destroys the bit array
     BTMemzero(bt,bt)      - zeros the entire bit array (sets all values to false)
     BTSet(bt,index)       - sets a particular entry as true
     BTClear(bt,index)     - sets a particular entry as false
     BTLookup(bt,index)    - returns the value 
     BTLookupSet(bt,index) - returns the value and then sets it true
     BTLength(m)           - returns number of bytes in array
     BTView(m,bt)

*/
#if !defined(__BITARRAY_H)
#define __BITARRAY_H

#if !defined(BITSPERBYTE)
#define BITSPERBYTE 8
#endif

typedef char*  BT;

extern char _mask, _BT_c;
extern int  _BT_idx;

#define BTView(m,bt) {\
  int __i; \
  for (__i=0; __i<m; __i++) { \
    printf("%d %d\n",__i,BTLookup(bt,__i)); \
  }}

#define BTLength(m)        ((m)/BITSPERBYTE+1)*sizeof(char)

#define BTCreate(m,array)  (array = (char *)PetscMalloc(((m)/BITSPERBYTE+1)*sizeof(char)),\
                           ( !array ) ? 1 : (BTMemzero(m,array),0) )

#define BTMemzero(m,array) PetscMemzero(array,(m)/BITSPERBYTE+1)

#define BTLookupSet(array, index)    (_BT_idx         = (index)/BITSPERBYTE, \
                                        _BT_c           = array[_BT_idx], \
                                        _mask           = (char)1 << ((index)%BITSPERBYTE), \
                                        array[_BT_idx]  = _BT_c | _mask, \
                                        _BT_c & _mask )

#define BTSet(array, index)    (_BT_idx         = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _mask           = (char)1 << ((index)%BITSPERBYTE), \
                                 array[_BT_idx]  = _BT_c | _mask,0)


#define BTClear(array, index)  (_BT_idx         = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _mask           = (char)1 << ((index)%BITSPERBYTE), \
                                 array[_BT_idx]  = _BT_c & (~_mask),0)

#define BTLookup(array, index) (_BT_idx         = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _mask           = (char)1 << ((index)%BITSPERBYTE), \
                                 (_BT_c & _mask) != 0 )


#define BTDestroy(array) (PetscFree(array),0)

#endif
