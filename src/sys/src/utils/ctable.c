#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ctable.c,v 1.2 1999/01/08 16:27:18 balay Exp balay $";
#endif
/* Contributed by - Mark Adams */

#include "petsc.h"
#include "src/sys/ctable.h" 

#undef __FUNC__  
#define __FUNC__ "TableCreate"
/* TableCreate() ********************************************
 * 
 * hash table for non-zero data and keys 
 *
 */
int TableCreate( Table *rta, const int n )
{
  Table ta;

  ta = (Table)PetscMalloc(sizeof(Table_struct));CHKPTRQ(ta);
  ta->tablesize = ( 3 * n ) / 2 + 17;
  ta->keytable = (int*)PetscMalloc( sizeof(int)*ta->tablesize ); 
  CHKPTRQ(ta->keytable);
  PetscMemzero( ta->keytable, sizeof(int)*ta->tablesize );
  ta->table = (int*)PetscMalloc( sizeof(int)*ta->tablesize );
  CHKPTRQ(ta->table);
  ta->head = 0;
  ta->count = 0;
  
  *rta = ta;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TableCreateCopy"
/* TableCreate() ********************************************
 * 
 * hash table for non-zero data and keys 
 *
 */
int TableCreateCopy( Table *rta, const Table intable )
{
  int ii;
  Table ta;

  ta = (Table)PetscMalloc(sizeof(Table_struct));CHKPTRQ(ta);
  ta->tablesize = intable->tablesize;
  ta->keytable = (int*)PetscMalloc( sizeof(int)*ta->tablesize ); 
  CHKPTRQ(ta->keytable);
  ta->table = (int*)PetscMalloc( sizeof(int)*ta->tablesize );
  CHKPTRQ(ta->table);
  for( ii = 0 ; ii < ta->tablesize ; ii++ ){
    ta->keytable = intable->keytable; ta->table = intable->table;
  }
  ta->head = 0;
  ta->count = 0;
  
  *rta = ta;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TableDelete"
/* TableDelete() ********************************************
 * 
 *
 */
int TableDelete( Table ta )
{
  int ierr;
  ierr = PetscFree(ta->keytable); CHKERRQ(ierr);
  ierr = PetscFree(ta->table); CHKERRQ(ierr);
  ierr = PetscFree(ta); CHKERRQ(ierr);
  
  return 0;
} 
#undef __FUNC__  
#define __FUNC__ "TableGetCount"
/* TableGetCount() ********************************************
 */
int TableGetCount( const Table ta ) 
{ 
  return ta->count; 
}

#undef __FUNC__  
#define __FUNC__ "TableIsEmpty"
/* TableIsEmpty() ********************************************
 */
int TableIsEmpty( const Table ta ) 
{ 
  return !( ta->count ); 
}

#undef __FUNC__  
#define __FUNC__ "TableAdd"
/* TableAdd() ********************************************
 *
 */
int TableAdd( Table ta, const int key, const int data )
{  
  int ii = 0, hash = HASHT( ta, key ), ierr;
  const int tsize = ta->tablesize, tcount = ta->count; 

  if( key == 0 || data == 0 ) SETERRQ(1,1,"Table zero key or data");
  
  if( ta->count < (5*ta->tablesize) / 6 - 1 ) {
    while( ii++ < ta->tablesize ){
      if( ta->keytable[hash] == key ){
	ta->table[hash] = data; /* over write */
	return 0; 
      }
      else if( !ta->keytable[hash] ) {
	ta->count++; /* add */
	ta->keytable[hash] = key; ta->table[hash] = data;
	return 0;
      }
      hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
    }  
    SETERRQ(1,1,"TABLE error"); 
  }
  else {
    int *oldtab = ta->table, *oldkt = ta->keytable, newk, ndata;
    
    /* alloc new (bigger) table */
    ta->tablesize = 2*tsize; 
    ta->table = (int*)PetscMalloc( sizeof(int)*ta->tablesize ); 
    CHKPTRQ(ta->table);
    ta->keytable = (int*)PetscMalloc( sizeof(int)*ta->tablesize ); 
    CHKPTRQ(ta->keytable);
    PetscMemzero( ta->keytable, sizeof(int)*ta->tablesize );
    ta->count = ta->head = 0; 
    
    ierr = TableAdd( ta, key, data ); CHKERRQ(ierr); 
    /* rehash */
    for( ii = 0 ; ii < tsize ; ii++ ) {
      newk = oldkt[ii];
      if( newk ) {
	ndata = oldtab[ii];
	ierr = TableAdd( ta, newk, ndata ); CHKERRQ(ierr); 
      }
    }
    if( ta->count != tcount + 1 ) SETERRQ(1,1,"Table error");
    
    ierr = PetscFree( oldtab );  CHKERRQ(ierr);
    ierr = PetscFree( oldkt );  CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TableRemoveAll"
/* TableRemoveAll() ********************************************
 *
 *
 */
int TableRemoveAll( Table ta )
{ 
  ta->head = 0;  
  if( ta->count ){ 
    ta->count = 0;
    PetscMemzero( ta->keytable, ta->tablesize*sizeof(int) ); 
  }

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TableFind"
/* TableFind() ********************************************
 *
 * returns data
 *
 */
int TableFind( Table ta, const int key ) 
{  
  int hash = HASHT( ta, key ), ii = 0;

  while( ii++ < ta->tablesize ){
    if ( ta->keytable[hash] == 0 ) return 0;
    else if ( ta->keytable[hash] == key ) return ta->table[hash];
    hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
  }
  
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TableGetHeadPosition"
/* TableGetHeadPosition() ********************************************
 *
 */
int TableGetHeadPosition( Table ta, CTablePos *ppos )
{
  int i = 0;

  *ppos = NULL;
  if ( !ta->count ) return 0;
  
  /* find first valid place */
  do{
    if( ta->keytable[i] ){
      *ppos = (CTablePos)&ta->table[i];
      break;
    }
  }while( i++ < ta->tablesize );
  if( !*ppos ) SETERRQ(1,1,"No head");

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TableGetNext"
/* TableGetNext() ********************************************
 *
 *  - iteration - CTablePos is always valid (points to a data)
 *  
 */
int TableGetNext( Table ta, CTablePos *rPosition, int *pkey, int *data )
{
  int index; 
  CTablePos pos;

  pos = *rPosition;
  if(!pos)SETERRQ(1,1,"TABLE null position"); 
  *data = *pos; 
  if(!*data)SETERRQ(1,1,"TABLE error"); 
  index = pos - ta->table;
  *pkey = ta->keytable[index]; 
  if( !*pkey ) SETERRQ(1,1,"TABLE error");  

  /* get next */
  do{
    pos++;  index++;
    if( index >= ta->tablesize ) {      
      pos = 0; /* end of list */
      break;
    }
    else if ( ta->keytable[index] ) {
      pos = ta->table + index;
      break;
    }
  }while( index < ta->tablesize );

  *rPosition = pos;

  return 0;
}




