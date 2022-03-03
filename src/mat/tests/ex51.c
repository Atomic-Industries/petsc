
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for MatBAIJ format.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,*submatA,*submatB;
  PetscInt       bs=1,m=43,ov=1,i,j,k,*rows,*cols,M,nd=5,*idx,mm,nn,lsize;
  PetscErrorCode ierr;
  PetscScalar    *vals,rval;
  IS             *is1,*is2;
  PetscRandom    rdm;
  Vec            xx,s1,s2;
  PetscReal      s1norm,s2norm,rnorm,tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  M    = m*bs;

  CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,NULL,&A));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,M,M,15,NULL,&B));
  CHKERRQ(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  CHKERRQ(PetscMalloc1(bs,&rows));
  CHKERRQ(PetscMalloc1(bs,&cols));
  CHKERRQ(PetscMalloc1(bs*bs,&vals));
  CHKERRQ(PetscMalloc1(M,&idx));

  /* Now set blocks of values */
  for (i=0; i<20*bs; i++) {
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    cols[0] = bs*(int)(PetscRealPart(rval)*m);
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    rows[0] = bs*(int)(PetscRealPart(rval)*m);
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      CHKERRQ(PetscRandomGetValue(rdm,&rval));
      vals[j] = rval;
    }
    CHKERRQ(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
    CHKERRQ(MatSetValues(B,bs,rows,bs,cols,vals,ADD_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Test MatIncreaseOverlap() */
  CHKERRQ(PetscMalloc1(nd,&is1));
  CHKERRQ(PetscMalloc1(nd,&is2));

  for (i=0; i<nd; i++) {
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    lsize = (int)(PetscRealPart(rval)*m);
    for (j=0; j<lsize; j++) {
      CHKERRQ(PetscRandomGetValue(rdm,&rval));
      idx[j*bs] = bs*(int)(PetscRealPart(rval)*m);
      for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,lsize*bs,idx,PETSC_COPY_VALUES,is1+i));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,lsize*bs,idx,PETSC_COPY_VALUES,is2+i));
  }
  CHKERRQ(MatIncreaseOverlap(A,nd,is1,ov));
  CHKERRQ(MatIncreaseOverlap(B,nd,is2,ov));

  for (i=0; i<nd; ++i) {
    CHKERRQ(ISEqual(is1[i],is2[i],&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"i=%" PetscInt_FMT ", flg =%d",i,(int)flg);
  }

  for (i=0; i<nd; ++i) {
    CHKERRQ(ISSort(is1[i]));
    CHKERRQ(ISSort(is2[i]));
  }

  CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA));
  CHKERRQ(MatCreateSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    CHKERRQ(MatGetSize(submatA[i],&mm,&nn));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    CHKERRQ(VecDuplicate(xx,&s1));
    CHKERRQ(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      CHKERRQ(VecSetRandom(xx,rdm));
      CHKERRQ(MatMult(submatA[i],xx,s1));
      CHKERRQ(MatMult(submatB[i],xx,s2));
      CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
      CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",(double)s1norm,(double)s2norm));
      }
    }
    CHKERRQ(VecDestroy(&xx));
    CHKERRQ(VecDestroy(&s1));
    CHKERRQ(VecDestroy(&s2));
  }
  /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
  CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA));
  CHKERRQ(MatCreateSubMatrices(B,nd,is2,is2,MAT_REUSE_MATRIX,&submatB));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    CHKERRQ(MatGetSize(submatA[i],&mm,&nn));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    CHKERRQ(VecDuplicate(xx,&s1));
    CHKERRQ(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      CHKERRQ(VecSetRandom(xx,rdm));
      CHKERRQ(MatMult(submatA[i],xx,s1));
      CHKERRQ(MatMult(submatB[i],xx,s2));
      CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
      CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",(double)s1norm,(double)s2norm));
      }
    }
    CHKERRQ(VecDestroy(&xx));
    CHKERRQ(VecDestroy(&s1));
    CHKERRQ(VecDestroy(&s2));
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    CHKERRQ(ISDestroy(&is1[i]));
    CHKERRQ(ISDestroy(&is2[i]));
  }
  CHKERRQ(MatDestroySubMatrices(nd,&submatA));
  CHKERRQ(MatDestroySubMatrices(nd,&submatB));
  CHKERRQ(PetscFree(is1));
  CHKERRQ(PetscFree(is2));
  CHKERRQ(PetscFree(idx));
  CHKERRQ(PetscFree(rows));
  CHKERRQ(PetscFree(cols));
  CHKERRQ(PetscFree(vals));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscRandomDestroy(&rdm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -mat_block_size {{1 2  5 7 8}} -ov {{1 3}} -mat_size {{11 13}} -nd {{7}}

TEST*/
