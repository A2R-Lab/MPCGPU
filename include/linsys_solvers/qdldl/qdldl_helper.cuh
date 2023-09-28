#pragma once
#include <cstdint>
#include "gpuassert.cuh"
#include "qdldl.h"


namespace qdl{


///TODO: pass in allocated mem
__host__
void qdldl_solve_schur(const QDLDL_int An,
					   QDLDL_int *h_col_ptr, QDLDL_int *h_row_ind, QDLDL_float *Ax, QDLDL_float *b, 
					   QDLDL_float *h_lambda,
					   QDLDL_int *Lp, QDLDL_int *Li, QDLDL_float *Lx, QDLDL_float *D, QDLDL_float *Dinv, QDLDL_int *Lnz, QDLDL_int *etree, QDLDL_bool *bwork, QDLDL_int *iwork, QDLDL_float *fwork){

	



    QDLDL_int i;

	const QDLDL_int *Ap = h_col_ptr;
	const QDLDL_int *Ai = h_row_ind;

    //data for L and D factors
	QDLDL_int Ln = An;

	

	//data for elim tree calculation

	


	//Data for results of A\b
	QDLDL_float *x = h_lambda;


	/*--------------------------------
	* pre-factorisation memory allocations
	*---------------------------------*/

	//These can happen *before* the etree is calculated
	//since the sizes are not sparsity pattern specific

	//For the elimination tree


	//For the L factors.   Li and Lx are sparsity dependent
	//so must be done after the etree is constructed


	//Working memory.  Note that both the etree and factor
	//calls requires a working vector of QDLDL_int, with
	//the factor function requiring 3*An elements and the
	//etree only An elements.   Just allocate the larger
	//amount here and use it in both places

	/*--------------------------------
	* elimination tree calculation
	*---------------------------------*/

	

	/*--------------------------------
	* LDL factorisation
	*---------------------------------*/

	//First allocate memory for Li and Lx
	

	//now factor
	QDLDL_factor(An,Ap,Ai,Ax,Lp,Li,Lx,D,Dinv,Lnz,etree,bwork,iwork,fwork);

	/*--------------------------------
	* solve
	*---------------------------------*/
	// x = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

	//when solving A\b, start with x = b
	for(i=0;i < Ln; i++) x[i] = b[i];


	QDLDL_solve(Ln,Lp,Li,Lx,Dinv,x);

}



}