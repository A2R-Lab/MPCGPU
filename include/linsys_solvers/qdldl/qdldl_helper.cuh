#pragma once
#include <cstdint>
#include "gpuassert.cuh"
#include "qdldl.h"


namespace qdl{

// template <typename ind_T, typename val_T> 
// __global__
// void bd_to_csr_lowertri(
//                 unsigned               n,           ///< number of rows
//                 ind_T               *row_ptr,    ///< row pointers (size m+1)
//                 ind_T               *col_ind,    ///< column indices (size nnz)
//                 val_T                  *val,        ///< numerical values (size nnz)
//                 T  *bdmat,
//                 unsigned bdim,
//                 unsigned mdim)
// {

//     const int brow_val_ct = bdim*bdim + ((bdim+1)*bdim)/2;
//     int row, col, csr_row_offset, basic_col_offset, bd_block_row, bd_block_col, bd_col, bd_row, bd_row_len;
//     int iter, bd_offset, row_adj;


//     for(row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID; row < n; row += GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS){

//         bd_block_row = row/bdim;

//         bd_row_len = (bd_block_row>0)*bdim + row%bdim+1;
        
//         if(row==0){
//             row_ptr[row] = 0;
//         }
        
//         row_adj = (row%bdim);    
//         // int thisthing = ((row_adj+1)*(2*(bdim-row_adj)+row_adj))/2;
//         int thisthing = ((row_adj+1)*row_adj)/2;
//         csr_row_offset = (bd_block_row>0)*((bdim+1)*bdim)/2 + (bd_block_row>0) * (bd_block_row-1)*brow_val_ct + (bd_block_row>0)*(row%bdim)*bdim + thisthing;

//         basic_col_offset = (bd_block_row>0)*(bd_block_row-1)*bdim;
//         row_ptr[row+1] = csr_row_offset+bd_row_len;

//         for(iter=0; iter<bd_row_len; iter++){

//             col = basic_col_offset+iter;
//             bd_block_col = ( col / bdim ) + 1 - bd_block_row;  // block col
//             bd_col = col % bdim;
//             bd_row = row % bdim;

//             bd_offset = bd_block_row*3*bdim*bdim + bd_block_col*bdim*bdim + bd_col*bdim + bd_row;
            
//             col_ind[csr_row_offset+iter] = col;
//             val[csr_row_offset+iter] = bdmat[bd_offset];
//         }

//     }
// }


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

	/*--------------------------------
	* print factors and solution
	*---------------------------------*/
/*	printf("\n");
	printf("A (CSC format):\n");
	print_line();
	print_arrayi(Ap, An + 1, "A.p");
	print_arrayi(Ai, Ap[An], "A.i");
	print_arrayf(Ax, Ap[An], "A.x");
	printf("\n\n");

	printf("elimination tree:\n");
	print_line();
	print_arrayi(etree, Ln, "etree");
	print_arrayi(Lnz, Ln, "Lnz");
	printf("\n\n");

	printf("L (CSC format):\n");
	print_line();
	print_arrayi(Lp, Ln + 1, "L.p");
	print_arrayi(Li, Lp[Ln], "L.i");
	print_arrayf(Lx, Lp[Ln], "L.x");
	printf("\n\n");

	printf("D:\n");
	print_line();
	print_arrayf(D, An,    "diag(D)     ");
	print_arrayf(Dinv, An, "diag(D^{-1})");
	printf("\n\n");

	printf("solve results:\n");
	print_line();
	print_arrayf(b, An, "b");
	print_arrayf(x, An, "A\\b");
	printf("\n\n");
*/

	/*--------------------------------
	* clean up
	*---------------------------------*/


}



}