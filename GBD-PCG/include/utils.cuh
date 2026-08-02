#pragma once
#include <stdint.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "glass.cuh"

namespace cgrps = cooperative_groups;

template <typename T, uint32_t block_dim, uint32_t max_block_id>
__device__
void loadbdVec(T *s_var, 
               const uint32_t block_id,
               T *d_var_b)
{

    // Load this block-row's own segment into the MIDDLE slot [block_dim, 2*block_dim).
    for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x){
        s_var[ind + block_dim] = *(d_var_b + ind);
    }

    if(block_id == 0){
        // No left neighbour: zero the PREV pad slot [0, block_dim) so the (zeroed) L
        // matrix strip multiplies zero; load NEXT into [2*block_dim, 3*block_dim).
        for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x){
            s_var[ind] = static_cast<T>(0);
            s_var[ind + 2*block_dim] = *(d_var_b + block_dim + ind);
        }
    }
    else if (block_id == max_block_id){
        // No right neighbour: load PREV into [0, block_dim); zero the NEXT pad slot
        // [2*block_dim, 3*block_dim) so the (zeroed) R matrix strip multiplies zero.
        for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x){
            s_var[ind] = *(d_var_b - block_dim + ind);
            s_var[ind + 2*block_dim] = static_cast<T>(0);
        }
    }
    else{
        T *dst, *src;
        for (unsigned ind = threadIdx.x; ind < 2*block_dim; ind += blockDim.x){
            dst = s_var + ind + (ind >= block_dim) * block_dim;
            src = d_var_b + ind - (ind < block_dim) * block_dim;
            *dst = *src;
        }
    }

}


//
// Block-tridiagonal matrix-vector product for one block-row (cooperative: one CUDA
// block per knot). With the absent L (block 0) / R (last block) matrix strips zeroed
// by the kernel's populate phase and the absent halo-vector pad slots zeroed by
// loadbdVec, every block-row is one uniform full-width matvec — no first/middle/last
// special-casing:
//   s_dst(b_dim) = strip(b_dim x 3*b_dim, column-major [L|D|R]) * s_vec(3*b_dim).
// That is exactly glass::gemv with ROW_MAJOR=false (column-major), which matches the
// strip's s_mat[b_dim*c + r] storage. No trailing sync (callers barrier after, as
// before). GBD-PCG stays the cooperative grid-wide analog of glass::bdmv / glass::pcg
// (which are single-block); only this in-block matvec primitive is shared with GLASS.
//
template <typename T>
__device__
void bdmv(T *s_dst,
          T *s_mat,
          T *s_vec,
          uint32_t b_dim,
          uint32_t max_block_id,
          uint32_t block_id)
{
    (void)max_block_id; (void)block_id;   // boundaries handled by zero-padding now
    glass::gemv<T, /*TRANSPOSE*/false, /*ROW_MAJOR*/false, /*TRAILING_SYNC*/false>(
        b_dim, 3 * b_dim, static_cast<T>(1), s_mat, s_vec, s_dst);
}

template <typename T>
__device__
void gato_memcpy(T *dst, T *src, unsigned size_Ts){
	unsigned ind;
	for(ind=threadIdx.x; ind < size_Ts; ind+=blockDim.x){
		dst[ind] = src[ind];
	}
}

template <typename T>
__device__
void load_block_bd(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned bcol, unsigned brow, bool transpose=false, cooperative_groups::thread_group g = cooperative_groups::this_thread_block()){
    
    if(bcol > 2 || brow > m_dim-1){
        printf("doing somehting wrong in load_block_bd\n");
        return;
    }
    

    unsigned block_row_offset, block_col_offset;

    block_row_offset = brow * (3 * b_dim * b_dim);
    block_col_offset = bcol*b_dim*b_dim;

    if(!transpose){

        gato_memcpy<T>(
            dst,
            src+block_row_offset+block_col_offset,
            b_dim*b_dim
        );

    }
    else{

        unsigned ind, transpose_col, transpose_row;

        for(ind=threadIdx.x; ind<b_dim*b_dim; ind+=blockDim.x){
            transpose_col = ind%b_dim * b_dim;
            transpose_row = ind/b_dim;
            dst[transpose_col + transpose_row] = src[block_row_offset + block_col_offset + ind];    
        }
    }
}

template <typename T>
__device__
void store_block_bd(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier=1, cooperative_groups::thread_group g = cooperative_groups::this_thread_block()){
    
    unsigned block_row_offset, block_col_offset, ind;


    block_row_offset = BLOCKNO * (3 * b_dim * b_dim);
    block_col_offset = col*b_dim*b_dim;


    if(multiplier==1){

        glass::copy<T>(b_dim*b_dim, src, &dst[block_row_offset+block_col_offset]);

        gato_memcpy<T>(
            dst+block_row_offset+block_col_offset,
            src,
            b_dim*b_dim
        );

    }
    else{
        
        for(ind=g.thread_rank(); ind<b_dim*b_dim; ind+=g.size()){
            dst[block_row_offset + block_col_offset + ind] = src[ind] * multiplier;
        }

    }
}

