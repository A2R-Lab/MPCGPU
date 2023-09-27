#pragma once
#include <cstdint>
#include <cooperative_groups.h>
#include "glass.cuh"
#include "qdldl.h"
#include <fstream>

void write_device_matrix_to_file(float* d_matrix, int rows, int cols, const char* filename, int filesuffix = 0) {
    
    char fname[100];
    snprintf(fname, sizeof(fname), "%s%d.txt", filename, filesuffix);
    
    // Allocate host memory for the matrix
    float* h_matrix = new float[rows * cols];

    // Copy the data from the device to the host memory
    size_t pitch = cols * sizeof(float);
    cudaMemcpy2D(h_matrix, pitch, d_matrix, pitch, pitch, rows, cudaMemcpyDeviceToHost);

    // Write the data to a file in column-major order
    std::ofstream outfile(fname);
    if (outfile.is_open()) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                outfile << std::setprecision(std::numeric_limits<float>::max_digits10+1) << h_matrix[col * rows + row] << "\t";
            }
            outfile << std::endl;
        }
        outfile.close();
    } else {
        std::cerr << "Unable to open file: " << fname << std::endl;
    }

    // Deallocate host memory
    delete[] h_matrix;
}


///TODO: this has maximum branching right now
template <typename T>
__device__
void store_block_csr_lowertri(uint32_t bdim, uint32_t mdim, T *d_src, QDLDL_float *d_val, bool col1, uint32_t bd_block_row, int32_t multiplier=1){
    
    const int brow_val_ct = bdim*bdim + ((bdim+1)*bdim)/2;
    int row, col, csr_row_offset, full_csr_offset;
    int write_len;
    int cur_triangle_offset;

    for(row = threadIdx.x; row < bdim; row += blockDim.x){


        cur_triangle_offset = ((row+1)*row)/2;
        csr_row_offset = (bd_block_row>0)*((bdim+1)*bdim)/2 +                   // add triangle if not first block row
                         (bd_block_row>0) * (bd_block_row-1)*brow_val_ct +      // add previous full block rows if not first block row
                         (bd_block_row>0)*row*bdim +                            // 
                         cur_triangle_offset;                                   // triangle offset


        write_len = (bd_block_row>0)*((!col1)*(bdim)+(col1)*(row+1)) + (col1)*(bd_block_row==0)*(row+1);
        
        for(col = 0; col<write_len; col++){
            full_csr_offset = csr_row_offset + (bd_block_row>0)*(col1)*bdim + col;
            d_val[full_csr_offset] = static_cast<QDLDL_float>(d_src[row + col*bdim]) * multiplier;
        }
    }
}


__global__
void prep_csr(uint32_t state_size, uint32_t knot_points, QDLDL_int *d_col_ptr, QDLDL_int *d_row_ind){
    
    for (uint32_t blockrow = blockIdx.x; blockrow < knot_points; blockrow+=gridDim.x)
    {
        const int brow_val_ct = state_size*state_size + ((state_size+1)*state_size)/2;
        int row, col, csr_row_offset, full_csr_offset, bd_row_len;
        int cur_triangle_offset;

        for(row = threadIdx.x; row < state_size; row += blockDim.x){


            if(blockrow==0 && row==0){
                d_col_ptr[0] = 0;
            }
            
            cur_triangle_offset = ((row+1)*row)/2;
            csr_row_offset = (blockrow>0)*((state_size+1)*state_size)/2 +                   // add triangle if not first block row
                            (blockrow>0) * (blockrow-1)*brow_val_ct +      // add previous full block rows if not first block row
                            (blockrow>0)*row*state_size +                            // 
                            cur_triangle_offset;                                   // triangle offset


            bd_row_len = (blockrow>0)*state_size + row+1;
            d_col_ptr[blockrow*state_size + row+1] = csr_row_offset+bd_row_len;
            
            for(col = 0; col < bd_row_len; col++){
                full_csr_offset = csr_row_offset + col;
                d_row_ind[full_csr_offset] = (blockrow>0)*(blockrow-1)*state_size + col;
            }

        }
    }
    
}