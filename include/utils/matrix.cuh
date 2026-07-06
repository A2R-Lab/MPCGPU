#pragma once
#include <cstdint>
// TODO: GBD-PCG utils include fix
#include "utils.cuh"

// The in-block linear-algebra primitives that used to live here (gato_ATx, gato_vec_dif/sum,
// mat_vec_prod, add_identity, set_identity, inv) have been migrated to GLASS:
//   gato_ATx(out,A,x,m,n)        -> glass::gemv<T, /*TRANSPOSE*/true,  /*ROW_MAJOR*/false>(m, n, 1, A, x, out)
//   mat_vec_prod(r,c,A,x,out)    -> glass::gemv<T, /*TRANSPOSE*/false, /*ROW_MAJOR*/false>(r, c, 1, A, x, out)
//   gato_vec_dif(out,a,b,n)      -> glass::axpby(n, 1, a, -1, b, out)
//   gato_vec_sum(out,a,b,n)      -> glass::axpby(n, 1, a,  1, b, out)
//   add_identity(A,dim,f)        -> glass::add_identity(dim, A, f)             (arg reorder)
//   set_identity(dim,A)          -> glass::set_identity(dim, A)        (fused multi-matrix -> N calls)
//   inv(...)            -> glass::inv(...)           (identical signatures)
// All replacements were validated bit-exact against the hand-rolled versions. Only the host-side
// debug dumper is kept here.


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
