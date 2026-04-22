/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_bsr2csr_bad_arg(const Arguments& arg)
{
    static const size_t              safe_size = 1;
    host_dense_vector<rocsparse_int> hbsr_row_ptr(safe_size + 1);
    hbsr_row_ptr[0] = 0;
    hbsr_row_ptr[1] = 1;
    device_dense_vector<rocsparse_int> dbsr_row_ptr(hbsr_row_ptr);
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create rocsparse descriptors
    rocsparse_local_mat_descr local_bsr_descr;
    rocsparse_local_mat_descr local_csr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_direction       dir         = rocsparse_direction_row;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             nb          = safe_size;
    const rocsparse_mat_descr bsr_descr   = local_bsr_descr;
    const T*                  bsr_val     = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr = (const rocsparse_int*)dbsr_row_ptr;

    const rocsparse_int*      bsr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim   = safe_size;
    const rocsparse_mat_descr csr_descr   = local_csr_descr;
    T*                        csr_val     = (T*)0x4;
    rocsparse_int*            csr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind = (rocsparse_int*)0x4;

#define PARAMS                                                                               \
    handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, csr_descr, \
        csr_val, csr_row_ptr, csr_col_ind

    bad_arg_analysis(rocsparse_bsr2csr<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(PARAMS), rocsparse_status_requires_sorted_storage);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_sorted));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_sorted));

    // Check block_dim == 0
    block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(PARAMS), rocsparse_status_invalid_size);

#undef PARAMS
}

template <typename T>
static rocsparse_status check_bsr_generation(const Arguments& arg)
{
    std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl;

    rocsparse_seedrand();
    rocsparse_matrix_factory<T> matrix_factory(arg);

    {
        rocsparse_int Mb = (arg.M + arg.block_dim - 1) / arg.block_dim;
        rocsparse_int Nb = (arg.N + arg.block_dim - 1) / arg.block_dim;
        rocsparse_int block_dim = arg.block_dim;
        rocsparse_index_base base = arg.baseA;
        rocsparse_direction dir = arg.direction;
        rocsparse_int nnzb;
        std::vector<rocsparse_int> hbsr_row_ptr;
        std::vector<rocsparse_int> hbsr_col_ind;
        std::vector<T> hbsr_val;
        matrix_factory.init_gebsr(hbsr_row_ptr,
            hbsr_col_ind,
            hbsr_val,
            dir,
            Mb,
            Nb,
            nnzb,
            block_dim,
            block_dim,
            base,
            bsr_construction_alg::convert_csr);
    }

    // Reference matrix
    rocsparse_seedrand();
    rocsparse_int Mb_ref = (arg.M + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int Nb_ref = (arg.N + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int block_dim_ref = arg.block_dim;
    rocsparse_index_base base_ref = arg.baseA;
    rocsparse_direction dir_ref = arg.direction;
    rocsparse_int nnzb_ref;
    std::vector<rocsparse_int> hbsr_row_ptr_ref;
    std::vector<rocsparse_int> hbsr_col_ind_ref;
    std::vector<T> hbsr_val_ref;
    matrix_factory.init_gebsr(hbsr_row_ptr_ref,
        hbsr_col_ind_ref,
        hbsr_val_ref,
        dir_ref,
        Mb_ref,
        Nb_ref,
        nnzb_ref,
        block_dim_ref,
        block_dim_ref,
        base_ref,
        bsr_construction_alg::convert_csr);

    // A matrix
    // rocsparse_seedrand();
    // rocsparse_int Mb_A = (arg.M + arg.block_dim - 1) / arg.block_dim;
    // rocsparse_int Nb_A = (arg.N + arg.block_dim - 1) / arg.block_dim;
    // rocsparse_int block_dim_A = arg.block_dim;
    // rocsparse_index_base base_A = arg.baseA;
    // rocsparse_direction dir_A = arg.direction;
    // host_gebsr_matrix<T>   hA(dir_A, Mb_A, Nb_A, 0, block_dim_A, block_dim_A, base_A);
    // matrix_factory.init_gebsr(hA, bsr_construction_alg::convert_csr);

    // B matrix
    rocsparse_seedrand();
    rocsparse_int Mb_B = (arg.M + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int Nb_B = (arg.N + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int block_dim_B = arg.block_dim;
    rocsparse_index_base base_B = arg.baseA;
    rocsparse_direction dir_B = arg.direction;
    host_gebsr_matrix<T>   hB(dir_B, Mb_B, Nb_B, 0, block_dim_B, block_dim_B, base_B);
    matrix_factory.init_gebsr(hB,
                                Mb_B,
                                Nb_B,
                                block_dim_B,
                                block_dim_B,
                                base_B,
                                bsr_construction_alg::convert_csr);

    std::cout << "dir_B: " << dir_B << " arg.direction: " << arg.direction << " hB.block_direction: " << hB.block_direction << " dir_ref: " << dir_ref << std::endl;

    // C matrix
    rocsparse_seedrand();
    rocsparse_int Mb_C = (arg.M + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int Nb_C = (arg.N + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int block_dim_C = arg.block_dim;
    rocsparse_index_base base_C = arg.baseA;
    rocsparse_direction dir_C = arg.direction;
    rocsparse_int nnzb_C = 0;
    host_gebsr_matrix<T>   hC(dir_C, Mb_C, Nb_C, nnzb_C, block_dim_C, block_dim_C, base_C);
    matrix_factory.init_gebsr(hC,
                            dir_C,
                            Mb_C,
                            Nb_C,
                            nnzb_C,
                            block_dim_C,
                            block_dim_C,
                            base_C,
                            bsr_construction_alg::convert_csr);

    // D matrix
    rocsparse_seedrand();
    rocsparse_int Mb_D = (arg.M + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int Nb_D = (arg.N + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int block_dim_D = arg.block_dim;
    rocsparse_index_base base_D = arg.baseA;
    rocsparse_direction dir_D = arg.direction;
    rocsparse_int nnzb_D;
    std::vector<rocsparse_int> hbsr_row_ptr_D;
    std::vector<rocsparse_int> hbsr_col_ind_D;
    std::vector<T> hbsr_val_D;
    matrix_factory.init_bsr(hbsr_row_ptr_D,
                            hbsr_col_ind_D,
                            hbsr_val_D,
                            dir_D,
                            Mb_D,
                            Nb_D,
                            nnzb_D,
                            block_dim_D,
                            base_D,
                            bsr_construction_alg::convert_csr);

    // E matrix
    rocsparse_seedrand();
    rocsparse_int Mb_E = (arg.M + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int Nb_E = (arg.N + arg.block_dim - 1) / arg.block_dim;
    rocsparse_int block_dim_E = arg.block_dim;
    rocsparse_index_base base_E = arg.baseA;
    rocsparse_direction dir_E = arg.direction;
    host_gebsr_matrix<T>   hE;
    device_gebsr_matrix<T> dE;
    matrix_factory.init_bsr(hE,
                            dE,
                            Mb_E,
                            Nb_E,
                            base_E,
                            bsr_construction_alg::convert_csr);


#define CHECK_MATCH(actual, ref)                                           \
    do                                                                     \
    {                                                                      \
        if((actual) != (ref))                                              \
        {                                                                  \
            std::cout << #actual " does not match " #ref << std::endl;     \
            return rocsparse_status_internal_error;                        \
        }                                                                  \
    } while(0)

    // std::cout << "Mb_A: " << Mb_A << " Mb_ref: " << Mb_ref << std::endl;
    // std::cout << "hA.row_block_dim: " << hA.row_block_dim << " hA.col_block_dim: " << hA.col_block_dim << " block_dim_ref: " << block_dim_ref << std::endl;

    // Check A
    // CHECK_MATCH(hA.mb, Mb_ref);
    // CHECK_MATCH(hA.nb, Nb_ref);
    // CHECK_MATCH(hA.row_block_dim, block_dim_ref);
    // CHECK_MATCH(hA.col_block_dim, block_dim_ref);
    // CHECK_MATCH(hA.base, base_ref);
    // CHECK_MATCH(hA.block_direction, dir_ref);

    // for(size_t i = 0; i < hbsr_row_ptr_ref.size(); i++)
    // {
    //     if(hbsr_row_ptr_ref[i] != hA.ptr[i])
    //     {
    //         std::cout << "Error at index " << i << " of A row pointer array" << std::endl;
    //         return rocsparse_status_internal_error;
    //     }
    // }
    // for(size_t i = 0; i < hbsr_col_ind_ref.size(); i++)
    // {
    //     if(hbsr_col_ind_ref[i] != hA.ind[i])
    //     {
    //         std::cout << "Error at index " << i << " of A column indices array" << std::endl;
    //         return rocsparse_status_internal_error;
    //     }
    // }
    // for(size_t i = 0; i < hbsr_val_ref.size(); i++)
    // {
    //     if(hbsr_val_ref[i] != hA.val[i])
    //     {
    //         std::cout << "Error at index " << i << " of A values array hbsr_val_ref[i]: " 
    //                   << hbsr_val_ref[i] << " hA.val[i]: " << hA.val[i] << std::endl;
    //         return rocsparse_status_internal_error;
    //     }
    // }


    // Check B
    CHECK_MATCH(Mb_B, Mb_ref);
    CHECK_MATCH(Nb_B, Nb_ref);
    CHECK_MATCH(block_dim_B, block_dim_ref);
    CHECK_MATCH(base_B, base_ref);
    CHECK_MATCH(dir_B, dir_ref);

    CHECK_MATCH(hB.mb, Mb_ref);
    CHECK_MATCH(hB.nb, Nb_ref);
    CHECK_MATCH(hB.row_block_dim, block_dim_ref);
    CHECK_MATCH(hB.col_block_dim, block_dim_ref);
    CHECK_MATCH(hB.base, base_ref);
    CHECK_MATCH(hB.block_direction, dir_ref);

    for(size_t i = 0; i < hbsr_row_ptr_ref.size(); i++)
    {
        if(hbsr_row_ptr_ref[i] != hB.ptr[i])
        {
            std::cout << "Error at index " << i << " of B row pointer array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }
    for(size_t i = 0; i < hbsr_col_ind_ref.size(); i++)
    {
        if(hbsr_col_ind_ref[i] != hB.ind[i])
        {
            std::cout << "Error at index " << i << " of B column indices array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }
    for(size_t i = 0; i < hbsr_val_ref.size(); i++)
    {
        if(hbsr_val_ref[i] != hB.val[i])
        {
            std::cout << "Error at index " << i << " of B values array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }

    // Check C
    CHECK_MATCH(Mb_C, Mb_ref);
    CHECK_MATCH(Nb_C, Nb_ref);
    CHECK_MATCH(nnzb_C, nnzb_ref);
    CHECK_MATCH(block_dim_C, block_dim_ref);
    CHECK_MATCH(base_C, base_ref);
    CHECK_MATCH(dir_C, dir_ref);

    CHECK_MATCH(hC.mb, Mb_ref);
    CHECK_MATCH(hC.nb, Nb_ref);
    CHECK_MATCH(hC.nnzb, nnzb_ref);
    CHECK_MATCH(hC.row_block_dim, block_dim_ref);
    CHECK_MATCH(hC.col_block_dim, block_dim_ref);
    CHECK_MATCH(hC.base, base_ref);
    CHECK_MATCH(hC.block_direction, dir_ref);

    for(size_t i = 0; i < hbsr_row_ptr_ref.size(); i++)
    {
        if(hbsr_row_ptr_ref[i] != hC.ptr[i])
        {
            std::cout << "Error at index " << i << " of C row pointer array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }
    for(size_t i = 0; i < hbsr_col_ind_ref.size(); i++)
    {
        if(hbsr_col_ind_ref[i] != hC.ind[i])
        {
            std::cout << "Error at index " << i << " of C column indices array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }
    for(size_t i = 0; i < hbsr_val_ref.size(); i++)
    {
        if(hbsr_val_ref[i] != hC.val[i])
        {
            std::cout << "Error at index " << i << " of C values array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }

    // Check D
    CHECK_MATCH(Mb_D, Mb_ref);
    CHECK_MATCH(Nb_D, Nb_ref);
    CHECK_MATCH(nnzb_D, nnzb_ref);
    CHECK_MATCH(block_dim_D, block_dim_ref);
    CHECK_MATCH(base_D, base_ref);
    CHECK_MATCH(dir_D, dir_ref);

    for(size_t i = 0; i < hbsr_row_ptr_ref.size(); i++)
    {
        if(hbsr_row_ptr_ref[i] != hbsr_row_ptr_D[i])
        {
            std::cout << "Error at index " << i << " of D row pointer array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }
    for(size_t i = 0; i < hbsr_col_ind_ref.size(); i++)
    {
        if(hbsr_col_ind_ref[i] != hbsr_col_ind_D[i])
        {
            std::cout << "Error at index " << i << " of D column indices array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }
    for(size_t i = 0; i < hbsr_val_ref.size(); i++)
    {
        if(hbsr_val_ref[i] != hbsr_val_D[i])
        {
            std::cout << "Error at index " << i << " of D values array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }

    // Check E
    CHECK_MATCH(Mb_E, Mb_ref);
    CHECK_MATCH(Nb_E, Nb_ref);
    CHECK_MATCH(block_dim_E, block_dim_ref);
    CHECK_MATCH(base_E, base_ref);
    CHECK_MATCH(dir_E, dir_ref);

    CHECK_MATCH(hE.mb, Mb_ref);
    CHECK_MATCH(hE.nb, Nb_ref);
    CHECK_MATCH(hE.row_block_dim, block_dim_ref);
    CHECK_MATCH(hE.col_block_dim, block_dim_ref);
    CHECK_MATCH(hE.base, base_ref);
    CHECK_MATCH(hE.block_direction, dir_ref);

    for(size_t i = 0; i < hbsr_row_ptr_ref.size(); i++)
    {
        if(hbsr_row_ptr_ref[i] != hE.ptr[i])
        {
            std::cout << "Error at index " << i << " of E row pointer array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }
    for(size_t i = 0; i < hbsr_col_ind_ref.size(); i++)
    {
        if(hbsr_col_ind_ref[i] != hE.ind[i])
        {
            std::cout << "Error at index " << i << " of E column indices array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }
    for(size_t i = 0; i < hbsr_val_ref.size(); i++)
    {
        if(hbsr_val_ref[i] != hE.val[i])
        {
            std::cout << "Error at index " << i << " of E values array" << std::endl;
            return rocsparse_status_internal_error;
        }
    }

    std::cout << "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB" << std::endl;

    return rocsparse_status_success;

    // {
    //     std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" << std::endl;
    //     rocsparse_seedrand();
    //     rocsparse_matrix_factory<T> matrix_factory(arg);

    //     rocsparse_int Mb = (arg.M + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int Nb = (arg.N + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int block_dim = arg.block_dim;
    //     rocsparse_index_base base = arg.baseA;
    //     rocsparse_direction dir = arg.direction;

    //     rocsparse_int nnzb;
    //     std::vector<rocsparse_int> hbsr_row_ptr;
    //     std::vector<rocsparse_int> hbsr_col_ind;
    //     std::vector<T> hbsr_val;
    //     matrix_factory.init_gebsr(hbsr_row_ptr,
    //         hbsr_col_ind,
    //         hbsr_val,
    //         dir,
    //         Mb,
    //         Nb,
    //         nnzb,
    //         block_dim,
    //         block_dim,
    //         base,
    //         bsr_construction_alg::convert_csr);

    //     std::cout << "BSR A" << std::endl;
    //     std::vector<T> hdense(Mb * block_dim * Nb * block_dim, 0);
    //     for(int i = 0; i < Mb; i++)
    //     {
    //         int start = hbsr_row_ptr[i] - base;
    //         int end = hbsr_row_ptr[i + 1] - base;

    //         for(int j = start; j < end; j++)
    //         {
    //             int bcol = hbsr_col_ind[j] - base;

    //             for(int r = 0; r < block_dim; r++)
    //             {
    //                 for(int c = 0; c < block_dim; c++)
    //                 {
    //                     int row = block_dim * i + r;
    //                     int col = block_dim * bcol + c;
    //                     hdense[Nb * block_dim * row + col] = hbsr_val[block_dim * block_dim * j + block_dim * r + c];
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hdense" << std::endl;
    //     for(int row = 0; row < block_dim * Mb; row++)
    //     {
    //         for(int col = 0; col < block_dim * Mb; col++)
    //         {
    //             std::cout << hdense[Nb * block_dim * row + col] << " ";
    //         }
    //         std::cout << "" << std::endl;
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hbsr_row_ptr" << std::endl;
    //     for(size_t i = 0; i < hbsr_row_ptr.size(); i++)
    //     {
    //         std::cout << hbsr_row_ptr[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hbsr_col_ind" << std::endl;
    //     for(size_t i = 0; i < hbsr_col_ind.size(); i++)
    //     {
    //         std::cout << hbsr_col_ind[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hbsr_val" << std::endl;
    //     for(size_t i = 0; i < hbsr_val.size(); i++)
    //     {
    //         std::cout << hbsr_val[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }






    // {
    //     std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl;
    //     rocsparse_seedrand();
    //     rocsparse_matrix_factory<T> matrix_factory(arg);

    //     rocsparse_int Mb = (arg.M + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int Nb = (arg.N + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int block_dim = arg.block_dim;
    //     rocsparse_index_base base = arg.baseA;
    //     rocsparse_direction dir = arg.direction;

    //     rocsparse_int nnzb;
    //     std::vector<rocsparse_int> hbsr_row_ptr;
    //     std::vector<rocsparse_int> hbsr_col_ind;
    //     std::vector<T> hbsr_val;
    //     matrix_factory.init_gebsr(hbsr_row_ptr,
    //         hbsr_col_ind,
    //         hbsr_val,
    //         dir,
    //         Mb,
    //         Nb,
    //         nnzb,
    //         block_dim,
    //         block_dim,
    //         base,
    //         bsr_construction_alg::convert_csr);

    //     std::cout << "BSR A" << std::endl;
    //     std::vector<T> hdense(Mb * block_dim * Nb * block_dim, 0);
    //     for(int i = 0; i < Mb; i++)
    //     {
    //         int start = hbsr_row_ptr[i] - base;
    //         int end = hbsr_row_ptr[i + 1] - base;

    //         for(int j = start; j < end; j++)
    //         {
    //             int bcol = hbsr_col_ind[j] - base;

    //             for(int r = 0; r < block_dim; r++)
    //             {
    //                 for(int c = 0; c < block_dim; c++)
    //                 {
    //                     int row = block_dim * i + r;
    //                     int col = block_dim * bcol + c;
    //                     hdense[Nb * block_dim * row + col] = hbsr_val[block_dim * block_dim * j + block_dim * r + c];
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hdense" << std::endl;
    //     for(int row = 0; row < block_dim * Mb; row++)
    //     {
    //         for(int col = 0; col < block_dim * Mb; col++)
    //         {
    //             std::cout << hdense[Nb * block_dim * row + col] << " ";
    //         }
    //         std::cout << "" << std::endl;
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hbsr_row_ptr" << std::endl;
    //     for(size_t i = 0; i < hbsr_row_ptr.size(); i++)
    //     {
    //         std::cout << hbsr_row_ptr[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hbsr_col_ind" << std::endl;
    //     for(size_t i = 0; i < hbsr_col_ind.size(); i++)
    //     {
    //         std::cout << hbsr_col_ind[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hbsr_val" << std::endl;
    //     for(size_t i = 0; i < hbsr_val.size(); i++)
    //     {
    //         std::cout << hbsr_val[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }

    // {
    //     std::cout << "BBBBBBBBBBBBBBBBBBBBBBBBBBBBB" << std::endl;
    //     rocsparse_seedrand();
    //     rocsparse_matrix_factory<T> matrix_factory(arg);

    //     rocsparse_int Mb = (arg.M + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int Nb = (arg.N + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int block_dim = arg.block_dim;
    //     rocsparse_index_base base = arg.baseA;
    //     rocsparse_direction dir = arg.direction;

    //     host_gebsr_matrix<T>   hA(dir, Mb, Nb, 0, block_dim, block_dim, base);
    //     matrix_factory.init_gebsr(hA, bsr_construction_alg::convert_csr);

    //     std::cout << "hA.mb: " << hA.mb << " hA.nb: " << hA.nb << " hA.nnzb: " << hA.nnzb << std::endl;
    //     std::cout << "hA.row_block_dim: " << hA.row_block_dim << " hA.col_block_dim: " << hA.col_block_dim << std::endl;
    //     std::cout << "hA.ptr.size(): " << hA.ptr.size() << std::endl;
    //     std::cout << "hA.ind.size(): " << hA.ind.size() << std::endl;
    //     std::cout << "hA.val.size(): " << hA.val.size() << std::endl;

    //     std::cout << "BSR B" << std::endl;

    //     std::vector<T> hdense(hA.mb * block_dim * hA.nb * block_dim, 0);
    //     for(int i = 0; i < hA.mb; i++)
    //     {
    //         int start = hA.ptr[i] - base;
    //         int end = hA.ptr[i + 1] - base;

    //         for(int j = start; j < end; j++)
    //         {
    //             int bcol = hA.ind[j] - base;

    //             for(int r = 0; r < block_dim; r++)
    //             {
    //                 for(int c = 0; c < block_dim; c++)
    //                 {
    //                     int row = block_dim * i + r;
    //                     int col = block_dim * bcol + c;

    //                     hdense[Nb * block_dim * row + col] = hA.val[block_dim * block_dim * j + block_dim * r + c];
    //                 }
    //             }
    //         }
    //     }

    //     std::cout << "hdense" << std::endl;
    //     for(int row = 0; row < block_dim * hA.mb; row++)
    //     {
    //         for(int col = 0; col < block_dim * hA.mb; col++)
    //         {
    //             std::cout << hdense[hA.nb * block_dim * row + col] << " ";
    //         }
    //         std::cout << "" << std::endl;
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hA.ptr" << std::endl;
    //     for(size_t i = 0; i < hA.ptr.size(); i++)
    //     {
    //         std::cout << hA.ptr[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hA.ind" << std::endl;
    //     for(size_t i = 0; i < hA.ind.size(); i++)
    //     {
    //         std::cout << hA.ind[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hA.val" << std::endl;
    //     for(size_t i = 0; i < hA.val.size(); i++)
    //     {
    //         std::cout << hA.val[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }

    // {
    //     std::cout << "CCCCCCCCCCCCCCCCCCCCCCCCCCCCC" << std::endl;
    //     rocsparse_seedrand();
    //     rocsparse_matrix_factory<T> matrix_factory(arg);

    //     rocsparse_int Mb = (arg.M + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int Nb = (arg.N + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int block_dim = arg.block_dim;
    //     rocsparse_index_base base = arg.baseA;
    //     rocsparse_direction dir = arg.direction;

    //     std::cout << "1" << std::endl;
    //     host_gebsr_matrix<T>   hA(dir, Mb, Nb, 0, block_dim, block_dim, base);
    //     std::cout << "2" << std::endl;
    //     matrix_factory.init_gebsr(hA,
    //                                 Mb,
    //                                 Nb,
    //                                 block_dim,
    //                                 block_dim,
    //                                 base,
    //                                 bsr_construction_alg::convert_csr);

    //     std::cout << "BSR C" << std::endl;
    //     std::vector<T> hdense(Mb * block_dim * Nb * block_dim, 0);
    //     for(int i = 0; i < Mb; i++)
    //     {
    //         int start = hA.ptr[i] - base;
    //         int end = hA.ptr[i + 1] - base;

    //         for(int j = start; j < end; j++)
    //         {
    //             int bcol = hA.ind[j] - base;

    //             for(int r = 0; r < block_dim; r++)
    //             {
    //                 for(int c = 0; c < block_dim; c++)
    //                 {
    //                     int row = block_dim * i + r;
    //                     int col = block_dim * bcol + c;
    //                     hdense[Nb * block_dim * row + col] = hA.val[block_dim * block_dim * j + block_dim * r + c];
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hdense" << std::endl;
    //     for(int row = 0; row < block_dim * Mb; row++)
    //     {
    //         for(int col = 0; col < block_dim * Mb; col++)
    //         {
    //             std::cout << hdense[Nb * block_dim * row + col] << " ";
    //         }
    //         std::cout << "" << std::endl;
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hA.ptr" << std::endl;
    //     for(size_t i = 0; i < hA.ptr.size(); i++)
    //     {
    //         std::cout << hA.ptr[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hA.ind" << std::endl;
    //     for(size_t i = 0; i < hA.ind.size(); i++)
    //     {
    //         std::cout << hA.ind[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hA.val" << std::endl;
    //     for(size_t i = 0; i < hA.val.size(); i++)
    //     {
    //         std::cout << hA.val[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }

    // {
    //     std::cout << "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD" << std::endl;
    //     rocsparse_seedrand();
    //     rocsparse_matrix_factory<T> matrix_factory(arg);

    //     rocsparse_int Mb = (arg.M + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int Nb = (arg.N + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int block_dim = arg.block_dim;
    //     rocsparse_index_base base = arg.baseA;
    //     rocsparse_direction dir = arg.direction;

    //     rocsparse_int nnzb = 0;
    //     host_gebsr_matrix<T>   hA(dir, Mb, Nb, nnzb, block_dim, block_dim, base);
        
    //     matrix_factory.init_gebsr(hA,
    //                             dir,
    //                             Mb,
    //                             Nb,
    //                             nnzb,
    //                             block_dim,
    //                             block_dim,
    //                             base,
    //                             bsr_construction_alg::convert_csr);

    //     std::cout << "BSR D" << std::endl;
    //     std::vector<T> hdense(Mb * block_dim * Nb * block_dim, 0);
    //     for(int i = 0; i < Mb; i++)
    //     {
    //         int start = hA.ptr[i] - base;
    //         int end = hA.ptr[i + 1] - base;

    //         for(int j = start; j < end; j++)
    //         {
    //             int bcol = hA.ind[j] - base;

    //             for(int r = 0; r < block_dim; r++)
    //             {
    //                 for(int c = 0; c < block_dim; c++)
    //                 {
    //                     int row = block_dim * i + r;
    //                     int col = block_dim * bcol + c;
    //                     hdense[Nb * block_dim * row + col] = hA.val[block_dim * block_dim * j + block_dim * r + c];
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hdense" << std::endl;
    //     for(int row = 0; row < block_dim * Mb; row++)
    //     {
    //         for(int col = 0; col < block_dim * Mb; col++)
    //         {
    //             std::cout << hdense[Nb * block_dim * row + col] << " ";
    //         }
    //         std::cout << "" << std::endl;
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hA.ptr" << std::endl;
    //     for(size_t i = 0; i < hA.ptr.size(); i++)
    //     {
    //         std::cout << hA.ptr[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hA.ind" << std::endl;
    //     for(size_t i = 0; i < hA.ind.size(); i++)
    //     {
    //         std::cout << hA.ind[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hA.val" << std::endl;
    //     for(size_t i = 0; i < hA.val.size(); i++)
    //     {
    //         std::cout << hA.val[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }

    // {
    //     std::cout << "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE" << std::endl;
    //     rocsparse_seedrand();
    //     rocsparse_matrix_factory<T> matrix_factory(arg);

    //     rocsparse_int Mb = (arg.M + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int Nb = (arg.N + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int block_dim = arg.block_dim;
    //     rocsparse_index_base base = arg.baseA;
    //     rocsparse_direction dir = arg.direction;

    //     rocsparse_int nnzb;
    //     std::vector<rocsparse_int> hbsr_row_ptr;
    //     std::vector<rocsparse_int> hbsr_col_ind;
    //     std::vector<T> hbsr_val;
    //     matrix_factory.init_bsr(hbsr_row_ptr,
    //                             hbsr_col_ind,
    //                             hbsr_val,
    //                             dir,
    //                             Mb,
    //                             Nb,
    //                             nnzb,
    //                             block_dim,
    //                             base,
    //                             bsr_construction_alg::convert_csr);

    //     std::cout << "BSR E" << std::endl;
    //     std::vector<T> hdense(Mb * block_dim * Nb * block_dim, 0);
    //     for(int i = 0; i < Mb; i++)
    //     {
    //         int start = hbsr_row_ptr[i] - base;
    //         int end = hbsr_row_ptr[i + 1] - base;

    //         for(int j = start; j < end; j++)
    //         {
    //             int bcol = hbsr_col_ind[j] - base;

    //             for(int r = 0; r < block_dim; r++)
    //             {
    //                 for(int c = 0; c < block_dim; c++)
    //                 {
    //                     int row = block_dim * i + r;
    //                     int col = block_dim * bcol + c;
    //                     hdense[Nb * block_dim * row + col] = hbsr_val[block_dim * block_dim * j + block_dim * r + c];
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hdense" << std::endl;
    //     for(int row = 0; row < block_dim * Mb; row++)
    //     {
    //         for(int col = 0; col < block_dim * Mb; col++)
    //         {
    //             std::cout << hdense[Nb * block_dim * row + col] << " ";
    //         }
    //         std::cout << "" << std::endl;
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hbsr_row_ptr" << std::endl;
    //     for(size_t i = 0; i < hbsr_row_ptr.size(); i++)
    //     {
    //         std::cout << hbsr_row_ptr[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hbsr_col_ind" << std::endl;
    //     for(size_t i = 0; i < hbsr_col_ind.size(); i++)
    //     {
    //         std::cout << hbsr_col_ind[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hbsr_val" << std::endl;
    //     for(size_t i = 0; i < hbsr_val.size(); i++)
    //     {
    //         std::cout << hbsr_val[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }

    // {
    //     std::cout << "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" << std::endl;
    //     rocsparse_seedrand();
    //     rocsparse_matrix_factory<T> matrix_factory(arg);

    //     rocsparse_int Mb = (arg.M + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int Nb = (arg.N + arg.block_dim - 1) / arg.block_dim;
    //     rocsparse_int block_dim = arg.block_dim;
    //     rocsparse_index_base base = arg.baseA;
    //     rocsparse_direction dir = arg.direction;

    //     host_gebsr_matrix<T>   hA;
    //     device_gebsr_matrix<T> dA;

    //     std::cout << "Mb: " << Mb << " Nb: " << Nb << std::endl;
        
    //     matrix_factory.init_bsr(hA,
    //                             dA,
    //                             Mb,
    //                             Nb,
    //                             base,
    //                             bsr_construction_alg::convert_csr);

    //     std::cout << "Mb: " << Mb << " Nb: " << Nb << std::endl;

    //     std::cout << "BSR F" << std::endl;
    //     std::vector<T> hdense(Mb * block_dim * Nb * block_dim, 0);
    //     for(int i = 0; i < Mb; i++)
    //     {
    //         int start = hA.ptr[i] - base;
    //         int end = hA.ptr[i + 1] - base;

    //         for(int j = start; j < end; j++)
    //         {
    //             int bcol = hA.ind[j] - base;

    //             for(int r = 0; r < block_dim; r++)
    //             {
    //                 for(int c = 0; c < block_dim; c++)
    //                 {
    //                     int row = block_dim * i + r;
    //                     int col = block_dim * bcol + c;
    //                     hdense[Nb * block_dim * row + col] = hA.val[block_dim * block_dim * j + block_dim * r + c];
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hdense" << std::endl;
    //     for(int row = 0; row < block_dim * Mb; row++)
    //     {
    //         for(int col = 0; col < block_dim * Mb; col++)
    //         {
    //             std::cout << hdense[Nb * block_dim * row + col] << " ";
    //         }
    //         std::cout << "" << std::endl;
    //     }
    //     std::cout << "" << std::endl;

    //     std::cout << "hA.ptr" << std::endl;
    //     for(size_t i = 0; i < hA.ptr.size(); i++)
    //     {
    //         std::cout << hA.ptr[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hA.ind" << std::endl;
    //     for(size_t i = 0; i < hA.ind.size(); i++)
    //     {
    //         std::cout << hA.ind[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    //     std::cout << "hA.val" << std::endl;
    //     for(size_t i = 0; i < hA.val.size(); i++)
    //     {
    //         std::cout << hA.val[i] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }
}



template <typename T>
void testing_bsr2csr(const Arguments& arg)
{
    CHECK_ROCSPARSE_ERROR(check_bsr_generation<T>(arg));





    rocsparse_int        M         = arg.M;
    rocsparse_int        N         = arg.N;
    rocsparse_index_base bsr_base  = arg.baseA;
    rocsparse_index_base csr_base  = arg.baseB;
    rocsparse_int        block_dim = arg.block_dim;

    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    rocsparse_local_mat_descr bsr_descr;
    rocsparse_local_mat_descr csr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, bsr_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));

    rocsparse_matrix_factory<T> matrix_factory(arg);

    // Declare and initialize matrices.
    host_gebsr_matrix<T>   hA;
    device_gebsr_matrix<T> dA;

    matrix_factory.init_bsr(hA, dA, Mb, Nb, bsr_base, bsr_construction_alg::convert_csr);

    M = dA.mb * dA.row_block_dim;
    N = dA.nb * dA.col_block_dim;

    rocsparse_int nnzb = hA.ind.size();

    // Allocate device memory for output CSR matrix
    device_csr_matrix<T> dC(M, N, size_t(nnzb) * block_dim * block_dim, csr_base);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsr2csr<T>(handle,
                                                            dA.block_direction,
                                                            dA.mb,
                                                            dA.nb,
                                                            bsr_descr,
                                                            dA.val,
                                                            dA.ptr,
                                                            dA.ind,
                                                            block_dim,
                                                            csr_descr,
                                                            dC.val,
                                                            dC.ptr,
                                                            dC.ind));

        host_csr_matrix<T> hC_gold(M, N, size_t(nnzb) * block_dim * block_dim, csr_base);
        host_bsr_to_csr<T>(hA.block_direction,
                           hA.mb,
                           hA.nb,
                           hA.nnzb,
                           hA.val,
                           hA.ptr,
                           hA.ind,
                           block_dim,
                           hA.base,
                           hC_gold.val,
                           hC_gold.ptr,
                           hC_gold.ind,
                           hC_gold.base);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC", dC);
        }

        hC_gold.near_check(dC);
    }

    if(arg.timing)
    {
        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_bsr2csr<T>,
                                                                      handle,
                                                                      dA.block_direction,
                                                                      dA.mb,
                                                                      dA.nb,
                                                                      bsr_descr,
                                                                      dA.val,
                                                                      dA.ptr,
                                                                      dA.ind,
                                                                      block_dim,
                                                                      csr_descr,
                                                                      dC.val,
                                                                      dC.ptr,
                                                                      dC.ind);

        double gbyte_count = bsr2csr_gbyte_count<T>(Mb, block_dim, nnzb);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::Mb,
                            Mb,
                            display_key_t::Nb,
                            Nb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::nnzb,
                            nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_bsr2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsr2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
#undef INSTANTIATE
void testing_bsr2csr_extra(const Arguments& arg) {}
