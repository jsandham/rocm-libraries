/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_cscsort.hpp"
#include "rocsparse_csrsort.hpp"
#include "rocsparse_utility.hpp"

// Sorting the row indices within each column of A is sorting the column indices within each
// row of the transpose of A, stored in CSR format.

rocsparse_status rocsparse::cscsort_buffer_size(rocsparse_handle      handle,
                                                rocsparse_cscsort_alg alg,
                                                int64_t               m,
                                                int64_t               n,
                                                int64_t               nnz,
                                                rocsparse_indextype   csc_col_ptr_indextype_A,
                                                const void*           csc_col_ptr_A,
                                                rocsparse_indextype   csc_row_indextype_A,
                                                const void*           csc_row_ind_A,
                                                rocsparse_datatype    csc_val_datatype_A,
                                                const void*           csc_val_A,
                                                rocsparse_indextype   csc_col_ptr_indextype_B,
                                                const void*           csc_col_ptr_B,
                                                rocsparse_indextype   csc_row_indextype_B,
                                                const void*           csc_row_ind_B,
                                                rocsparse_datatype    csc_val_datatype_B,
                                                const void*           csc_val_B,
                                                size_t*               buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_buffer_size(handle,
                                                             rocsparse::rocsparse_csrsort_alg_default,
                                                             n,
                                                             m,
                                                             nnz,
                                                             csc_col_ptr_indextype_A,
                                                             csc_col_ptr_A,
                                                             csc_row_indextype_A,
                                                             csc_row_ind_A,
                                                             csc_val_datatype_A,
                                                             csc_val_A,
                                                             csc_col_ptr_indextype_B,
                                                             csc_col_ptr_B,
                                                             csc_row_indextype_B,
                                                             csc_row_ind_B,
                                                             csc_val_datatype_B,
                                                             csc_val_B,
                                                             buffer_size));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsort(rocsparse_handle      handle,
                                    rocsparse_cscsort_alg alg,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               nnz,
                                    int64_t               batch_count_A,
                                    int64_t               offsets_batch_stride_A,
                                    int64_t               rows_values_batch_stride_A,
                                    rocsparse_index_base  idx_base_A,
                                    rocsparse_indextype   csc_col_ptr_indextype_A,
                                    const void*           csc_col_ptr_A,
                                    rocsparse_indextype   csc_row_indextype_A,
                                    const void*           csc_row_ind_A,
                                    rocsparse_datatype    csc_val_datatype_A,
                                    const void*           csc_val_A,
                                    int64_t               batch_count_B,
                                    int64_t               offsets_batch_stride_B,
                                    int64_t               rows_values_batch_stride_B,
                                    rocsparse_index_base  idx_base_B,
                                    rocsparse_indextype   csc_col_ptr_indextype_B,
                                    void*                 csc_col_ptr_B,
                                    rocsparse_indextype   csc_row_indextype_B,
                                    void*                 csc_row_ind_B,
                                    rocsparse_datatype    csc_val_datatype_B,
                                    void*                 csc_val_B,
                                    void*                 temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort(handle,
                                                 rocsparse::rocsparse_csrsort_alg_default,
                                                 n,
                                                 m,
                                                 nnz,
                                                 batch_count_A,
                                                 offsets_batch_stride_A,
                                                 rows_values_batch_stride_A,
                                                 idx_base_A,
                                                 csc_col_ptr_indextype_A,
                                                 csc_col_ptr_A,
                                                 csc_row_indextype_A,
                                                 csc_row_ind_A,
                                                 csc_val_datatype_A,
                                                 csc_val_A,
                                                 batch_count_B,
                                                 offsets_batch_stride_B,
                                                 rows_values_batch_stride_B,
                                                 idx_base_B,
                                                 csc_col_ptr_indextype_B,
                                                 csc_col_ptr_B,
                                                 csc_row_indextype_B,
                                                 csc_row_ind_B,
                                                 csc_val_datatype_B,
                                                 csc_val_B,
                                                 temp_buffer));
    return rocsparse_status_success;
}
