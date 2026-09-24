/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrsort.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "rocsparse_gcreate_identity_permutation.hpp"
#include "rocsparse_utility.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <tuple>

namespace rocsparse
{
    typedef rocsparse_status (*csrsort_buffer_size_t)(
        rocsparse_handle, int64_t, int64_t, int64_t, const void*, const void*, size_t*);

    using csrsort_buffer_size_tuple = std::tuple<rocsparse_indextype, rocsparse_indextype>;

    // clang-format off
#define CSRSORT_BUFFER_SIZE_CONFIG(I, J)                                         \
{                                                                       \
    csrsort_buffer_size_tuple(I, J), csrsort_buffer_size_template<typename rocsparse::indextype_traits<I>::type_t, typename rocsparse::indextype_traits<J>::type_t>  \
}
    // clang-format on

    static const std::map<csrsort_buffer_size_tuple, csrsort_buffer_size_t>
        s_csrsort_buffer_size_dispatch{{
            CSRSORT_BUFFER_SIZE_CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i32),
            CSRSORT_BUFFER_SIZE_CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i32),
            CSRSORT_BUFFER_SIZE_CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i64),
        }};

    static rocsparse_status csrsort_buffer_size_find(csrsort_buffer_size_t* function_,
                                                     rocsparse_indextype    i_type_,
                                                     rocsparse_indextype    j_type_)
    {
        const auto& it = rocsparse::s_csrsort_buffer_size_dispatch.find(
            rocsparse::csrsort_buffer_size_tuple(i_type_, j_type_));

        if(it != rocsparse::s_csrsort_buffer_size_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "i_type: " << rocsparse::enum_utils::to_string(i_type_)
                      << ", j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_csrsort_buffer_size_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  j_type = std::get<1>(t);
                std::cout << std::endl
                          << std::endl
                          << "i_type: " << rocsparse::enum_utils::to_string(i_type)
                          << ", j_type: " << rocsparse::enum_utils::to_string(j_type)
                          << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

    typedef rocsparse_status (*csrsort_t)(rocsparse_handle,
                                          int64_t,
                                          int64_t,
                                          int64_t,
                                          rocsparse_index_base,
                                          const void*,
                                          void*,
                                          void*,
                                          void*);

    using csrsort_tuple = std::tuple<rocsparse_indextype, rocsparse_indextype>;

    // clang-format off
#define CSRSORT_CONFIG(I, J)                                         \
{                                                                       \
    csrsort_tuple(I, J), csrsort_template<typename rocsparse::indextype_traits<I>::type_t, typename rocsparse::indextype_traits<J>::type_t>  \
}
    // clang-format on

    static const std::map<csrsort_tuple, csrsort_t> s_csrsort_dispatch{{
        CSRSORT_CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i32),
        CSRSORT_CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i32),
        CSRSORT_CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i64),
    }};

    static rocsparse_status
        csrsort_find(csrsort_t* function_, rocsparse_indextype i_type_, rocsparse_indextype j_type_)
    {
        const auto& it
            = rocsparse::s_csrsort_dispatch.find(rocsparse::csrsort_tuple(i_type_, j_type_));

        if(it != rocsparse::s_csrsort_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "i_type: " << rocsparse::enum_utils::to_string(i_type_)
                      << ", j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_csrsort_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  j_type = std::get<1>(t);
                std::cout << std::endl
                          << std::endl
                          << "i_type: " << rocsparse::enum_utils::to_string(i_type)
                          << ", j_type: " << rocsparse::enum_utils::to_string(j_type)
                          << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

    // The buffer starts with the permutation array, which tracks where each entry moves
    // while the column indices are sorted, followed by scratch space shared by the index
    // sort and the value permutation.
    static size_t csrsort_perm_size(int64_t nnz, rocsparse_indextype perm_indextype)
    {
        return rocsparse::align_size<char>(rocsparse::indextype_sizeof(perm_indextype) * nnz);
    }

    // The row pointer and column indices are copied and the values gathered without any
    // type conversion.
    static rocsparse_status csrsort_check_types(rocsparse_indextype csr_row_ptr_indextype_A,
                                                rocsparse_indextype csr_col_indextype_A,
                                                rocsparse_datatype  csr_val_datatype_A,
                                                rocsparse_indextype csr_row_ptr_indextype_B,
                                                rocsparse_indextype csr_col_indextype_B,
                                                rocsparse_datatype  csr_val_datatype_B)
    {
        if(csr_row_ptr_indextype_B != csr_row_ptr_indextype_A
           || csr_col_indextype_B != csr_col_indextype_A)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the index types of the output matrix must match the index types of the input "
                "matrix");
        }
        if(csr_val_datatype_B != csr_val_datatype_A)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the data type of the output matrix must match the data type of the input matrix");
        }
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::csrsort_buffer_size(rocsparse_handle      handle,
                                                rocsparse_csrsort_alg alg,
                                                int64_t               m,
                                                int64_t               n,
                                                int64_t               nnz,
                                                rocsparse_indextype   csr_row_ptr_indextype_A,
                                                const void*           csr_row_ptr_A,
                                                rocsparse_indextype   csr_col_indextype_A,
                                                const void*           csr_col_ind_A,
                                                rocsparse_datatype    csr_val_datatype_A,
                                                const void*           csr_val_A,
                                                rocsparse_indextype   csr_row_ptr_indextype_B,
                                                const void*           csr_row_ptr_B,
                                                rocsparse_indextype   csr_col_indextype_B,
                                                const void*           csr_col_ind_B,
                                                rocsparse_datatype    csr_val_datatype_B,
                                                const void*           csr_val_B,
                                                size_t*               buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_check_types(csr_row_ptr_indextype_A,
                                                             csr_col_indextype_A,
                                                             csr_val_datatype_A,
                                                             csr_row_ptr_indextype_B,
                                                             csr_col_indextype_B,
                                                             csr_val_datatype_B));

    rocsparse::csrsort_buffer_size_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_buffer_size_find(
        &f, csr_row_ptr_indextype_B, csr_col_indextype_B));

    size_t sort_buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        f(handle, m, n, nnz, csr_row_ptr_B, csr_col_ind_B, &sort_buffer_size));

    // Values sorted in place are gathered into scratch space first, since the gather cannot
    // write over its own input.
    const size_t gather_buffer_size
        = (csr_val_B == csr_val_A)
              ? rocsparse::align_size<char>(rocsparse::datatype_sizeof(csr_val_datatype_B) * nnz)
              : 0;

    *buffer_size = rocsparse::csrsort_perm_size(nnz, csr_row_ptr_indextype_B)
                   + rocsparse::max(sort_buffer_size, gather_buffer_size);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsort(rocsparse_handle      handle,
                                    rocsparse_csrsort_alg alg,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               nnz,
                                    int64_t               batch_count_A,
                                    int64_t               offsets_batch_stride_A,
                                    int64_t               columns_values_batch_stride_A,
                                    rocsparse_index_base  idx_base_A,
                                    rocsparse_indextype   csr_row_ptr_indextype_A,
                                    const void*           csr_row_ptr_A,
                                    rocsparse_indextype   csr_col_indextype_A,
                                    const void*           csr_col_ind_A,
                                    rocsparse_datatype    csr_val_datatype_A,
                                    const void*           csr_val_A,
                                    int64_t               batch_count_B,
                                    int64_t               offsets_batch_stride_B,
                                    int64_t               columns_values_batch_stride_B,
                                    rocsparse_index_base  idx_base_B,
                                    rocsparse_indextype   csr_row_ptr_indextype_B,
                                    void*                 csr_row_ptr_B,
                                    rocsparse_indextype   csr_col_indextype_B,
                                    void*                 csr_col_ind_B,
                                    rocsparse_datatype    csr_val_datatype_B,
                                    void*                 csr_val_B,
                                    void*                 temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_check_types(csr_row_ptr_indextype_A,
                                                             csr_col_indextype_A,
                                                             csr_val_datatype_A,
                                                             csr_row_ptr_indextype_B,
                                                             csr_col_indextype_B,
                                                             csr_val_datatype_B));

    if(batch_count_B != batch_count_A)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the batch count of the output matrix must match the batch count of the input matrix");
    }

    if(idx_base_B != idx_base_A)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the index base of the output matrix must match the index base of the input matrix");
    }

    rocsparse::csrsort_t f;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrsort_find(&f, csr_row_ptr_indextype_B, csr_col_indextype_B));

    const rocsparse_indextype perm_indextype = csr_row_ptr_indextype_B;

    const size_t row_ptr_size_A = rocsparse::indextype_sizeof(csr_row_ptr_indextype_A);
    const size_t col_size_A     = rocsparse::indextype_sizeof(csr_col_indextype_A);
    const size_t val_size_A     = rocsparse::datatype_sizeof(csr_val_datatype_A);
    const size_t row_ptr_size_B = rocsparse::indextype_sizeof(csr_row_ptr_indextype_B);
    const size_t col_size_B     = rocsparse::indextype_sizeof(csr_col_indextype_B);
    const size_t val_size_B     = rocsparse::datatype_sizeof(csr_val_datatype_B);

    void* perm = temp_buffer;
    void* sort_buffer
        = reinterpret_cast<char*>(temp_buffer) + rocsparse::csrsort_perm_size(nnz, perm_indextype);

    // The batches run one after the other on the handle stream, so they share temp_buffer.
    for(int64_t batch = 0; batch < batch_count_A; ++batch)
    {
        const int64_t offsets_offset_A = batch * offsets_batch_stride_A;
        const int64_t offsets_offset_B = batch * offsets_batch_stride_B;
        const int64_t cv_offset_A      = batch * columns_values_batch_stride_A;
        const int64_t cv_offset_B      = batch * columns_values_batch_stride_B;

        const void* row_ptr_A
            = reinterpret_cast<const char*>(csr_row_ptr_A) + offsets_offset_A * row_ptr_size_A;
        const void* col_ind_A
            = reinterpret_cast<const char*>(csr_col_ind_A) + cv_offset_A * col_size_A;
        const void* val_A = reinterpret_cast<const char*>(csr_val_A) + cv_offset_A * val_size_A;
        void*       row_ptr_B
            = reinterpret_cast<char*>(csr_row_ptr_B) + offsets_offset_B * row_ptr_size_B;
        void* col_ind_B = reinterpret_cast<char*>(csr_col_ind_B) + cv_offset_B * col_size_B;
        void* val_B     = reinterpret_cast<char*>(csr_val_B) + cv_offset_B * val_size_B;

        // The column sort works in place, so the row pointer and column indices of A are
        // first copied into B. A matrix without rows may have a null row pointer.
        if(m > 0 && row_ptr_B != row_ptr_A)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(row_ptr_B,
                                               row_ptr_A,
                                               row_ptr_size_A * (m + 1),
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }
        if(col_ind_B != col_ind_A)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(col_ind_B,
                                               col_ind_A,
                                               col_size_A * nnz,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }

        // The column sort applies its reordering to perm, so it must start as the identity.
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gcreate_identity_permutation(handle, nnz, perm_indextype, perm));

        RETURN_IF_ROCSPARSE_ERROR(
            f(handle, m, n, nnz, idx_base_B, row_ptr_B, col_ind_B, perm, sort_buffer));

        // The gather cannot write over its own input, so in place values go through scratch.
        const bool in_place_val = (val_B == val_A);
        void*      sorted_val   = in_place_val ? sort_buffer : val_B;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthr(handle,
                                                  nnz,
                                                  csr_val_datatype_A,
                                                  val_A,
                                                  csr_val_datatype_B,
                                                  sorted_val,
                                                  perm_indextype,
                                                  perm,
                                                  rocsparse_index_base_zero));

        if(in_place_val)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(val_B,
                                               sorted_val,
                                               val_size_B * nnz,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }
    }

    return rocsparse_status_success;
}
