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

#include "rocsparse_coosort.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "rocsparse_gcreate_identity_permutation.hpp"
#include "rocsparse_utility.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <tuple>

namespace rocsparse
{
    typedef rocsparse_status (*coosort_buffer_size_t)(
        rocsparse_handle, int64_t, int64_t, int64_t, const void*, const void*, size_t*);

    using coosort_buffer_size_tuple = std::tuple<rocsparse_indextype>;

    // clang-format off
#define COOSORT_BUFFER_SIZE_CONFIG(J)                                         \
{                                                                       \
    coosort_buffer_size_tuple(J), coosort_buffer_size_template<typename rocsparse::indextype_traits<J>::type_t>  \
}
    // clang-format on

    static const std::map<coosort_buffer_size_tuple, coosort_buffer_size_t>
        s_coosort_buffer_size_dispatch{{
            COOSORT_BUFFER_SIZE_CONFIG(rocsparse_indextype_i32),
            COOSORT_BUFFER_SIZE_CONFIG(rocsparse_indextype_i64),
        }};

    static rocsparse_status coosort_buffer_size_find(coosort_buffer_size_t* function_,
                                                     rocsparse_indextype    j_type_)
    {

        const auto& it = rocsparse::s_coosort_buffer_size_dispatch.find(
            rocsparse::coosort_buffer_size_tuple(j_type_));

        if(it != rocsparse::s_coosort_buffer_size_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {

#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_coosort_buffer_size_dispatch)
            {
                const auto& t      = p.first;
                const auto  j_type = std::get<0>(t);
                std::cout << std::endl
                          << std::endl
                          << "j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "j_type: " << rocsparse::enum_utils::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

    typedef rocsparse_status (*coosort_by_row_t)(
        rocsparse_handle, int64_t, int64_t, int64_t, void*, void*, void*, void*);

    using coosort_by_row_tuple = std::tuple<rocsparse_indextype>;

    // clang-format off
#define COOSORT_BY_ROW_CONFIG(J)                                         \
{                                                                       \
    coosort_by_row_tuple(J), coosort_by_row_template<typename rocsparse::indextype_traits<J>::type_t>  \
}
    // clang-format on

    static const std::map<coosort_by_row_tuple, coosort_by_row_t> s_coosort_by_row_dispatch{{
        COOSORT_BY_ROW_CONFIG(rocsparse_indextype_i32),
        COOSORT_BY_ROW_CONFIG(rocsparse_indextype_i64),
    }};

    static rocsparse_status coosort_by_row_find(coosort_by_row_t*   function_,
                                                rocsparse_indextype j_type_)
    {

        const auto& it
            = rocsparse::s_coosort_by_row_dispatch.find(rocsparse::coosort_by_row_tuple(j_type_));

        if(it != rocsparse::s_coosort_by_row_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {

#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_coosort_by_row_dispatch)
            {
                const auto& t      = p.first;
                const auto  j_type = std::get<0>(t);
                std::cout << std::endl
                          << std::endl
                          << "j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "j_type: " << rocsparse::enum_utils::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

    typedef rocsparse_status (*coosort_by_column_t)(
        rocsparse_handle, int64_t, int64_t, int64_t, void*, void*, void*, void*);

    using coosort_by_column_tuple = std::tuple<rocsparse_indextype>;

    // clang-format off
#define COOSORT_BY_COLUMN_CONFIG(J)                                         \
{                                                                       \
    coosort_by_column_tuple(J), coosort_by_column_template<typename rocsparse::indextype_traits<J>::type_t>  \
}
    // clang-format on

    static const std::map<coosort_by_column_tuple, coosort_by_column_t>
        s_coosort_by_column_dispatch{{
            COOSORT_BY_COLUMN_CONFIG(rocsparse_indextype_i32),
            COOSORT_BY_COLUMN_CONFIG(rocsparse_indextype_i64),
        }};

    static rocsparse_status coosort_by_column_find(coosort_by_column_t* function_,
                                                   rocsparse_indextype  j_type_)
    {

        const auto& it = rocsparse::s_coosort_by_column_dispatch.find(
            rocsparse::coosort_by_column_tuple(j_type_));

        if(it != rocsparse::s_coosort_by_column_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {

#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_coosort_by_column_dispatch)
            {
                const auto& t      = p.first;
                const auto  j_type = std::get<0>(t);
                std::cout << std::endl
                          << std::endl
                          << "j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "j_type: " << rocsparse::enum_utils::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

    // The buffer starts with the permutation array, which tracks where each entry moves
    // while the indices are sorted, followed by scratch space shared by the index sort
    // and the value permutation.
    static size_t coosort_perm_size(int64_t nnz, rocsparse_indextype perm_indextype)
    {
        return rocsparse::align_size<char>(rocsparse::indextype_sizeof(perm_indextype) * nnz);
    }

    // The indices are copied and the values gathered without any type conversion.
    static rocsparse_status coosort_check_types(rocsparse_indextype coo_row_indextype_A,
                                                rocsparse_indextype coo_col_indextype_A,
                                                rocsparse_datatype  coo_val_datatype_A,
                                                rocsparse_indextype coo_row_indextype_B,
                                                rocsparse_indextype coo_col_indextype_B,
                                                rocsparse_datatype  coo_val_datatype_B)
    {
        if(coo_row_indextype_B != coo_row_indextype_A
           || coo_col_indextype_B != coo_col_indextype_A)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the index types of the output matrix must match the index types of the input "
                "matrix");
        }
        if(coo_val_datatype_B != coo_val_datatype_A)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the data type of the output matrix must match the data type of the input matrix");
        }
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::coosort_buffer_size(rocsparse_handle      handle,
                                                rocsparse_coosort_alg alg,
                                                rocsparse_direction   dir,
                                                int64_t               m,
                                                int64_t               n,
                                                int64_t               nnz,
                                                rocsparse_indextype   coo_row_indextype_A,
                                                const void*           coo_row_ind_A,
                                                rocsparse_indextype   coo_col_indextype_A,
                                                const void*           coo_col_ind_A,
                                                rocsparse_datatype    coo_val_datatype_A,
                                                const void*           coo_val_A,
                                                rocsparse_indextype   coo_row_indextype_B,
                                                const void*           coo_row_ind_B,
                                                rocsparse_indextype   coo_col_indextype_B,
                                                const void*           coo_col_ind_B,
                                                rocsparse_datatype    coo_val_datatype_B,
                                                const void*           coo_val_B,
                                                size_t*               buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_check_types(coo_row_indextype_A,
                                                             coo_col_indextype_A,
                                                             coo_val_datatype_A,
                                                             coo_row_indextype_B,
                                                             coo_col_indextype_B,
                                                             coo_val_datatype_B));

    rocsparse::coosort_buffer_size_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_buffer_size_find(&f, coo_row_indextype_B));

    size_t sort_buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        f(handle, m, n, nnz, coo_row_ind_B, coo_col_ind_B, &sort_buffer_size));

    // Values sorted in place are gathered into scratch space first, since the gather cannot
    // write over its own input.
    const size_t gather_buffer_size
        = (coo_val_B == coo_val_A)
              ? rocsparse::align_size<char>(rocsparse::datatype_sizeof(coo_val_datatype_B) * nnz)
              : 0;

    *buffer_size = rocsparse::coosort_perm_size(nnz, coo_row_indextype_B)
                   + rocsparse::max(sort_buffer_size, gather_buffer_size);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::coosort(rocsparse_handle      handle,
                                    rocsparse_coosort_alg alg,
                                    rocsparse_direction   dir,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               nnz,
                                    int64_t               batch_count_A,
                                    int64_t               batch_stride_A,
                                    rocsparse_indextype   coo_row_indextype_A,
                                    const void*           coo_row_ind_A,
                                    rocsparse_indextype   coo_col_indextype_A,
                                    const void*           coo_col_ind_A,
                                    rocsparse_datatype    coo_val_datatype_A,
                                    const void*           coo_val_A,
                                    int64_t               batch_count_B,
                                    int64_t               batch_stride_B,
                                    rocsparse_indextype   coo_row_indextype_B,
                                    void*                 coo_row_ind_B,
                                    rocsparse_indextype   coo_col_indextype_B,
                                    void*                 coo_col_ind_B,
                                    rocsparse_datatype    coo_val_datatype_B,
                                    void*                 coo_val_B,
                                    void*                 temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_check_types(coo_row_indextype_A,
                                                             coo_col_indextype_A,
                                                             coo_val_datatype_A,
                                                             coo_row_indextype_B,
                                                             coo_col_indextype_B,
                                                             coo_val_datatype_B));

    if(batch_count_B != batch_count_A)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the batch count of the output matrix must match the batch count of the input matrix");
    }

    const rocsparse_indextype perm_indextype = coo_row_indextype_B;

    const size_t row_size_A = rocsparse::indextype_sizeof(coo_row_indextype_A);
    const size_t col_size_A = rocsparse::indextype_sizeof(coo_col_indextype_A);
    const size_t val_size_A = rocsparse::datatype_sizeof(coo_val_datatype_A);
    const size_t row_size_B = rocsparse::indextype_sizeof(coo_row_indextype_B);
    const size_t col_size_B = rocsparse::indextype_sizeof(coo_col_indextype_B);
    const size_t val_size_B = rocsparse::datatype_sizeof(coo_val_datatype_B);

    void* perm = temp_buffer;
    void* sort_buffer
        = reinterpret_cast<char*>(temp_buffer) + rocsparse::coosort_perm_size(nnz, perm_indextype);

    // The batches run one after the other on the handle stream, so they share temp_buffer.
    for(int64_t batch = 0; batch < batch_count_A; ++batch)
    {
        const int64_t offset_A = batch * batch_stride_A;
        const int64_t offset_B = batch * batch_stride_B;

        const void* row_ind_A = reinterpret_cast<const char*>(coo_row_ind_A) + offset_A * row_size_A;
        const void* col_ind_A = reinterpret_cast<const char*>(coo_col_ind_A) + offset_A * col_size_A;
        const void* val_A     = reinterpret_cast<const char*>(coo_val_A) + offset_A * val_size_A;
        void*       row_ind_B = reinterpret_cast<char*>(coo_row_ind_B) + offset_B * row_size_B;
        void*       col_ind_B = reinterpret_cast<char*>(coo_col_ind_B) + offset_B * col_size_B;
        void*       val_B     = reinterpret_cast<char*>(coo_val_B) + offset_B * val_size_B;

        // The index sort works in place, so the indices of A are first copied into B.
        if(row_ind_B != row_ind_A)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(row_ind_B,
                                               row_ind_A,
                                               row_size_A * nnz,
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

        // The index sort applies its reordering to perm, so it must start as the identity.
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gcreate_identity_permutation(handle, nnz, perm_indextype, perm));

        switch(dir)
        {
        case rocsparse_direction_row:
        {
            rocsparse::coosort_by_row_t f;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_row_find(&f, coo_row_indextype_B));
            RETURN_IF_ROCSPARSE_ERROR(f(handle, m, n, nnz, row_ind_B, col_ind_B, perm, sort_buffer));
            break;
        }

        case rocsparse_direction_column:
        {
            rocsparse::coosort_by_column_t f;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::coosort_by_column_find(&f, coo_col_indextype_B));
            RETURN_IF_ROCSPARSE_ERROR(f(handle, m, n, nnz, row_ind_B, col_ind_B, perm, sort_buffer));
            break;
        }

        // LCOV_EXCL_START
        default:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
            // LCOV_EXCL_STOP
        }

        // The gather cannot write over its own input, so in place values go through scratch.
        const bool in_place_val = (val_B == val_A);
        void*      sorted_val   = in_place_val ? sort_buffer : val_B;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthr(handle,
                                                  nnz,
                                                  coo_val_datatype_A,
                                                  val_A,
                                                  coo_val_datatype_B,
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
