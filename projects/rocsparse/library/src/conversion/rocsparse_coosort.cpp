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
}

rocsparse_status rocsparse::coosort_buffer_size(rocsparse_handle      handle,
                                                rocsparse_coosort_alg alg,
                                                rocsparse_direction   dir,
                                                int64_t               m,
                                                int64_t               n,
                                                int64_t               nnz,
                                                rocsparse_indextype   coo_row_indextype,
                                                const void*           coo_row_ind,
                                                rocsparse_indextype   coo_col_indextype,
                                                const void*           coo_col_ind,
                                                rocsparse_datatype    coo_val_datatype,
                                                const void*           coo_val,
                                                size_t*               buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::coosort_buffer_size_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_buffer_size_find(&f, coo_row_indextype));

    size_t sort_buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(f(handle, m, n, nnz, coo_row_ind, coo_col_ind, &sort_buffer_size));

    // The sorted values are gathered into the scratch space before being copied back.
    const size_t gather_buffer_size
        = rocsparse::align_size<char>(rocsparse::datatype_sizeof(coo_val_datatype) * nnz);

    *buffer_size = rocsparse::coosort_perm_size(nnz, coo_row_indextype)
                   + rocsparse::max(sort_buffer_size, gather_buffer_size);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::coosort(rocsparse_handle      handle,
                                    rocsparse_coosort_alg alg,
                                    rocsparse_direction   dir,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               nnz,
                                    rocsparse_indextype   coo_row_indextype,
                                    void*                 coo_row_ind,
                                    rocsparse_indextype   coo_col_indextype,
                                    void*                 coo_col_ind,
                                    rocsparse_datatype    coo_val_datatype,
                                    void*                 coo_val,
                                    void*                 temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_indextype perm_indextype = coo_row_indextype;

    void* perm = temp_buffer;
    void* sort_buffer
        = reinterpret_cast<char*>(temp_buffer) + rocsparse::coosort_perm_size(nnz, perm_indextype);

    // The index sort applies its reordering to perm, so it must start as the identity.
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gcreate_identity_permutation(handle, nnz, perm_indextype, perm));

    switch(dir)
    {
    case rocsparse_direction_row:
    {
        rocsparse::coosort_by_row_t f;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_row_find(&f, coo_row_indextype));
        RETURN_IF_ROCSPARSE_ERROR(
            f(handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, sort_buffer));
        break;
    }

    case rocsparse_direction_column:
    {
        rocsparse::coosort_by_column_t f;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_column_find(&f, coo_col_indextype));
        RETURN_IF_ROCSPARSE_ERROR(
            f(handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, sort_buffer));
        break;
    }

    // LCOV_EXCL_START
    default:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
        // LCOV_EXCL_STOP
    }

    void* sorted_val = sort_buffer;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthr(handle,
                                              nnz,
                                              coo_val_datatype,
                                              coo_val,
                                              coo_val_datatype,
                                              sorted_val,
                                              perm_indextype,
                                              perm,
                                              rocsparse_index_base_zero));

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(coo_val,
                                       sorted_val,
                                       rocsparse::datatype_sizeof(coo_val_datatype) * nnz,
                                       hipMemcpyDeviceToDevice,
                                       handle->stream));

    return rocsparse_status_success;
}