/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_permute.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "rocsparse_utility.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <tuple>

#include <vector>

template <typename T>
rocsparse_status rocsparse::permute_buffer_size_template(rocsparse_handle handle,
                                                         int64_t          nnz,
                                                         const void*      values,
                                                         size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(1, nnz);

    ROCSPARSE_CHECKARG_POINTER(3, buffer_size);

    // Quick return if possible
    if(nnz == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(2, nnz, values);

    *buffer_size = ((sizeof(T) * nnz - 1) / 256 + 1) * 256;

    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse::permute_template(
    rocsparse_handle handle, int64_t nnz, void* values, const void* perm, void* temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // Permutation vector given
    T* permuted_values = reinterpret_cast<T*>(ptr);
    ptr += ((sizeof(T) * nnz - 1) / 256 + 1) * 256;

    RETURN_IF_ROCSPARSE_ERROR((gthr_template<I, T>(
        handle, nnz, values, permuted_values, perm, rocsparse_index_base_zero)));

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        values, permuted_values, sizeof(T) * nnz, hipMemcpyDeviceToDevice, handle->stream));

    return rocsparse_status_success;
}

#define INSTANTIATE_BUFFER_SIZE(T)                                        \
    template rocsparse_status rocsparse::permute_buffer_size_template<T>( \
        rocsparse_handle, int64_t, const void*, size_t* buffer_size)

INSTANTIATE_BUFFER_SIZE(float);
INSTANTIATE_BUFFER_SIZE(double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex);
#undef INSTANTIATE_BUFFER_SIZE

#define INSTANTIATE(T, I)                                        \
    template rocsparse_status rocsparse::permute_template<T, I>( \
        rocsparse_handle, int64_t, void*, const void*, void*)

INSTANTIATE(float, int32_t);
INSTANTIATE(float, int64_t);
INSTANTIATE(double, int32_t);
INSTANTIATE(double, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t);
#undef INSTANTIATE

namespace rocsparse
{
    typedef rocsparse_status (*permute_buffer_size_t)(rocsparse_handle,
                                                      int64_t,
                                                      const void*,
                                                      size_t*);

    using permute_buffer_size_tuple = std::tuple<rocsparse_datatype>;

    // clang-format off
#define PERMUTE_BUFFER_SIZE_CONFIG(T)                                                              \
    {                                                                                              \
        permute_buffer_size_tuple(T),                                                              \
            permute_buffer_size_template<typename rocsparse::datatype_traits<T>::type_t>           \
    }
    // clang-format on

    static const std::map<permute_buffer_size_tuple, permute_buffer_size_t>
        s_permute_buffer_size_dispatch{{
            PERMUTE_BUFFER_SIZE_CONFIG(rocsparse_datatype_f32_r),
            PERMUTE_BUFFER_SIZE_CONFIG(rocsparse_datatype_f64_r),
            PERMUTE_BUFFER_SIZE_CONFIG(rocsparse_datatype_f32_c),
            PERMUTE_BUFFER_SIZE_CONFIG(rocsparse_datatype_f64_c),
        }};

    static rocsparse_status permute_buffer_size_find(permute_buffer_size_t* function_,
                                                     rocsparse_datatype     t_type_)
    {

        const auto& it = rocsparse::s_permute_buffer_size_dispatch.find(
            rocsparse::permute_buffer_size_tuple(t_type_));

        if(it != rocsparse::s_permute_buffer_size_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {

#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "t_type: " << rocsparse::enum_utils::to_string(t_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_permute_buffer_size_dispatch)
            {
                const auto& t      = p.first;
                const auto  t_type = std::get<0>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::enum_utils::to_string(t_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

    typedef rocsparse_status (*permute_t)(rocsparse_handle, int64_t, void*, const void*, void*);

    using permute_tuple = std::tuple<rocsparse_datatype, rocsparse_indextype>;

    // clang-format off
#define PERMUTE_CONFIG(T, J)                                                                       \
    {                                                                                              \
        permute_tuple(T, J),                                                                       \
            permute_template<typename rocsparse::datatype_traits<T>::type_t,                       \
                             typename rocsparse::indextype_traits<J>::type_t>                      \
    }
    // clang-format on

    static const std::map<permute_tuple, permute_t> s_permute_dispatch{{
        PERMUTE_CONFIG(rocsparse_datatype_f32_r, rocsparse_indextype_i32),
        PERMUTE_CONFIG(rocsparse_datatype_f64_r, rocsparse_indextype_i32),
        PERMUTE_CONFIG(rocsparse_datatype_f32_c, rocsparse_indextype_i32),
        PERMUTE_CONFIG(rocsparse_datatype_f64_c, rocsparse_indextype_i32),

        PERMUTE_CONFIG(rocsparse_datatype_f32_r, rocsparse_indextype_i64),
        PERMUTE_CONFIG(rocsparse_datatype_f64_r, rocsparse_indextype_i64),
        PERMUTE_CONFIG(rocsparse_datatype_f32_c, rocsparse_indextype_i64),
        PERMUTE_CONFIG(rocsparse_datatype_f64_c, rocsparse_indextype_i64),
    }};

    static rocsparse_status
        permute_find(permute_t* function_, rocsparse_datatype t_type_, rocsparse_indextype j_type_)
    {

        const auto& it
            = rocsparse::s_permute_dispatch.find(rocsparse::permute_tuple(t_type_, j_type_));

        if(it != rocsparse::s_permute_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {

#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "t_type: " << rocsparse::enum_utils::to_string(t_type_)
                      << "j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_permute_dispatch)
            {
                const auto& t      = p.first;
                const auto  t_type = std::get<0>(t);
                const auto  j_type = std::get<1>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl
                          << "j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::enum_utils::to_string(t_type_)
                 << "j_type: " << rocsparse::enum_utils::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::permute_buffer_size(rocsparse_handle   handle,
                                                int64_t            nnz,
                                                rocsparse_datatype values_datatype,
                                                const void*        values,
                                                size_t*            buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::permute_buffer_size_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::permute_buffer_size_find(&f, values_datatype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle, nnz, values, buffer_size));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::permute(rocsparse_handle    handle,
                                    int64_t             nnz,
                                    rocsparse_datatype  values_datatype,
                                    void*               values,
                                    rocsparse_indextype perm_indextype,
                                    void*               perm,
                                    void*               temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::permute_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::permute_find(&f, values_datatype, perm_indextype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle, nnz, values, perm, temp_buffer));

    return rocsparse_status_success;
}