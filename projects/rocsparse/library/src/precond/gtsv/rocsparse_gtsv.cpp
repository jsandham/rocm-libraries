/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/precond/rocsparse_gtsv.h"
#include "rocsparse_grid.hpp"
#include "rocsparse_gtsv.hpp"
#include "rocsparse_gtsv_spike_solver.hpp"
#include "rocsparse_gtsv_spike_solver_small.hpp"

template <typename T>
rocsparse_status rocsparse::gtsv_buffer_size_template(rocsparse_handle handle,
                                                      rocsparse_int    m,
                                                      rocsparse_int    n,
                                                      const T*         dl,
                                                      const T*         d,
                                                      const T*         du,
                                                      const T*         B,
                                                      rocsparse_int    ldb,
                                                      size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgtsv_buffer_size"),
                         m,
                         n,
                         (const void*&)dl,
                         (const void*&)d,
                         (const void*&)du,
                         (const void*&)B,
                         ldb,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG(1, m, (m <= 1), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(7, ldb);
    ROCSPARSE_CHECKARG(7,
                       ldb,
                       ldb < rocsparse::max(static_cast<rocsparse_int>(1), m),
                       rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(3, n, dl);
    ROCSPARSE_CHECKARG_ARRAY(4, n, d);
    ROCSPARSE_CHECKARG_ARRAY(5, n, du);
    ROCSPARSE_CHECKARG_ARRAY(6, n, B);
    ROCSPARSE_CHECKARG_POINTER(8, buffer_size);

    // Quick return if possible
    if(n == 0)
    {
        buffer_size[0] = 0;
        return rocsparse_status_success;
    }

    *buffer_size = 0;

    int current_m = m;
    for(int level = 0; level < gtsv_buffer_data<T>::MAX_RECURSION_LEVELS; level++)
    {
        const int BLOCKDIM = gtsv_buffer_data<T>::BLOCKDIM;

        const int m_pad = pad_to_blockdim(current_m, BLOCKDIM);

        *buffer_size += align_size<T>(m_pad); // dl_pad
        *buffer_size += align_size<T>(m_pad); // d_pad
        *buffer_size += align_size<T>(m_pad); // du_pad
        *buffer_size += align_size<T>(m_pad * n); // B_pad
        *buffer_size += align_size<T>(m_pad); // w_pad
        *buffer_size += align_size<T>(m_pad); // v_pad
        *buffer_size += align_size<T>(m_pad); // mt_pad

        const int S_size = 2 * m_pad / BLOCKDIM;

        *buffer_size += align_size<T>(S_size); // sl
        *buffer_size += align_size<T>(S_size); // s
        *buffer_size += align_size<T>(S_size); // su
        *buffer_size += align_size<T>(S_size * n); // sB

        if(level == 0)
        {
            *buffer_size += align_size<T>(gtsv_buffer_data<T>::FUSED_V2_FACTOR_SIZE); // factor_pad
        }

        current_m = S_size;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::gtsv_template(rocsparse_handle handle,
                                          rocsparse_int    m,
                                          rocsparse_int    n,
                                          const T*         dl,
                                          const T*         d,
                                          const T*         du,
                                          T*               B,
                                          rocsparse_int    ldb,
                                          void*            temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgtsv"),
                         m,
                         n,
                         (const void*&)dl,
                         (const void*&)d,
                         (const void*&)du,
                         (const void*&)B,
                         ldb,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG(1, m, (m <= 1), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG(7,
                       ldb,
                       (ldb < rocsparse::max(static_cast<rocsparse_int>(1), m)),
                       rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(3, n, dl);
    ROCSPARSE_CHECKARG_ARRAY(4, n, d);
    ROCSPARSE_CHECKARG_ARRAY(5, n, du);
    ROCSPARSE_CHECKARG_ARRAY(6, n, B);
    ROCSPARSE_CHECKARG_ARRAY(8, n, temp_buffer);

    if(n == 0)
    {
        return rocsparse_status_success;
    }

    char* ptr = reinterpret_cast<char*>(temp_buffer);

    gtsv_buffer_data<T> data;

    int current_m = m;
    for(int level = 0; level < gtsv_buffer_data<T>::MAX_RECURSION_LEVELS; level++)
    {
        const int BLOCKDIM = gtsv_buffer_data<T>::BLOCKDIM;

        const int m_pad = pad_to_blockdim(current_m, BLOCKDIM);

        data.dl_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(m_pad);
        data.d_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(m_pad);
        data.du_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(m_pad);
        data.B_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(m_pad * n);
        data.w_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(m_pad);
        data.v_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(m_pad);
        data.mt_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(m_pad);

        const int S_size = 2 * m_pad / BLOCKDIM;

        data.sl[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(S_size);
        data.s[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(S_size);
        data.su[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(S_size);
        data.sB[level] = reinterpret_cast<T*>(ptr);
        ptr += align_size<T>(S_size * n);

        data.factor_pad[level] = nullptr;
        if(level == 0)
        {
            data.factor_pad[level] = reinterpret_cast<T*>(ptr);
            ptr += align_size<T>(gtsv_buffer_data<T>::FUSED_V2_FACTOR_SIZE);
        }

        current_m = S_size;
    }

    if(m <= 1024)
    {
        RETURN_IF_ROCSPARSE_ERROR((gtsv_spike_solver_small_template<0>(handle,
            m,
            n,
            ldb,
            dl, //lower_diag,
            d, //main_diag,
            du, //upper_diag,
            B,
            data.dl_pad, //lower_pad,
            data.d_pad, //main_pad,
            data.du_pad, //upper_pad,
            data.B_pad,
            data.mt_pad,
            data.factor_pad)));
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR((gtsv_spike_solver_template<0>(handle,
            m,
            n,
            ldb,
            dl, //lower_diag,
            d, //main_diag,
            du, //upper_diag,
            B,
            data.dl_pad, //lower_pad,
            data.d_pad, //main_pad,
            data.du_pad, //upper_pad,
            data.B_pad,
            data.w_pad,
            data.v_pad,
            data.mt_pad,
            data.sl, //S_lower,
            data.s, //S_main,
            data.su, //S_upper,
            data.sB)));
    }
    // RETURN_IF_ROCSPARSE_ERROR((gtsv_spike_solver_template2<0>(handle,
    //                                                           m,
    //                                                           n,
    //                                                           ldb,
    //                                                           dl, //lower_diag,
    //                                                           d, //main_diag,
    //                                                           du, //upper_diag,
    //                                                           B,
    //                                                           data.dl_pad, //lower_pad,
    //                                                           data.d_pad, //main_pad,
    //                                                           data.du_pad, //upper_pad,
    //                                                           data.B_pad,
    //                                                           data.w_pad,
    //                                                           data.v_pad,
    //                                                           data.mt_pad,
    //                                                           data.sl, //S_lower,
    //                                                           data.s, //S_main,
    //                                                           data.su, //S_upper,
    //                                                           data.sB)));

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_sgtsv_buffer_size(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        const float*     dl,
                                                        const float*     d,
                                                        const float*     du,
                                                        const float*     B,
                                                        rocsparse_int    ldb,
                                                        size_t*          buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gtsv_buffer_size_template(handle, m, n, dl, d, du, B, ldb, buffer_size));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
extern "C" rocsparse_status rocsparse_dgtsv_buffer_size(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        const double*    dl,
                                                        const double*    d,
                                                        const double*    du,
                                                        const double*    B,
                                                        rocsparse_int    ldb,
                                                        size_t*          buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gtsv_buffer_size_template(handle, m, n, dl, d, du, B, ldb, buffer_size));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
extern "C" rocsparse_status rocsparse_cgtsv_buffer_size(rocsparse_handle               handle,
                                                        rocsparse_int                  m,
                                                        rocsparse_int                  n,
                                                        const rocsparse_float_complex* dl,
                                                        const rocsparse_float_complex* d,
                                                        const rocsparse_float_complex* du,
                                                        const rocsparse_float_complex* B,
                                                        rocsparse_int                  ldb,
                                                        size_t*                        buffer_size)
try
{
    // ROCSPARSE_ROUTINE_TRACE;
    // RETURN_IF_ROCSPARSE_ERROR(
    // rocsparse::gtsv_buffer_size_template(handle, m, n, dl, d, du, B, ldb, buffer_size));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
extern "C" rocsparse_status rocsparse_zgtsv_buffer_size(rocsparse_handle                handle,
                                                        rocsparse_int                   m,
                                                        rocsparse_int                   n,
                                                        const rocsparse_double_complex* dl,
                                                        const rocsparse_double_complex* d,
                                                        const rocsparse_double_complex* du,
                                                        const rocsparse_double_complex* B,
                                                        rocsparse_int                   ldb,
                                                        size_t*                         buffer_size)
try
{
    // ROCSPARSE_ROUTINE_TRACE;
    // RETURN_IF_ROCSPARSE_ERROR(
    // rocsparse::gtsv_buffer_size_template(handle, m, n, dl, d, du, B, ldb, buffer_size));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_sgtsv(rocsparse_handle handle,
                                            rocsparse_int    m,
                                            rocsparse_int    n,
                                            const float*     dl,
                                            const float*     d,
                                            const float*     du,
                                            float*           B,
                                            rocsparse_int    ldb,
                                            void*            temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gtsv_template(handle, m, n, dl, d, du, B, ldb, temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
extern "C" rocsparse_status rocsparse_dgtsv(rocsparse_handle handle,
                                            rocsparse_int    m,
                                            rocsparse_int    n,
                                            const double*    dl,
                                            const double*    d,
                                            const double*    du,
                                            double*          B,
                                            rocsparse_int    ldb,
                                            void*            temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gtsv_template(handle, m, n, dl, d, du, B, ldb, temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
extern "C" rocsparse_status rocsparse_cgtsv(rocsparse_handle               handle,
                                            rocsparse_int                  m,
                                            rocsparse_int                  n,
                                            const rocsparse_float_complex* dl,
                                            const rocsparse_float_complex* d,
                                            const rocsparse_float_complex* du,
                                            rocsparse_float_complex*       B,
                                            rocsparse_int                  ldb,
                                            void*                          temp_buffer)
try
{
    // ROCSPARSE_ROUTINE_TRACE;
    // RETURN_IF_ROCSPARSE_ERROR(
    // rocsparse::gtsv_template(handle, m, n, dl, d, du, B, ldb, temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
extern "C" rocsparse_status rocsparse_zgtsv(rocsparse_handle                handle,
                                            rocsparse_int                   m,
                                            rocsparse_int                   n,
                                            const rocsparse_double_complex* dl,
                                            const rocsparse_double_complex* d,
                                            const rocsparse_double_complex* du,
                                            rocsparse_double_complex*       B,
                                            rocsparse_int                   ldb,
                                            void*                           temp_buffer)
try
{
    // ROCSPARSE_ROUTINE_TRACE;
    // RETURN_IF_ROCSPARSE_ERROR(
    // rocsparse::gtsv_template(handle, m, n, dl, d, du, B, ldb, temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

// #define C_IMPL(NAME, TYPE)                                                                       \
//     extern "C" rocsparse_status NAME(rocsparse_handle handle,                                    \
//                                      rocsparse_int    m,                                         \
//                                      rocsparse_int    n,                                         \
//                                      const TYPE*      dl,                                        \
//                                      const TYPE*      d,                                         \
//                                      const TYPE*      du,                                        \
//                                      const TYPE*      B,                                         \
//                                      rocsparse_int    ldb,                                       \
//                                      size_t*          buffer_size)                               \
//     try                                                                                          \
//     {                                                                                            \
//         ROCSPARSE_ROUTINE_TRACE;                                                                 \
//         RETURN_IF_ROCSPARSE_ERROR(                                                               \
//             rocsparse::gtsv_buffer_size_template(handle, m, n, dl, d, du, B, ldb, buffer_size)); \
//         return rocsparse_status_success;                                                         \
//     }                                                                                            \
//     catch(...)                                                                                   \
//     {                                                                                            \
//         RETURN_ROCSPARSE_EXCEPTION();                                                            \
//     }

// C_IMPL(rocsparse_sgtsv_buffer_size, float);
// C_IMPL(rocsparse_dgtsv_buffer_size, double);
// C_IMPL(rocsparse_cgtsv_buffer_size, rocsparse_float_complex);
// C_IMPL(rocsparse_zgtsv_buffer_size, rocsparse_double_complex);

// #undef C_IMPL

// #define C_IMPL(NAME, TYPE)                                                           \
//     extern "C" rocsparse_status NAME(rocsparse_handle handle,                        \
//                                      rocsparse_int    m,                             \
//                                      rocsparse_int    n,                             \
//                                      const TYPE*      dl,                            \
//                                      const TYPE*      d,                             \
//                                      const TYPE*      du,                            \
//                                      TYPE*            B,                             \
//                                      rocsparse_int    ldb,                           \
//                                      void*            temp_buffer)                   \
//     try                                                                              \
//     {                                                                                \
//         ROCSPARSE_ROUTINE_TRACE;                                                     \
//         RETURN_IF_ROCSPARSE_ERROR(                                                   \
//             rocsparse::gtsv_template(handle, m, n, dl, d, du, B, ldb, temp_buffer)); \
//         return rocsparse_status_success;                                             \
//     }                                                                                \
//     catch(...)                                                                       \
//     {                                                                                \
//         RETURN_ROCSPARSE_EXCEPTION();                                                \
//     }

// C_IMPL(rocsparse_sgtsv, float);
// C_IMPL(rocsparse_dgtsv, double);
// C_IMPL(rocsparse_cgtsv, rocsparse_float_complex);
// C_IMPL(rocsparse_zgtsv, rocsparse_double_complex);

// #undef C_IMPL
