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

#include <map>
#include <vector>

#include "internal/precond/rocsparse_gtsv.h"
#include "rocsparse_gtsv.hpp"

#include "gtsv_device.h"

namespace rocsparse
{
    static uint64_t next_power_of_two(uint64_t m)
    {
        if(m == 0)
        {
            return 1;
        }

        m--;

        m |= m >> 1;
        m |= m >> 2;
        m |= m >> 4;
        m |= m >> 8;
        m |= m >> 16;
        m |= m >> 32;

        return m + 1;
    }

    template <typename T>
    inline size_t align256(size_t size)
    {
        return ((sizeof(T) * size - 1) / 256 + 1) * 256;
    }

    template <typename T>
    struct gtsv_buffer_data
    {
        // The reduced system shrinks by BLOCKDIM/2 at each level. With the S_solve leaf
        // limited to 32 rows, six levels covers any m representable in an int.
        static constexpr int MAX_RECURSION_LEVELS = 6;

        static constexpr int BLOCKDIM[MAX_RECURSION_LEVELS] = {32, 32, 32, 32, 32, 32};

        T* dl_pad[MAX_RECURSION_LEVELS];
        T* d_pad[MAX_RECURSION_LEVELS];
        T* du_pad[MAX_RECURSION_LEVELS];
        T* B_pad[MAX_RECURSION_LEVELS];

        T* w_pad[MAX_RECURSION_LEVELS];
        T* v_pad[MAX_RECURSION_LEVELS];
        T* mt_pad[MAX_RECURSION_LEVELS];

        T* sl[MAX_RECURSION_LEVELS];
        T* s[MAX_RECURSION_LEVELS];
        T* su[MAX_RECURSION_LEVELS];
        T* sB[MAX_RECURSION_LEVELS];

        pivot_mask<256>* pivot[MAX_RECURSION_LEVELS];
    };
}

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
        const int BLOCKDIM = gtsv_buffer_data<T>::BLOCKDIM[level];

        int m_pad = static_cast<int>(next_power_of_two(static_cast<uint64_t>(current_m)));
        m_pad     = rocsparse::max(m_pad, BLOCKDIM);

        *buffer_size += align256<T>(m_pad); // dl_pad
        *buffer_size += align256<T>(m_pad); // d_pad
        *buffer_size += align256<T>(m_pad); // du_pad
        *buffer_size += align256<T>(m_pad * n); // B_pad
        *buffer_size += align256<T>(m_pad); // w_pad
        *buffer_size += align256<T>(m_pad); // v_pad
        *buffer_size += align256<T>(m_pad); // mt_pad

        const int S_size = 2 * m_pad / BLOCKDIM;

        *buffer_size += align256<T>(S_size); // sl
        *buffer_size += align256<T>(S_size); // s
        *buffer_size += align256<T>(S_size); // su
        *buffer_size += align256<T>(S_size * n); // sB

        const int nblocks = m_pad / BLOCKDIM;

        *buffer_size += align256<pivot_mask<256>>(nblocks); // pivot

        current_m = S_size;
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_data_marshaling(rocsparse_handle handle,
                                                   int              m,
                                                   int              m_pad,
                                                   const T*         dl,
                                                   const T*         d,
                                                   const T*         du,
                                                   T*               dl_pad,
                                                   T*               d_pad,
                                                   T*               du_pad)
    {
        constexpr uint32_t TILE    = rocsparse::gtsv_marshal_tile<BLOCKSIZE>::TILE;
        const int          nblocks = m_pad / static_cast<int>(BLOCKDIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::data_marshaling_kernel<BLOCKSIZE, BLOCKDIM>),
            dim3((BLOCKDIM + TILE - 1) / TILE, (nblocks + TILE - 1) / TILE),
            dim3(TILE, BLOCKSIZE / TILE),
            0,
            handle->stream,
            m,
            m_pad,
            dl,
            d,
            du,
            dl_pad,
            d_pad,
            du_pad);

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_data_marshaling_B(
        rocsparse_handle handle, int m, int m_pad, int n, int ldb, const T* B, T* B_pad)
    {
        constexpr uint32_t TILE    = rocsparse::gtsv_marshal_tile<BLOCKSIZE>::TILE;
        const int          nblocks = m_pad / static_cast<int>(BLOCKDIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::data_marshaling_B_kernel<BLOCKSIZE, BLOCKDIM>),
            dim3((BLOCKDIM + TILE - 1) / TILE, (nblocks + TILE - 1) / TILE, std::min(n, 65535)),
            dim3(TILE, BLOCKSIZE / TILE),
            0,
            handle->stream,
            m,
            m_pad,
            n,
            ldb,
            B,
            B_pad);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_LBMT_solve_wvmt(rocsparse_handle handle,
                                                   int              m_pad,
                                                   T*               dl_pad,
                                                   T*               d_pad,
                                                   T*               du_pad,
                                                   T*               w_pad,
                                                   T*               v_pad,
                                                   T*               mt_pad,
                                                   pivot_mask<256>* pivot)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::LBMT_solve_wvmt_kernel<BLOCKSIZE, BLOCKDIM>),
                                           dim3(((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           m_pad,
                                           dl_pad,
                                           d_pad,
                                           du_pad,
                                           w_pad,
                                           v_pad,
                                           mt_pad,
                                           pivot);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_LBMT_solve_rhs(rocsparse_handle       handle,
                                                  int                    m_pad,
                                                  int                    n,
                                                  const T*               dl_pad,
                                                  const T*               d_pad,
                                                  const T*               du_pad,
                                                  const T*               mt_pad,
                                                  T*                     B_pad,
                                                  const pivot_mask<256>* pivot)
    {
        //std::cout << "x gridsize: " << (((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1) << std::endl;
        // if(n % 2 == 0)
        // {
        //     RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        //         (rocsparse::LBMT_solve_rhs_kernel<BLOCKSIZE, BLOCKDIM, 2>),
        //         dim3(((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1, n / 2, 1),
        //         dim3(BLOCKSIZE, 1, 1),
        //         0,
        //         handle->stream,
        //         m_pad,
        //         n,
        //         dl_pad,
        //         d_pad,
        //         du_pad,
        //         mt_pad,
        //         B_pad,
        //         pivot);
        // }
        // else
        // {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::LBMT_solve_rhs_kernel<BLOCKSIZE, BLOCKDIM, 1>),
            dim3(((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1, n, 1),
            dim3(BLOCKSIZE, 1, 1),
            0,
            handle->stream,
            m_pad,
            n,
            dl_pad,
            d_pad,
            du_pad,
            mt_pad,
            B_pad,
            pivot);
        // }

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_fill_s_matrix(rocsparse_handle handle,
                                                 int              m_pad,
                                                 int              n,
                                                 const T*         w_pad,
                                                 const T*         v_pad,
                                                 const T*         B_pad,
                                                 T*               sl,
                                                 T*               s,
                                                 T*               su,
                                                 T*               sB)
    {
        const int s_size = 2 * m_pad / BLOCKDIM;
        const int s_grid = (s_size - 1) / BLOCKSIZE + 1;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::fill_s_matrix_kernel<BLOCKSIZE, BLOCKDIM>),
                                           dim3(s_grid, std::min(n, 65535), 1),
                                           dim3(BLOCKSIZE, 1, 1),
                                           0,
                                           handle->stream,
                                           m_pad,
                                           n,
                                           w_pad,
                                           v_pad,
                                           B_pad,
                                           sl,
                                           s,
                                           su,
                                           sB);
        return rocsparse_status_success;
    }

    template <typename T, uint32_t S_SIZE>
    static rocsparse_status launch_s_solve_kernel(
        rocsparse_handle handle, int m, int n, const T* sl, const T* s, const T* su, T* sB)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::S_solve_kernel<S_SIZE>),
                                           dim3(n),
                                           dim3(1),
                                           0,
                                           handle->stream,
                                           m,
                                           n,
                                           sl,
                                           s,
                                           su,
                                           sB);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKDIM, uint32_t BLOCKSIZE, typename T>
    static rocsparse_status launch_scatter_S_B_to_B_pad(
        rocsparse_handle handle, int m_pad, int n, const T* sB, T* B_pad)
    {
        const int s_size = 2 * m_pad / BLOCKDIM;
        dim3      scatter_grid((s_size / 2 + BLOCKSIZE - 1) / BLOCKSIZE, std::min(n, 65535));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::scatter_S_B_to_B_pad_kernel<BLOCKDIM, BLOCKSIZE>),
            scatter_grid,
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            s_size,
            m_pad,
            n,
            sB,
            B_pad);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_backward_solve(
        rocsparse_handle handle, int m_pad, int n, const T* w_pad, const T* v_pad, T* B_pad)
    {
        const int grid = (m_pad - 1) / BLOCKSIZE + 1;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::backward_solve_kernel<BLOCKSIZE, BLOCKDIM>),
                                           dim3(grid, 1, 1),
                                           dim3(BLOCKSIZE, 1, 1),
                                           0,
                                           handle->stream,
                                           m_pad,
                                           n,
                                           w_pad,
                                           v_pad,
                                           B_pad);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_data_marshaling2(
        rocsparse_handle handle, int m, int m_pad, int n, int ldb, const T* B_pad, T* B)
    {
        constexpr uint32_t TILE    = rocsparse::gtsv_marshal_tile<BLOCKSIZE>::TILE;
        const int          nblocks = m_pad / static_cast<int>(BLOCKDIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::data_marshaling_kernel2<BLOCKSIZE, BLOCKDIM>),
            dim3((BLOCKDIM + TILE - 1) / TILE, (nblocks + TILE - 1) / TILE, std::min(n, 65535)),
            dim3(TILE, BLOCKSIZE / TILE),
            0,
            handle->stream,
            m,
            m_pad,
            n,
            ldb,
            B_pad,
            B);
        return rocsparse_status_success;
    }

    template <uint32_t LEVEL, typename T>
    static rocsparse_status gtsv_spike_solver_template(rocsparse_handle  handle,
                                                       rocsparse_int     m,
                                                       rocsparse_int     n,
                                                       rocsparse_int     ldb,
                                                       const T*          dl, //lower_diag,
                                                       const T*          d, //main_diag,
                                                       const T*          du, //upper_diag,
                                                       T*                B,
                                                       T**               dl_pad, //lower_pad,
                                                       T**               d_pad, //main_pad,
                                                       T**               du_pad, //upper_pad,
                                                       T**               B_pad,
                                                       T**               w_pad,
                                                       T**               v_pad,
                                                       T**               mt_pad,
                                                       T**               sl, //S_lower,
                                                       T**               s, //S_main,
                                                       T**               su, //S_upper,
                                                       T**               sB,
                                                       pivot_mask<256>** pivot)
    {
        ROCSPARSE_ROUTINE_TRACE;

        constexpr uint32_t BLOCKDIM  = gtsv_buffer_data<T>::BLOCKDIM[LEVEL];
        constexpr int      BLOCKSIZE = 128;

        int m_pad = next_power_of_two(m);
        m_pad     = rocsparse::max(m_pad, (int)BLOCKDIM);

        const int s_size  = 2 * m_pad / BLOCKDIM;
        const int nblocks = m_pad / BLOCKDIM;

        RETURN_IF_ROCSPARSE_ERROR((launch_data_marshaling<BLOCKSIZE, BLOCKDIM>(
            handle, m, m_pad, dl, d, du, dl_pad[LEVEL], d_pad[LEVEL], du_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_data_marshaling_B<BLOCKSIZE, BLOCKDIM>(
            handle, m, m_pad, n, ldb, B, B_pad[LEVEL])));

        RETURN_IF_HIP_ERROR(hipMemsetAsync(w_pad[LEVEL], 0, sizeof(T) * m_pad, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(v_pad[LEVEL], 0, sizeof(T) * m_pad, handle->stream));
        RETURN_IF_HIP_ERROR(
            hipMemsetAsync(pivot[LEVEL], 0, sizeof(pivot_mask<256>) * nblocks, handle->stream));

        RETURN_IF_ROCSPARSE_ERROR((launch_LBMT_solve_wvmt<BLOCKSIZE, BLOCKDIM>(handle,
                                                                               m_pad,
                                                                               dl_pad[LEVEL],
                                                                               d_pad[LEVEL],
                                                                               du_pad[LEVEL],
                                                                               w_pad[LEVEL],
                                                                               v_pad[LEVEL],
                                                                               mt_pad[LEVEL],
                                                                               pivot[LEVEL])));

        for(int i = 0; i < n; i += 65535)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (launch_LBMT_solve_rhs<BLOCKSIZE, BLOCKDIM>(handle,
                                                            m_pad,
                                                            rocsparse::min(n - i, 65535),
                                                            dl_pad[LEVEL],
                                                            d_pad[LEVEL],
                                                            du_pad[LEVEL],
                                                            mt_pad[LEVEL],
                                                            B_pad[LEVEL] + m_pad * i,
                                                            pivot[LEVEL])));
        }

        RETURN_IF_ROCSPARSE_ERROR((launch_fill_s_matrix<BLOCKSIZE, BLOCKDIM>(handle,
                                                                             m_pad,
                                                                             n,
                                                                             w_pad[LEVEL],
                                                                             v_pad[LEVEL],
                                                                             B_pad[LEVEL],
                                                                             sl[LEVEL],
                                                                             s[LEVEL],
                                                                             su[LEVEL],
                                                                             sB[LEVEL])));

        using S_solve_launch_ptr
            = rocsparse_status (*)(rocsparse_handle, int, int, const T*, const T*, const T*, T*);

        static const std::map<int, S_solve_launch_ptr> s_solve_dispatch = {
            {2, launch_s_solve_kernel<T, 2>},
            {4, launch_s_solve_kernel<T, 4>},
            {8, launch_s_solve_kernel<T, 8>},
            {16, launch_s_solve_kernel<T, 16>},
            {32, launch_s_solve_kernel<T, 32>},
        };

        //std::cout << "s_size: " << s_size << " BLOCKDIM: " << BLOCKDIM << std::endl;

        auto dispatch_it = s_solve_dispatch.lower_bound(s_size);
        if(dispatch_it != s_solve_dispatch.end())
        {
            RETURN_IF_ROCSPARSE_ERROR(
                dispatch_it->second(handle, s_size, n, sl[LEVEL], s[LEVEL], su[LEVEL], sB[LEVEL]));
        }
        else if constexpr(LEVEL + 1 < gtsv_buffer_data<T>::MAX_RECURSION_LEVELS)
        {
            RETURN_IF_ROCSPARSE_ERROR((gtsv_spike_solver_template<LEVEL + 1>(handle,
                                                                             s_size,
                                                                             n,
                                                                             s_size,
                                                                             sl[LEVEL],
                                                                             s[LEVEL],
                                                                             su[LEVEL],
                                                                             sB[LEVEL],
                                                                             dl_pad,
                                                                             d_pad,
                                                                             du_pad,
                                                                             B_pad,
                                                                             w_pad,
                                                                             v_pad,
                                                                             mt_pad,
                                                                             sl,
                                                                             s,
                                                                             su,
                                                                             sB,
                                                                             pivot)));
        }
        else
        {
            return rocsparse_status_internal_error;
        }

        RETURN_IF_ROCSPARSE_ERROR((launch_scatter_S_B_to_B_pad<BLOCKDIM, BLOCKSIZE>(
            handle, m_pad, n, sB[LEVEL], B_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_backward_solve<BLOCKSIZE, BLOCKDIM>(
            handle, m_pad, n, w_pad[LEVEL], v_pad[LEVEL], B_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_data_marshaling2<BLOCKSIZE, BLOCKDIM>(
            handle, m, m_pad, n, ldb, B_pad[LEVEL], B)));

        return rocsparse_status_success;
    }
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

    // {
    //     constexpr uint32_t BLOCKSIZE = 256;

    //     rocsparse_int block_dim = 2;
    //     rocsparse_int m_pad     = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
    //     rocsparse_int gridsize  = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);

    //     while(gridsize > 512)
    //     {
    //         block_dim *= 2;
    //         m_pad    = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
    //         gridsize = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
    //     }

    //     // round up to next power of 2
    //     gridsize = fnp2(gridsize);

    //     std::cout << "gridsize: " << gridsize << " block_dim: " << block_dim << std::endl;
    // }

    char* ptr = reinterpret_cast<char*>(temp_buffer);

    gtsv_buffer_data<T> data;

    int current_m = m;
    for(int level = 0; level < gtsv_buffer_data<T>::MAX_RECURSION_LEVELS; level++)
    {
        const int BLOCKDIM = gtsv_buffer_data<T>::BLOCKDIM[level];

        int m_pad = static_cast<int>(next_power_of_two(static_cast<uint64_t>(current_m)));
        m_pad     = rocsparse::max(m_pad, BLOCKDIM);

        data.dl_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(m_pad);
        data.d_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(m_pad);
        data.du_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(m_pad);
        data.B_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(m_pad * n);
        data.w_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(m_pad);
        data.v_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(m_pad);
        data.mt_pad[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(m_pad);

        const int S_size = 2 * m_pad / BLOCKDIM;

        data.sl[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(S_size);
        data.s[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(S_size);
        data.su[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(S_size);
        data.sB[level] = reinterpret_cast<T*>(ptr);
        ptr += align256<T>(S_size * n);

        const int nblocks = m_pad / BLOCKDIM;

        data.pivot[level] = reinterpret_cast<pivot_mask<256>*>(ptr);
        ptr += align256<pivot_mask<256>>(nblocks);

        current_m = S_size;
    }

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
                                                             data.sB,
                                                             data.pivot)));

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                                    \
                                     rocsparse_int    m,                                         \
                                     rocsparse_int    n,                                         \
                                     const TYPE*      dl,                                        \
                                     const TYPE*      d,                                         \
                                     const TYPE*      du,                                        \
                                     const TYPE*      B,                                         \
                                     rocsparse_int    ldb,                                       \
                                     size_t*          buffer_size)                               \
    try                                                                                          \
    {                                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                                 \
        RETURN_IF_ROCSPARSE_ERROR(                                                               \
            rocsparse::gtsv_buffer_size_template(handle, m, n, dl, d, du, B, ldb, buffer_size)); \
        return rocsparse_status_success;                                                         \
    }                                                                                            \
    catch(...)                                                                                   \
    {                                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                                            \
    }

C_IMPL(rocsparse_sgtsv_buffer_size, float);
C_IMPL(rocsparse_dgtsv_buffer_size, double);
C_IMPL(rocsparse_cgtsv_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_buffer_size, rocsparse_double_complex);

#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                           \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                        \
                                     rocsparse_int    m,                             \
                                     rocsparse_int    n,                             \
                                     const TYPE*      dl,                            \
                                     const TYPE*      d,                             \
                                     const TYPE*      du,                            \
                                     TYPE*            B,                             \
                                     rocsparse_int    ldb,                           \
                                     void*            temp_buffer)                   \
    try                                                                              \
    {                                                                                \
        ROCSPARSE_ROUTINE_TRACE;                                                     \
        RETURN_IF_ROCSPARSE_ERROR(                                                   \
            rocsparse::gtsv_template(handle, m, n, dl, d, du, B, ldb, temp_buffer)); \
        return rocsparse_status_success;                                             \
    }                                                                                \
    catch(...)                                                                       \
    {                                                                                \
        RETURN_ROCSPARSE_EXCEPTION();                                                \
    }

C_IMPL(rocsparse_sgtsv, float);
C_IMPL(rocsparse_dgtsv, double);
C_IMPL(rocsparse_cgtsv, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv, rocsparse_double_complex);

#undef C_IMPL
