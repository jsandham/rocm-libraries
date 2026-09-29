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

#include <vector>

#include "internal/precond/rocsparse_gtsv.h"
#include "rocsparse_gtsv.hpp"

#include "gtsv_device.h"

namespace rocsparse
{
    // The block kernels index complete partitions of BLOCKDIM rows, so padding only
    // has to reach the next multiple of BLOCKDIM. Power-of-two padding adds fully
    // artificial partitions whose spike tips are zero.
    static int pad_to_blockdim(int m, int blockdim)
    {
        const int64_t value  = static_cast<int64_t>(m);
        const int64_t padded = ((value + blockdim - 1) / blockdim) * blockdim;
        return static_cast<int>(padded);
    }

    template <typename T>
    inline size_t align256(size_t size)
    {
        return ((sizeof(T) * size - 1) / 256 + 1) * 256;
    }

    template <typename T>
    struct gtsv_buffer_data
    {
        // Each level keeps two interface rows per BLOCKDIM-row partition. The leaf
        // solver handles every reduced size up to BLOCKDIM, so six levels cover any
        // m whose padded length fits in a 32-bit int.
        static constexpr int MAX_RECURSION_LEVELS = 6;

        static constexpr int BLOCKDIM = 32;

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
        const int BLOCKDIM = gtsv_buffer_data<T>::BLOCKDIM;

        const int m_pad = pad_to_blockdim(current_m, BLOCKDIM);

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

        current_m = S_size;
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
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
        const int nblocks = m_pad / BLOCKDIM;

        const int grid_size = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::data_marshaling_kernel<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
            dim3(grid_size),
            dim3(BLOCKSIZE),
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_data_marshaling_B(
        rocsparse_handle handle, int m, int m_pad, int n, int ldb, const T* B, T* B_pad)
    {
        const int nblocks = m_pad / BLOCKDIM;

        const int grid_x = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;
        const int grid_y = std::min(n, 65535);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::data_marshaling_B_kernel<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
            grid,
            block,
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
                                                   const T*         dl_pad,
                                                   const T*         d_pad,
                                                   const T*         du_pad,
                                                   T*               w_pad,
                                                   T*               v_pad,
                                                   T*               mt_pad)
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
                                           mt_pad);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_LBMT_solve_rhs(rocsparse_handle handle,
                                                  int              m_pad,
                                                  int              n,
                                                  const T*         dl_pad,
                                                  const T*         d_pad,
                                                  const T*         du_pad,
                                                  const T*         mt_pad,
                                                  T*               B_pad)
    {
        const int nblocks = m_pad / BLOCKDIM;

        const int grid_x = (nblocks - 1) / BLOCKSIZE + 1;
        const int grid_y = std::min(n, 65535);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::LBMT_solve_rhs_kernel<BLOCKSIZE, BLOCKDIM>),
                                           grid,
                                           block,
                                           0,
                                           handle->stream,
                                           m_pad,
                                           n,
                                           dl_pad,
                                           d_pad,
                                           du_pad,
                                           mt_pad,
                                           B_pad);
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

    template <typename T>
    static rocsparse_status launch_s_solve_kernel(
        rocsparse_handle handle, int m, int n, const T* sl, const T* s, const T* su, T* sB)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::S_solve_kernel<T>),
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_scatter_S_B_to_B_pad(
        rocsparse_handle handle, int m_pad, int n, const T* sB, T* B_pad)
    {
        const int s_size = 2 * m_pad / BLOCKDIM;
        dim3      scatter_grid((s_size / 2 + BLOCKSIZE - 1) / BLOCKSIZE, std::min(n, 65535));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::scatter_S_B_to_B_pad_kernel<BLOCKSIZE, BLOCKDIM>),
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
        rocsparse_handle handle, int m_pad, int n, const T* w, const T* v, T* rhs)
    {
        const int grid_x = (m_pad - 1) / BLOCKSIZE + 1;
        const int grid_y = std::min(n, 32768);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::backward_solve_kernel<BLOCKSIZE, BLOCKDIM>),
                                           grid,
                                           block,
                                           0,
                                           handle->stream,
                                           m_pad,
                                           n,
                                           w,
                                           v,
                                           rhs);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_reverse_data_marshaling_B(
        rocsparse_handle handle, int m, int m_pad, int n, int ldb, const T* B_pad, T* B)
    {
        const int nblocks = m_pad / BLOCKDIM;

        const int grid_x = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;
        const int grid_y = std::min(n, 32768);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::
                 reverse_data_marshaling_B_kernel<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
            grid,
            block,
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
    static rocsparse_status gtsv_spike_solver_template(rocsparse_handle handle,
                                                       rocsparse_int    m,
                                                       rocsparse_int    n,
                                                       rocsparse_int    ldb,
                                                       const T*         dl, //lower_diag,
                                                       const T*         d, //main_diag,
                                                       const T*         du, //upper_diag,
                                                       T*               B,
                                                       T**              dl_pad, //lower_pad,
                                                       T**              d_pad, //main_pad,
                                                       T**              du_pad, //upper_pad,
                                                       T**              B_pad,
                                                       T**              w_pad,
                                                       T**              v_pad,
                                                       T**              mt_pad,
                                                       T**              sl, //S_lower,
                                                       T**              s, //S_main,
                                                       T**              su, //S_upper,
                                                       T**              sB)
    {
        ROCSPARSE_ROUTINE_TRACE;

        constexpr uint32_t BLOCKDIM  = gtsv_buffer_data<T>::BLOCKDIM;
        constexpr int      BLOCKSIZE = 256;

        const int m_pad = pad_to_blockdim(m, BLOCKDIM);

        const int s_size = 2 * m_pad / BLOCKDIM;

        RETURN_IF_ROCSPARSE_ERROR((launch_data_marshaling<128, BLOCKDIM, 256>(
            handle, m, m_pad, dl, d, du, dl_pad[LEVEL], d_pad[LEVEL], du_pad[LEVEL])));
        RETURN_IF_ROCSPARSE_ERROR((launch_data_marshaling_B<128, BLOCKDIM, 256>(
            handle, m, m_pad, n, ldb, B, B_pad[LEVEL])));

        RETURN_IF_HIP_ERROR(hipMemsetAsync(w_pad[LEVEL], 0, sizeof(T) * m_pad, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(v_pad[LEVEL], 0, sizeof(T) * m_pad, handle->stream));

        RETURN_IF_ROCSPARSE_ERROR((launch_LBMT_solve_wvmt<BLOCKSIZE, BLOCKDIM>(handle,
                                                                               m_pad,
                                                                               dl_pad[LEVEL],
                                                                               d_pad[LEVEL],
                                                                               du_pad[LEVEL],
                                                                               w_pad[LEVEL],
                                                                               v_pad[LEVEL],
                                                                               mt_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_LBMT_solve_rhs<BLOCKSIZE, BLOCKDIM>(handle,
                                                                              m_pad,
                                                                              n,
                                                                              dl_pad[LEVEL],
                                                                              d_pad[LEVEL],
                                                                              du_pad[LEVEL],
                                                                              mt_pad[LEVEL],
                                                                              B_pad[LEVEL])));

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

        if(s_size <= static_cast<int>(BLOCKDIM))
        {
            RETURN_IF_ROCSPARSE_ERROR((launch_s_solve_kernel<T>(
                handle, s_size, n, sl[LEVEL], s[LEVEL], su[LEVEL], sB[LEVEL])));
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
                                                                             sB)));
        }
        else
        {
            return rocsparse_status_internal_error;
        }

        RETURN_IF_ROCSPARSE_ERROR((launch_scatter_S_B_to_B_pad<BLOCKSIZE, BLOCKDIM>(
            handle, m_pad, n, sB[LEVEL], B_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_backward_solve<BLOCKSIZE, BLOCKDIM>(
            handle, m_pad, n, w_pad[LEVEL], v_pad[LEVEL], B_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_reverse_data_marshaling_B<128, BLOCKDIM, 256>(
            handle, m, m_pad, n, ldb, B_pad[LEVEL], B)));

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_data_marshaling2(rocsparse_handle handle,
                                                    int              m,
                                                    int              m_pad,
                                                    const T*         dl,
                                                    const T*         d,
                                                    const T*         du,
                                                    T*               dl_pad,
                                                    T*               d_pad,
                                                    T*               du_pad)
    {
        const int nblocks = m_pad / BLOCKDIM;

        const int grid_size = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::data_marshaling_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
            dim3(grid_size),
            dim3(BLOCKSIZE),
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_data_marshaling_B2(
        rocsparse_handle handle, int m, int m_pad, int n, int ldb, const T* B, T* B_pad)
    {
        const int nblocks = m_pad / BLOCKDIM;

        const int grid_x = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;
        const int grid_y = std::min(n, 65535);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::data_marshaling_B_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
            grid,
            block,
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_LBMT_solve_wvmt2(rocsparse_handle handle,
                                                    int              m_pad,
                                                    const T*         dl_pad,
                                                    const T*         d_pad,
                                                    const T*         du_pad,
                                                    T*               w_pad,
                                                    T*               v_pad,
                                                    T*               mt_pad)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::LBMT_solve_wvmt_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
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
            mt_pad);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_LBMT_solve_rhs2(rocsparse_handle handle,
                                                   int              m_pad,
                                                   int              n,
                                                   const T*         dl_pad,
                                                   const T*         d_pad,
                                                   const T*         du_pad,
                                                   const T*         mt_pad,
                                                   T*               B_pad)
    {
        const int nblocks = m_pad / BLOCKDIM;

        const int grid_x = (nblocks - 1) / BLOCKSIZE + 1;
        const int grid_y = std::min(n, 65535);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::LBMT_solve_rhs_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
            grid,
            block,
            0,
            handle->stream,
            m_pad,
            n,
            dl_pad,
            d_pad,
            du_pad,
            mt_pad,
            B_pad);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_fill_s_matrix2(rocsparse_handle handle,
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

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::fill_s_matrix_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
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

    template <typename T>
    static rocsparse_status launch_s_solve_kernel2(
        rocsparse_handle handle, int m, int n, const T* sl, const T* s, const T* su, T* sB)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::S_solve_kernel2<T>),
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_scatter_S_B_to_B_pad2(
        rocsparse_handle handle, int m_pad, int n, const T* sB, T* B_pad)
    {
        const int s_size = 2 * m_pad / BLOCKDIM;
        dim3      scatter_grid((s_size / 2 + BLOCKSIZE - 1) / BLOCKSIZE, std::min(n, 65535));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::scatter_S_B_to_B_pad_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_reverse_data_marshaling_B2(
        rocsparse_handle handle, int m, int m_pad, int n, int ldb, const T* B_pad, T* B)
    {
        const int nblocks = m_pad / BLOCKDIM;

        const int grid_x = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;
        const int grid_y = std::min(n, 32768);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::
                 reverse_data_marshaling_B_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
            grid,
            block,
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static rocsparse_status launch_backward_solve2(
        rocsparse_handle handle, int m_pad, int n, const T* w, const T* v, T* rhs)
    {
        const int grid_x = (m_pad - 1) / BLOCKSIZE + 1;
        const int grid_y = std::min(n, 32768);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::backward_solve_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
            grid,
            block,
            0,
            handle->stream,
            m_pad,
            n,
            w,
            v,
            rhs);
        return rocsparse_status_success;
    }

    template <uint32_t LEVEL, typename T>
    static rocsparse_status gtsv_spike_solver_template2(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        rocsparse_int    ldb,
                                                        const T*         dl, //lower_diag,
                                                        const T*         d, //main_diag,
                                                        const T*         du, //upper_diag,
                                                        T*               B,
                                                        T**              dl_pad, //lower_pad,
                                                        T**              d_pad, //main_pad,
                                                        T**              du_pad, //upper_pad,
                                                        T**              B_pad,
                                                        T**              w_pad,
                                                        T**              v_pad,
                                                        T**              mt_pad,
                                                        T**              sl, //S_lower,
                                                        T**              s, //S_main,
                                                        T**              su, //S_upper,
                                                        T**              sB)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // // Temporary test code
        // {
        //     constexpr int BLOCKDIM             = 4;
        //     constexpr int BLOCKSIZE            = 4;
        //     constexpr int PARTITIONS_PER_GROUP = 8;
        //     const int     m                    = 64;
        //     const int     m_pad                = 64;
        //     const int     nblocks              = m_pad / BLOCKDIM;

        //     std::vector<float> hdata(m, 0);
        //     std::vector<float> hdata_pad(m_pad, 0);

        //     for(int i = 0; i < m; i++)
        //     {
        //         hdata[i] = i;
        //     }

        //     float* ddata     = nullptr;
        //     float* ddata_pad = nullptr;
        //     RETURN_IF_HIP_ERROR(hipMalloc((void**)&ddata, sizeof(float) * m));
        //     RETURN_IF_HIP_ERROR(hipMalloc((void**)&ddata_pad, sizeof(float) * m_pad));
        //     RETURN_IF_HIP_ERROR(
        //         hipMemcpy(ddata, hdata.data(), sizeof(float) * m, hipMemcpyHostToDevice));

        //     const int grid = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;

        //     RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        //         (rocsparse::data_marshaling_B_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
        //         grid,
        //         BLOCKSIZE,
        //         0,
        //         handle->stream,
        //         m,
        //         m_pad,
        //         1,
        //         m,
        //         ddata,
        //         ddata_pad);

        //     RETURN_IF_HIP_ERROR(hipMemcpy(
        //         hdata_pad.data(), ddata_pad, sizeof(float) * m_pad, hipMemcpyDeviceToHost));

        //     std::cout << "hdata_pad" << std::endl;
        //     for(int i = 0; i < m_pad; i++)
        //     {
        //         std::cout << hdata_pad[i] << " ";
        //     }
        //     std::cout << "" << std::endl;

        //     RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        //         (rocsparse::
        //              reverse_data_marshaling_B_kernel2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>),
        //         grid,
        //         BLOCKSIZE,
        //         0,
        //         handle->stream,
        //         m,
        //         m_pad,
        //         1,
        //         m,
        //         ddata_pad,
        //         ddata);

        //     RETURN_IF_HIP_ERROR(
        //         hipMemcpy(hdata.data(), ddata, sizeof(float) * m, hipMemcpyDeviceToHost));

        //     std::cout << "hdata" << std::endl;
        //     for(int i = 0; i < m; i++)
        //     {
        //         std::cout << hdata[i] << " ";
        //     }
        //     std::cout << "" << std::endl;

        //     RETURN_IF_HIP_ERROR(hipFree(ddata));
        //     RETURN_IF_HIP_ERROR(hipFree(ddata_pad));
        // }

        constexpr uint32_t BLOCKDIM  = gtsv_buffer_data<T>::BLOCKDIM;
        constexpr int      BLOCKSIZE = 256;

        const int m_pad = pad_to_blockdim(m, BLOCKDIM);

        const int s_size = 2 * m_pad / BLOCKDIM;

        RETURN_IF_ROCSPARSE_ERROR((launch_data_marshaling2<128, BLOCKDIM, 256>(
            handle, m, m_pad, dl, d, du, dl_pad[LEVEL], d_pad[LEVEL], du_pad[LEVEL])));
        RETURN_IF_ROCSPARSE_ERROR((launch_data_marshaling_B2<128, BLOCKDIM, 256>(
            handle, m, m_pad, n, ldb, B, B_pad[LEVEL])));

        RETURN_IF_HIP_ERROR(hipMemsetAsync(w_pad[LEVEL], 0, sizeof(T) * m_pad, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(v_pad[LEVEL], 0, sizeof(T) * m_pad, handle->stream));

        RETURN_IF_ROCSPARSE_ERROR(
            (launch_LBMT_solve_wvmt2<BLOCKSIZE, BLOCKDIM, 256>(handle,
                                                               m_pad,
                                                               dl_pad[LEVEL],
                                                               d_pad[LEVEL],
                                                               du_pad[LEVEL],
                                                               w_pad[LEVEL],
                                                               v_pad[LEVEL],
                                                               mt_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_LBMT_solve_rhs2<BLOCKSIZE, BLOCKDIM, 256>(handle,
                                                                                    m_pad,
                                                                                    n,
                                                                                    dl_pad[LEVEL],
                                                                                    d_pad[LEVEL],
                                                                                    du_pad[LEVEL],
                                                                                    mt_pad[LEVEL],
                                                                                    B_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_fill_s_matrix2<BLOCKSIZE, BLOCKDIM, 256>(handle,
                                                                                   m_pad,
                                                                                   n,
                                                                                   w_pad[LEVEL],
                                                                                   v_pad[LEVEL],
                                                                                   B_pad[LEVEL],
                                                                                   sl[LEVEL],
                                                                                   s[LEVEL],
                                                                                   su[LEVEL],
                                                                                   sB[LEVEL])));

        if(s_size <= static_cast<int>(BLOCKDIM))
        {
            RETURN_IF_ROCSPARSE_ERROR((launch_s_solve_kernel2<T>(
                handle, s_size, n, sl[LEVEL], s[LEVEL], su[LEVEL], sB[LEVEL])));
        }
        else if constexpr(LEVEL + 1 < gtsv_buffer_data<T>::MAX_RECURSION_LEVELS)
        {
            RETURN_IF_ROCSPARSE_ERROR((gtsv_spike_solver_template2<LEVEL + 1>(handle,
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
                                                                              sB)));
        }
        else
        {
            return rocsparse_status_internal_error;
        }

        RETURN_IF_ROCSPARSE_ERROR((launch_scatter_S_B_to_B_pad2<BLOCKSIZE, BLOCKDIM, 256>(
            handle, m_pad, n, sB[LEVEL], B_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_backward_solve2<BLOCKSIZE, BLOCKDIM, 256>(
            handle, m_pad, n, w_pad[LEVEL], v_pad[LEVEL], B_pad[LEVEL])));

        RETURN_IF_ROCSPARSE_ERROR((launch_reverse_data_marshaling_B2<128, BLOCKDIM, 256>(
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

    char* ptr = reinterpret_cast<char*>(temp_buffer);

    gtsv_buffer_data<T> data;

    int current_m = m;
    for(int level = 0; level < gtsv_buffer_data<T>::MAX_RECURSION_LEVELS; level++)
    {
        const int BLOCKDIM = gtsv_buffer_data<T>::BLOCKDIM;

        const int m_pad = pad_to_blockdim(current_m, BLOCKDIM);

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
                                                             data.sB)));
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
