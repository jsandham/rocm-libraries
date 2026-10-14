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

#include <vector>

#include "internal/precond/rocsparse_gtsv.h"
#include "rocsparse_grid.hpp"
#include "rocsparse_gtsv.hpp"
#include "rocsparse_gtsv_spike_solver.hpp"

#include "gtsv_device.h"

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
}



template <uint32_t LEVEL, typename T>
rocsparse_status rocsparse::gtsv_spike_solver_template(rocsparse_handle handle,
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

template <uint32_t LEVEL, typename T>
rocsparse_status rocsparse::gtsv_spike_solver_template2(rocsparse_handle handle,
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

// Level 0 is the entry point from gtsv_template. Instantiating it also emits the
// deeper levels, because each level calls the next one under if constexpr.
#define INSTANTIATE(T)                                                                   \
    template rocsparse_status rocsparse::gtsv_spike_solver_template<0, T>(               \
        rocsparse_handle handle,                                                         \
        rocsparse_int    m,                                                              \
        rocsparse_int    n,                                                              \
        rocsparse_int    ldb,                                                            \
        const T*         dl,                                                             \
        const T*         d,                                                              \
        const T*         du,                                                             \
        T*               B,                                                               \
        T**              dl_pad,                                                          \
        T**              d_pad,                                                          \
        T**              du_pad,                                                         \
        T**              B_pad,                                                          \
        T**              w_pad,                                                          \
        T**              v_pad,                                                          \
        T**              mt_pad,                                                         \
        T**              sl,                                                             \
        T**              s,                                                              \
        T**              su,                                                            \
        T**              sB);

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE