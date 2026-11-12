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

#include <cstdlib>
#include <map>
#include <vector>

#include "internal/precond/rocsparse_gtsv.h"
#include "rocsparse_grid.hpp"
#include "rocsparse_gtsv.hpp"
#include "rocsparse_gtsv_spike_solver.hpp"
#include "rocsparse_gtsv_spike_solver_small.hpp"

#include "gtsv_spike_fused_device.h"
#include "gtsv_spike_one_thread_per_system_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_transpose(rocsparse_handle handle,
                                             rocsparse_int    m,
                                             rocsparse_int    n,
                                             rocsparse_int    ldb,
                                             rocsparse_int    ldx,
                                             const T*         B,
                                             T*               X)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static_assert(BLOCKSIZE % BLOCKDIM == 0);
        static_assert(BLOCKSIZE >= BLOCKDIM);

        constexpr rocsparse_int TILE_Y = BLOCKSIZE / BLOCKDIM;
        constexpr rocsparse_int TILE_X = BLOCKDIM;

        const rocsparse_int gridx = (m + TILE_X - 1) / TILE_X;
        const rocsparse_int gridy = (n + TILE_Y - 1) / TILE_Y;

        dim3 threadsPerBlock(TILE_X, TILE_Y);
        dim3 numBlocks(gridx, std::min(gridy, 32768));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::transpose_kernel<TILE_X, TILE_Y>),
                                           numBlocks,
                                           threadsPerBlock,
                                           0,
                                           handle->stream,
                                           m,
                                           n,
                                           ldb,
                                           ldx,
                                           B,
                                           X);
        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status launch_LBMT_solve_mt(rocsparse_handle handle,
                                                 rocsparse_int    m,
                                                 const T* __restrict__ dl,
                                                 const T* __restrict__ d,
                                                 const T* __restrict__ du,
                                                 T* __restrict__ mt)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::LBMT_solve_mt_kernel), 1, 1, 0, handle->stream, m, dl, d, du, mt);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_LBMT_solve_rhs_one_thread_per_system(rocsparse_handle handle,
                                                                        rocsparse_int    m,
                                                                        rocsparse_int    n,
                                                                        const T* __restrict__ dl,
                                                                        const T* __restrict__ d,
                                                                        const T* __restrict__ du,
                                                                        const T* __restrict__ mt,
                                                                        T* __restrict__ rhs)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::LBMT_solve_rhs_one_thread_per_system_kernel<BLOCKSIZE, BLOCKDIM>),
            (n - 1) / BLOCKSIZE + 1,
            BLOCKSIZE,
            0,
            handle->stream,
            m,
            n,
            dl,
            d,
            du,
            mt,
            rhs);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status tridiagonal_spike_solver_one_thread_per_system(rocsparse_handle handle,
                                                                           rocsparse_int    m,
                                                                           rocsparse_int    n,
                                                                           rocsparse_int    ldb,
                                                                           const T* __restrict__ dl,
                                                                           const T* __restrict__ d,
                                                                           const T* __restrict__ du,
                                                                           T* __restrict__ B,
                                                                           T* B_pad,
                                                                           T* mt)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // B is m by n with leading dimension ldb. The solve works on the transpose,
        // packed with leading dimension n, then the result is written back with ldb.
        RETURN_IF_ROCSPARSE_ERROR(
            (launch_transpose<BLOCKSIZE, BLOCKDIM>(handle, m, n, ldb, n, B, B_pad)));
        RETURN_IF_ROCSPARSE_ERROR(launch_LBMT_solve_mt(handle, m, dl, d, du, mt));
        RETURN_IF_ROCSPARSE_ERROR((launch_LBMT_solve_rhs_one_thread_per_system<BLOCKSIZE, BLOCKDIM>(
            handle, m, n, dl, d, du, mt, B_pad)));
        RETURN_IF_ROCSPARSE_ERROR(
            (launch_transpose<BLOCKSIZE, BLOCKDIM>(handle, n, m, n, ldb, B_pad, B)));
        return rocsparse_status_success;
    }

    // Version 3 of the fused solver: factor once, then apply to the right hand sides.
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_solve_fused_wvmt(rocsparse_handle handle,
                                                    rocsparse_int    m,
                                                    const T* __restrict__ lower,
                                                    const T* __restrict__ main,
                                                    const T* __restrict__ upper,
                                                    T* __restrict__ coeff1,
                                                    T* __restrict__ coeff2,
                                                    T* __restrict__ coeff3,
                                                    T* __restrict__ coeff4,
                                                    T* __restrict__ coeff5,
                                                    T* __restrict__ w,
                                                    T* __restrict__ v,
                                                    T* __restrict__ mt,
                                                    int* __restrict__ pivot_mask,
                                                    T* __restrict__ s_coeff1,
                                                    T* __restrict__ s_coeff2,
                                                    T* __restrict__ s_coeff3,
                                                    T* __restrict__ s_coeff4,
                                                    T* __restrict__ s_coeff5,
                                                    T* __restrict__ s_mt,
                                                    int* __restrict__ s_pivot_mask)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::solve_fused_wvmt_kernel<BLOCKSIZE, BLOCKDIM>),
            1,
            BLOCKSIZE,
            0,
            handle->stream,
            m,
            lower,
            main,
            upper,
            coeff1,
            coeff2,
            coeff3,
            coeff4,
            coeff5,
            w,
            v,
            mt,
            pivot_mask,
            s_coeff1,
            s_coeff2,
            s_coeff3,
            s_coeff4,
            s_coeff5,
            s_mt,
            s_pivot_mask);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    static rocsparse_status launch_solve_rhs_fused(rocsparse_handle handle,
                                                   rocsparse_int    m,
                                                   rocsparse_int    n,
                                                   rocsparse_int    ldb,
                                                   const T* __restrict__ coeff1,
                                                   const T* __restrict__ coeff2,
                                                   const T* __restrict__ coeff3,
                                                   const T* __restrict__ coeff4,
                                                   const T* __restrict__ coeff5,
                                                   const T* __restrict__ w,
                                                   const T* __restrict__ v,
                                                   const T* __restrict__ mt,
                                                   const int* __restrict__ mask,
                                                   const T* __restrict__ s_coeff1,
                                                   const T* __restrict__ s_coeff2,
                                                   const T* __restrict__ s_coeff3,
                                                   const T* __restrict__ s_coeff4,
                                                   const T* __restrict__ s_coeff5,
                                                   const T* __restrict__ s_mt,
                                                   const int* __restrict__ s_mask,
                                                   T* __restrict__ B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::solve_fused_rhs_kernel<BLOCKSIZE, BLOCKDIM, NCOLS>),
            n / NCOLS,
            BLOCKSIZE,
            0,
            handle->stream,
            m,
            ldb,
            coeff1,
            coeff2,
            coeff3,
            coeff4,
            coeff5,
            w,
            v,
            mt,
            mask,
            s_coeff1,
            s_coeff2,
            s_coeff3,
            s_coeff4,
            s_coeff5,
            s_mt,
            s_mask,
            B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKDIM, typename T>
    static constexpr uint32_t fused_ncols()
    {
        if(BLOCKDIM >= 32)
        {
            return 2;
        }
        if(BLOCKDIM >= 16)
        {
            return (sizeof(T) <= 4) ? 2 : 4;
        }
        if(BLOCKDIM >= 8)
        {
            return (sizeof(T) <= 4) ? 4 : 8;
        }
        return 8;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status tridiagonal_spike_solver_fused(rocsparse_handle handle,
                                                           rocsparse_int    m,
                                                           rocsparse_int    n,
                                                           rocsparse_int    ldb,
                                                           const T* __restrict__ lower,
                                                           const T* __restrict__ main,
                                                           const T* __restrict__ upper,
                                                           T* __restrict__ B,
                                                           T* __restrict__ scratch,
                                                           int* __restrict__ mask)
    {
        T* coeff1 = scratch + 0 * BLOCKSIZE * BLOCKDIM;
        T* coeff2 = scratch + 1 * BLOCKSIZE * BLOCKDIM;
        T* coeff3 = scratch + 2 * BLOCKSIZE * BLOCKDIM;
        T* coeff4 = scratch + 3 * BLOCKSIZE * BLOCKDIM;
        T* coeff5 = scratch + 4 * BLOCKSIZE * BLOCKDIM;
        T* w      = scratch + 5 * BLOCKSIZE * BLOCKDIM;
        T* v      = scratch + 6 * BLOCKSIZE * BLOCKDIM;
        T* mt     = scratch + 7 * BLOCKSIZE * BLOCKDIM;

        constexpr int S_SIZE = 2 * BLOCKSIZE;

        T* s_coeff1 = scratch + 8 * BLOCKSIZE * BLOCKDIM + 0 * S_SIZE;
        T* s_coeff2 = scratch + 8 * BLOCKSIZE * BLOCKDIM + 1 * S_SIZE;
        T* s_coeff3 = scratch + 8 * BLOCKSIZE * BLOCKDIM + 2 * S_SIZE;
        T* s_coeff4 = scratch + 8 * BLOCKSIZE * BLOCKDIM + 3 * S_SIZE;
        T* s_coeff5 = scratch + 8 * BLOCKSIZE * BLOCKDIM + 4 * S_SIZE;
        T* s_mt     = scratch + 8 * BLOCKSIZE * BLOCKDIM + 5 * S_SIZE;

        int* s_mask = mask + 32;

        RETURN_IF_HIP_ERROR(hipMemsetAsync(w, 0, sizeof(T) * BLOCKSIZE * BLOCKDIM, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(v, 0, sizeof(T) * BLOCKSIZE * BLOCKDIM, handle->stream));
        // RETURN_IF_HIP_ERROR(hipMemsetAsync(scratch, 0, sizeof(T) * 8 * BLOCKSIZE * BLOCKDIM, handle->stream));

        constexpr rocsparse_int NCOLS = fused_ncols<BLOCKDIM, T>();

        RETURN_IF_ROCSPARSE_ERROR((launch_solve_fused_wvmt<BLOCKSIZE, BLOCKDIM>(handle,
                                                                                m,
                                                                                lower,
                                                                                main,
                                                                                upper,
                                                                                coeff1,
                                                                                coeff2,
                                                                                coeff3,
                                                                                coeff4,
                                                                                coeff5,
                                                                                w,
                                                                                v,
                                                                                mt,
                                                                                mask,
                                                                                s_coeff1,
                                                                                s_coeff2,
                                                                                s_coeff3,
                                                                                s_coeff4,
                                                                                s_coeff5,
                                                                                s_mt,
                                                                                s_mask)));

        if(n < 500)
        {
            // For small number of columns we want to launch enough blocks to try and saturate the GPU
            // as much as possible. TODO: We should tune this based on the card being used.
            RETURN_IF_ROCSPARSE_ERROR((launch_solve_rhs_fused<BLOCKSIZE, BLOCKDIM, 1>(handle,
                                                                                      m,
                                                                                      n,
                                                                                      ldb,
                                                                                      coeff1,
                                                                                      coeff2,
                                                                                      coeff3,
                                                                                      coeff4,
                                                                                      coeff5,
                                                                                      w,
                                                                                      v,
                                                                                      mt,
                                                                                      mask,
                                                                                      s_coeff1,
                                                                                      s_coeff2,
                                                                                      s_coeff3,
                                                                                      s_coeff4,
                                                                                      s_coeff5,
                                                                                      s_mt,
                                                                                      s_mask,
                                                                                      B)));
        }
        else
        {
            int remainder = n % NCOLS;
            int nn        = n - remainder;

            RETURN_IF_ROCSPARSE_ERROR((launch_solve_rhs_fused<BLOCKSIZE, BLOCKDIM, NCOLS>(handle,
                                                                                          m,
                                                                                          nn,
                                                                                          ldb,
                                                                                          coeff1,
                                                                                          coeff2,
                                                                                          coeff3,
                                                                                          coeff4,
                                                                                          coeff5,
                                                                                          w,
                                                                                          v,
                                                                                          mt,
                                                                                          mask,
                                                                                          s_coeff1,
                                                                                          s_coeff2,
                                                                                          s_coeff3,
                                                                                          s_coeff4,
                                                                                          s_coeff5,
                                                                                          s_mt,
                                                                                          s_mask,
                                                                                          B)));

            if(remainder > 0)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (launch_solve_rhs_fused<BLOCKSIZE, BLOCKDIM, 1>(handle,
                                                                    m,
                                                                    remainder,
                                                                    ldb,
                                                                    coeff1,
                                                                    coeff2,
                                                                    coeff3,
                                                                    coeff4,
                                                                    coeff5,
                                                                    w,
                                                                    v,
                                                                    mt,
                                                                    mask,
                                                                    s_coeff1,
                                                                    s_coeff2,
                                                                    s_coeff3,
                                                                    s_coeff4,
                                                                    s_coeff5,
                                                                    s_mt,
                                                                    s_mask,
                                                                    B + ldb * nn)));
            }
        }
        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::gtsv_spike_solver_small_template(rocsparse_handle handle,
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
                                                             T**              mt_pad,
                                                             T**              scratch,
                                                             int**            mask,
                                                             rocsparse_int    level)
{
    ROCSPARSE_ROUTINE_TRACE;

    using solve_launch_ptr = rocsparse_status (*)(rocsparse_handle,
                                                  rocsparse_int,
                                                  rocsparse_int,
                                                  rocsparse_int,
                                                  const T*,
                                                  const T*,
                                                  const T*,
                                                  T*,
                                                  T*,
                                                  T*);

    using solve_fused_launch_ptr = rocsparse_status (*)(rocsparse_handle,
                                                        rocsparse_int,
                                                        rocsparse_int,
                                                        rocsparse_int,
                                                        const T*,
                                                        const T*,
                                                        const T*,
                                                        T*,
                                                        T*,
                                                        int*);

    static const std::map<rocsparse_int, solve_launch_ptr> solve_dispatch = {
        {4, tridiagonal_spike_solver_one_thread_per_system<256, 4>},
        {8, tridiagonal_spike_solver_one_thread_per_system<256, 8>},
        {16, tridiagonal_spike_solver_one_thread_per_system<256, 16>},
        {32, tridiagonal_spike_solver_one_thread_per_system<256, 32>},
        {64, tridiagonal_spike_solver_one_thread_per_system<256, 64>},
    };

    static const std::map<rocsparse_int, solve_fused_launch_ptr> solve_fused_dispatch = {
        {128, tridiagonal_spike_solver_fused<32, 4>},
        {256, tridiagonal_spike_solver_fused<32, 8>},
        {512, tridiagonal_spike_solver_fused<32, 16>},
        {1024, tridiagonal_spike_solver_fused<32, 32>},
    };

    if(m <= 64)
    {
        auto dispatch_it = solve_dispatch.lower_bound(m);
        if(dispatch_it != solve_dispatch.end())
        {
            RETURN_IF_ROCSPARSE_ERROR(
                dispatch_it->second(handle, m, n, ldb, dl, d, du, B, B_pad[level], mt_pad[level]));
        }
        else
        {
            return rocsparse_status_internal_error;
        }
    }
    else
    {
        auto dispatch_it = solve_fused_dispatch.lower_bound(m);
        if(dispatch_it != solve_fused_dispatch.end())
        {
            RETURN_IF_ROCSPARSE_ERROR(
                dispatch_it->second(handle, m, n, ldb, dl, d, du, B, scratch[level], mask[level]));
        }
        else
        {
            return rocsparse_status_internal_error;
        }
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(T)                                                        \
    template rocsparse_status rocsparse::gtsv_spike_solver_small_template<T>( \
        rocsparse_handle handle,                                              \
        rocsparse_int    m,                                                   \
        rocsparse_int    n,                                                   \
        rocsparse_int    ldb,                                                 \
        const T*         dl,                                                  \
        const T*         d,                                                   \
        const T*         du,                                                  \
        T*               B,                                                   \
        T**              dl_pad,                                              \
        T**              d_pad,                                               \
        T**              du_pad,                                              \
        T**              B_pad,                                               \
        T**              mt_pad,                                              \
        T**              scratch,                                             \
        int**            mask,                                                \
        rocsparse_int    level);

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE