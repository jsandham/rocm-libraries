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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    static rocsparse_status launch_solve_fused(rocsparse_handle handle,
                                               rocsparse_int    m,
                                               rocsparse_int    n,
                                               const T* __restrict__ dl,
                                               const T* __restrict__ d,
                                               const T* __restrict__ du,
                                               T* __restrict__ B)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::solve_fused_kernel<BLOCKSIZE, BLOCKDIM, NCOLS>),
            n / NCOLS,
            BLOCKSIZE,
            0,
            handle->stream,
            m,
            dl,
            d,
            du,
            B);
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
                                                                           T**           B_pad,
                                                                           T**           mt,
                                                                           rocsparse_int level)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // B is m by n with leading dimension ldb. The solve works on the transpose,
        // packed with leading dimension n, then the result is written back with ldb.
        RETURN_IF_ROCSPARSE_ERROR(
            (launch_transpose<BLOCKSIZE, BLOCKDIM>(handle, m, n, ldb, n, B, B_pad[level])));
        RETURN_IF_ROCSPARSE_ERROR(launch_LBMT_solve_mt(handle, m, dl, d, du, mt[level]));
        RETURN_IF_ROCSPARSE_ERROR((launch_LBMT_solve_rhs_one_thread_per_system<BLOCKSIZE, BLOCKDIM>(
            handle, m, n, dl, d, du, mt[level], B_pad[level])));
        RETURN_IF_ROCSPARSE_ERROR(
            (launch_transpose<BLOCKSIZE, BLOCKDIM>(handle, n, m, n, ldb, B_pad[level], B)));
        return rocsparse_status_success;
    }

    // Version 2 of the fused solver: factor once, then apply to the right hand sides.
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_solve_fused_factor_v2(rocsparse_handle handle,
                                                         rocsparse_int    m,
                                                         const T* __restrict__ dl,
                                                         const T* __restrict__ d,
                                                         const T* __restrict__ du,
                                                         T* __restrict__ factors)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static_assert(fused_v2_layout<BLOCKSIZE, BLOCKDIM>::size
                      <= gtsv_buffer_data<T>::FUSED_V2_FACTOR_SIZE);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::solve_fused_factor_kernel_v2<BLOCKSIZE, BLOCKDIM>),
            1,
            BLOCKSIZE,
            0,
            handle->stream,
            m,
            dl,
            d,
            du,
            factors);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    static rocsparse_status launch_solve_fused_v2(rocsparse_handle handle,
                                                  rocsparse_int    m,
                                                  rocsparse_int    n,
                                                  rocsparse_int    ldb,
                                                  const T* __restrict__ factors,
                                                  T* __restrict__ B)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::solve_fused_kernel_v2<BLOCKSIZE, BLOCKDIM, NCOLS>),
            n / NCOLS,
            BLOCKSIZE,
            0,
            handle->stream,
            m,
            ldb,
            factors,
            B);
        return rocsparse_status_success;
    }

    // Number of right hand sides handled by each block in version 2. Each thread keeps
    // NCOLS * BLOCKDIM values in registers. The values were tuned on gfx1201 by
    // benchmarking n = 100000 columns.
    template <uint32_t BLOCKDIM, typename T>
    static constexpr uint32_t fused_v2_ncols()
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
    static rocsparse_status tridiagonal_spike_solver_fused_v2(rocsparse_handle handle,
                                                              rocsparse_int    m,
                                                              rocsparse_int    n,
                                                              rocsparse_int    ldb,
                                                              const T* __restrict__ dl,
                                                              const T* __restrict__ d,
                                                              const T* __restrict__ du,
                                                              T* __restrict__ B,
                                                              T* __restrict__ factors)
    {
        ROCSPARSE_ROUTINE_TRACE;

        constexpr rocsparse_int NCOLS = fused_v2_ncols<BLOCKDIM, T>();

        RETURN_IF_ROCSPARSE_ERROR(
            (launch_solve_fused_factor_v2<BLOCKSIZE, BLOCKDIM>(handle, m, dl, d, du, factors)));

        if(n < 500)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (launch_solve_fused_v2<BLOCKSIZE, BLOCKDIM, 1>(handle, m, n, ldb, factors, B)));
        }
        else
        {
            const rocsparse_int remainder = n % NCOLS;
            const rocsparse_int nn        = n - remainder;

            RETURN_IF_ROCSPARSE_ERROR((launch_solve_fused_v2<BLOCKSIZE, BLOCKDIM, NCOLS>(
                handle, m, nn, ldb, factors, B)));

            if(remainder > 0)
            {
                RETURN_IF_ROCSPARSE_ERROR((launch_solve_fused_v2<BLOCKSIZE, BLOCKDIM, 1>(
                    handle, m, remainder, ldb, factors, B + static_cast<int64_t>(ldb) * nn)));
            }
        }
        return rocsparse_status_success;
    }

    // Temporary switch for A/B testing: set ROCSPARSE_GTSV_FUSED_V2=1 to use version 2.
    static bool gtsv_fused_v2_enabled()
    {
        const char* env = std::getenv("ROCSPARSE_GTSV_FUSED_V2");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status tridiagonal_spike_solver_fused(rocsparse_handle handle,
                                                           rocsparse_int    m,
                                                           rocsparse_int    n,
                                                           rocsparse_int    ldb,
                                                           const T* __restrict__ dl,
                                                           const T* __restrict__ d,
                                                           const T* __restrict__ du,
                                                           T* __restrict__ B,
                                                           T* __restrict__ factors)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Version 1 does not use ldb (it assumes ldb == m) nor the factor workspace.
        (void)ldb;
        (void)factors;

        if(n < 500)
        {
            // For small number of columns we want to launch enough blocks to try and saturate the GPU
            // as much as possible. TODO: We should tune this based on the card being used.
            RETURN_IF_ROCSPARSE_ERROR(
                (launch_solve_fused<BLOCKSIZE, BLOCKDIM, 1>(handle, m, n, dl, d, du, B)));
        }
        else
        {
            rocsparse_int remainder = n % 8;
            rocsparse_int nn        = n - remainder;

            RETURN_IF_ROCSPARSE_ERROR(
                (launch_solve_fused<BLOCKSIZE, BLOCKDIM, 8>(handle, m, nn, dl, d, du, B)));

            if(remainder > 0)
            {
                RETURN_IF_ROCSPARSE_ERROR((launch_solve_fused<BLOCKSIZE, BLOCKDIM, 1>(
                    handle, m, remainder, dl, d, du, B + m * nn)));
            }
        }
        return rocsparse_status_success;
    }
}

template <uint32_t LEVEL, typename T>
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
                                                             T**              factor_pad)
{
    ROCSPARSE_ROUTINE_TRACE;

    using solve_launch_ptr       = rocsparse_status (*)(rocsparse_handle,
                                                  rocsparse_int,
                                                  rocsparse_int,
                                                  rocsparse_int,
                                                  const T*,
                                                  const T*,
                                                  const T*,
                                                  T*,
                                                  T**,
                                                  T**,
                                                  rocsparse_int);
    using solve_fused_launch_ptr = rocsparse_status (*)(rocsparse_handle,
                                                        rocsparse_int,
                                                        rocsparse_int,
                                                        rocsparse_int,
                                                        const T*,
                                                        const T*,
                                                        const T*,
                                                        T*,
                                                        T*);

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

    static const std::map<rocsparse_int, solve_fused_launch_ptr> solve_fused_v2_dispatch = {
        {128, tridiagonal_spike_solver_fused_v2<32, 4>},
        {256, tridiagonal_spike_solver_fused_v2<32, 8>},
        {512, tridiagonal_spike_solver_fused_v2<32, 16>},
        {1024, tridiagonal_spike_solver_fused_v2<32, 32>},
    };

    static const bool use_fused_v2 = gtsv_fused_v2_enabled();

    if(m <= 64)
    {
        auto dispatch_it = solve_dispatch.lower_bound(m);
        if(dispatch_it != solve_dispatch.end())
        {
            RETURN_IF_ROCSPARSE_ERROR(
                dispatch_it->second(handle, m, n, ldb, dl, d, du, B, B_pad, mt_pad, LEVEL));
        }
        else
        {
            return rocsparse_status_internal_error;
        }
    }
    else
    {
        const auto& fused_dispatch = use_fused_v2 ? solve_fused_v2_dispatch : solve_fused_dispatch;

        auto dispatch_it = fused_dispatch.lower_bound(m);
        if(dispatch_it != fused_dispatch.end())
        {
            RETURN_IF_ROCSPARSE_ERROR(
                dispatch_it->second(handle, m, n, ldb, dl, d, du, B, factor_pad[LEVEL]));
        }
        else
        {
            return rocsparse_status_internal_error;
        }
    }

    return rocsparse_status_success;
}

// Level 0 is the entry point from gtsv_template. Instantiating it also emits the
// deeper levels, because each level calls the next one under if constexpr.
#define INSTANTIATE(T)                                                           \
    template rocsparse_status rocsparse::gtsv_spike_solver_small_template<0, T>( \
        rocsparse_handle handle,                                                 \
        rocsparse_int    m,                                                      \
        rocsparse_int    n,                                                      \
        rocsparse_int    ldb,                                                    \
        const T*         dl,                                                     \
        const T*         d,                                                      \
        const T*         du,                                                     \
        T*               B,                                                      \
        T**              dl_pad,                                                 \
        T**              d_pad,                                                  \
        T**              du_pad,                                                 \
        T**              B_pad,                                                  \
        T**              mt_pad,                                                 \
        T**              factor_pad);

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE