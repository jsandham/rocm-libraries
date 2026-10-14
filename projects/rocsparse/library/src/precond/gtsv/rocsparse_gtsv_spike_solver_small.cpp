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
#include <map>

#include "internal/precond/rocsparse_gtsv.h"
#include "rocsparse_grid.hpp"
#include "rocsparse_gtsv.hpp"
#include "rocsparse_gtsv_spike_solver.hpp"

#include "gtsv_spike_one_thread_per_system_device.h"
#include "gtsv_spike_fused_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_transpose(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const T* B, T* X)
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

        const rocsparse_int ldb = m;
        const rocsparse_int ldx = n;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::transpose_kernel<TILE_X, TILE_Y>),
                                            numBlocks, threadsPerBlock,
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
    static rocsparse_status launch_LBMT_solve_mt(rocsparse_handle handle, rocsparse_int m,
                                     const T* __restrict__ dl,
                                     const T* __restrict__ d,
                                     const T* __restrict__ du,
                                     T* __restrict__ mt)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::LBMT_solve_mt_kernel),
                                            1, 1,
                                           0,
                                           handle->stream,
                                           m,
                                           dl,
                                           d,
                                           du,
                                           mt);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status launch_LBMT_solve_rhs_one_thread_per_system(rocsparse_handle handle, rocsparse_int m,
                                                            rocsparse_int n,
                                                            const T* __restrict__ dl,
                                                            const T* __restrict__ d,
                                                            const T* __restrict__ du,
                                                            const T* __restrict__ mt,
                                                            T* __restrict__ rhs)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::LBMT_solve_rhs_one_thread_per_system_kernel<BLOCKSIZE, BLOCKDIM>),
                                            (n - 1) / BLOCKSIZE + 1, BLOCKSIZE,
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
    static rocsparse_status launch_solve_fused(rocsparse_handle handle, rocsparse_int m,
                                    rocsparse_int n,
                                    const T* __restrict__ dl,
                                    const T* __restrict__ d,
                                    const T* __restrict__ du,
                                    const T* __restrict__ B,
                                    T* __restrict__ X)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::solve_fused_kernel<BLOCKSIZE, BLOCKDIM, NCOLS>),
                                            n / NCOLS, BLOCKSIZE,
                                           0,
                                           handle->stream,
                                           m,
                                           dl,
                                           d,
                                           du,
                                           B,
                                           X);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status tridiagonal_spike_solver_one_thread_per_system(rocsparse_handle handle, rocsparse_int m,
                                                               rocsparse_int n,
                                                               const T* __restrict__ dl,
                                                               const T* __restrict__ d,
                                                               const T* __restrict__ du,
                                                               const T* __restrict__ B,
                                                               T* __restrict__ X,
                                                               T** B_pad,
                                                               T** mt,
                                                               rocsparse_int level)
    {
        ROCSPARSE_ROUTINE_TRACE;

        launch_transpose<BLOCKSIZE, BLOCKDIM>(handle, m, n, B, B_pad[level]);
        launch_LBMT_solve_mt(handle, m, dl, d, du, mt[level]);
        launch_LBMT_solve_rhs_one_thread_per_system<BLOCKSIZE, BLOCKDIM>(
            handle, m, n, dl, d, du, mt[level], B_pad[level]);
        launch_transpose<BLOCKSIZE, BLOCKDIM>(handle, n, m, B_pad[level], X);
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static rocsparse_status tridiagonal_spike_solver_fused(rocsparse_handle handle, rocsparse_int m,
                                    rocsparse_int n,
                                    const T* __restrict__ dl,
                                    const T* __restrict__ d,
                                    const T* __restrict__ du,
                                    const T* __restrict__ B,
                                    T* __restrict__ X)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(n < 500)
        {
            // For small number of columns we want to launch enough blocks to try and saturate the GPU
            // as much as possible. TODO: We should tune this based on the card being used.
            launch_solve_fused<BLOCKSIZE, BLOCKDIM, 1>(handle, m, n, dl, d, du, B, X);
        }
        else
        {
            rocsparse_int remainder = n % 8;
            rocsparse_int nn        = n - remainder;

            launch_solve_fused<BLOCKSIZE, BLOCKDIM, 8>(handle, m, nn, dl, d, du, B, X);

            if(remainder > 0)
            {
                launch_solve_fused<BLOCKSIZE, BLOCKDIM, 1>(handle, m, remainder, dl, d, du, B + m * nn, X + m * nn);
            }
        }
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
                                                        T**              mt_pad)
{
    ROCSPARSE_ROUTINE_TRACE;

    using solve_launch_ptr = void (*)(rocsparse_handle, rocsparse_int, rocsparse_int, const T*, const T*, const T*, const T*, T*, T**, T**, rocsparse_int);
    using solve_fused_launch_ptr = void (*)(rocsparse_handle, rocsparse_int, rocsparse_int, const T*, const T*, const T*, const T*, T*);

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
            dispatch_it->second(handle, m, n, dl, d, du, B, X, B_pad, mt_pad, level);
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
            dispatch_it->second(handle, m, n, dl, d, du, B, X);
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
 #define INSTANTIATE(T)                                                                   \
     template rocsparse_status rocsparse::gtsv_spike_solver_small_template<0, T>(               \
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
         T**              mt_pad);
 
 INSTANTIATE(float);
 INSTANTIATE(double);
 
 #undef INSTANTIATE