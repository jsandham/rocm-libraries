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

#pragma once

#include <assert.h>

#include "rocsparse_common.hpp"

#include "pivot_mask.h"

namespace rocsparse
{
    template <uint32_t TILE_X, uint32_t TILE_Y, typename T>
    ROCSPARSE_KERNEL(TILE_X* TILE_Y)
    void transpose_kernel(
        int m, int n, int ldb, int ldx, const T* __restrict__ B, T* __restrict__ X)
    {
        for(int b = 0; b < n; b += TILE_Y * gridDim.y)
        {
            __shared__ T s_tile[TILE_Y][TILE_X + 1];

            const int tx = threadIdx.x; // [0, TILE_X - 1]
            const int ty = threadIdx.y; // [0, TILE_Y - 1]

            const int thread_id = ty * TILE_X + tx;

            const int block_row_B = blockIdx.x * TILE_X;
            const int block_col_B = blockIdx.y * TILE_Y;

            const int row_B = block_row_B + thread_id % TILE_X;
            const int col_B = b + block_col_B + thread_id / TILE_X;

            if(row_B < m && col_B < n)
            {
                s_tile[thread_id / TILE_X][thread_id % TILE_X] = B[row_B + col_B * ldb];
            }

            __syncthreads();

            const int row_X = b + block_col_B + thread_id % TILE_Y;
            const int col_X = block_row_B + thread_id / TILE_Y;

            if(row_X < n && col_X < m)
            {
                X[row_X + col_X * ldx] = s_tile[thread_id % TILE_Y][thread_id / TILE_Y];
            }

            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void solve_one_thread_per_system_kernel(int m,
                                            int n,
                                            const T* __restrict__ lower,
                                            const T* __restrict__ main,
                                            const T* __restrict__ upper,
                                            T* __restrict__ rhs)
    {
        static_assert(BLOCKDIM >= 2);
        static_assert(BLOCKDIM <= BLOCKSIZE);

        const int gid = BLOCKSIZE * blockIdx.x + threadIdx.x;

        if(gid >= n)
        {
            return;
        }

        T mt[BLOCKDIM];

        pivot_mask<BLOCKDIM> pivot{};

        int k  = 0;
        T   bk = main[k];

        while(k < m)
        {
            T ck   = upper[k];
            T ck_1 = (k < (m - 1)) ? upper[k + 1] : static_cast<T>(0);
            T bk_1 = (k < (m - 1)) ? main[k + 1] : static_cast<T>(0);
            T ak_1 = (k < (m - 1)) ? lower[k + 1] : static_cast<T>(0);
            T ak_2 = (k < (m - 2)) ? lower[k + 2] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            // 1x1 pivoting
            if(use_1x1_pivot || k == (m - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                mt[k] = ck * inv_bk;

                pivot.set_pivoting_to_1x1(k);

                // L * B * x = y
                const T rhsk = rhs[n * k + gid] * inv_bk;

                rhs[n * k + gid] = rhsk;

                if(k < (m - 1))
                {
                    rhs[n * (k + 1) + gid] += -(ak_1 * rhsk);

                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                mt[k] = -ck * ck_1 * det;

                pivot.set_pivoting_to2x2(k);

                if(k < (m - 1))
                {
                    mt[k + 1] = bk * ck_1 * det;

                    pivot.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                // L * B * x = y
                const T rhsk   = rhs[n * k + gid] * det;
                const T rhsk_1 = rhs[n * (k + 1) + gid] * det;

                rhs[n * k + gid]       = (bk_1 * rhsk - ck * rhsk_1);
                rhs[n * (k + 1) + gid] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (m - 2))
                {
                    rhs[n * (k + 2) + gid] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                    bk_2 = main[k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        assert(k == m);
        // at this point k = m. Could just set k = m - 1 here
        k--;

        k -= pivot.get_pivoting(k);

        // backward solve (M^T * rhs = rhs)
        while(k >= 0)
        {
            if(pivot.get_pivoting(k) == 1)
            {
                const T tmp = mt[k];

                rhs[n * k + gid] += -tmp * rhs[n * (k + 1) + gid];

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[k];
                const T tmp2 = mt[k - 1];

                rhs[n * k + gid] += -tmp1 * rhs[n * (k + 1) + gid];
                rhs[n * (k - 1) + gid] += -tmp2 * rhs[n * (k + 1) + gid];

                k -= 2;
            }
        }
    }

    template <typename T>
    ROCSPARSE_KERNEL(1)
    void LBMT_solve_mt_kernel(int m,
                              const T* __restrict__ lower,
                              const T* __restrict__ main,
                              const T* __restrict__ upper,
                              T* __restrict__ mt)
    {
        int k  = 0;
        T   bk = main[k];

        while(k < m)
        {
            T ck   = upper[k];
            T ck_1 = (k < (m - 1)) ? upper[k + 1] : static_cast<T>(0);
            T bk_1 = (k < (m - 1)) ? main[k + 1] : static_cast<T>(0);
            T ak_1 = (k < (m - 1)) ? lower[k + 1] : static_cast<T>(0);
            T ak_2 = (k < (m - 2)) ? lower[k + 2] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            // 1x1 pivoting
            if(use_1x1_pivot || k == (m - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                mt[k] = ck * inv_bk;

                if(k < (m - 1))
                {
                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                mt[k] = -ck * ck_1 * det;

                if(k < (m - 1))
                {
                    mt[k + 1] = bk * ck_1 * det;
                }

                T bk_2 = static_cast<T>(0);

                if(k < (m - 2))
                {
                    bk_2 = main[k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void LBMT_solve_rhs_one_thread_per_system_kernel(int m,
                                                     int n,
                                                     const T* __restrict__ lower,
                                                     const T* __restrict__ main,
                                                     const T* __restrict__ upper,
                                                     const T* __restrict__ mt,
                                                     T* __restrict__ rhs)
    {
        static_assert(BLOCKDIM >= 2);
        static_assert(BLOCKDIM <= BLOCKSIZE);

        const int gid = BLOCKSIZE * blockIdx.x + threadIdx.x;

        __shared__ T shared_mt[BLOCKDIM];

        if(threadIdx.x < m)
        {
            shared_mt[threadIdx.x] = mt[threadIdx.x];
        }

        __syncthreads();

        if(gid >= n)
        {
            return;
        }

        pivot_mask<BLOCKDIM> pivot{};

        int k  = 0;
        T   bk = main[k];

        while(k < m)
        {
            T ck   = upper[k];
            T ck_1 = (k < (m - 1)) ? upper[k + 1] : static_cast<T>(0);
            T bk_1 = (k < (m - 1)) ? main[k + 1] : static_cast<T>(0);
            T ak_1 = (k < (m - 1)) ? lower[k + 1] : static_cast<T>(0);
            T ak_2 = (k < (m - 2)) ? lower[k + 2] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            // 1x1 pivoting
            if(use_1x1_pivot || k == (m - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                pivot.set_pivoting_to_1x1(k);

                // L * B * x = y
                const T rhsk = rhs[n * k + gid] * inv_bk;

                rhs[n * k + gid] = rhsk;

                if(k < (m - 1))
                {
                    rhs[n * (k + 1) + gid] += -(ak_1 * rhsk);

                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                pivot.set_pivoting_to2x2(k);

                if(k < (m - 1))
                {
                    pivot.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                // L * B * x = y
                const T rhsk   = rhs[n * k + gid] * det;
                const T rhsk_1 = rhs[n * (k + 1) + gid] * det;

                rhs[n * k + gid]       = (bk_1 * rhsk - ck * rhsk_1);
                rhs[n * (k + 1) + gid] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (m - 2))
                {
                    rhs[n * (k + 2) + gid] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                    bk_2 = main[k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        assert(k == m);
        // at this point k = m. Could just set k = m - 1 here
        k--;

        k -= pivot.get_pivoting(k);

        // backward solve (M^T * rhs = rhs)
        while(k >= 0)
        {
            if(pivot.get_pivoting(k) == 1)
            {
                const T tmp = shared_mt[k];

                rhs[n * k + gid] += -tmp * rhs[n * (k + 1) + gid];

                k -= 1;
            }
            else
            {
                const T tmp1 = shared_mt[k];
                const T tmp2 = shared_mt[k - 1];

                rhs[n * k + gid] += -tmp1 * rhs[n * (k + 1) + gid];
                rhs[n * (k - 1) + gid] += -tmp2 * rhs[n * (k + 1) + gid];

                k -= 2;
            }
        }
    }
}