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
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void solve_fused_kernel(int m,
                            const T* __restrict__ lower,
                            const T* __restrict__ main,
                            const T* __restrict__ upper,
                            T* __restrict__ B)
    {
        static_assert(BLOCKDIM >= 2);
        static_assert(BLOCKDIM % 2 == 0);
        static_assert(BLOCKSIZE % 2 == 0);

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        T rhs_reg[BLOCKDIM * NCOLS];
        for(int c = 0; c < NCOLS; c++)
        {
            for(int i = 0; i < BLOCKDIM; i++)
            {
                rhs_reg[BLOCKDIM * c + i] = (BLOCKDIM * tid + i < m)
                                                ? B[BLOCKDIM * tid + i + m * (NCOLS * bid + c)]
                                                : static_cast<T>(0);
            }
        }

        T w_reg[BLOCKDIM];
        T v_reg[BLOCKDIM];
        for(int i = 0; i < BLOCKDIM; i++)
        {
            w_reg[i] = static_cast<T>(0);
            v_reg[i] = static_cast<T>(0);
        }

        {
            T lower_reg[BLOCKDIM];
            T main_reg[BLOCKDIM];
            T upper_reg[BLOCKDIM];
            T mt_reg[BLOCKDIM];

            for(int i = 0; i < BLOCKDIM; i++)
            {
                lower_reg[i]
                    = (BLOCKDIM * tid + i < m) ? lower[BLOCKDIM * tid + i] : static_cast<T>(0);
                main_reg[i]
                    = (BLOCKDIM * tid + i < m) ? main[BLOCKDIM * tid + i] : static_cast<T>(1);
                upper_reg[i]
                    = (BLOCKDIM * tid + i < m) ? upper[BLOCKDIM * tid + i] : static_cast<T>(0);
                mt_reg[i] = static_cast<T>(0);
            }

            // LBMT solve
            T bk = main_reg[0];

            pivot_mask<BLOCKDIM> pivot{};

            w_reg[0]            = lower_reg[0];
            v_reg[BLOCKDIM - 1] = upper_reg[(BLOCKDIM - 1)];

            int k = 0;
            while(k < BLOCKDIM)
            {
                T ck   = upper_reg[k];
                T ck_1 = (k < (BLOCKDIM - 1)) ? upper_reg[(k + 1)] : static_cast<T>(0);
                T bk_1 = (k < (BLOCKDIM - 1)) ? main_reg[(k + 1)] : static_cast<T>(0);
                T ak_1 = (k < (BLOCKDIM - 1)) ? lower_reg[(k + 1)] : static_cast<T>(0);
                T ak_2 = (k < (BLOCKDIM - 2)) ? lower_reg[(k + 2)] : static_cast<T>(0);

                // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
                // pivoting criteria
                const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

                // 1x1 pivoting
                if(use_1x1_pivot || k == (BLOCKDIM - 1))
                {
                    const T inv_bk = static_cast<T>(1) / bk;

                    const T wk = w_reg[k];
                    const T vk = v_reg[k];

                    w_reg[k]  = wk * inv_bk;
                    v_reg[k]  = vk * inv_bk;
                    mt_reg[k] = ck * inv_bk;

                    pivot.set_pivoting_to_1x1(k);

                    if(k < (BLOCKDIM - 1))
                    {
                        w_reg[k + 1] += -ak_1 * wk * inv_bk;
                    }

                    // L * B * x = y
                    for(int c = 0; c < NCOLS; c++)
                    {
                        const T rhsk = rhs_reg[BLOCKDIM * c + k] * inv_bk;

                        rhs_reg[BLOCKDIM * c + k] = rhsk;

                        if(k < (BLOCKDIM - 1))
                        {
                            rhs_reg[BLOCKDIM * c + (k + 1)] += -(ak_1 * rhsk);
                        }
                    }

                    if(k < (BLOCKDIM - 1))
                    {
                        bk_1 = bk_1 - ak_1 * ck * inv_bk;
                    }

                    bk = bk_1;

                    k += 1;
                }
                else
                {
                    const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                    const T wk   = w_reg[k];
                    const T wk_1 = w_reg[k + 1];
                    const T vk   = v_reg[k];
                    const T vk_1 = v_reg[k + 1];

                    w_reg[k]  = (bk_1 * wk - ck * wk_1) * det;
                    v_reg[k]  = (bk_1 * vk - ck * vk_1) * det;
                    mt_reg[k] = -ck * ck_1 * det;

                    pivot.set_pivoting_to2x2(k);

                    if(k < (BLOCKDIM - 1))
                    {
                        w_reg[k + 1]  = (-ak_1 * wk + bk * wk_1) * det;
                        v_reg[k + 1]  = (-ak_1 * vk + bk * vk_1) * det;
                        mt_reg[k + 1] = bk * ck_1 * det;

                        pivot.set_pivoting_to2x2(k + 1);
                    }

                    T bk_2 = static_cast<T>(0);

                    if(k < (BLOCKDIM - 2))
                    {
                        w_reg[k + 2] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
                    }

                    // |bk   ck  ||xk  |   |rhsk   |
                    // |ak_1 bk_1||xk_1| = |rhsk _1|
                    //
                    //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
                    //                                  |-ak_1  bk |

                    // L * B * x = y
                    for(int c = 0; c < NCOLS; c++)
                    {
                        const T rhsk   = rhs_reg[BLOCKDIM * c + k] * det;
                        const T rhsk_1 = rhs_reg[BLOCKDIM * c + k + 1] * det;

                        rhs_reg[BLOCKDIM * c + k]     = (bk_1 * rhsk - ck * rhsk_1);
                        rhs_reg[BLOCKDIM * c + k + 1] = (-ak_1 * rhsk + bk * rhsk_1);

                        if(k < (BLOCKDIM - 2))
                        {
                            rhs_reg[BLOCKDIM * c + k + 2]
                                += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);
                        }
                    }

                    if(k < (BLOCKDIM - 2))
                    {
                        bk_2 = main_reg[k + 2];
                        bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                    }

                    bk = bk_2;
                    k += 2;
                }
            }

            assert(k == BLOCKDIM);
            // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
            k--;

            k -= pivot.get_pivoting(k);

            // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
            while(k >= 0)
            {
                if(pivot.get_pivoting(k) == 1)
                {
                    const T tmp = mt_reg[k];

                    w_reg[k] += -tmp * w_reg[k + 1];
                    v_reg[k] += -tmp * v_reg[k + 1];

                    for(int c = 0; c < NCOLS; c++)
                    {
                        rhs_reg[BLOCKDIM * c + k] += -tmp * rhs_reg[BLOCKDIM * c + k + 1];
                    }

                    k -= 1;
                }
                else
                {
                    const T tmp1 = mt_reg[k];
                    const T tmp2 = mt_reg[k - 1];

                    w_reg[k] += -tmp1 * w_reg[k + 1];
                    w_reg[k - 1] += -tmp2 * w_reg[k + 1];
                    v_reg[k] += -tmp1 * v_reg[k + 1];
                    v_reg[k - 1] += -tmp2 * v_reg[k + 1];

                    for(int c = 0; c < NCOLS; c++)
                    {
                        rhs_reg[BLOCKDIM * c + k] += -tmp1 * rhs_reg[BLOCKDIM * c + k + 1];
                        rhs_reg[BLOCKDIM * c + k - 1] += -tmp2 * rhs_reg[BLOCKDIM * c + k + 1];
                    }

                    k -= 2;
                }
            }
        }

        // Fill S system
        constexpr int S_SIZE = 2 * BLOCKSIZE;

        __shared__ T S_lower[S_SIZE];
        __shared__ T S_main[S_SIZE];
        __shared__ T S_upper[S_SIZE];
        __shared__ T S_rhs[S_SIZE * NCOLS];

        S_lower[2 * tid]     = static_cast<T>(1);
        S_lower[2 * tid + 1] = w_reg[(BLOCKDIM - 1)];

        S_upper[2 * tid]     = v_reg[0];
        S_upper[2 * tid + 1] = static_cast<T>(1);

        S_main[2 * tid]     = w_reg[0];
        S_main[2 * tid + 1] = v_reg[(BLOCKDIM - 1)];

        if(tid == 0)
        {
            S_lower[0] = static_cast<T>(0);
            S_lower[1] = static_cast<T>(0);

            S_main[0] = static_cast<T>(1);
        }

        if(tid == (BLOCKSIZE - 1))
        {
            S_upper[2 * (BLOCKSIZE - 1)]     = static_cast<T>(0);
            S_upper[2 * (BLOCKSIZE - 1) + 1] = static_cast<T>(0);

            S_main[2 * (BLOCKSIZE - 1) + 1] = static_cast<T>(1);
        }

        for(int c = 0; c < NCOLS; c++)
        {
            S_rhs[S_SIZE * c + 2 * tid]     = rhs_reg[BLOCKDIM * c + 0];
            S_rhs[S_SIZE * c + 2 * tid + 1] = rhs_reg[BLOCKDIM * c + (BLOCKDIM - 1)];
        }

        __syncthreads();

        __shared__ T S_mt[S_SIZE];

        // Solve S system (solved by 1 thread)
        if(tid == 0)
        {
            pivot_mask<S_SIZE> pivot{};

            int k  = 0;
            T   bk = S_main[k];

            while(k < S_SIZE)
            {
                T ck   = S_upper[k];
                T ck_1 = (k < (S_SIZE - 1)) ? S_upper[k + 1] : static_cast<T>(0);
                T bk_1 = (k < (S_SIZE - 1)) ? S_main[k + 1] : static_cast<T>(0);
                T ak_1 = (k < (S_SIZE - 1)) ? S_lower[k + 1] : static_cast<T>(0);
                T ak_2 = (k < (S_SIZE - 2)) ? S_lower[k + 2] : static_cast<T>(0);

                // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
                // pivoting criteria
                const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

                // 1x1 pivoting
                if(use_1x1_pivot || k == (S_SIZE - 1))
                {
                    const T inv_bk = static_cast<T>(1) / bk;

                    S_mt[k] = ck * inv_bk;

                    pivot.set_pivoting_to_1x1(k);

                    // L * B * x = y
                    for(int c = 0; c < NCOLS; c++)
                    {
                        const T rhsk = S_rhs[S_SIZE * c + k] * inv_bk;

                        S_rhs[S_SIZE * c + k] = rhsk;

                        if(k < (S_SIZE - 1))
                        {
                            S_rhs[S_SIZE * c + k + 1] += -(ak_1 * rhsk);
                        }
                    }

                    if(k < (S_SIZE - 1))
                    {
                        bk_1 = bk_1 - ak_1 * ck * inv_bk;
                    }

                    bk = bk_1;

                    k += 1;
                }
                else
                {
                    const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                    S_mt[k] = -ck * ck_1 * det;

                    pivot.set_pivoting_to2x2(k);

                    if(k < (S_SIZE - 1))
                    {
                        S_mt[k + 1] = bk * ck_1 * det;

                        pivot.set_pivoting_to2x2(k + 1);
                    }

                    T bk_2 = static_cast<T>(0);

                    // L * B * x = y
                    for(int c = 0; c < NCOLS; c++)
                    {
                        const T rhsk   = S_rhs[S_SIZE * c + k] * det;
                        const T rhsk_1 = S_rhs[S_SIZE * c + k + 1] * det;

                        S_rhs[S_SIZE * c + k]     = (bk_1 * rhsk - ck * rhsk_1);
                        S_rhs[S_SIZE * c + k + 1] = (-ak_1 * rhsk + bk * rhsk_1);

                        if(k < (S_SIZE - 2))
                        {
                            S_rhs[S_SIZE * c + k + 2]
                                += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);
                        }
                    }

                    if(k < (S_SIZE - 2))
                    {
                        bk_2 = S_main[k + 2];
                        bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                    }

                    bk = bk_2;
                    k += 2;
                }
            }

            assert(k == S_SIZE);
            // at this point k = S_SIZE. Could just set k = S_SIZE - 1 here
            k--;

            k -= pivot.get_pivoting(k);

            // backward solve (M^T * rhs = rhs)
            while(k >= 0)
            {
                if(pivot.get_pivoting(k) == 1)
                {
                    const T tmp = S_mt[k];

                    for(int c = 0; c < NCOLS; c++)
                    {
                        S_rhs[S_SIZE * c + k] += -tmp * S_rhs[S_SIZE * c + k + 1];
                    }

                    k -= 1;
                }
                else
                {
                    const T tmp1 = S_mt[k];
                    const T tmp2 = S_mt[k - 1];

                    for(int c = 0; c < NCOLS; c++)
                    {
                        S_rhs[S_SIZE * c + k] += -tmp1 * S_rhs[S_SIZE * c + k + 1];
                        S_rhs[S_SIZE * c + k - 1] += -tmp2 * S_rhs[S_SIZE * c + k + 1];
                    }

                    k -= 2;
                }
            }
        }

        __syncthreads();

        // Scatter
        for(int c = 0; c < NCOLS; c++)
        {
            // After the swap loop, element at position 2*tid is:
            //   tid == 0  -> S_rhs[0]       (not touched by the swap)
            //   tid  > 0  -> S_rhs[2*tid - 1] (position 2*tid was swapped with 2*tid-1)
            const T val_even = (tid == 0) ? S_rhs[S_SIZE * c + 0] : S_rhs[S_SIZE * c + 2 * tid - 1];

            // After the swap loop, element at position 2*i+1 is:
            //   2*tid+1 < S_SIZE-1  -> S_rhs[2*tid + 2] (swapped with its right neighbour)
            //   2*tid+1 == S_SIZE-1 -> S_rhs[S_SIZE-1] (last element, not touched)
            const T val_odd = (2 * tid + 1 < S_SIZE - 1) ? S_rhs[S_SIZE * c + 2 * tid + 2]
                                                         : S_rhs[S_SIZE * c + S_SIZE - 1];

            rhs_reg[BLOCKDIM * c + 0]            = val_even;
            rhs_reg[BLOCKDIM * c + BLOCKDIM - 1] = val_odd;

            // backward solve (S * x = rhs)
            const T tmp1 = __shfl_sync(
                0xFFFFFFFFULL, rhs_reg[BLOCKDIM * c + (BLOCKDIM - 1)], tid - 1, BLOCKSIZE);
            const T tmp2
                = __shfl_sync(0xFFFFFFFFULL, rhs_reg[BLOCKDIM * c + 0], tid + 1, BLOCKSIZE);
            const T x1 = (tid >= 1) ? tmp1 : static_cast<T>(0);
            const T x2 = (tid < (BLOCKSIZE - 1)) ? tmp2 : static_cast<T>(0);

            for(int j = 1; j < BLOCKDIM - 1; j++)
            {
                rhs_reg[BLOCKDIM * c + j]
                    = rhs_reg[BLOCKDIM * c + j] - w_reg[j] * x1 - v_reg[j] * x2;
            }
        }

        // Transpose back
        for(int c = 0; c < NCOLS; c++)
        {
            for(int i = 0; i < BLOCKDIM; i++)
            {
                if((BLOCKDIM * tid + i) < m)
                {
                    B[BLOCKDIM * tid + i + m * (NCOLS * bid + c)] = rhs_reg[BLOCKDIM * c + i];
                }
            }
        }
    }
}