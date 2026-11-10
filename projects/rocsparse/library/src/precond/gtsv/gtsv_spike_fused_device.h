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
    // template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    // ROCSPARSE_KERNEL(BLOCKSIZE)
    // void wvmt_solve_fused_kernel(int m,
    //                              const T* __restrict__ lower,
    //                              const T* __restrict__ main,
    //                              const T* __restrict__ upper,
    //                              T* __restrict__ B)
    // {
    //     static_assert(BLOCKDIM >= 2);
    //     static_assert(BLOCKDIM % 2 == 0);
    //     static_assert(BLOCKSIZE % 2 == 0);

    //     const int tid = threadIdx.x;
    //     const int bid = blockIdx.x;

    //     T w_reg[BLOCKDIM];
    //     T v_reg[BLOCKDIM];
    //     for(int i = 0; i < BLOCKDIM; i++)
    //     {
    //         w_reg[i] = static_cast<T>(0);
    //         v_reg[i] = static_cast<T>(0);
    //     }

    //     {
    //         T lower_reg[BLOCKDIM];
    //         T main_reg[BLOCKDIM];
    //         T upper_reg[BLOCKDIM];
    //         T mt_reg[BLOCKDIM];

    //         for(int i = 0; i < BLOCKDIM; i++)
    //         {
    //             lower_reg[i]
    //                 = (BLOCKDIM * tid + i < m) ? lower[BLOCKDIM * tid + i] : static_cast<T>(0);
    //             main_reg[i]
    //                 = (BLOCKDIM * tid + i < m) ? main[BLOCKDIM * tid + i] : static_cast<T>(1);
    //             upper_reg[i]
    //                 = (BLOCKDIM * tid + i < m) ? upper[BLOCKDIM * tid + i] : static_cast<T>(0);
    //             mt_reg[i] = static_cast<T>(0);
    //         }

    //         // LBMT solve
    //         T bk = main_reg[0];

    //         pivot_mask<BLOCKDIM> pivot{};

    //         w_reg[0]            = lower_reg[0];
    //         v_reg[BLOCKDIM - 1] = upper_reg[(BLOCKDIM - 1)];

    //         int k = 0;
    //         while(k < BLOCKDIM)
    //         {
    //             T ck   = upper_reg[k];
    //             T ck_1 = (k < (BLOCKDIM - 1)) ? upper_reg[(k + 1)] : static_cast<T>(0);
    //             T bk_1 = (k < (BLOCKDIM - 1)) ? main_reg[(k + 1)] : static_cast<T>(0);
    //             T ak_1 = (k < (BLOCKDIM - 1)) ? lower_reg[(k + 1)] : static_cast<T>(0);
    //             T ak_2 = (k < (BLOCKDIM - 2)) ? lower_reg[(k + 2)] : static_cast<T>(0);

    //             // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
    //             // pivoting criteria
    //             const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

    //             // 1x1 pivoting
    //             if(use_1x1_pivot || k == (BLOCKDIM - 1))
    //             {
    //                 const T inv_bk = static_cast<T>(1) / bk;

    //                 const T wk = w_reg[k];
    //                 const T vk = v_reg[k];

    //                 w_reg[k]  = wk * inv_bk;
    //                 v_reg[k]  = vk * inv_bk;
    //                 mt_reg[k] = ck * inv_bk;

    //                 pivot.set_pivoting_to_1x1(k);

    //                 if(k < (BLOCKDIM - 1))
    //                 {
    //                     w_reg[k + 1] += -ak_1 * wk * inv_bk;
    //                 }

    //                 // L * B * x = y
    //                 for(int c = 0; c < NCOLS; c++)
    //                 {
    //                     const T rhsk = rhs_reg[BLOCKDIM * c + k] * inv_bk;

    //                     rhs_reg[BLOCKDIM * c + k] = rhsk;

    //                     if(k < (BLOCKDIM - 1))
    //                     {
    //                         rhs_reg[BLOCKDIM * c + (k + 1)] += -(ak_1 * rhsk);
    //                     }
    //                 }

    //                 if(k < (BLOCKDIM - 1))
    //                 {
    //                     bk_1 = bk_1 - ak_1 * ck * inv_bk;
    //                 }

    //                 bk = bk_1;

    //                 k += 1;
    //             }
    //             else
    //             {
    //                 const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

    //                 const T wk   = w_reg[k];
    //                 const T wk_1 = w_reg[k + 1];
    //                 const T vk   = v_reg[k];
    //                 const T vk_1 = v_reg[k + 1];

    //                 w_reg[k]  = (bk_1 * wk - ck * wk_1) * det;
    //                 v_reg[k]  = (bk_1 * vk - ck * vk_1) * det;
    //                 mt_reg[k] = -ck * ck_1 * det;

    //                 pivot.set_pivoting_to2x2(k);

    //                 if(k < (BLOCKDIM - 1))
    //                 {
    //                     w_reg[k + 1]  = (-ak_1 * wk + bk * wk_1) * det;
    //                     v_reg[k + 1]  = (-ak_1 * vk + bk * vk_1) * det;
    //                     mt_reg[k + 1] = bk * ck_1 * det;

    //                     pivot.set_pivoting_to2x2(k + 1);
    //                 }

    //                 T bk_2 = static_cast<T>(0);

    //                 if(k < (BLOCKDIM - 2))
    //                 {
    //                     w_reg[k + 2] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
    //                 }

    //                 // |bk   ck  ||xk  |   |rhsk   |
    //                 // |ak_1 bk_1||xk_1| = |rhsk _1|
    //                 //
    //                 //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
    //                 //                                  |-ak_1  bk |

    //                 // L * B * x = y
    //                 for(int c = 0; c < NCOLS; c++)
    //                 {
    //                     const T rhsk   = rhs_reg[BLOCKDIM * c + k] * det;
    //                     const T rhsk_1 = rhs_reg[BLOCKDIM * c + k + 1] * det;

    //                     rhs_reg[BLOCKDIM * c + k]     = (bk_1 * rhsk - ck * rhsk_1);
    //                     rhs_reg[BLOCKDIM * c + k + 1] = (-ak_1 * rhsk + bk * rhsk_1);

    //                     if(k < (BLOCKDIM - 2))
    //                     {
    //                         rhs_reg[BLOCKDIM * c + k + 2]
    //                             += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);
    //                     }
    //                 }

    //                 if(k < (BLOCKDIM - 2))
    //                 {
    //                     bk_2 = main_reg[k + 2];
    //                     bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
    //                 }

    //                 bk = bk_2;
    //                 k += 2;
    //             }
    //         }

    //         assert(k == BLOCKDIM);
    //         // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
    //         k--;

    //         k -= pivot.get_pivoting(k);

    //         // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
    //         while(k >= 0)
    //         {
    //             if(pivot.get_pivoting(k) == 1)
    //             {
    //                 const T tmp = mt_reg[k];

    //                 w_reg[k] += -tmp * w_reg[k + 1];
    //                 v_reg[k] += -tmp * v_reg[k + 1];

    //                 k -= 1;
    //             }
    //             else
    //             {
    //                 const T tmp1 = mt_reg[k];
    //                 const T tmp2 = mt_reg[k - 1];

    //                 w_reg[k] += -tmp1 * w_reg[k + 1];
    //                 w_reg[k - 1] += -tmp2 * w_reg[k + 1];
    //                 v_reg[k] += -tmp1 * v_reg[k + 1];
    //                 v_reg[k - 1] += -tmp2 * v_reg[k + 1];

    //                 k -= 2;
    //             }
    //         }
    //     }
    // }

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

    // =========================================================================
    // Version 2 of the fused solver.
    //
    // The matrix factorization (pivot decisions, per-step coefficients, spike
    // vectors w and v, and the factorization of the reduced system S) does not
    // depend on the right hand sides, so it is computed once by
    // solve_fused_factor_kernel_v2 and stored in a workspace. The RHS kernel
    // solve_fused_kernel_v2 then only applies the stored factors:
    //
    //  * coefficients (and w, v) are streamed from global memory one step at a
    //    time, in a layout [step][partition] so that lanes read consecutive
    //    addresses. They are never held in BLOCKDIM sized register arrays.
    //  * all register arrays are indexed with compile time constants (the loops
    //    are fully unrolled and the data dependent 1x1/2x2 pivot choice is
    //    handled with predication), so nothing is placed in scratch memory.
    //  * the reduced system S is solved with one lane per RHS column instead of
    //    one lane for all columns, using the precomputed factorization.
    // =========================================================================

    // Workspace layout (in units of T) for the v2 factorization.
    //
    //   coef  : 6 arrays of [BLOCKDIM][BLOCKSIZE]. Array 0..4 hold the forward
    //           step coefficients, array 5 holds mt (the M^T factor).
    //             1x1 step at row k : [0][k] = 1/bk,        [1][k] = ak_1
    //             2x2 step at row k : [0][k] = bk_1 * det,  [1][k] = -ck * det,
    //                                 [2][k] = -ak_1 * det, [3][k] = bk * det,
    //                                 [4][k] = ak_2
    //   wv    : 2 arrays of [BLOCKDIM][BLOCKSIZE]: w and v
    //   scoef : the same 6 coefficient arrays for the reduced system, [S_SIZE]
    //   mask  : BLOCKSIZE pivot masks of the partitions followed by 2 words
    //           holding the 64 bit pivot mask of the reduced system.
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM>
    struct fused_v2_layout
    {
        static constexpr int S_SIZE = 2 * BLOCKSIZE;

        static constexpr size_t coef_offset  = 0;
        static constexpr size_t wv_offset    = coef_offset + 6 * BLOCKDIM * BLOCKSIZE;
        static constexpr size_t scoef_offset = wv_offset + 2 * BLOCKDIM * BLOCKSIZE;
        static constexpr size_t mask_offset  = scoef_offset + 6 * S_SIZE;
        static constexpr size_t size         = mask_offset + BLOCKSIZE + 2;
    };

    template <typename T>
    struct fused_v2_coef
    {
        const T* base;
        int      row_stride;
        int      which_stride;

        __device__ __forceinline__ T operator()(int which, int k) const
        {
            return base[which * which_stride + k * row_stride];
        }
    };

    // Computes the LBM^T factorization (with Bunch-Kaufman 1x1/2x2 pivoting) of a
    // N x N tridiagonal matrix given by the contiguous arrays lo, ma and up
    // (lo[i] = A[i][i-1], ma[i] = A[i][i], up[i] = A[i][i+1]).
    // The coefficient arrays are written to
    //   out[which * which_stride + k * row_stride]
    // and the pivot mask is returned (bit k set <=> row k belongs to a 2x2 pivot).
    template <int N, typename T>
    __device__ __forceinline__ uint64_t
        lbmt_factor(const T* lo, const T* ma, const T* up, T* out, int row_stride, int which_stride)
    {
        uint64_t mask = 0;

        int k  = 0;
        T   bk = ma[0];

        while(k < N)
        {
            const T ck   = up[k];
            const T ck_1 = (k < (N - 1)) ? up[k + 1] : static_cast<T>(0);
            T       bk_1 = (k < (N - 1)) ? ma[k + 1] : static_cast<T>(0);
            const T ak_1 = (k < (N - 1)) ? lo[k + 1] : static_cast<T>(0);
            const T ak_2 = (k < (N - 2)) ? lo[k + 2] : static_cast<T>(0);

            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            if(use_1x1_pivot || k == (N - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                out[0 * which_stride + k * row_stride] = inv_bk;
                out[1 * which_stride + k * row_stride] = ak_1;
                out[5 * which_stride + k * row_stride] = ck * inv_bk;

                if(k < (N - 1))
                {
                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;
                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                out[0 * which_stride + k * row_stride]       = bk_1 * det;
                out[1 * which_stride + k * row_stride]       = -ck * det;
                out[2 * which_stride + k * row_stride]       = -ak_1 * det;
                out[3 * which_stride + k * row_stride]       = bk * det;
                out[4 * which_stride + k * row_stride]       = ak_2;
                out[5 * which_stride + k * row_stride]       = -ck * ck_1 * det;
                out[5 * which_stride + (k + 1) * row_stride] = bk * ck_1 * det;

                mask |= (static_cast<uint64_t>(3) << k);

                T bk_2 = static_cast<T>(0);
                if(k < (N - 2))
                {
                    bk_2 = ma[k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        return mask;
    }

    // Applies the stored LBM^T factorization to the NV vectors x[0..NV-1]:
    //   L * B * y = x   (forward),  M^T * x = y  (backward).
    // All indices into x are compile time constants after unrolling, so x lives
    // entirely in registers. The data dependent pivoting is handled with
    // predication.
    template <int N, int NV, typename T, typename COEF>
    __device__ __forceinline__ void lbmt_apply(const COEF& cf, uint64_t mask, T (&x)[NV][N])
    {
        // Forward: L * B * y = x
        bool skip = false;
#pragma unroll
        for(int k = 0; k < N; k++)
        {
            if(skip)
            {
                skip = false;
                continue;
            }

            const int k1 = (k + 1 < N) ? (k + 1) : k;
            const int k2 = (k + 2 < N) ? (k + 2) : k;

            if(((mask >> k) & 1) == 0)
            {
                // 1x1 pivot
                const T inv_bk = cf(0, k);
                const T ak_1   = cf(1, k);
#pragma unroll
                for(int c = 0; c < NV; c++)
                {
                    const T r = x[c][k] * inv_bk;
                    x[c][k]   = r;
                    if(k + 1 < N)
                    {
                        x[c][k1] -= ak_1 * r;
                    }
                }
            }
            else
            {
                // 2x2 pivot
                const T p00  = cf(0, k);
                const T p01  = cf(1, k);
                const T p10  = cf(2, k);
                const T p11  = cf(3, k);
                const T ak_2 = cf(4, k);
#pragma unroll
                for(int c = 0; c < NV; c++)
                {
                    const T r0 = x[c][k];
                    const T r1 = x[c][k1];
                    const T n0 = p00 * r0 + p01 * r1;
                    const T n1 = p10 * r0 + p11 * r1;
                    x[c][k]    = n0;
                    x[c][k1]   = n1;
                    if(k + 2 < N)
                    {
                        x[c][k2] -= ak_2 * n1;
                    }
                }
                skip = true;
            }
        }

        // Backward: M^T * x = y. The last row (or the last 2x2 block) needs no update.
        int kc = N - 1 - ((((mask >> (N - 1)) & 1) != 0) ? 2 : 1);
#pragma unroll
        for(int k = N - 2; k >= 0; k--)
        {
            if(k != kc)
            {
                continue;
            }

            const int km1 = (k > 0) ? (k - 1) : 0;

            const T mt = cf(5, k);
#pragma unroll
            for(int c = 0; c < NV; c++)
            {
                x[c][k] -= mt * x[c][k + 1];
            }

            if(((mask >> k) & 1) == 0)
            {
                kc -= 1;
            }
            else
            {
                const T mt1 = cf(5, km1);
#pragma unroll
                for(int c = 0; c < NV; c++)
                {
                    x[c][km1] -= mt1 * x[c][k + 1];
                }
                kc -= 2;
            }
        }
    }

    // Computes the factorization of the (single) tridiagonal matrix and stores
    // it in `factors`. Launched with a single block of BLOCKSIZE threads.
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void solve_fused_factor_kernel_v2(int m,
                                      const T* __restrict__ lower,
                                      const T* __restrict__ main,
                                      const T* __restrict__ upper,
                                      T* __restrict__ factors)
    {
        static_assert(BLOCKDIM >= 2);
        static_assert(BLOCKDIM % 2 == 0);
        static_assert(BLOCKDIM <= 32);
        static_assert(BLOCKSIZE % 2 == 0);

        using layout         = fused_v2_layout<BLOCKSIZE, BLOCKDIM>;
        constexpr int S_SIZE = layout::S_SIZE;
        const int     tid    = threadIdx.x;
        const int     first  = BLOCKDIM * tid;

        T*            coef  = factors + layout::coef_offset;
        T*            wv    = factors + layout::wv_offset;
        T*            scoef = factors + layout::scoef_offset;
        unsigned int* masks = reinterpret_cast<unsigned int*>(factors + layout::mask_offset);

        // Copy the matrix into shared memory, padding rows beyond m with the identity.
        __shared__ T P_lower[BLOCKSIZE * BLOCKDIM];
        __shared__ T P_main[BLOCKSIZE * BLOCKDIM];
        __shared__ T P_upper[BLOCKSIZE * BLOCKDIM];

        for(int i = tid; i < BLOCKSIZE * BLOCKDIM; i += BLOCKSIZE)
        {
            P_lower[i] = (i < m) ? lower[i] : static_cast<T>(0);
            P_main[i]  = (i < m) ? main[i] : static_cast<T>(1);
            P_upper[i] = (i < m) ? upper[i] : static_cast<T>(0);
        }

        __syncthreads();

        const T* lo = P_lower + first;
        const T* ma = P_main + first;
        const T* up = P_upper + first;

        const uint64_t mask
            = lbmt_factor<BLOCKDIM>(lo, ma, up, coef + tid, BLOCKSIZE, BLOCKDIM * BLOCKSIZE);
        masks[tid] = static_cast<unsigned int>(mask);

        // Spike vectors w = A^-1 * (lower[0] * e_0) and v = A^-1 * (upper[BLOCKDIM-1] * e_{BLOCKDIM-1})
        T x[2][BLOCKDIM];
#pragma unroll
        for(int i = 0; i < BLOCKDIM; i++)
        {
            x[0][i] = (i == 0) ? lo[0] : static_cast<T>(0);
            x[1][i] = (i == BLOCKDIM - 1) ? up[BLOCKDIM - 1] : static_cast<T>(0);
        }

        lbmt_apply<BLOCKDIM, 2>(
            fused_v2_coef<T>{coef + tid, BLOCKSIZE, BLOCKDIM * BLOCKSIZE}, mask, x);

#pragma unroll
        for(int i = 0; i < BLOCKDIM; i++)
        {
            wv[(0 * BLOCKDIM + i) * BLOCKSIZE + tid] = x[0][i];
            wv[(1 * BLOCKDIM + i) * BLOCKSIZE + tid] = x[1][i];
        }

        // Reduced system S
        __shared__ T            S_lower[S_SIZE];
        __shared__ T            S_main[S_SIZE];
        __shared__ T            S_upper[S_SIZE];
        __shared__ T            S_coef[6 * S_SIZE];
        __shared__ unsigned int S_mask[2];

        S_lower[2 * tid]     = static_cast<T>(1);
        S_lower[2 * tid + 1] = x[0][BLOCKDIM - 1];

        S_upper[2 * tid]     = x[1][0];
        S_upper[2 * tid + 1] = static_cast<T>(1);

        S_main[2 * tid]     = x[0][0];
        S_main[2 * tid + 1] = x[1][BLOCKDIM - 1];

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

        __syncthreads();

        if(tid == 0)
        {
            const uint64_t smask = lbmt_factor<S_SIZE>(S_lower, S_main, S_upper, S_coef, 1, S_SIZE);

            S_mask[0] = static_cast<unsigned int>(smask);
            S_mask[1] = static_cast<unsigned int>(smask >> 32);
        }

        __syncthreads();

        for(int i = tid; i < 6 * S_SIZE; i += BLOCKSIZE)
        {
            scoef[i] = S_coef[i];
        }

        if(tid < 2)
        {
            masks[BLOCKSIZE + tid] = S_mask[tid];
        }
    }

    // Applies the factorization computed by solve_fused_factor_kernel_v2 to NCOLS
    // right hand sides per block. B is accessed with column stride ldb.
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void solve_fused_kernel_v2(int m, int ldb, const T* __restrict__ factors, T* __restrict__ B)
    {
        static_assert(BLOCKDIM >= 2);
        static_assert(BLOCKDIM % 2 == 0);
        static_assert(BLOCKDIM <= 32);
        static_assert(BLOCKSIZE % 2 == 0);
        static_assert(NCOLS <= BLOCKSIZE);

        using layout         = fused_v2_layout<BLOCKSIZE, BLOCKDIM>;
        constexpr int S_SIZE = layout::S_SIZE;

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        const T*            coef  = factors + layout::coef_offset;
        const T*            wv    = factors + layout::wv_offset;
        const T*            scoef = factors + layout::scoef_offset;
        const unsigned int* masks
            = reinterpret_cast<const unsigned int*>(factors + layout::mask_offset);

        __shared__ T            S_coef[6 * S_SIZE];
        __shared__ T            S_rhs[S_SIZE * NCOLS];
        __shared__ unsigned int S_mask[2];

        for(int i = tid; i < 6 * S_SIZE; i += BLOCKSIZE)
        {
            S_coef[i] = scoef[i];
        }

        if(tid < 2)
        {
            S_mask[tid] = masks[BLOCKSIZE + tid];
        }

        const uint64_t mask = masks[tid];

        // Load the right hand sides
        const int64_t col0 = static_cast<int64_t>(NCOLS) * bid;

        T rhs[NCOLS][BLOCKDIM];
#pragma unroll
        for(int c = 0; c < NCOLS; c++)
        {
#pragma unroll
            for(int i = 0; i < BLOCKDIM; i++)
            {
                const int row = BLOCKDIM * tid + i;
                rhs[c][i]     = (row < m) ? B[row + ldb * (col0 + c)] : static_cast<T>(0);
            }
        }

        // Local solve of each partition
        lbmt_apply<BLOCKDIM, NCOLS>(
            fused_v2_coef<T>{coef + tid, BLOCKSIZE, BLOCKDIM * BLOCKSIZE}, mask, rhs);

        // Gather the right hand side of S
#pragma unroll
        for(int c = 0; c < NCOLS; c++)
        {
            S_rhs[S_SIZE * c + 2 * tid]     = rhs[c][0];
            S_rhs[S_SIZE * c + 2 * tid + 1] = rhs[c][BLOCKDIM - 1];
        }

        __syncthreads();

        // Solve S, one lane per column
        if(tid < NCOLS)
        {
            T xs[1][S_SIZE];
#pragma unroll
            for(int i = 0; i < S_SIZE; i++)
            {
                xs[0][i] = S_rhs[S_SIZE * tid + i];
            }

            const uint64_t smask
                = static_cast<uint64_t>(S_mask[0]) | (static_cast<uint64_t>(S_mask[1]) << 32);

            lbmt_apply<S_SIZE, 1>(fused_v2_coef<T>{S_coef, 1, S_SIZE}, smask, xs);

#pragma unroll
            for(int i = 0; i < S_SIZE; i++)
            {
                S_rhs[S_SIZE * tid + i] = xs[0][i];
            }
        }

        __syncthreads();

        // Scatter
        T x1[NCOLS];
        T x2[NCOLS];
#pragma unroll
        for(int c = 0; c < NCOLS; c++)
        {
            const T val_even = (tid == 0) ? S_rhs[S_SIZE * c + 0] : S_rhs[S_SIZE * c + 2 * tid - 1];
            const T val_odd  = (2 * tid + 1 < S_SIZE - 1) ? S_rhs[S_SIZE * c + 2 * tid + 2]
                                                          : S_rhs[S_SIZE * c + S_SIZE - 1];

            rhs[c][0]            = val_even;
            rhs[c][BLOCKDIM - 1] = val_odd;

            const T tmp1 = __shfl_sync(0xFFFFFFFFULL, rhs[c][BLOCKDIM - 1], tid - 1, BLOCKSIZE);
            const T tmp2 = __shfl_sync(0xFFFFFFFFULL, rhs[c][0], tid + 1, BLOCKSIZE);

            x1[c] = (tid >= 1) ? tmp1 : static_cast<T>(0);
            x2[c] = (tid < (BLOCKSIZE - 1)) ? tmp2 : static_cast<T>(0);
        }

        // rhs = rhs - w * x1 - v * x2, streaming w and v
#pragma unroll
        for(int j = 1; j < BLOCKDIM - 1; j++)
        {
            const T wj = wv[(0 * BLOCKDIM + j) * BLOCKSIZE + tid];
            const T vj = wv[(1 * BLOCKDIM + j) * BLOCKSIZE + tid];
#pragma unroll
            for(int c = 0; c < NCOLS; c++)
            {
                rhs[c][j] = rhs[c][j] - wj * x1[c] - vj * x2[c];
            }
        }

        // Store the solution
#pragma unroll
        for(int c = 0; c < NCOLS; c++)
        {
#pragma unroll
            for(int i = 0; i < BLOCKDIM; i++)
            {
                const int row = BLOCKDIM * tid + i;
                if(row < m)
                {
                    B[row + ldb * (col0 + c)] = rhs[c][i];
                }
            }
        }
    }
}