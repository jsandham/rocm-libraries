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
    // Sets bit k to 0 to record a 1x1 pivot at row k.
    __device__ __forceinline__ void set_pivoting_to_1x1(int* mask, int k)
    {
        *mask &= ~(1u << k);
    }

    // Sets bit k to 1 to record 2x2 pivoting at row k.
    __device__ __forceinline__ void set_pivoting_to_2x2(int* mask, int k)
    {
        *mask |= (1u << k);
    }

    // Returns 1 if row k used 1x1 pivoting, 2 if row k is part of a 2x2 pivot.
    __device__ __forceinline__ int get_pivoting(int mask, int k)
    {
        return ((mask >> k) & 1u) ? 2 : 1;
    }

    // Need to store:
    // bk_1 * det
    // ck * det
    // ak_2
    // -ak_1 * det
    // bk * det
    // mt
    // w
    // v
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void solve_fused_wvmt_kernel(int m,
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
        static_assert(BLOCKDIM >= 2);
        static_assert(BLOCKDIM % 2 == 0);
        static_assert(BLOCKSIZE % 2 == 0);

        const int tid = threadIdx.x;

        __shared__ T slower[BLOCKSIZE * BLOCKDIM];
        __shared__ T smain[BLOCKSIZE * BLOCKDIM];
        __shared__ T supper[BLOCKSIZE * BLOCKDIM];

        for(int i = tid; i < BLOCKSIZE * BLOCKDIM; i += BLOCKSIZE)
        {
            slower[i] = (i < m) ? lower[i] : static_cast<T>(0);
            smain[i]  = (i < m) ? main[i] : static_cast<T>(1);
            supper[i] = (i < m) ? upper[i] : static_cast<T>(0);
        }

        __syncthreads();

        // LBMT solve
        T bk = smain[BLOCKDIM * tid];

        w[tid]                              = slower[BLOCKDIM * tid];
        v[BLOCKSIZE * (BLOCKDIM - 1) + tid] = supper[BLOCKDIM * tid + (BLOCKDIM - 1)];

        bool skip = false;
#pragma unroll
        for(int k = 0; k < BLOCKDIM; k++)
        {
            if(skip)
            {
                skip = false;
                continue;
            }

            T ck   = supper[BLOCKDIM * tid + k];
            T ck_1 = (k < (BLOCKDIM - 1)) ? supper[BLOCKDIM * tid + (k + 1)] : static_cast<T>(0);
            T bk_1 = (k < (BLOCKDIM - 1)) ? smain[BLOCKDIM * tid + (k + 1)] : static_cast<T>(0);
            T ak_1 = (k < (BLOCKDIM - 1)) ? slower[BLOCKDIM * tid + (k + 1)] : static_cast<T>(0);
            T ak_2 = (k < (BLOCKDIM - 2)) ? slower[BLOCKDIM * tid + (k + 2)] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            // 1x1 pivoting
            if(use_1x1_pivot || k == (BLOCKDIM - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                const T wk = w[BLOCKSIZE * k + tid];
                const T vk = v[BLOCKSIZE * k + tid];

                w[BLOCKSIZE * k + tid]  = wk * inv_bk;
                v[BLOCKSIZE * k + tid]  = vk * inv_bk;
                mt[BLOCKSIZE * k + tid] = ck * inv_bk;

                set_pivoting_to_1x1(&pivot_mask[tid], k);

                if(k < (BLOCKDIM - 1))
                {
                    w[BLOCKSIZE * (k + 1) + tid] += -ak_1 * wk * inv_bk;
                }

                coeff1[BLOCKSIZE * k + tid] = inv_bk;
                coeff2[BLOCKSIZE * k + tid] = ak_1;

                if(k < (BLOCKDIM - 1))
                {
                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                const T wk   = w[BLOCKSIZE * k + tid];
                const T wk_1 = w[BLOCKSIZE * (k + 1) + tid];
                const T vk   = v[BLOCKSIZE * k + tid];
                const T vk_1 = v[BLOCKSIZE * (k + 1) + tid];

                w[BLOCKSIZE * k + tid]  = (bk_1 * wk - ck * wk_1) * det;
                v[BLOCKSIZE * k + tid]  = (bk_1 * vk - ck * vk_1) * det;
                mt[BLOCKSIZE * k + tid] = -ck * ck_1 * det;

                set_pivoting_to_2x2(&pivot_mask[tid], k);

                if(k < (BLOCKDIM - 1))
                {
                    w[BLOCKSIZE * (k + 1) + tid]  = (-ak_1 * wk + bk * wk_1) * det;
                    v[BLOCKSIZE * (k + 1) + tid]  = (-ak_1 * vk + bk * vk_1) * det;
                    mt[BLOCKSIZE * (k + 1) + tid] = bk * ck_1 * det;

                    set_pivoting_to_2x2(&pivot_mask[tid], k + 1);
                }

                T bk_2 = static_cast<T>(0);

                if(k < (BLOCKDIM - 2))
                {
                    w[BLOCKSIZE * (k + 2) + tid] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
                }

                // |bk   ck  ||xk  |   |rhsk   |
                // |ak_1 bk_1||xk_1| = |rhsk _1|
                //
                //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
                //                                  |-ak_1  bk |
                coeff1[BLOCKSIZE * k + tid] = bk_1 * det;
                coeff2[BLOCKSIZE * k + tid] = ck * det;
                coeff3[BLOCKSIZE * k + tid] = ak_2;
                coeff4[BLOCKSIZE * k + tid] = -ak_1 * det;
                coeff5[BLOCKSIZE * k + tid] = bk * det;

                if(k < (BLOCKDIM - 2))
                {
                    bk_2 = smain[BLOCKDIM * tid + k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk   = bk_2;
                skip = true;
            }
        }

        const int part_mask = pivot_mask[tid];
        int       kc        = (BLOCKDIM - 1) - get_pivoting(part_mask, BLOCKDIM - 1);

        // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
#pragma unroll
        for(int k = BLOCKDIM - 2; k >= 0; --k)
        {
            if(k != kc)
            {
                continue;
            }

            if(get_pivoting(part_mask, k) == 1)
            {
                const T tmp = mt[BLOCKSIZE * k + tid];

                w[BLOCKSIZE * k + tid] += -tmp * w[BLOCKSIZE * (k + 1) + tid];
                v[BLOCKSIZE * k + tid] += -tmp * v[BLOCKSIZE * (k + 1) + tid];

                kc -= 1;
            }
            else
            {
                const T tmp1 = mt[BLOCKSIZE * k + tid];
                const T tmp2 = mt[BLOCKSIZE * (k - 1) + tid];

                w[BLOCKSIZE * k + tid] += -tmp1 * w[BLOCKSIZE * (k + 1) + tid];
                w[BLOCKSIZE * (k - 1) + tid] += -tmp2 * w[BLOCKSIZE * (k + 1) + tid];
                v[BLOCKSIZE * k + tid] += -tmp1 * v[BLOCKSIZE * (k + 1) + tid];
                v[BLOCKSIZE * (k - 1) + tid] += -tmp2 * v[BLOCKSIZE * (k + 1) + tid];

                kc -= 2;
            }
        }

        // Fill S system
        constexpr int S_SIZE = 2 * BLOCKSIZE;

        __shared__ T S_lower[S_SIZE];
        __shared__ T S_main[S_SIZE];
        __shared__ T S_upper[S_SIZE];

        S_lower[2 * tid]     = static_cast<T>(1);
        S_lower[2 * tid + 1] = w[BLOCKSIZE * (BLOCKDIM - 1) + tid];

        S_upper[2 * tid]     = v[tid];
        S_upper[2 * tid + 1] = static_cast<T>(1);

        S_main[2 * tid]     = w[tid];
        S_main[2 * tid + 1] = v[BLOCKSIZE * (BLOCKDIM - 1) + tid];

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

        // Solve S system (solved by 1 thread)
        if(tid == 0)
        {
            T bk = S_main[0];

            bool skip = false;
#pragma unroll
            for(int k = 0; k < S_SIZE; k++)
            {
                if(skip)
                {
                    skip = false;
                    continue;
                }

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

                    s_mt[k] = ck * inv_bk;

                    set_pivoting_to_1x1(&s_pivot_mask[k / 32], (k % 32));

                    s_coeff1[k] = inv_bk;
                    s_coeff2[k] = ak_1;

                    if(k < (S_SIZE - 1))
                    {
                        bk_1 = bk_1 - ak_1 * ck * inv_bk;
                    }

                    bk = bk_1;
                }
                else
                {
                    const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                    s_mt[k] = -ck * ck_1 * det;

                    set_pivoting_to_2x2(&s_pivot_mask[k / 32], (k % 32));

                    if(k < (S_SIZE - 1))
                    {
                        s_mt[k + 1] = bk * ck_1 * det;

                        set_pivoting_to_2x2(&s_pivot_mask[(k + 1) / 32], ((k + 1) % 32));
                    }

                    T bk_2 = static_cast<T>(0);

                    s_coeff1[k] = bk_1 * det;
                    s_coeff2[k] = ck * det;
                    s_coeff3[k] = ak_2;
                    s_coeff4[k] = -ak_1 * det;
                    s_coeff5[k] = bk * det;

                    if(k < (S_SIZE - 2))
                    {
                        bk_2 = S_main[k + 2];
                        bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                    }

                    bk   = bk_2;
                    skip = true;
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void solve_fused_rhs_kernel(int m,
                                int ldb,
                                const T* __restrict__ coeff1,
                                const T* __restrict__ coeff2,
                                const T* __restrict__ coeff3,
                                const T* __restrict__ coeff4,
                                const T* __restrict__ coeff5,
                                const T* __restrict__ w,
                                const T* __restrict__ v,
                                const T* __restrict__ mt,
                                const int* mask,
                                const T* __restrict__ s_coeff1,
                                const T* __restrict__ s_coeff2,
                                const T* __restrict__ s_coeff3,
                                const T* __restrict__ s_coeff4,
                                const T* __restrict__ s_coeff5,
                                const T* __restrict__ s_mt,
                                const int* s_mask,
                                T* __restrict__ B)
    {
        static_assert(BLOCKDIM >= 2);
        static_assert(BLOCKDIM <= 32);
        static_assert(BLOCKDIM % 2 == 0);
        static_assert(BLOCKSIZE % 2 == 0);
        static_assert(NCOLS <= BLOCKSIZE);

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        // Column offset is 64-bit so the compiler keeps B's base in a scalar
        // register and folds each row into an immediate offset, matching version 2.
        const int64_t col0 = static_cast<int64_t>(NCOLS) * bid;

        T rhs_reg[NCOLS][BLOCKDIM];
        for(int c = 0; c < NCOLS; c++)
        {
            for(int i = 0; i < BLOCKDIM; i++)
            {
                const int row = BLOCKDIM * tid + i;
                rhs_reg[c][i] = (row < m) ? B[row + ldb * (col0 + c)] : static_cast<T>(0);
            }
        }

        // LBMT solve. Slots are inv_bk or bk_1*det, ak_1 or ck*det,
        // -ak_1*det, bk*det, ak_2, mt. Both arms read slots 0 and 1.
        const int part_mask = mask[tid];

        bool skip = false;
#pragma unroll
        for(int k = 0; k < BLOCKDIM; k++)
        {
            if(skip)
            {
                skip = false;
                continue;
            }

            const int k1 = (k + 1 < BLOCKDIM) ? (k + 1) : k;
            const int k2 = (k + 2 < BLOCKDIM) ? (k + 2) : k;

            // 1x1 pivoting
            if(((part_mask >> k) & 1) == 0)
            {
                const T inv_bk = coeff1[BLOCKSIZE * k + tid];
                const T ak_1   = coeff2[BLOCKSIZE * k + tid];

                // L * B * x = y
                for(int c = 0; c < NCOLS; c++)
                {
                    const T rhsk = rhs_reg[c][k] * inv_bk;

                    rhs_reg[c][k] = rhsk;

                    if(k < (BLOCKDIM - 1))
                    {
                        rhs_reg[c][k1] -= ak_1 * rhsk;
                    }
                }
            }
            else
            {
                const T bk_1_det  = coeff1[BLOCKSIZE * k + tid];
                const T ck_det    = coeff2[BLOCKSIZE * k + tid];
                const T nak_1_det = coeff4[BLOCKSIZE * k + tid];
                const T bk_det    = coeff5[BLOCKSIZE * k + tid];
                const T ak_2      = coeff3[BLOCKSIZE * k + tid];

                // L * B * x = y
                for(int c = 0; c < NCOLS; c++)
                {
                    const T rhsk   = rhs_reg[c][k];
                    const T rhsk_1 = rhs_reg[c][k1];

                    const T tmp0 = bk_1_det * rhsk - ck_det * rhsk_1;
                    const T tmp1 = nak_1_det * rhsk + bk_det * rhsk_1;

                    rhs_reg[c][k]  = tmp0;
                    rhs_reg[c][k1] = tmp1;

                    if(k < (BLOCKDIM - 2))
                    {
                        rhs_reg[c][k2] -= ak_2 * tmp1;
                    }
                }

                skip = true;
            }
        }

        int kc = BLOCKDIM - 1 - ((((part_mask >> (BLOCKDIM - 1)) & 1) != 0) ? 2 : 1);

        // backward solve (M^T * rhs = rhs)
#pragma unroll
        for(int k = BLOCKDIM - 2; k >= 0; k--)
        {
            if(k != kc)
            {
                continue;
            }

            // Both pivot types update row k, so do it before branching.
            const int km1 = (k > 0) ? (k - 1) : 0;
            const T   tmp = mt[BLOCKSIZE * k + tid];
            for(int c = 0; c < NCOLS; c++)
            {
                rhs_reg[c][k] -= tmp * rhs_reg[c][k + 1];
            }

            if(((part_mask >> k) & 1) == 0)
            {
                kc -= 1;
            }
            else
            {
                const T tmp2 = mt[BLOCKSIZE * km1 + tid];
                for(int c = 0; c < NCOLS; c++)
                {
                    rhs_reg[c][km1] -= tmp2 * rhs_reg[c][k + 1];
                }

                kc -= 2;
            }
        }

        // Fill S system
        constexpr int S_SIZE = 2 * BLOCKSIZE;

        __shared__ T S_rhs[S_SIZE * NCOLS];

        for(int c = 0; c < NCOLS; c++)
        {
            S_rhs[S_SIZE * c + 2 * tid]     = rhs_reg[c][0];
            S_rhs[S_SIZE * c + 2 * tid + 1] = rhs_reg[c][BLOCKDIM - 1];
        }

        // Pack the reduced system the way lbmt_apply reads it:
        // [0] inv_bk or bk_1*det, [1] ak_1 or ck*det, [2] -ak_1*det,
        // [3] bk*det, [4] ak_2, [5] mt.
        __shared__ T ss_coef[6 * S_SIZE];

        for(int k = tid; k < S_SIZE; k += BLOCKSIZE)
        {
            ss_coef[0 * S_SIZE + k] = s_coeff1[k];
            ss_coef[1 * S_SIZE + k] = s_coeff2[k];
            ss_coef[2 * S_SIZE + k] = s_coeff4[k];
            ss_coef[3 * S_SIZE + k] = s_coeff5[k];
            ss_coef[4 * S_SIZE + k] = s_coeff3[k];
            ss_coef[5 * S_SIZE + k] = s_mt[k];
        }

        __shared__ unsigned int ss_mask[2];
        if(tid < 2)
        {
            ss_mask[tid] = s_mask[tid];
        }

        __syncthreads();

        // Solve S system (solved by 1 thread)
        if(tid < NCOLS)
        {
            int            c = tid;
            const uint64_t smask
                = static_cast<uint64_t>(ss_mask[0]) | (static_cast<uint64_t>(ss_mask[1]) << 32);

            T xs[S_SIZE];
#pragma unroll
            for(int i = 0; i < S_SIZE; ++i)
            {
                xs[i] = S_rhs[S_SIZE * c + i];
            }

            bool skip = false;
#pragma unroll
            for(int k = 0; k < S_SIZE; k++)
            {
                if(skip)
                {
                    skip = false;
                    continue;
                }

                // 1x1 pivoting
                if(((smask >> k) & 1) == 0)
                {
                    const T inv_bk = ss_coef[0 * S_SIZE + k];
                    const T ak_1   = ss_coef[1 * S_SIZE + k];

                    // L * B * x = y
                    const T rhsk = xs[k] * inv_bk;

                    xs[k] = rhsk;

                    if(k < (S_SIZE - 1))
                    {
                        xs[k + 1] -= ak_1 * rhsk;
                    }
                }
                else
                {
                    const T bk_1_det  = ss_coef[0 * S_SIZE + k];
                    const T ck_det    = ss_coef[1 * S_SIZE + k];
                    const T nak_1_det = ss_coef[2 * S_SIZE + k];
                    const T bk_det    = ss_coef[3 * S_SIZE + k];
                    const T ak_2      = ss_coef[4 * S_SIZE + k];

                    // L * B * x = y
                    const T rhsk   = xs[k];
                    const T rhsk_1 = xs[k + 1];

                    const T tmp0 = bk_1_det * rhsk - ck_det * rhsk_1;
                    const T tmp1 = nak_1_det * rhsk + bk_det * rhsk_1;

                    xs[k]     = tmp0;
                    xs[k + 1] = tmp1;

                    if(k < (S_SIZE - 2))
                    {
                        xs[k + 2] -= ak_2 * tmp1;
                    }

                    skip = true;
                }
            }

            int kc = S_SIZE - 1 - ((((smask >> (S_SIZE - 1)) & 1) != 0) ? 2 : 1);

            // backward solve (M^T * rhs = rhs)
#pragma unroll
            for(int k = S_SIZE - 2; k >= 0; k--)
            {
                if(k != kc)
                {
                    continue;
                }

                const int km1 = (k > 0) ? (k - 1) : 0;
                const T   tmp = ss_coef[5 * S_SIZE + k];
                xs[k] -= tmp * xs[k + 1];

                if(((smask >> k) & 1) == 0)
                {
                    kc -= 1;
                }
                else
                {
                    const T tmp2 = ss_coef[5 * S_SIZE + km1];
                    xs[km1] -= tmp2 * xs[k + 1];
                    kc -= 2;
                }
            }

#pragma unroll
            for(int i = 0; i < S_SIZE; ++i)
            {
                S_rhs[S_SIZE * c + i] = xs[i];
            }
        }

        __syncthreads();

        // Scatter
        T x1[NCOLS];
        T x2[NCOLS];
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

            rhs_reg[c][0]            = val_even;
            rhs_reg[c][BLOCKDIM - 1] = val_odd;

            // backward solve (S * x = rhs).
            const T tmp1 = rocsparse::shfl_up(rhs_reg[c][BLOCKDIM - 1], 1, BLOCKSIZE);
            const T tmp2 = rocsparse::shfl_down(rhs_reg[c][0], 1, BLOCKSIZE);
            x1[c]        = (tid >= 1) ? tmp1 : static_cast<T>(0);
            x2[c]        = (tid < (BLOCKSIZE - 1)) ? tmp2 : static_cast<T>(0);
        }

        for(int j = 1; j < BLOCKDIM - 1; j++)
        {
            const T wj = w[BLOCKSIZE * j + tid];
            const T vj = v[BLOCKSIZE * j + tid];

            for(int c = 0; c < NCOLS; c++)
            {
                rhs_reg[c][j] = rhs_reg[c][j] - wj * x1[c] - vj * x2[c];
            }
        }

        // Transpose back
        for(int c = 0; c < NCOLS; c++)
        {
            for(int i = 0; i < BLOCKDIM; i++)
            {
                const int row = BLOCKDIM * tid + i;
                if(row < m)
                {
                    B[row + ldb * (col0 + c)] = rhs_reg[c][i];
                }
            }
        }
    }
}