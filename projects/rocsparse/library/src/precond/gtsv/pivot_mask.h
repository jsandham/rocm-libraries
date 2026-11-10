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

#include "rocsparse_common.hpp"

namespace rocsparse
{
    template <int BLOCKDIM>
    struct pivot_mask
    {
        static constexpr int WORDS = (BLOCKDIM + 31) / 32;
        unsigned int         bits[WORDS];

        // Sets bit k to 0 to record a 1x1 pivot at row k.
        __device__ __forceinline__ void set_pivoting_to_1x1(int k)
        {
            bits[k >> 5] &= ~(1u << (k & 31));
        }

        // Sets bit k to 1 to record 2x2 pivoting at row k.
        __device__ __forceinline__ void set_pivoting_to2x2(int k)
        {
            bits[k >> 5] |= (1u << (k & 31));
        }

        // Returns 1 if row k used 1x1 pivoting, 2 if row k is part of a 2x2 pivot.
        __device__ __forceinline__ int get_pivoting(int k) const
        {
            return ((bits[k >> 5] >> (k & 31)) & 1u) ? 2 : 1;
        }
    };

    template <typename T>
    ROCSPARSE_DEVICE_ILF bool bunch_kaufman_criterion(T ak_1, T ak_2, T bk, T bk_1, T ck, T ck_1)
    {
        const T kappa
            = static_cast<T>(0.5) * (rocsparse::sqrt(static_cast<T>(5.0)) - static_cast<T>(1.0));

        T sigma = static_cast<T>(0);
        sigma   = rocsparse::max((rocsparse::abs(ak_1)), (rocsparse::abs(ak_2)));
        sigma   = rocsparse::max((rocsparse::abs(bk_1)), sigma);
        sigma   = rocsparse::max((rocsparse::abs(ck)), sigma);
        sigma   = rocsparse::max((rocsparse::abs(ck_1)), sigma);

        return rocsparse::abs(bk) * sigma >= kappa * rocsparse::abs(ak_1 * ck);
    }
}