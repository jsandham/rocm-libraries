/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // The block kernels index complete partitions of BLOCKDIM rows, so padding only
    // has to reach the next multiple of BLOCKDIM. Power-of-two padding adds fully
    // artificial partitions whose spike tips are zero.
    inline int pad_to_blockdim(int m, int blockdim)
    {
        const int64_t value  = static_cast<int64_t>(m);
        const int64_t padded = ((value + blockdim - 1) / blockdim) * blockdim;
        return static_cast<int>(padded);
    }

    template <typename T>
    struct gtsv_buffer_data
    {
        // Each level keeps two interface rows per BLOCKDIM-row partition. The leaf
        // solver handles every reduced size up to BLOCKDIM, so six levels cover any
        // m whose padded length fits in a 32-bit int.
        static constexpr int MAX_RECURSION_LEVELS = 6;

        static constexpr int BLOCKDIM = 32;

        // Number of elements of T in the factorization workspace used by the fused
        // small solver (version 2), which must be at least
        // fused_v2_layout<32, 32>::size (8610).
        static constexpr int FUSED_V2_FACTOR_SIZE = 8704;

        T* dl_pad[MAX_RECURSION_LEVELS];
        T* d_pad[MAX_RECURSION_LEVELS];
        T* du_pad[MAX_RECURSION_LEVELS];
        T* B_pad[MAX_RECURSION_LEVELS];

        T* w_pad[MAX_RECURSION_LEVELS];
        T* v_pad[MAX_RECURSION_LEVELS];
        T* mt_pad[MAX_RECURSION_LEVELS];

        // Only level 0 is allocated.
        T* factor_pad[MAX_RECURSION_LEVELS];

        T* sl[MAX_RECURSION_LEVELS];
        T* s[MAX_RECURSION_LEVELS];
        T* su[MAX_RECURSION_LEVELS];
        T* sB[MAX_RECURSION_LEVELS];
    };


    template <typename T>
    rocsparse_status gtsv_buffer_size_template(rocsparse_handle handle,
                                               rocsparse_int    m,
                                               rocsparse_int    n,
                                               const T*         dl,
                                               const T*         d,
                                               const T*         du,
                                               const T*         B,
                                               rocsparse_int    ldb,
                                               size_t*          buffer_size);

    template <typename T>
    rocsparse_status gtsv_template(rocsparse_handle handle,
                                   rocsparse_int    m,
                                   rocsparse_int    n,
                                   const T*         dl,
                                   const T*         d,
                                   const T*         du,
                                   T*               B,
                                   rocsparse_int    ldb,
                                   void*            temp_buffer);
}
