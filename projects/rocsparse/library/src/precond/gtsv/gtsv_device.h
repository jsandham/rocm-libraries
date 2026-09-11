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

#include "rocsparse_common.hpp"

namespace rocsparse
{
    // Consider tridiagonal linear system A * x = rhs where A is m x m. Matrix A is represented by the three
    // arrays (the diagonal, upper diagonal, and lower diagonal) each of length m. The first entry in the
    // lower diagonal must be zero and the last entry in the upper diagonal must be zero. We solve this linear
    // system using the "Spike-Diagonal Pivoting" algorithm as detailed in the thesis:
    //
    // "Scalable Parallel Tridiagonal Algorithms with diagonal pivoting and their optimizations for many-core
    // architectures by L. Chang"
    //
    // See also:
    //
    // "L. Chang, J. A. Stratton, H. Kim and W. W. Hwu, "A scalable, numerically stable, high-performance tridiagonal
    // solver using GPUs," SC '12: Proceedings of the International Conference on High Performance Computing, Networking,
    // Storage and Analysis, Salt Lake City, UT, USA, 2012, pp. 1-11, doi: 10.1109/SC.2012.12."
    //
    // Here we give a rough outline:
    //
    // Given the tridiagonal linear system A * x = rhs. We first decompose this into A * x = D * S * x = rhs where
    // D is a block diagonal matrix and S is the "spike" matrix. We then define y = S * x which then allows us to
    // first solve D * y = rhs and then solve S * x = y. Because each block in the block diagonal matrix D is indendent,
    // we can solve each block in parallel. Specifically we use one thread for each diagonal block in D. We could use
    // the Thomas algorithm here, however because we want to incorporate some pivoting mechanism, we instead have each thread
    // decompose its diagonal block in to L * B * M^T where both L and M are lower triangular matrices and B is a diagonal
    // matrix. Specifically if the matrix D has the form:
    //
    // D = |D1  0  0  0  0  .  0 |
    //     |0   D2 0  0  0  .  0 |
    //     |0   0  D3 0  0  .  0 |
    //     |0   0  0  D4 0  .  0 |
    //     |.   .  .  .  .  .  . |
    //     |.               .  . |
    //     |0   .  .  .  .  .  Dm|
    //
    // Then D_i = L_i * B_i * M_i^T for i = 1..m. Note that each Di is itself tridiagonal. The matrices L, B, and M can
    // be computed by noting that:
    //
    // D_i = |Pd C | = |Id      0   | |Pd  0 | |Id Pd^-1*C|
    //       |A  Tr|   |A*Pd^-1 In-d| |0   Ts| |0  In-d   |
    //
    // where Pd is either 1x1 or 2x2 and Id is either 1x1 or 2x2 identity matrix. Ts is computed as Ts = Tr - A*Pd^-1*C.
    // For example consider one of the block diagonal matrices:
    //
    // Di = |2 1 0 0 0 0|
    //      |1 2 1 0 0 0|
    //      |0 1 2 1 0 0|
    //      |0 0 1 2 1 0|
    //      |0 0 0 1 2 1|
    //      |0 0 0 0 1 2|
    //
    // Then if using no pivoting (i.e. Pd is 1x1) we get:
    //
    // Pd = 2, Pd^-1 = 1/2, A = |1|, C = |1 0 0 0 0|, and Ts = |3/2 1  0  0  0|
    //                          |0|                            |1   2  1  0  0|
    //                          |0|                            |0   1  2  1  0|
    //                          |0|                            |0   0  1  2  1|
    //                          |0|                            |0   0  0  1  2|
    //
    // We can then recursively perform this on each subsequent Ts until we get:
    //
    // Di = |1   0   0   0   0   0| |2  0   0   0   0   0  | |1   1/2 0   0   0   0  |
    //      |1/2 1   0   0   0   0| |0  3/2 0   0   0   0  | |0   1   2/3 0   0   0  |
    //      |0   2/3 1   0   0   0| |0  0   4/3 0   0   0  | |0   0   1   3/4 0   0  |
    //      |0   0   3/4 1   0   0| |0  0   0   5/4 0   0  | |0   0   0   1   4/5 0  |
    //      |0   0   0   4/5 1   0| |0  0   0   0   6/5 0  | |0   0   0   0   1   5/6|
    //      |0   0   0   0   5/6 1| |0  0   0   0   0   7/6| |0   0   0   0   0   1  |
    //
    // Solving each of these systems then is just a matter of solving L * B * yi = rhsi
    // followed by M^T * xi = yi. The determination of whether we should use Pd as 1x1 or 2x2
    // is based off the Bunch-Kaufmann pivoting criteria. See cited sources above.
    //
    // Let us now return to our factoization of the original tridiagonal linear system,
    // A * x = D * S * x = rhs. We broke up finding the solution into the two phases. First
    // solving D * y = rhs and then secondly solving S * x = y. We now know how to solve
    // the first phase which is also the phase that performs the pivoting. We therefore focus on
    // solving the "spike" linear system. If the original matrix A is:
    //
    // A = |2 1 0 0 0 0 0 0| = |2 1 0 0 0 0 0 0| |1   0   v11   0   0   0   0   0|
    //     |1 2 1 0 0 0 0 0|   |1 2 0 0 0 0 0 0| |0   1   v12   0   0   0   0   0|
    //     |0 1 2 1 0 0 0 0|   |0 0 2 1 0 0 0 0| |0   w21 1     0   v21 0   0   0|
    //     |0 0 1 2 1 0 0 0|   |0 0 1 2 0 0 0 0| |0   w22 0     1   v22 0   0   0|
    //     |0 0 0 1 2 1 0 0|   |0 0 0 0 2 1 0 0| |0   0   0     w31 1   0   v31 0|
    //     |0 0 0 0 1 2 0 0|   |0 0 0 0 1 2 0 0| |0   0   0     w32 0   1   v32 0|
    //     |0 0 0 0 0 1 2 1|   |0 0 0 0 0 0 2 1| |0   0   0     0   0   w41 1   0|
    //     |0 0 0 0 0 0 1 2|   |0 0 0 0 0 0 1 2| |0   0   0     0   0   w42 0   1|
    //                         --------D-------- ----------------S----------------
    //
    // Here we use 2x2 blocks in D but in practice we use much larger blocks, for example 128x128.
    // We can solve for all the v and w unknowns by solving the following:
    //
    // |2 1||v11| = |0| and |2 1||w21| = |1| etc.
    // |1 2||v12|   |1|     |1 2||w22|   |0|
    //
    // Solving S * x = y then involves recursively decomposing the "spike" matrix like so:
    //
    // S = |1   0   v11   0   0   0   0   0| = |1  0   v11 0  0  0  0   0| |1 0 0 0 v11' 0 0 0|
    //     |0   1   v12   0   0   0   0   0|   |0  1   v12 0  0  0  0   0| |0 1 0 0 v12' 0 0 0|
    //     |0   w21 1     0   v21 0   0   0|   |0  w21 1   0  0  0  0   0| |0 0 1 0 v13' 0 0 0|
    //     |0   w22 0     1   v22 0   0   0|   |0  w22 0   1  0  0  0   0| |0 0 0 1 v14' 0 0 0|
    //     |0   0   0     w31 1   0   v31 0|   |0  0   0   0  1  0  v31 0| |0 0 0 w21' 1 0 0 0|
    //     |0   0   0     w32 0   1   v32 0|   |0  0   0   0  0  1  v32 0| |0 0 0 w22' 0 1 0 0|
    //     |0   0   0     0   0   w41 1   0|   |0  0   0   0  0  w41 1  0| |0 0 0 w23' 0 0 1 0|
    //     |0   0   0     0   0   w42 0   1|   |0  0   0   0  0  w42 0  0| |0 0 0 w24' 0 0 0 1|
    //
    // In the above the non-prime w and v values (i.e. v11, v12, w21, w22 etc) have been previously
    // computed. The primed w and v values (i.e. v11', v12', w21', w22' etc) can be found by solving:
    //
    // |1   0   v11 0||v11'| = |0| etc...
    // |0   1   v12 0||v12'|   |0|
    // |0   w21 1   0||v13'|   |0|
    // |0   w22 0   1||v14'|   |1|

    // Tiled shared-memory transpose between natural order and the padded
    // block-cyclic layout used by the LBMT kernels:
    //   padded[gwid * nblocks + glid] <-> original[glid * BLOCKDIM + gwid]
    template <uint32_t BLOCKSIZE>
    struct gtsv_marshal_tile
    {
        static constexpr uint32_t TILE       = 32;
        static constexpr uint32_t BLOCK_ROWS = BLOCKSIZE / TILE;
        static_assert(BLOCKSIZE % TILE == 0, "BLOCKSIZE must be a multiple of the transpose tile.");
    };

    // Load a TILE x TILE tile coalesced from an nblocks x BLOCKDIM matrix and store
    // it coalesced into the transposed BLOCKDIM x nblocks padded layout.
    template <uint32_t TILE, uint32_t BLOCK_ROWS, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void gtsv_tiled_marshal_to_padded(
        int m, int nblocks, const T* src, T* dst, T pad_value, T tile[TILE][TILE + 1])
    {
        const int tx = static_cast<int>(threadIdx.x);
        const int ty = static_cast<int>(threadIdx.y);

        for(int j = 0; j < static_cast<int>(TILE); j += static_cast<int>(BLOCK_ROWS))
        {
            const int row  = static_cast<int>(blockIdx.y) * static_cast<int>(TILE) + ty + j;
            const int col  = static_cast<int>(blockIdx.x) * static_cast<int>(TILE) + tx;
            const int orig = row * static_cast<int>(BLOCKDIM) + col;

            T val = pad_value;
            if(col < static_cast<int>(BLOCKDIM) && row < nblocks && orig < m)
            {
                val = src[orig];
            }
            tile[ty + j][tx] = val;
        }

        __syncthreads();

        for(int j = 0; j < static_cast<int>(TILE); j += static_cast<int>(BLOCK_ROWS))
        {
            const int row = static_cast<int>(blockIdx.x) * static_cast<int>(TILE) + ty + j;
            const int col = static_cast<int>(blockIdx.y) * static_cast<int>(TILE) + tx;

            if(row < static_cast<int>(BLOCKDIM) && col < nblocks)
            {
                dst[row * nblocks + col] = tile[tx][ty + j];
            }
        }
    }

    // Inverse of gtsv_tiled_marshal_to_padded: coalesced load from padded layout,
    // coalesced store back to the original nblocks x BLOCKDIM order.
    template <uint32_t TILE, uint32_t BLOCK_ROWS, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void gtsv_tiled_marshal_from_padded(
        int m, int nblocks, const T* src, T* dst, T tile[TILE][TILE + 1])
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        for(int j = 0; j < TILE; j += BLOCK_ROWS)
        {
            const int row = blockIdx.x * TILE + ty + j;
            const int col = blockIdx.y * TILE + tx;

            T val = static_cast<T>(0);
            if(row < BLOCKDIM && col < nblocks)
            {
                val = src[row * nblocks + col];
            }
            tile[ty + j][tx] = val;
        }

        __syncthreads();

        for(int j = 0; j < TILE; j += BLOCK_ROWS)
        {
            const int row  = blockIdx.y * TILE + ty + j;
            const int col  = blockIdx.x * TILE + tx;
            const int orig = row * BLOCKDIM + col;

            if(col < BLOCKDIM && row < nblocks && orig < m)
            {
                dst[orig] = tile[tx][ty + j];
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void data_marshaling_kernel(int m,
                                int m_pad,
                                const T* __restrict__ lower,
                                const T* __restrict__ main,
                                const T* __restrict__ upper,
                                T* __restrict__ lower_pad,
                                T* __restrict__ main_pad,
                                T* __restrict__ upper_pad)
    {
        constexpr uint32_t TILE       = gtsv_marshal_tile<BLOCKSIZE>::TILE;
        constexpr uint32_t BLOCK_ROWS = gtsv_marshal_tile<BLOCKSIZE>::BLOCK_ROWS;

        __shared__ T tile[TILE][TILE + 1];

        const int nblocks = m_pad / static_cast<int>(BLOCKDIM);

        gtsv_tiled_marshal_to_padded<TILE, BLOCK_ROWS, BLOCKDIM>(
            m, nblocks, lower, lower_pad, static_cast<T>(0), tile);
        __syncthreads();
        gtsv_tiled_marshal_to_padded<TILE, BLOCK_ROWS, BLOCKDIM>(
            m, nblocks, main, main_pad, static_cast<T>(1), tile);
        __syncthreads();
        gtsv_tiled_marshal_to_padded<TILE, BLOCK_ROWS, BLOCKDIM>(
            m, nblocks, upper, upper_pad, static_cast<T>(0), tile);
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void data_marshaling_B_device(
        int m, int m_pad, int n, const T* __restrict__ B, T* __restrict__ B_pad)
    {
        constexpr uint32_t TILE       = gtsv_marshal_tile<BLOCKSIZE>::TILE;
        constexpr uint32_t BLOCK_ROWS = gtsv_marshal_tile<BLOCKSIZE>::BLOCK_ROWS;

        __shared__ T tile[TILE][TILE + 1];

        const int nblocks = m_pad / static_cast<int>(BLOCKDIM);

        gtsv_tiled_marshal_to_padded<TILE, BLOCK_ROWS, BLOCKDIM>(
            m, nblocks, B, B_pad, static_cast<T>(0), tile);
        __syncthreads();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void data_marshaling_B_kernel(
        int m, int m_pad, int n, int ldb, const T* __restrict__ B, T* __restrict__ B_pad)
    {
        for(int64_t batch = hipBlockIdx_z; batch < n; batch += hipGridDim_z)
        {
            rocsparse::data_marshaling_B_device<BLOCKSIZE, BLOCKDIM>(
                m,
                m_pad,
                n,
                load_pointer(B, batch, static_cast<int64_t>(ldb)),
                load_pointer(B_pad, batch, static_cast<int64_t>(m_pad)));
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void data_marshaling_device2(
        int m, int m_pad, int n, const T* __restrict__ B_pad, T* __restrict__ B)
    {
        constexpr uint32_t TILE       = gtsv_marshal_tile<BLOCKSIZE>::TILE;
        constexpr uint32_t BLOCK_ROWS = gtsv_marshal_tile<BLOCKSIZE>::BLOCK_ROWS;

        __shared__ T tile[TILE][TILE + 1];

        const int nblocks = m_pad / static_cast<int>(BLOCKDIM);

        gtsv_tiled_marshal_from_padded<TILE, BLOCK_ROWS, BLOCKDIM>(m, nblocks, B_pad, B, tile);
        __syncthreads();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void data_marshaling_kernel2(
        int m, int m_pad, int n, int ldb, const T* __restrict__ B_pad, T* __restrict__ B)
    {
        for(int64_t batch = hipBlockIdx_z; batch < n; batch += hipGridDim_z)
        {
            rocsparse::data_marshaling_device2<BLOCKSIZE, BLOCKDIM>(
                m,
                m_pad,
                n,
                load_pointer(B_pad, batch, static_cast<int64_t>(m_pad)),
                load_pointer(B, batch, static_cast<int64_t>(ldb)));
        }
    }

    template <typename T>
    __device__ bool bunch_kaufman_criterion(T ak_1, T ak_2, T bk, T bk_1, T ck, T ck_1)
    {
        const double kappa = double(0.5) * (rocsparse::sqrt(double(5.0)) - double(1.0));

        double sigma = double(0);
        sigma        = rocsparse::max(double(rocsparse::abs(ak_1)), double(rocsparse::abs(ak_2)));
        sigma        = rocsparse::max(double(rocsparse::abs(bk_1)), sigma);
        sigma        = rocsparse::max(double(rocsparse::abs(ck)), sigma);
        sigma        = rocsparse::max(double(rocsparse::abs(ck_1)), sigma);

        return rocsparse::abs(bk) * sigma >= kappa * rocsparse::abs(ak_1 * ck);
    }

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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void LBMT_solve_wvmt_kernel(int m_pad,
                                T* __restrict__ lower,
                                T* __restrict__ main,
                                T* __restrict__ upper,
                                T* __restrict__ w,
                                T* __restrict__ v,
                                T* __restrict__ mt,
                                pivot_mask<256>* pivot)
    {
        static_assert(BLOCKDIM >= 2);

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int gid = tid + BLOCKSIZE * bid;

        const int nblocks = m_pad / BLOCKDIM;

        if(gid >= nblocks)
        {
            return;
        }

        T bk = main[gid];

        w[gid]                            = lower[gid];
        v[gid + (BLOCKDIM - 1) * nblocks] = upper[gid + (BLOCKDIM - 1) * nblocks];

        int k = 0;
        while(k < BLOCKDIM)
        {
            T ck   = upper[nblocks * k + gid];
            T ck_1 = (k < (BLOCKDIM - 1)) ? upper[nblocks * (k + 1) + gid] : static_cast<T>(0);
            T bk_1 = (k < (BLOCKDIM - 1)) ? main[nblocks * (k + 1) + gid] : static_cast<T>(0);
            T ak_1 = (k < (BLOCKDIM - 1)) ? lower[nblocks * (k + 1) + gid] : static_cast<T>(0);
            T ak_2 = (k < (BLOCKDIM - 2)) ? lower[nblocks * (k + 2) + gid] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            // 1x1 pivoting
            if(use_1x1_pivot || k == (BLOCKDIM - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                main[nblocks * k + gid] = inv_bk;

                T wk = w[nblocks * k + gid];
                T vk = v[nblocks * k + gid];

                w[nblocks * k + gid]  = wk * inv_bk;
                v[nblocks * k + gid]  = vk * inv_bk;
                mt[nblocks * k + gid] = ck * inv_bk;

                pivot[gid].set_pivoting_to_1x1(k);

                if(k < (BLOCKDIM - 1))
                {
                    w[nblocks * (k + 1) + gid] += -ak_1 * wk * inv_bk;
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

                main[nblocks * k + gid]  = bk_1 * det;
                upper[nblocks * k + gid] = -ck * det;

                if(k < (BLOCKDIM - 1))
                {
                    main[nblocks * (k + 1) + gid]  = bk * det;
                    lower[nblocks * (k + 1) + gid] = -ak_1 * det;
                }

                T wk   = w[nblocks * k + gid];
                T wk_1 = w[nblocks * (k + 1) + gid];
                T vk   = v[nblocks * k + gid];
                T vk_1 = v[nblocks * (k + 1) + gid];

                w[nblocks * k + gid]  = (bk_1 * wk - ck * wk_1) * det;
                v[nblocks * k + gid]  = (bk_1 * vk - ck * vk_1) * det;
                mt[nblocks * k + gid] = -ck * ck_1 * det;

                pivot[gid].set_pivoting_to2x2(k);

                if(k < (BLOCKDIM - 1))
                {
                    w[nblocks * (k + 1) + gid]  = (-ak_1 * wk + bk * wk_1) * det;
                    v[nblocks * (k + 1) + gid]  = (-ak_1 * vk + bk * vk_1) * det;
                    mt[nblocks * (k + 1) + gid] = bk * ck_1 * det;

                    pivot[gid].set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                if(k < (BLOCKDIM - 2))
                {
                    w[nblocks * (k + 2) + gid] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
                }

                if(k < (BLOCKDIM - 2))
                {
                    bk_2 = main[nblocks * (k + 2) + gid];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        assert(k == BLOCKDIM);
        // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
        k--;

        k -= pivot[gid].get_pivoting(k);

        // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
        while(k >= 0)
        {
            if(pivot[gid].get_pivoting(k) == 1)
            {
                const T tmp = mt[nblocks * k + gid];

                w[nblocks * k + gid] += -tmp * w[nblocks * (k + 1) + gid];
                v[nblocks * k + gid] += -tmp * v[nblocks * (k + 1) + gid];

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[nblocks * k + gid];
                const T tmp2 = mt[nblocks * (k - 1) + gid];

                w[nblocks * k + gid] += -tmp1 * w[nblocks * (k + 1) + gid];
                w[nblocks * (k - 1) + gid] += -tmp2 * w[nblocks * (k + 1) + gid];
                v[nblocks * k + gid] += -tmp1 * v[nblocks * (k + 1) + gid];
                v[nblocks * (k - 1) + gid] += -tmp2 * v[nblocks * (k + 1) + gid];

                k -= 2;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t COLS, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void LBMT_solve_rhs_kernel(int m_pad,
                               int n,
                               const T* __restrict__ lower,
                               const T* __restrict__ main,
                               const T* __restrict__ upper,
                               const T* __restrict__ mt,
                               T* __restrict__ rhs,
                               const pivot_mask<256>* pivot)
    {
        static_assert(BLOCKDIM >= 2);

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int gid = tid + BLOCKSIZE * bid;

        const int nblocks = m_pad / BLOCKDIM;

        if(gid >= nblocks)
        {
            return;
        }

        pivot_mask<256> p = pivot[gid];

        int k = 0;
        while(k < BLOCKDIM)
        {
            // 1x1 pivoting
            if(p.get_pivoting(k) == 1 || k == (BLOCKDIM - 1))
            {
                const T ak_1
                    = (k < (BLOCKDIM - 1)) ? lower[nblocks * (k + 1) + gid] : static_cast<T>(0);
                const T inv_bk = main[nblocks * k + gid];

                // L * B * x = y
                for(uint32_t i = 0; i < COLS; i++)
                {
                    const T rhsk = rhs[nblocks * k + gid + m_pad * (COLS * blockIdx.y + i)];

                    rhs[nblocks * k + gid + m_pad * (COLS * blockIdx.y + i)] = rhsk * inv_bk;

                    if(k < (BLOCKDIM - 1))
                    {
                        rhs[nblocks * (k + 1) + gid + m_pad * (COLS * blockIdx.y + i)]
                            += -(ak_1 * rhsk * inv_bk);
                    }
                }

                k += 1;
            }
            else
            {
                // |bk   ck  ||xk  |   |rhsk   |
                // |ak_1 bk_1||xk_1| = |rhsk _1|
                //
                //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
                //                                  |-ak_1  bk |

                const T bk_1_det = main[nblocks * k + gid]; // stores bk_1 * det
                const T bk_det   = main[nblocks * (k + 1) + gid]; // stores bk * det
                const T ck_det   = upper[nblocks * k + gid]; // stores -ck * det
                const T ak_1_det = lower[nblocks * (k + 1) + gid]; // stores = -ak_1 * det

                const T ak_2
                    = (k < (BLOCKDIM - 2)) ? lower[nblocks * (k + 2) + gid] : static_cast<T>(0);

                // L * B * x = y
                for(uint32_t i = 0; i < COLS; i++)
                {
                    const T rhsk   = rhs[nblocks * k + gid + m_pad * (COLS * blockIdx.y + i)];
                    const T rhsk_1 = rhs[nblocks * (k + 1) + gid + m_pad * (COLS * blockIdx.y + i)];

                    rhs[nblocks * k + gid + m_pad * (COLS * blockIdx.y + i)]
                        = (bk_1_det * rhsk + ck_det * rhsk_1);
                    rhs[nblocks * (k + 1) + gid + m_pad * (COLS * blockIdx.y + i)]
                        = (ak_1_det * rhsk + bk_det * rhsk_1);

                    if(k < (BLOCKDIM - 2))
                    {
                        rhs[nblocks * (k + 2) + gid + m_pad * (COLS * blockIdx.y + i)]
                            += -(ak_1_det * ak_2 * rhsk + ak_2 * bk_det * rhsk_1);
                    }
                }

                k += 2;
            }
        }

        assert(k == BLOCKDIM);
        // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
        k--;

        k -= p.get_pivoting(k);

        // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
        while(k >= 0)
        {
            if(p.get_pivoting(k) == 1)
            {
                const T tmp = mt[nblocks * k + gid];

                for(uint32_t i = 0; i < COLS; i++)
                {
                    rhs[nblocks * k + gid + m_pad * (COLS * blockIdx.y + i)]
                        += -tmp * rhs[nblocks * (k + 1) + gid + m_pad * (COLS * blockIdx.y + i)];
                }

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[nblocks * k + gid];
                const T tmp2 = mt[nblocks * (k - 1) + gid];

                for(uint32_t i = 0; i < COLS; i++)
                {
                    rhs[nblocks * k + gid + m_pad * (COLS * blockIdx.y + i)]
                        += -tmp1 * rhs[nblocks * (k + 1) + gid + m_pad * (COLS * blockIdx.y + i)];
                    rhs[nblocks * (k - 1) + gid + m_pad * (COLS * blockIdx.y + i)]
                        += -tmp2 * rhs[nblocks * (k + 1) + gid + m_pad * (COLS * blockIdx.y + i)];
                }

                k -= 2;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void fill_s_matrix_device(int m_pad,
                                                   int n,
                                                   const T* __restrict__ w,
                                                   const T* __restrict__ v,
                                                   const T* __restrict__ rhs,
                                                   T* __restrict__ S_lower,
                                                   T* __restrict__ S_main,
                                                   T* __restrict__ S_upper,
                                                   T* __restrict__ S_rhs)
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int gid = tid + BLOCKSIZE * bid;

        const int s_size = 2 * m_pad / BLOCKDIM;

        if(gid < s_size)
        {
            S_upper[gid] = (gid % 2 == 0) ? v[gid / 2] : static_cast<T>(1);
            S_lower[gid] = (gid % 2 == 0) ? static_cast<T>(1)
                                          : w[gid / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }

        if(gid == 0)
        {
            S_lower[0]          = static_cast<T>(0);
            S_lower[1]          = static_cast<T>(0);
            S_upper[s_size - 2] = static_cast<T>(0);
            S_upper[s_size - 1] = static_cast<T>(0);
            S_main[0]           = static_cast<T>(1);
            S_main[s_size - 1]  = static_cast<T>(1);
        }

        if(gid >= 1 && gid < s_size - 1)
        {
            S_main[gid]
                = (gid % 2 == 0) ? w[gid / 2] : v[gid / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }

        if(gid < s_size / 2)
        {
            S_rhs[2 * gid]     = rhs[gid];
            S_rhs[2 * gid + 1] = rhs[gid + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void fill_s_matrix_kernel(int m_pad,
                              int n,
                              const T* __restrict__ w,
                              const T* __restrict__ v,
                              const T* __restrict__ rhs,
                              T* __restrict__ S_lower,
                              T* __restrict__ S_main,
                              T* __restrict__ S_upper,
                              T* __restrict__ S_rhs)
    {
        const int s_size = 2 * m_pad / BLOCKDIM;

        for(int64_t batch = hipBlockIdx_y; batch < n; batch += hipGridDim_y)
        {
            rocsparse::fill_s_matrix_device<BLOCKSIZE, BLOCKDIM>(
                m_pad,
                n,
                w,
                v,
                load_pointer(rhs, batch, static_cast<int64_t>(m_pad)),
                S_lower,
                S_main,
                S_upper,
                load_pointer(S_rhs, batch, static_cast<int64_t>(s_size)));
        }
    }

    template <uint32_t S_SIZE, typename T>
    ROCSPARSE_KERNEL(S_SIZE)
    void S_solve_kernel(int m,
                        int n,
                        const T* __restrict__ S_lower,
                        const T* __restrict__ S_main,
                        const T* __restrict__ S_upper,
                        T* __restrict__ rhs)
    {
        static_assert(S_SIZE >= 2);

        const int batch = blockIdx.x;

        T mt[S_SIZE];

        pivot_mask<S_SIZE> pivot;

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

                mt[k] = ck * inv_bk;

                pivot.set_pivoting_to_1x1(k);

                // L * B * x = y
                T rhsk = rhs[k + m * batch] * inv_bk;

                rhs[k + m * batch] = rhsk;

                if(k < (S_SIZE - 1))
                {
                    rhs[k + 1 + m * batch] += -(ak_1 * rhsk);

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

                if(k < (S_SIZE - 1))
                {
                    mt[k + 1] = bk * ck_1 * det;

                    pivot.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                // L * B * x = y
                T rhsk   = rhs[k + m * batch] * det;
                T rhsk_1 = rhs[k + 1 + m * batch] * det;

                rhs[k + m * batch]     = (bk_1 * rhsk - ck * rhsk_1);
                rhs[k + 1 + m * batch] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (S_SIZE - 2))
                {
                    rhs[k + 2 + m * batch] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

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
                const T tmp = mt[k];

                rhs[k + m * batch] += -tmp * rhs[k + 1 + m * batch];

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[k];
                const T tmp2 = mt[k - 1];

                rhs[k + m * batch] += -tmp1 * rhs[k + 1 + m * batch];
                rhs[k - 1 + m * batch] += -tmp2 * rhs[k + 1 + m * batch];

                k -= 2;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void backward_solve_kernel(
        int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ rhs)
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        const int gid = tid + BLOCKSIZE * bid;

        const int nblocks = m_pad / BLOCKDIM;

        const int lid = gid % nblocks;
        const int wid = gid / nblocks;

        const T w_val = w[(m_pad / BLOCKDIM) * wid + lid];
        const T v_val = v[(m_pad / BLOCKDIM) * wid + lid];

        for(int j = 0; j < n; j++)
        {
            // backward solve (S * x = B_pad)
            const T x1 = (lid >= 1)
                             ? rhs[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (lid - 1) + m_pad * j]
                             : static_cast<T>(0);
            const T x2
                = (lid < (m_pad / BLOCKDIM - 1)) ? rhs[lid + 1 + m_pad * j] : static_cast<T>(0);

            if(wid >= 1 && wid < BLOCKDIM - 1)
            {
                rhs[(m_pad / BLOCKDIM) * wid + lid + m_pad * j]
                    = rhs[(m_pad / BLOCKDIM) * wid + lid + m_pad * j] - w_val * x1 - v_val * x2;
            }
        }

        // if(gid >= nblocks)
        // {
        //     return;
        // }

        // // backward solve (S * x = B_pad)
        // T x1
        //     = (gid >= 1) ? rhs[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (gid - 1)] : static_cast<T>(0);
        // T x2 = (gid < (m_pad / BLOCKDIM - 1)) ? rhs[gid + 1] : static_cast<T>(0);

        // for(int j = 1; j < BLOCKDIM - 1; j++)
        // {
        //     rhs[(m_pad / BLOCKDIM) * j + gid] = rhs[(m_pad / BLOCKDIM) * j + gid]
        //                                         - w[(m_pad / BLOCKDIM) * j + gid] * x1
        //                                         - v[(m_pad / BLOCKDIM) * j + gid] * x2;
        // }
    }

    // template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    // ROCSPARSE_KERNEL(BLOCKSIZE)
    // void backward_solve_kernel(
    //     int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ rhs)
    // {
    //     for(int64_t batch = hipBlockIdx_y; batch < n; batch += hipGridDim_y)
    //     {
    //         rocsparse::backward_solve_device<BLOCKSIZE, BLOCKDIM>(
    //             m_pad, n, w_val, v_val, load_pointer(rhs, batch, static_cast<int64_t>(m_pad)));
    //     }
    // }

    template <uint32_t BLOCKDIM, uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void scatter_S_B_to_B_pad_device(
        int s_size, int m_pad, int n, const T* __restrict__ S_B, T* __restrict__ B_pad)
    {
        const int i = blockIdx.x * BLOCKSIZE + threadIdx.x; // [0, s_size/2)

        if(i >= s_size / 2)
            return;

        const int stride = (m_pad / BLOCKDIM) * (BLOCKDIM - 1);

        // After the swap loop, element at position 2*i is:
        //   i == 0  -> S_B[0]       (not touched by the swap)
        //   i  > 0  -> S_B[2*i - 1] (position 2*i was swapped with 2*i-1)
        const T val_even = (i == 0) ? S_B[0] : S_B[2 * i - 1];

        // After the swap loop, element at position 2*i+1 is:
        //   2*i+1 < s_size-1  -> S_B[2*i + 2] (swapped with its right neighbour)
        //   2*i+1 == s_size-1 -> S_B[s_size-1] (last element, not touched)
        const T val_odd = (2 * i + 1 < s_size - 1) ? S_B[2 * i + 2] : S_B[s_size - 1];

        B_pad[i]          = val_even;
        B_pad[i + stride] = val_odd;
    }

    template <uint32_t BLOCKDIM, uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scatter_S_B_to_B_pad_kernel(
        int s_size, int m_pad, int n, const T* __restrict__ S_B, T* __restrict__ B_pad)
    {
        for(int64_t batch = hipBlockIdx_y; batch < n; batch += hipGridDim_y)
        {
            rocsparse::scatter_S_B_to_B_pad_device<BLOCKDIM, BLOCKSIZE>(
                s_size,
                m_pad,
                n,
                load_pointer(S_B, batch, static_cast<int64_t>(s_size)),
                load_pointer(B_pad, batch, static_cast<int64_t>(m_pad)));
        }
    }
}
