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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void data_transpose_device(
        int m, int m_pad, const T* __restrict__ d_in, T* __restrict__ d_out, T pad_value)
    {
        // Each CUDA block handles PARTITIONS_PER_GROUP (e.g., 256) total partitions,
        // but processes them in smaller chunks dictated by BLOCKSIZE (e.g., 128).
        const int group_start_partition = blockIdx.x * PARTITIONS_PER_GROUP;
        const int nblocks               = m_pad / BLOCKDIM;

        // Prevent completely out-of-bounds blocks from executing
        if(group_start_partition >= nblocks)
            return;

        // Shared memory now only needs to hold BLOCKSIZE partitions at a time (~33 KB)
        __shared__ T smem[BLOCKSIZE][BLOCKDIM + 1];

        const int tid = threadIdx.x;

        // Loop over the group in chunks of BLOCKSIZE (e.g., 0, then 128)
        for(int chunk = 0; chunk < PARTITIONS_PER_GROUP; chunk += BLOCKSIZE)
        {
            // ---------------------------------------------------------
            // PHASE 1: Coalesced Read from Global -> Shared Memory
            // ---------------------------------------------------------
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                int linear_idx    = i * BLOCKSIZE + tid;
                int partition_idx = linear_idx / BLOCKDIM;
                int element_idx   = linear_idx % BLOCKDIM;

                // Offset the global read by the current chunk
                int global_partition = group_start_partition + chunk + partition_idx;
                int global_in_idx    = global_partition * BLOCKDIM + element_idx;

                T val = pad_value;

                if(global_partition < nblocks && global_in_idx < m)
                {
                    val = d_in[global_in_idx];
                }

                smem[partition_idx][element_idx] = val;
            }

            __syncthreads();

            // ---------------------------------------------------------
            // PHASE 2: Coalesced Write from Shared -> Global Memory
            // ---------------------------------------------------------
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                int out_partition_idx = tid;
                int out_element_idx   = i;

                int linear_out
                    = out_element_idx * nblocks + group_start_partition + chunk + out_partition_idx;

                if(group_start_partition + chunk + out_partition_idx < nblocks)
                {
                    d_out[linear_out] = smem[out_partition_idx][out_element_idx];
                }
            }

            // Synchronize before overwriting shared memory with the next chunk
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void data_untranspose_backward_solve_device(int m,
                                                                     int m_pad,
                                                                     const T* __restrict__ w,
                                                                     const T* __restrict__ v,
                                                                     const T* __restrict__ d_in,
                                                                     T* __restrict__ d_out)
    {
        const int group_start_partition = blockIdx.x * PARTITIONS_PER_GROUP;
        const int nblocks               = m_pad / BLOCKDIM;

        if(group_start_partition >= nblocks)
            return;

        __shared__ T smem[BLOCKSIZE][BLOCKDIM + 1];

        const int tid = threadIdx.x;

        for(int chunk = 0; chunk < PARTITIONS_PER_GROUP; chunk += BLOCKSIZE)
        {
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                const int global_partition = group_start_partition + chunk + tid;
                const int linear_in        = i * nblocks + global_partition;

                T val = T(0);
                if(global_partition < nblocks)
                {
                    val = d_in[linear_in];
                }

                smem[tid][i] = val;
            }

            __syncthreads();

            for(int i = 0; i < BLOCKDIM; ++i)
            {
                const int linear_out       = i * BLOCKSIZE + tid;
                const int partition_idx    = linear_out / BLOCKDIM;
                const int element_idx      = linear_out % BLOCKDIM;
                const int global_partition = group_start_partition + chunk + partition_idx;
                const int global_out_idx   = global_partition * BLOCKDIM + element_idx;

                if(global_partition < nblocks && global_out_idx < m)
                {
                    T value = smem[partition_idx][element_idx];

                    if(element_idx > 0 && element_idx < BLOCKDIM - 1)
                    {
                        const T x1 = (global_partition > 0)
                                         ? d_in[(BLOCKDIM - 1) * nblocks + global_partition - 1]
                                         : T(0);
                        const T x2
                            = (global_partition < nblocks - 1) ? d_in[global_partition + 1] : T(0);
                        const int coefficient_idx = element_idx * nblocks + global_partition;

                        value -= w[coefficient_idx] * x1 + v[coefficient_idx] * x2;
                    }

                    d_out[global_out_idx] = value;
                }
            }

            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
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
        data_transpose_device<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
            m, m_pad, lower, lower_pad, static_cast<T>(0));
        __syncthreads();
        data_transpose_device<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
            m, m_pad, main, main_pad, static_cast<T>(1));
        __syncthreads();
        data_transpose_device<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
            m, m_pad, upper, upper_pad, static_cast<T>(0));
        __syncthreads();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void data_marshaling_B_kernel(
        int m, int m_pad, int n, int ldb, const T* __restrict__ B, T* __restrict__ B_pad)
    {
        for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
        {
            data_transpose_device<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                m,
                m_pad,
                load_pointer(B, batch, ldb),
                load_pointer(B_pad, batch, m_pad),
                static_cast<T>(0));
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void data_untranspose_backward_solve_kernel(int m,
                                                int m_pad,
                                                int n,
                                                int ldb,
                                                const T* __restrict__ w,
                                                const T* __restrict__ v,
                                                const T* __restrict__ B_pad,
                                                T* __restrict__ X)
    {
        for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
        {
            data_untranspose_backward_solve_device<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                m, m_pad, w, v, load_pointer(B_pad, batch, m_pad), load_pointer(X, batch, ldb));
            __syncthreads();
        }
    }

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
                                const T* __restrict__ lower,
                                const T* __restrict__ main,
                                const T* __restrict__ upper,
                                T* __restrict__ w,
                                T* __restrict__ v,
                                T* __restrict__ mt)
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

        pivot_mask<BLOCKDIM> pivot{};

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

                const T wk = w[nblocks * k + gid];
                const T vk = v[nblocks * k + gid];

                w[nblocks * k + gid]  = wk * inv_bk;
                v[nblocks * k + gid]  = vk * inv_bk;
                mt[nblocks * k + gid] = ck * inv_bk;

                pivot.set_pivoting_to_1x1(k);

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

                const T wk   = w[nblocks * k + gid];
                const T wk_1 = w[nblocks * (k + 1) + gid];
                const T vk   = v[nblocks * k + gid];
                const T vk_1 = v[nblocks * (k + 1) + gid];

                w[nblocks * k + gid]  = (bk_1 * wk - ck * wk_1) * det;
                v[nblocks * k + gid]  = (bk_1 * vk - ck * vk_1) * det;
                mt[nblocks * k + gid] = -ck * ck_1 * det;

                pivot.set_pivoting_to2x2(k);

                if(k < (BLOCKDIM - 1))
                {
                    w[nblocks * (k + 1) + gid]  = (-ak_1 * wk + bk * wk_1) * det;
                    v[nblocks * (k + 1) + gid]  = (-ak_1 * vk + bk * vk_1) * det;
                    mt[nblocks * (k + 1) + gid] = bk * ck_1 * det;

                    pivot.set_pivoting_to2x2(k + 1);
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

        k -= pivot.get_pivoting(k);

        // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
        while(k >= 0)
        {
            if(pivot.get_pivoting(k) == 1)
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void LBMT_solve_rhs_device(int m_pad,
                                                    int n,
                                                    const T* __restrict__ lower,
                                                    const T* __restrict__ main,
                                                    const T* __restrict__ upper,
                                                    const T* __restrict__ mt,
                                                    T* __restrict__ rhs)
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

        pivot_mask<BLOCKDIM> pivot{};

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

                pivot.set_pivoting_to_1x1(k);

                // L * B * x = y
                const T rhsk = rhs[nblocks * k + gid] * inv_bk;

                rhs[nblocks * k + gid] = rhsk;

                if(k < (BLOCKDIM - 1))
                {
                    rhs[nblocks * (k + 1) + gid] += -(ak_1 * rhsk);

                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                pivot.set_pivoting_to2x2(k);

                if(k < (BLOCKDIM - 1))
                {
                    pivot.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                // |bk   ck  ||xk  |   |rhsk   |
                // |ak_1 bk_1||xk_1| = |rhsk _1|
                //
                //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
                //                                  |-ak_1  bk |

                // L * B * x = y
                const T rhsk   = rhs[nblocks * k + gid] * det;
                const T rhsk_1 = rhs[nblocks * (k + 1) + gid] * det;

                rhs[nblocks * k + gid]       = (bk_1 * rhsk - ck * rhsk_1);
                rhs[nblocks * (k + 1) + gid] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (BLOCKDIM - 2))
                {
                    rhs[nblocks * (k + 2) + gid] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

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

        k -= pivot.get_pivoting(k);

        // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
        while(k >= 0)
        {
            if(pivot.get_pivoting(k) == 1)
            {
                const T tmp = mt[nblocks * k + gid];

                rhs[nblocks * k + gid] += -tmp * rhs[nblocks * (k + 1) + gid];

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[nblocks * k + gid];
                const T tmp2 = mt[nblocks * (k - 1) + gid];

                rhs[nblocks * k + gid] += -tmp1 * rhs[nblocks * (k + 1) + gid];
                rhs[nblocks * (k - 1) + gid] += -tmp2 * rhs[nblocks * (k + 1) + gid];

                k -= 2;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void LBMT_solve_rhs_kernel(int m_pad,
                               int n,
                               const T* __restrict__ lower,
                               const T* __restrict__ main,
                               const T* __restrict__ upper,
                               const T* __restrict__ mt,
                               T* __restrict__ rhs)
    {
        for(int64_t batch = hipBlockIdx_y; batch < n; batch += hipGridDim_y)
        {
            rocsparse::LBMT_solve_rhs_device<BLOCKSIZE, BLOCKDIM>(
                m_pad,
                n,
                lower,
                main,
                upper,
                mt,
                load_pointer(rhs, batch, static_cast<int64_t>(m_pad)));
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
                                                   T* __restrict__ S_rhs,
                                                   bool write_s_matrix)
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int gid = tid + BLOCKSIZE * bid;

        const int s_size = 2 * m_pad / BLOCKDIM;

        if(write_s_matrix && gid < s_size)
        {
            S_upper[gid] = (gid % 2 == 0) ? v[gid / 2] : static_cast<T>(1);
            S_lower[gid] = (gid % 2 == 0) ? static_cast<T>(1)
                                          : w[gid / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }

        if(write_s_matrix && gid >= 1 && gid < s_size - 1)
        {
            S_main[gid]
                = (gid % 2 == 0) ? w[gid / 2] : v[gid / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }

        if(gid < s_size / 2)
        {
            S_rhs[2 * gid]     = rhs[gid];
            S_rhs[2 * gid + 1] = rhs[gid + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }

        if(write_s_matrix && gid == 0)
        {
            S_lower[0] = static_cast<T>(0);
            S_main[0]  = static_cast<T>(1);
        }
        if(write_s_matrix && gid == 1)
        {
            S_lower[1] = static_cast<T>(0);
        }
        if(write_s_matrix && gid == s_size - 2)
        {
            S_upper[s_size - 2] = static_cast<T>(0);
        }
        if(write_s_matrix && gid == s_size - 1)
        {
            S_upper[s_size - 1] = static_cast<T>(0);
            S_main[s_size - 1]  = static_cast<T>(1);
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
                load_pointer(S_rhs, batch, static_cast<int64_t>(s_size)),
                batch == 0);
        }
    }

    template <typename T>
    ROCSPARSE_KERNEL(1)
    void S_solve_kernel(int m,
                        int n,
                        const T* __restrict__ S_lower,
                        const T* __restrict__ S_main,
                        const T* __restrict__ S_upper,
                        T* __restrict__ rhs)
    {
        // The host only launches this leaf when the reduced system has at most one
        // partition worth of interface rows.
        constexpr int max_rows = 32;
        if(m < 2 || m > max_rows)
        {
            return;
        }

        const int batch = blockIdx.x;

        T mt[max_rows];

        pivot_mask<max_rows> pivot{};

        int k  = 0;
        T   bk = S_main[k];

        while(k < m)
        {
            T ck   = S_upper[k];
            T ck_1 = (k < (m - 1)) ? S_upper[k + 1] : static_cast<T>(0);
            T bk_1 = (k < (m - 1)) ? S_main[k + 1] : static_cast<T>(0);
            T ak_1 = (k < (m - 1)) ? S_lower[k + 1] : static_cast<T>(0);
            T ak_2 = (k < (m - 2)) ? S_lower[k + 2] : static_cast<T>(0);

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
                T rhsk = rhs[k + m * batch] * inv_bk;

                rhs[k + m * batch] = rhsk;

                if(k < (m - 1))
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

                if(k < (m - 1))
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

                if(k < (m - 2))
                {
                    rhs[k + 2 + m * batch] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                    bk_2 = S_main[k + 2];
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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scatter_S_B_to_B_pad_kernel(
        int s_size, int m_pad, int n, const T* __restrict__ S_B, T* __restrict__ B_pad)
    {
        for(int64_t batch = hipBlockIdx_y; batch < n; batch += hipGridDim_y)
        {
            rocsparse::scatter_S_B_to_B_pad_device<BLOCKSIZE, BLOCKDIM>(
                s_size,
                m_pad,
                n,
                load_pointer(S_B, batch, static_cast<int64_t>(s_size)),
                load_pointer(B_pad, batch, static_cast<int64_t>(m_pad)));
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void backward_solve_device(
        int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ B_pad)
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int gid = tid + BLOCKSIZE * bid;

        const int lid = gid % (m_pad / BLOCKDIM);
        const int wid = gid / (m_pad / BLOCKDIM);

        if(gid >= m_pad)
        {
            return;
        }

        // backward solve (S * x = B_pad)
        const T x1 = (lid >= 1) ? B_pad[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (lid - 1)]
                                : static_cast<T>(0);
        const T x2 = (lid < (m_pad / BLOCKDIM - 1)) ? B_pad[lid + 1] : static_cast<T>(0);

        if(wid >= 1 && wid < BLOCKDIM - 1)
        {
            B_pad[(m_pad / BLOCKDIM) * wid + lid] = B_pad[(m_pad / BLOCKDIM) * wid + lid]
                                                    - w[(m_pad / BLOCKDIM) * wid + lid] * x1
                                                    - v[(m_pad / BLOCKDIM) * wid + lid] * x2;
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void backward_solve_kernel(
        int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ B_pad)
    {
        for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
        {
            backward_solve_device<BLOCKSIZE, BLOCKDIM>(
                m_pad, n, w, v, load_pointer(B_pad, batch, m_pad));
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void
        data_untranspose_device(int m, int m_pad, const T* __restrict__ d_in, T* __restrict__ d_out)
    {
        // Each CUDA block handles PARTITIONS_PER_GROUP partitions,
        // processed in smaller chunks dictated by BLOCKSIZE.
        const int group_start_partition = blockIdx.x * PARTITIONS_PER_GROUP;
        const int nblocks               = m_pad / BLOCKDIM;

        // Prevent completely out-of-bounds blocks from executing
        if(group_start_partition >= nblocks)
            return;

        __shared__ T smem[BLOCKSIZE][BLOCKDIM + 1];

        const int tid = threadIdx.x;

        // Loop over the group in chunks of BLOCKSIZE
        for(int chunk = 0; chunk < PARTITIONS_PER_GROUP; chunk += BLOCKSIZE)
        {
            // ---------------------------------------------------------
            // PHASE 1: Coalesced Read from Padded Layout -> Shared Memory
            // ---------------------------------------------------------
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                int in_partition_idx = tid;
                int in_element_idx   = i;

                int linear_in
                    = in_element_idx * nblocks + group_start_partition + chunk + in_partition_idx;

                T val = T(0);
                if(group_start_partition + chunk + in_partition_idx < nblocks)
                {
                    val = d_in[linear_in];
                }

                smem[in_partition_idx][in_element_idx] = val;
            }

            __syncthreads();

            // ---------------------------------------------------------
            // PHASE 2: Coalesced Write from Shared -> Global Standard Memory
            // ---------------------------------------------------------
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                int linear_out    = i * BLOCKSIZE + tid;
                int partition_idx = linear_out / BLOCKDIM;
                int element_idx   = linear_out % BLOCKDIM;

                int global_partition = group_start_partition + chunk + partition_idx;
                int global_out_idx   = global_partition * BLOCKDIM + element_idx;

                // Only write back to the unpadded, original system size 'm'.
                // The padded tail elements are naturally discarded.
                if(global_partition < nblocks && global_out_idx < m)
                {
                    d_out[global_out_idx] = smem[partition_idx][element_idx];
                }
            }

            // Synchronize before overwriting shared memory with the next chunk
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void reverse_data_marshaling_B_kernel(
        int m, int m_pad, int n, int ldb, const T* __restrict__ B_pad, T* __restrict__ B)
    {
        for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
        {
            data_untranspose_device<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                m, m_pad, load_pointer(B_pad, batch, m_pad), load_pointer(B, batch, ldb));
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE) void S_solve_fused_kernel(const T* __restrict__ lower,
                                        const T* __restrict__ main,
                                        const T* __restrict__ upper,
                                        T* __restrict__ rhs)
    {
        static_assert(BLOCKDIM >= 2);
        static_assert(BLOCKDIM % 2 == 0);
        static_assert(BLOCKSIZE % 2 == 0);

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        const int batch = bid;

        constexpr int M = BLOCKSIZE * BLOCKDIM;

        // Transpose data
        __shared__ T slower[M];
        __shared__ T smain[M];
        __shared__ T supper[M];
        __shared__ T srhs[M];

        __shared__ T sw[M];
        __shared__ T sv[M];
        __shared__ T smt[M];

        for(int i = 0; i < BLOCKDIM; i++)
        {
            slower[BLOCKSIZE * i + tid] = lower[BLOCKDIM * tid + i];
            smain[BLOCKSIZE * i + tid]  = main[BLOCKDIM * tid + i];
            supper[BLOCKSIZE * i + tid] = upper[BLOCKDIM * tid + i];
            srhs[BLOCKSIZE * i + tid]   = rhs[BLOCKDIM * tid + i + M * batch];

            sw[BLOCKSIZE * i + tid] = static_cast<T>(0);
            sv[BLOCKSIZE * i + tid] = static_cast<T>(0);
            smt[BLOCKSIZE * i + tid] = static_cast<T>(0);
        }

        __syncthreads();

        // LBMT solve
        T bk = smain[tid];

        pivot_mask<BLOCKDIM> pivot{};

        sw[tid]                              = slower[tid];
        sv[tid + (BLOCKDIM - 1) * BLOCKSIZE] = supper[tid + (BLOCKDIM - 1) * BLOCKSIZE];

        int k = 0;
        while(k < BLOCKDIM)
        {
            T ck   = supper[BLOCKSIZE * k + tid];
            T ck_1 = (k < (BLOCKDIM - 1)) ? supper[BLOCKSIZE * (k + 1) + tid] : static_cast<T>(0);
            T bk_1 = (k < (BLOCKDIM - 1)) ? smain[BLOCKSIZE * (k + 1) + tid] : static_cast<T>(0);
            T ak_1 = (k < (BLOCKDIM - 1)) ? slower[BLOCKSIZE * (k + 1) + tid] : static_cast<T>(0);
            T ak_2 = (k < (BLOCKDIM - 2)) ? slower[BLOCKSIZE * (k + 2) + tid] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            // 1x1 pivoting
            if(use_1x1_pivot || k == (BLOCKDIM - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                const T wk = sw[BLOCKSIZE * k + tid];
                const T vk = sv[BLOCKSIZE * k + tid];

                sw[BLOCKSIZE * k + tid]  = wk * inv_bk;
                sv[BLOCKSIZE * k + tid]  = vk * inv_bk;
                smt[BLOCKSIZE * k + tid] = ck * inv_bk;

                pivot.set_pivoting_to_1x1(k);

                if(k < (BLOCKDIM - 1))
                {
                    sw[BLOCKSIZE * (k + 1) + tid] += -ak_1 * wk * inv_bk;
                }

                // L * B * x = y
                const T rhsk = srhs[BLOCKSIZE * k + tid] * inv_bk;

                srhs[BLOCKSIZE * k + tid] = rhsk;

                if(k < (BLOCKDIM - 1))
                {
                    srhs[BLOCKSIZE * (k + 1) + tid] += -(ak_1 * rhsk);

                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                const T wk   = sw[BLOCKSIZE * k + tid];
                const T wk_1 = sw[BLOCKSIZE * (k + 1) + tid];
                const T vk   = sv[BLOCKSIZE * k + tid];
                const T vk_1 = sv[BLOCKSIZE * (k + 1) + tid];

                sw[BLOCKSIZE * k + tid]  = (bk_1 * wk - ck * wk_1) * det;
                sv[BLOCKSIZE * k + tid]  = (bk_1 * vk - ck * vk_1) * det;
                smt[BLOCKSIZE * k + tid] = -ck * ck_1 * det;

                pivot.set_pivoting_to2x2(k);

                if(k < (BLOCKDIM - 1))
                {
                    sw[BLOCKSIZE * (k + 1) + tid]  = (-ak_1 * wk + bk * wk_1) * det;
                    sv[BLOCKSIZE * (k + 1) + tid]  = (-ak_1 * vk + bk * vk_1) * det;
                    smt[BLOCKSIZE * (k + 1) + tid] = bk * ck_1 * det;

                    pivot.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                if(k < (BLOCKDIM - 2))
                {
                    sw[BLOCKSIZE * (k + 2) + tid] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
                }

                // |bk   ck  ||xk  |   |rhsk   |
                // |ak_1 bk_1||xk_1| = |rhsk _1|
                //
                //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
                //                                  |-ak_1  bk |

                // L * B * x = y
                const T rhsk   = srhs[BLOCKSIZE * k + tid] * det;
                const T rhsk_1 = srhs[BLOCKSIZE * (k + 1) + tid] * det;

                srhs[BLOCKSIZE * k + tid]       = (bk_1 * rhsk - ck * rhsk_1);
                srhs[BLOCKSIZE * (k + 1) + tid] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (BLOCKDIM - 2))
                {
                    srhs[BLOCKSIZE * (k + 2) + tid] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                    bk_2 = smain[BLOCKSIZE * (k + 2) + tid];
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
                const T tmp = smt[BLOCKSIZE * k + tid];

                sw[BLOCKSIZE * k + tid] += -tmp * sw[BLOCKSIZE * (k + 1) + tid];
                sv[BLOCKSIZE * k + tid] += -tmp * sv[BLOCKSIZE * (k + 1) + tid];
                srhs[BLOCKSIZE * k + tid] += -tmp * srhs[BLOCKSIZE * (k + 1) + tid];

                k -= 1;
            }
            else
            {
                const T tmp1 = smt[BLOCKSIZE * k + tid];
                const T tmp2 = smt[BLOCKSIZE * (k - 1) + tid];

                sw[BLOCKSIZE * k + tid] += -tmp1 * sw[BLOCKSIZE * (k + 1) + tid];
                sw[BLOCKSIZE * (k - 1) + tid] += -tmp2 * sw[BLOCKSIZE * (k + 1) + tid];
                sv[BLOCKSIZE * k + tid] += -tmp1 * sv[BLOCKSIZE * (k + 1) + tid];
                sv[BLOCKSIZE * (k - 1) + tid] += -tmp2 * sv[BLOCKSIZE * (k + 1) + tid];
                srhs[BLOCKSIZE * k + tid] += -tmp1 * srhs[BLOCKSIZE * (k + 1) + tid];
                srhs[BLOCKSIZE * (k - 1) + tid] += -tmp2 * srhs[BLOCKSIZE * (k + 1) + tid];

                k -= 2;
            }
        }

        __syncthreads();

        // Fill S system
        constexpr int S_SIZE = 2 * BLOCKSIZE;

        __shared__ T S_lower[S_SIZE];
        __shared__ T S_main[S_SIZE];
        __shared__ T S_upper[S_SIZE];
        __shared__ T S_rhs[S_SIZE];

        for(int i = 0; i < 2; i++)
        {
            const int gid = BLOCKSIZE * i + tid;

            S_upper[gid] = (gid % 2 == 0) ? sv[gid / 2] : static_cast<T>(1);
            S_lower[gid]
                = (gid % 2 == 0) ? static_cast<T>(1) : sw[gid / 2 + BLOCKSIZE * (BLOCKDIM - 1)];

            if(gid >= 1 && gid < S_SIZE - 1)
            {
                S_main[gid] = (gid % 2 == 0) ? sw[gid / 2] : sv[gid / 2 + BLOCKSIZE * (BLOCKDIM - 1)];
            }

            if(gid < S_SIZE / 2)
            {
                S_rhs[2 * gid]     = srhs[gid];
                S_rhs[2 * gid + 1] = srhs[gid + BLOCKSIZE * (BLOCKDIM - 1)];
            }
        }

        if(tid == 0)
        {
            S_lower[0] = static_cast<T>(0);
            S_main[0]  = static_cast<T>(1);
        }
        if(tid == 1)
        {
            S_lower[1] = static_cast<T>(0);
        }
        if(tid == S_SIZE - BLOCKSIZE - 2)
        {
            S_upper[S_SIZE - 2] = static_cast<T>(0);
        }
        if(tid == S_SIZE - BLOCKSIZE - 1)
        {
            S_upper[S_SIZE - 1] = static_cast<T>(0);
            S_main[S_SIZE - 1]  = static_cast<T>(1);
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
                    const T rhsk = S_rhs[k] * inv_bk;

                    S_rhs[k] = rhsk;

                    if(k < (S_SIZE - 1))
                    {
                        S_rhs[k + 1] += -(ak_1 * rhsk);

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
                    const T rhsk   = S_rhs[k] * det;
                    const T rhsk_1 = S_rhs[k + 1] * det;

                    S_rhs[k]     = (bk_1 * rhsk - ck * rhsk_1);
                    S_rhs[k + 1] = (-ak_1 * rhsk + bk * rhsk_1);

                    if(k < (S_SIZE - 2))
                    {
                        S_rhs[k + 2] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

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

                    S_rhs[k] += -tmp * S_rhs[k + 1];

                    k -= 1;
                }
                else
                {
                    const T tmp1 = S_mt[k];
                    const T tmp2 = S_mt[k - 1];

                    S_rhs[k] += -tmp1 * S_rhs[k + 1];
                    S_rhs[k - 1] += -tmp2 * S_rhs[k + 1];

                    k -= 2;
                }
            }
        }

        __syncthreads();

        // Scatter
        constexpr int STRIDE = BLOCKSIZE * (BLOCKDIM - 1);

        // After the swap loop, element at position 2*tid is:
        //   tid == 0  -> S_rhs[0]       (not touched by the swap)
        //   tid  > 0  -> S_rhs[2*tid - 1] (position 2*tid was swapped with 2*tid-1)
        const T val_even = (tid == 0) ? S_rhs[0] : S_rhs[2 * tid - 1];

        // After the swap loop, element at position 2*i+1 is:
        //   2*tid+1 < S_SIZE-1  -> S_rhs[2*tid + 2] (swapped with its right neighbour)
        //   2*tid+1 == S_SIZE-1 -> S_rhs[S_SIZE-1] (last element, not touched)
        const T val_odd = (2 * tid + 1 < S_SIZE - 1) ? S_rhs[2 * tid + 2] : S_rhs[S_SIZE - 1];

        srhs[tid]          = val_even;
        srhs[tid + STRIDE] = val_odd;

        __syncthreads();

        // backward solve (S * x = rhs)
        const T x1 = (tid >= 1) ? srhs[BLOCKSIZE * (BLOCKDIM - 1) + (tid - 1)] : static_cast<T>(0);
        const T x2 = (tid < (BLOCKSIZE - 1)) ? srhs[tid + 1] : static_cast<T>(0);

        for(int j = 1; j < BLOCKDIM - 1; j++)
        {
            srhs[BLOCKSIZE * j + tid] = srhs[BLOCKSIZE * j + tid] - sw[BLOCKSIZE * j + tid] * x1
                                        - sv[BLOCKSIZE * j + tid] * x2;
        }

        __syncthreads();

        // Transpose back
        for(int i = 0; i < BLOCKDIM; i++)
        {
            rhs[BLOCKDIM * tid + i + M * batch] = srhs[BLOCKSIZE * i + tid];
        }
    }





























































    // Address of element `element` within partition `partition` in the grouped layout
    // written by data_transpose_device2.
    template <uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP>
    struct gtsv_partition_index
    {
        int group_width;
        int base;

        __device__ __forceinline__ int operator()(int element) const
        {
            return base + element * group_width;
        }
    };

    template <uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP>
    ROCSPARSE_DEVICE_ILF gtsv_partition_index<BLOCKDIM, PARTITIONS_PER_GROUP>
                         gtsv_partition(int partition, int nblocks)
    {
        const int partitions_per_group = static_cast<int>(PARTITIONS_PER_GROUP);
        const int group_start = (partition / partitions_per_group) * partitions_per_group;
        const int remaining   = nblocks - group_start;
        const int group_width
            = remaining < partitions_per_group ? remaining : partitions_per_group;

        return {group_width,
                group_start * static_cast<int>(BLOCKDIM) + (partition - group_start)};
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void LBMT_solve_wvmt_kernel2(int m_pad,
                                 const T* __restrict__ lower,
                                 const T* __restrict__ main,
                                 const T* __restrict__ upper,
                                 T* __restrict__ w,
                                 T* __restrict__ v,
                                 T* __restrict__ mt)
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

        const auto row = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(gid, nblocks);

        T bk = main[row(0)];

        pivot_mask<BLOCKDIM> pivot{};

        w[row(0)]                 = lower[row(0)];
        v[row(BLOCKDIM - 1)]      = upper[row(BLOCKDIM - 1)];

        int k = 0;
        while(k < BLOCKDIM)
        {
            T ck   = upper[row(k)];
            T ck_1 = (k < (BLOCKDIM - 1)) ? upper[row(k + 1)] : static_cast<T>(0);
            T bk_1 = (k < (BLOCKDIM - 1)) ? main[row(k + 1)] : static_cast<T>(0);
            T ak_1 = (k < (BLOCKDIM - 1)) ? lower[row(k + 1)] : static_cast<T>(0);
            T ak_2 = (k < (BLOCKDIM - 2)) ? lower[row(k + 2)] : static_cast<T>(0);

            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            if(use_1x1_pivot || k == (BLOCKDIM - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                const T wk = w[row(k)];
                const T vk = v[row(k)];

                w[row(k)]  = wk * inv_bk;
                v[row(k)]  = vk * inv_bk;
                mt[row(k)] = ck * inv_bk;

                pivot.set_pivoting_to_1x1(k);

                if(k < (BLOCKDIM - 1))
                {
                    w[row(k + 1)] += -ak_1 * wk * inv_bk;
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

                const T wk   = w[row(k)];
                const T wk_1 = w[row(k + 1)];
                const T vk   = v[row(k)];
                const T vk_1 = v[row(k + 1)];

                w[row(k)]  = (bk_1 * wk - ck * wk_1) * det;
                v[row(k)]  = (bk_1 * vk - ck * vk_1) * det;
                mt[row(k)] = -ck * ck_1 * det;

                pivot.set_pivoting_to2x2(k);

                if(k < (BLOCKDIM - 1))
                {
                    w[row(k + 1)]  = (-ak_1 * wk + bk * wk_1) * det;
                    v[row(k + 1)]  = (-ak_1 * vk + bk * vk_1) * det;
                    mt[row(k + 1)] = bk * ck_1 * det;

                    pivot.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                if(k < (BLOCKDIM - 2))
                {
                    w[row(k + 2)] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
                }

                if(k < (BLOCKDIM - 2))
                {
                    bk_2 = main[row(k + 2)];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        assert(k == BLOCKDIM);
        k--;

        k -= pivot.get_pivoting(k);

        while(k >= 0)
        {
            if(pivot.get_pivoting(k) == 1)
            {
                const T tmp = mt[row(k)];

                w[row(k)] += -tmp * w[row(k + 1)];
                v[row(k)] += -tmp * v[row(k + 1)];

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[row(k)];
                const T tmp2 = mt[row(k - 1)];

                w[row(k)] += -tmp1 * w[row(k + 1)];
                w[row(k - 1)] += -tmp2 * w[row(k + 1)];
                v[row(k)] += -tmp1 * v[row(k + 1)];
                v[row(k - 1)] += -tmp2 * v[row(k + 1)];

                k -= 2;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void LBMT_solve_rhs_device2(int m_pad,
                                                     int n,
                                                     const T* __restrict__ lower,
                                                     const T* __restrict__ main,
                                                     const T* __restrict__ upper,
                                                     const T* __restrict__ mt,
                                                     T* __restrict__ rhs)
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

        const auto row = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(gid, nblocks);

        T bk = main[row(0)];

        pivot_mask<BLOCKDIM> pivot{};

        int k = 0;
        while(k < BLOCKDIM)
        {
            T ck   = upper[row(k)];
            T ck_1 = (k < (BLOCKDIM - 1)) ? upper[row(k + 1)] : static_cast<T>(0);
            T bk_1 = (k < (BLOCKDIM - 1)) ? main[row(k + 1)] : static_cast<T>(0);
            T ak_1 = (k < (BLOCKDIM - 1)) ? lower[row(k + 1)] : static_cast<T>(0);
            T ak_2 = (k < (BLOCKDIM - 2)) ? lower[row(k + 2)] : static_cast<T>(0);

            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            if(use_1x1_pivot || k == (BLOCKDIM - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                pivot.set_pivoting_to_1x1(k);

                const T rhsk = rhs[row(k)] * inv_bk;

                rhs[row(k)] = rhsk;

                if(k < (BLOCKDIM - 1))
                {
                    rhs[row(k + 1)] += -(ak_1 * rhsk);

                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                pivot.set_pivoting_to2x2(k);

                if(k < (BLOCKDIM - 1))
                {
                    pivot.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                const T rhsk   = rhs[row(k)] * det;
                const T rhsk_1 = rhs[row(k + 1)] * det;

                rhs[row(k)]     = (bk_1 * rhsk - ck * rhsk_1);
                rhs[row(k + 1)] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (BLOCKDIM - 2))
                {
                    rhs[row(k + 2)] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                    bk_2 = main[row(k + 2)];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        assert(k == BLOCKDIM);
        k--;

        k -= pivot.get_pivoting(k);

        while(k >= 0)
        {
            if(pivot.get_pivoting(k) == 1)
            {
                const T tmp = mt[row(k)];

                rhs[row(k)] += -tmp * rhs[row(k + 1)];

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[row(k)];
                const T tmp2 = mt[row(k - 1)];

                rhs[row(k)] += -tmp1 * rhs[row(k + 1)];
                rhs[row(k - 1)] += -tmp2 * rhs[row(k + 1)];

                k -= 2;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void LBMT_solve_rhs_kernel2(int m_pad,
                                int n,
                                const T* __restrict__ lower,
                                const T* __restrict__ main,
                                const T* __restrict__ upper,
                                const T* __restrict__ mt,
                                T* __restrict__ rhs)
    {
        for(int64_t batch = hipBlockIdx_y; batch < n; batch += hipGridDim_y)
        {
            rocsparse::LBMT_solve_rhs_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                m_pad,
                n,
                lower,
                main,
                upper,
                mt,
                load_pointer(rhs, batch, static_cast<int64_t>(m_pad)));
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void fill_s_matrix_device2(int m_pad,
                                                    int n,
                                                    const T* __restrict__ w,
                                                    const T* __restrict__ v,
                                                    const T* __restrict__ rhs,
                                                    T* __restrict__ S_lower,
                                                    T* __restrict__ S_main,
                                                    T* __restrict__ S_upper,
                                                    T* __restrict__ S_rhs,
                                                    bool write_s_matrix)
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int gid = tid + BLOCKSIZE * bid;

        const int nblocks = m_pad / BLOCKDIM;
        const int s_size  = 2 * nblocks;

        if(write_s_matrix && gid < s_size)
        {
            const auto part = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(gid / 2, nblocks);

            S_upper[gid] = (gid % 2 == 0) ? v[part(0)] : static_cast<T>(1);
            S_lower[gid] = (gid % 2 == 0) ? static_cast<T>(1) : w[part(BLOCKDIM - 1)];
        }

        if(write_s_matrix && gid >= 1 && gid < s_size - 1)
        {
            const auto part = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(gid / 2, nblocks);

            S_main[gid] = (gid % 2 == 0) ? w[part(0)] : v[part(BLOCKDIM - 1)];
        }

        if(gid < s_size / 2)
        {
            const auto part = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(gid, nblocks);

            S_rhs[2 * gid]     = rhs[part(0)];
            S_rhs[2 * gid + 1] = rhs[part(BLOCKDIM - 1)];
        }

        if(write_s_matrix && gid == 0)
        {
            S_lower[0] = static_cast<T>(0);
            S_main[0]  = static_cast<T>(1);
        }
        if(write_s_matrix && gid == 1)
        {
            S_lower[1] = static_cast<T>(0);
        }
        if(write_s_matrix && gid == s_size - 2)
        {
            S_upper[s_size - 2] = static_cast<T>(0);
        }
        if(write_s_matrix && gid == s_size - 1)
        {
            S_upper[s_size - 1] = static_cast<T>(0);
            S_main[s_size - 1]  = static_cast<T>(1);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void fill_s_matrix_kernel2(int m_pad,
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
            rocsparse::fill_s_matrix_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                m_pad,
                n,
                w,
                v,
                load_pointer(rhs, batch, static_cast<int64_t>(m_pad)),
                S_lower,
                S_main,
                S_upper,
                load_pointer(S_rhs, batch, static_cast<int64_t>(s_size)),
                batch == 0);
        }
    }

    template <typename T>
    ROCSPARSE_KERNEL(1)
    void S_solve_kernel2(int m,
                         int n,
                         const T* __restrict__ S_lower,
                         const T* __restrict__ S_main,
                         const T* __restrict__ S_upper,
                         T* __restrict__ rhs)
    {
        // fill_s writes the reduced system contiguously, so this leaf does not use
        // the grouped padded layout.
        constexpr int max_rows = 32;
        if(m < 2 || m > max_rows)
        {
            return;
        }

        const int batch = blockIdx.x;

        T mt[max_rows];

        pivot_mask<max_rows> pivot{};

        int k  = 0;
        T   bk = S_main[k];

        while(k < m)
        {
            T ck   = S_upper[k];
            T ck_1 = (k < (m - 1)) ? S_upper[k + 1] : static_cast<T>(0);
            T bk_1 = (k < (m - 1)) ? S_main[k + 1] : static_cast<T>(0);
            T ak_1 = (k < (m - 1)) ? S_lower[k + 1] : static_cast<T>(0);
            T ak_2 = (k < (m - 2)) ? S_lower[k + 2] : static_cast<T>(0);

            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            if(use_1x1_pivot || k == (m - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                mt[k] = ck * inv_bk;

                pivot.set_pivoting_to_1x1(k);

                T rhsk = rhs[k + m * batch] * inv_bk;

                rhs[k + m * batch] = rhsk;

                if(k < (m - 1))
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

                if(k < (m - 1))
                {
                    mt[k + 1] = bk * ck_1 * det;

                    pivot.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                T rhsk   = rhs[k + m * batch] * det;
                T rhsk_1 = rhs[k + 1 + m * batch] * det;

                rhs[k + m * batch]     = (bk_1 * rhsk - ck * rhsk_1);
                rhs[k + 1 + m * batch] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (m - 2))
                {
                    rhs[k + 2 + m * batch] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                    bk_2 = S_main[k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        assert(k == m);
        k--;

        k -= pivot.get_pivoting(k);

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

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void scatter_S_B_to_B_pad_device2(
        int s_size, int m_pad, int n, const T* __restrict__ S_B, T* __restrict__ B_pad)
    {
        const int i = blockIdx.x * BLOCKSIZE + threadIdx.x;

        if(i >= s_size / 2)
            return;

        const int  nblocks = m_pad / BLOCKDIM;
        const auto part    = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(i, nblocks);

        const T val_even = (i == 0) ? S_B[0] : S_B[2 * i - 1];
        const T val_odd  = (2 * i + 1 < s_size - 1) ? S_B[2 * i + 2] : S_B[s_size - 1];

        B_pad[part(0)]             = val_even;
        B_pad[part(BLOCKDIM - 1)]  = val_odd;
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scatter_S_B_to_B_pad_kernel2(
        int s_size, int m_pad, int n, const T* __restrict__ S_B, T* __restrict__ B_pad)
    {
        for(int64_t batch = hipBlockIdx_y; batch < n; batch += hipGridDim_y)
        {
            rocsparse::scatter_S_B_to_B_pad_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                s_size,
                m_pad,
                n,
                load_pointer(S_B, batch, static_cast<int64_t>(s_size)),
                load_pointer(B_pad, batch, static_cast<int64_t>(m_pad)));
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void backward_solve_device2(
        int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ B_pad)
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int gid = tid + BLOCKSIZE * bid;

        const int nblocks = m_pad / BLOCKDIM;

        if(gid >= m_pad)
        {
            return;
        }

        const int lid = gid % nblocks;
        const int wid = gid / nblocks;

        const auto part = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(lid, nblocks);

        T x1 = static_cast<T>(0);
        if(lid >= 1)
        {
            const auto previous
                = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(lid - 1, nblocks);
            x1 = B_pad[previous(BLOCKDIM - 1)];
        }

        T x2 = static_cast<T>(0);
        if(lid < nblocks - 1)
        {
            const auto next = gtsv_partition<BLOCKDIM, PARTITIONS_PER_GROUP>(lid + 1, nblocks);
            x2 = B_pad[next(0)];
        }

        if(wid >= 1 && wid < BLOCKDIM - 1)
        {
            const int at = part(wid);

            B_pad[at] = B_pad[at] - w[at] * x1 - v[at] * x2;
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void backward_solve_kernel2(
        int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ B_pad)
    {
        for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
        {
            backward_solve_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                m_pad, n, w, v, load_pointer(B_pad, batch, m_pad));
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void data_transpose_device2(
        int m, int m_pad, const T* __restrict__ d_in, T* __restrict__ d_out, T pad_value)
    {
        // Each CUDA block handles PARTITIONS_PER_GROUP (e.g., 256) total partitions,
        // but processes them in smaller chunks dictated by BLOCKSIZE (e.g., 128).
        const int group_start_partition = blockIdx.x * PARTITIONS_PER_GROUP;
        const int nblocks               = m_pad / BLOCKDIM;

        // Prevent completely out-of-bounds blocks from executing
        if(group_start_partition >= nblocks)
            return;

        // Shared memory now only needs to hold BLOCKSIZE partitions at a time (~33 KB)
        __shared__ T smem[BLOCKSIZE][BLOCKDIM + 1];

        const int tid               = threadIdx.x;
        const int global_offset_out = group_start_partition * BLOCKDIM;
        const int remaining         = nblocks - group_start_partition;
        const int group_width       = remaining < static_cast<int>(PARTITIONS_PER_GROUP)
                                          ? remaining
                                          : static_cast<int>(PARTITIONS_PER_GROUP);

        // Loop over the group in chunks of BLOCKSIZE (e.g., 0, then 128).
        // A full group keeps the PARTITIONS_PER_GROUP stride. The final partial
        // group is packed into group_width partitions so every index stays in m_pad.
        for(int chunk = 0; chunk < group_width; chunk += BLOCKSIZE)
        {
            // ---------------------------------------------------------
            // PHASE 1: Coalesced Read from Global -> Shared Memory
            // ---------------------------------------------------------
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                int linear_idx    = i * BLOCKSIZE + tid;
                int partition_idx = linear_idx / BLOCKDIM;
                int element_idx   = linear_idx % BLOCKDIM;

                // Offset the global read by the current chunk
                int global_partition = group_start_partition + chunk + partition_idx;
                int global_in_idx    = global_partition * BLOCKDIM + element_idx;

                T val = pad_value;

                if(global_partition < nblocks && global_in_idx < m)
                {
                    val = d_in[global_in_idx];
                }

                smem[partition_idx][element_idx] = val;
            }

            __syncthreads();

            // ---------------------------------------------------------
            // PHASE 2: Coalesced Write from Shared -> Global Memory
            // ---------------------------------------------------------
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                int out_partition_idx = tid;
                int out_element_idx   = i;
                int local_partition   = chunk + out_partition_idx;

                int linear_out
                    = global_offset_out + out_element_idx * group_width + local_partition;

                if(local_partition < group_width)
                {
                    d_out[linear_out] = smem[out_partition_idx][out_element_idx];
                }
            }

            // Synchronize before overwriting shared memory with the next chunk
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_DEVICE_ILF void data_untranspose_device2(int m,
                                                       int m_pad,
                                                       const T* __restrict__ d_in,
                                                       T* __restrict__ d_out)
    {
        // Inverse of data_transpose_device2. Padded layout within a group is
        // group_start * BLOCKDIM + element * group_width + local partition.
        const int group_start_partition = blockIdx.x * PARTITIONS_PER_GROUP;
        const int nblocks               = m_pad / BLOCKDIM;

        if(group_start_partition >= nblocks)
            return;

        __shared__ T smem[BLOCKSIZE][BLOCKDIM + 1];

        const int tid         = threadIdx.x;
        const int remaining   = nblocks - group_start_partition;
        const int group_width = remaining < static_cast<int>(PARTITIONS_PER_GROUP)
                                    ? remaining
                                    : static_cast<int>(PARTITIONS_PER_GROUP);

        for(int chunk = 0; chunk < group_width; chunk += BLOCKSIZE)
        {
            // Coalesced read of the padded layout. Thread tid owns one partition.
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                int in_partition_idx = tid;
                int in_element_idx   = i;
                int local_partition  = chunk + in_partition_idx;

                int linear_in = group_start_partition * BLOCKDIM + in_element_idx * group_width
                                + local_partition;

                T val = T(0);
                if(local_partition < group_width)
                {
                    val = d_in[linear_in];
                }

                smem[in_partition_idx][in_element_idx] = val;
            }

            __syncthreads();

            // Coalesced write back to standard layout. Padding past m is dropped.
            for(int i = 0; i < BLOCKDIM; ++i)
            {
                int linear_out    = i * BLOCKSIZE + tid;
                int partition_idx = linear_out / BLOCKDIM;
                int element_idx   = linear_out % BLOCKDIM;

                int global_partition = group_start_partition + chunk + partition_idx;
                int global_out_idx   = global_partition * BLOCKDIM + element_idx;

                if(global_partition < nblocks && global_out_idx < m)
                {
                    d_out[global_out_idx] = smem[partition_idx][element_idx];
                }
            }

            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void data_marshaling_kernel2(int m,
                                 int m_pad,
                                 const T* __restrict__ lower,
                                 const T* __restrict__ main,
                                 const T* __restrict__ upper,
                                 T* __restrict__ lower_pad,
                                 T* __restrict__ main_pad,
                                 T* __restrict__ upper_pad)
    {
        data_transpose_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
            m, m_pad, lower, lower_pad, static_cast<T>(0));
        __syncthreads();
        data_transpose_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
            m, m_pad, main, main_pad, static_cast<T>(1));
        __syncthreads();
        data_transpose_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
            m, m_pad, upper, upper_pad, static_cast<T>(0));
        __syncthreads();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void data_marshaling_B_kernel2(
        int m, int m_pad, int n, int ldb, const T* __restrict__ B, T* __restrict__ B_pad)
    {
        for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
        {
            data_transpose_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                m,
                m_pad,
                load_pointer(B, batch, ldb),
                load_pointer(B_pad, batch, m_pad),
                static_cast<T>(0));
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void reverse_data_marshaling_B_kernel2(
        int m, int m_pad, int n, int ldb, const T* __restrict__ B_pad, T* __restrict__ B)
    {
        for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
        {
            data_untranspose_device2<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
                m, m_pad, load_pointer(B_pad, batch, m_pad), load_pointer(B, batch, ldb));
            __syncthreads();
        }
    }
}
