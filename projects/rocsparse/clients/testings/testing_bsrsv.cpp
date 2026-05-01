/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

#include "rocsparse_enum.hpp"

// BSR indexing macros
#define BSR_IND(j, bi, bj) BSR_IND_R(j, bi, bj)
#define BSR_IND_R(j, bi, bj) (block_dim * block_dim * (j) + (bi) * block_dim + (bj))

template <bool SLEEP>
__device__ __forceinline__ int32_t spin_loop(int32_t* __restrict__ done, int scope)
{
    // int32_t  local_done    = __hip_atomic_load(done, __ATOMIC_RELAXED, scope);
    int32_t  local_done    = __hip_atomic_load(done, __ATOMIC_ACQUIRE, scope);
    uint32_t times_through = 0;
    while(!local_done)
    {
        if(SLEEP)
        {
            for(uint32_t i = 0; i < times_through; ++i)
            {
                __builtin_amdgcn_s_sleep(1);
            }

            if(times_through < 3907)
            {
                ++times_through;
            }
        }
        // local_done = __hip_atomic_load(done, __ATOMIC_RELAXED, scope);
        local_done = __hip_atomic_load(done, __ATOMIC_ACQUIRE, scope);
    }
    return local_done;
}

template <typename T>
__device__ __forceinline__ T atomic_min(T* ptr, T val)
{
    return atomicMin(ptr, val);
}

template <typename T>
__device__ __forceinline__ T fma(T p, T q, T r);

template <>
__device__ __forceinline__ float fma(float p, float q, float r)
{
    return std::fma(p, q, r);
}

template <>
__device__ __forceinline__ double fma(double p, double q, double r)
{
    return std::fma(p, q, r);
}

template <>
__device__ __forceinline__ rocsparse_float_complex fma(rocsparse_float_complex p,
                                                    rocsparse_float_complex q,
                                                    rocsparse_float_complex r)
{
    return std::fma(p, q, r);
}

template <>
__device__ __forceinline__ rocsparse_double_complex fma(rocsparse_double_complex p,
                                                        rocsparse_double_complex q,
                                                        rocsparse_double_complex r)
{
    return std::fma(p, q, r);
}

template <typename T>
__device__ __forceinline__ T fma(T p, T q, T r);

template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
static __device__ __forceinline__
void bsrsv_lower_general_device(rocsparse_int mb,
                                T             alpha,
                                const rocsparse_int* __restrict__ bsr_row_ptr,
                                const rocsparse_int* __restrict__ bsr_col_ind,
                                const T* __restrict__ bsr_val,
                                rocsparse_int block_dim,
                                const T* __restrict__ x,
                                T* __restrict__ y,
                                int* __restrict__ done_array,
                                rocsparse_int* __restrict__ map,
                                rocsparse_int* __restrict__ zero_pivot)
{
    const int lid = hipThreadIdx_x & (WFSIZE - 1);
    const int wid = hipThreadIdx_x / WFSIZE;

    // Index into the row map
    const rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= mb)
    {
        return;
    }

    // Get the BSR row this wavefront will operate on
    const rocsparse_int row = map[idx];

    // Current row entry and exit point
    const rocsparse_int row_begin = bsr_row_ptr[row];
    const rocsparse_int row_end   = bsr_row_ptr[row + 1];

    // Initialize local_col with mb
    rocsparse_int local_col = mb;

    // Initialize y with alpha and x
    for(rocsparse_int bi = lid; bi < block_dim; bi += WFSIZE)
    {
        y[row * block_dim + bi] = alpha * x[row * block_dim + bi];
    }

    // Loop over the current row
    rocsparse_int j;
    for(j = row_begin; j < row_end; ++j)
    {
        // Current column index
        local_col = bsr_col_ind[j];

        // Processing lower triangular

        // Ignore all diagonal entries and above
        if(local_col >= row)
        {
            break;
        }

        // Spin loop until dependency has been resolved
        spin_loop<SLEEP>(&done_array[local_col], __HIP_MEMORY_SCOPE_AGENT);

        // Wait for y to be visible globally
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

        // Local sum computation
        for(rocsparse_int bi = lid; bi < block_dim; bi += WFSIZE)
        {
            // Local sum accumulator
            T local_sum = static_cast<T>(0);

            for(rocsparse_int bj = 0; bj < block_dim; ++bj)
            {
                local_sum = fma(
                    bsr_val[BSR_IND(j, bi, bj)], y[local_col * block_dim + bj], local_sum);
            }

            // Write local sum to y
            y[row * block_dim + bi] -= local_sum;
        }
    }

    bool pivot = false;

    // Process diagonal
    if(local_col == row)
    {
        for(rocsparse_int bi = 0; bi < block_dim; ++bi)
        {
            // Load diagonal matrix entry
            const T diag = bsr_val[block_dim * block_dim * j + bi + bi * block_dim];

            // Load result of bi-th BSR row
            T val = y[row * block_dim + bi];
            // Check for numerical pivot
            if(diag == static_cast<T>(0))
            {
                pivot = true;
            }
            else
            {
                // Divide result of bi-th BSR row by diagonal entry
                y[row * block_dim + bi] = val /= diag;
            }

            // Update remaining non-diagonal entries
            for(rocsparse_int bj = bi + lid + 1; bj < block_dim; bj += WFSIZE)
            {
                y[row * block_dim + bj] -= val * bsr_val[BSR_IND(j, bj, bi)];
            }
        }
    }

    // Write "row is done" flag
    if(lid == 0)
    {
        __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

        if(pivot == true)
        {
            atomic_min(zero_pivot, row);
        }
    }
}

template <typename T>
union const_host_device_scalar
{
    T                        value;
    const T*                 pointer;
    __forceinline__ __host__ const_host_device_scalar(const T* scalar)
        : pointer(scalar){};
    __forceinline__ __host__ const_host_device_scalar(const T& scalar)
        : value(scalar){};
};

template <typename T>
__forceinline__ __host__ const_host_device_scalar<T>
                        to_const_host_device_scalar(rocsparse_pointer_mode mode, const T* s_)
{
    return (mode == rocsparse_pointer_mode_host) ? const_host_device_scalar<T>(*s_)
                                                    : const_host_device_scalar<T>(s_);
}

template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
__launch_bounds__(BLOCKSIZE) static __global__
void bsrsv_lower_general(rocsparse_int mb,
            const_host_device_scalar<T> alpha_union,
                        const rocsparse_int* __restrict__ bsr_row_ptr,
                        const rocsparse_int* __restrict__ bsr_col_ind,
                        const T* __restrict__ bsr_val,
                        rocsparse_int block_dim,
                        const T* __restrict__ x,
                        T* __restrict__ y,
                        int* __restrict__ done_array,
                        rocsparse_int* __restrict__ map,
                        rocsparse_int* __restrict__ zero_pivot,
                        bool                 is_host_mode)
{
    const auto alpha = (is_host_mode) ? alpha_union.value : *alpha_union.pointer;
    bsrsv_lower_general_device<BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                        alpha,
                                                        bsr_row_ptr,
                                                        bsr_col_ind,
                                                        bsr_val,
                                                        block_dim,
                                                        x,
                                                        y,
                                                        done_array,
                                                        map,
                                                        zero_pivot);
}



template <typename T>
void testing_bsrsv_bad_arg(const Arguments& arg)
{
}

template <typename T>
void testing_bsrsv(const Arguments& arg)
{
    int                       mb        = 5;
    int                       nb        = 5;
    int                       block_dim = 2;
    int                       M         = 10;
    int                       N         = 10;
    int                       nnzb      = 16;

    T h_alpha = static_cast<T>(1);

    std::vector<int> hbsr_row_ptr = {0, 3, 6, 11, 13, 16}; // size mb+1
    std::vector<int> hbsr_col_ind = {0, 3, 4, 0, 1, 4, 0, 1, 2, 3, 4, 1, 3, 0, 1, 4}; // size nnzb
    std::vector<T>   hbsr_val(64, 1); // size block_dim*block_dim*nnzb
    std::vector<T>   hx(N, 0);
    std::vector<T>   hy(M, 1);

    int* dbsr_row_ptr = nullptr;
    int* dbsr_col_ind = nullptr;
    T*   dbsr_val     = nullptr;
    T*   dx           = nullptr;
    T*   dy           = nullptr;
    CHECK_HIP_ERROR(hipMalloc((void**)&dbsr_row_ptr, sizeof(int) * (mb + 1)));
    CHECK_HIP_ERROR(hipMalloc((void**)&dbsr_col_ind, sizeof(int) * 16));
    CHECK_HIP_ERROR(hipMalloc((void**)&dbsr_val, sizeof(T) * 64));
    CHECK_HIP_ERROR(hipMalloc((void**)&dx, sizeof(T) * N));
    CHECK_HIP_ERROR(hipMalloc((void**)&dy, sizeof(T) * M));

    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * 16, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val, hbsr_val.data(), sizeof(T) * 64, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * M, hipMemcpyHostToDevice));

    std::cout << "hbsr_row_ptr" << std::endl;
    for(size_t i = 0; i < hbsr_row_ptr.size(); i++)
    {
        std::cout << hbsr_row_ptr[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "hbsr_col_ind" << std::endl;
    for(size_t i = 0; i < hbsr_col_ind.size(); i++)
    {
        std::cout << hbsr_col_ind[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "hbsr_val" << std::endl;
    for(size_t i = 0; i < hbsr_val.size(); i++)
    {
        std::cout << hbsr_val[i] << " ";
    }
    std::cout << "" << std::endl;

    std::vector<int> hdone_array(mb, 0);
    std::vector<int> hrow_map = {0, 1, 2, 3, 4};

    int* dzero_pivot = nullptr;
    int* ddone_array = nullptr;
    int* drow_map    = nullptr;
    CHECK_HIP_ERROR(hipMalloc((void**)&dzero_pivot, sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc((void**)&ddone_array, sizeof(int) * mb));
    CHECK_HIP_ERROR(hipMalloc((void**)&drow_map, sizeof(int) * mb));

    CHECK_HIP_ERROR(
        hipMemcpy(ddone_array, hdone_array.data(), sizeof(int) * mb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(drow_map, hrow_map.data(), sizeof(int) * mb, hipMemcpyHostToDevice));

    // Obtain required buffer size
    void*  dbuffer;
    size_t buffer_size = 1280;
    std::cout << "buffer_size: " << buffer_size << std::endl;
    CHECK_HIP_ERROR(hipMalloc((void**)&dbuffer, buffer_size));

    hipLaunchKernelGGL((bsrsv_lower_general<128, 32, false, T>),
                    dim3((32 * mb - 1) / 128 + 1),
                    dim3(128),
                    0,
                    0,
                    mb,
                    to_const_host_device_scalar(rocsparse_pointer_mode_host, &h_alpha),
                    dbsr_row_ptr,
                    dbsr_col_ind,
                    dbsr_val,
                    block_dim,
                    dx,
                    dy,
                    ddone_array,
                    drow_map,
                    dzero_pivot,
                    true);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    std::cout << "IIII" << std::endl;

    // Free buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
    CHECK_HIP_ERROR(hipFree(dzero_pivot));
    CHECK_HIP_ERROR(hipFree(ddone_array));
    CHECK_HIP_ERROR(hipFree(drow_map));

    CHECK_HIP_ERROR(hipFree(dbsr_row_ptr));
    CHECK_HIP_ERROR(hipFree(dbsr_col_ind));
    CHECK_HIP_ERROR(hipFree(dbsr_val));
    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_bsrsv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrsv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsrsv_extra(const Arguments& arg) {}