/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_gemvi.h"
#include "rocsparse_common.h"
#include "rocsparse_gemvi.hpp"

#include "gemvi_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemvi_kernel(I m,
                      I n,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                      const T* __restrict__ A,
                      int64_t lda,
                      I       nnz,
                      const T* __restrict__ x_val,
                      const I* __restrict__ x_ind,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                      T* __restrict__ y,
                      rocsparse_index_base idx_base,
                      bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
        {
            rocsparse::gemvi_device<BLOCKSIZE, WFSIZE>(
                m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t UNROLL, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemvi_kernel2(I m,
                       I n,
                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                       const T* __restrict__ A,
                       int64_t lda,
                       I       nnz,
                       const T* __restrict__ x_val,
                       const I* __restrict__ x_ind,
                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                       T* __restrict__ y,
                       rocsparse_index_base idx_base,
                       bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
        {
            rocsparse::gemvi_device2<BLOCKSIZE, WFSIZE, UNROLL>(
                m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t UNROLL, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemvi_kernel_part1(I m,
                            I n,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const T* __restrict__ A,
                            int64_t lda,
                            I       nnz,
                            const T* __restrict__ x_val,
                            const I* __restrict__ x_ind,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                            T* __restrict__ y,
                            T* __restrict__ workspace,
                            rocsparse_index_base idx_base,
                            bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
        {
            rocsparse::gemvi_device_part1<BLOCKSIZE, WFSIZE, UNROLL>(
                m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, workspace, idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemvi_kernel_part2(I   m,
                            I   n,
                            int grid_y,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                            const T* __restrict__ workspace,
                            T*   y,
                            bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
        {
            rocsparse::gemvi_device_part2<BLOCKSIZE, WFSIZE>(
                m, n, grid_y, alpha, beta, workspace, y);
        }
    }

    constexpr uint32_t gemvi_part1_blocksize  = 256;
    constexpr uint32_t gemvi_part1_wfsize     = 32;
    constexpr uint32_t gemvi_part1_unroll     = 8;
    constexpr int      gemvi_part1_max_grid_y = 256;
    constexpr int      gemvi_part1_oversub    = 2;

    inline int gemvi_part1_grid_x(rocsparse_int m)
    {
        return static_cast<int>((m - 1) / static_cast<rocsparse_int>(gemvi_part1_wfsize) + 1);
    }

    // Split-k y-grid. Grow only until part1 has enough row-blocks to fill the
    // device (occupancy * OVERSUB), and never beyond one nnz per wavefront
    // worker, which is the most split-k that still does useful A loads.
    inline int gemvi_part1_grid_y(const hipDeviceProp_t& prop, rocsparse_int m, rocsparse_int nnz)
    {
        constexpr int nwf = static_cast<int>(gemvi_part1_blocksize / gemvi_part1_wfsize);

        const int ny_work = rocsparse::min((nnz - 1) / nwf + 1, gemvi_part1_max_grid_y);

        int per_cu = 1;
        if(prop.maxThreadsPerMultiProcessor > 0)
        {
            per_cu = rocsparse::max(
                prop.maxThreadsPerMultiProcessor / static_cast<int>(gemvi_part1_blocksize), 1);
        }

        const int sat    = gemvi_part1_oversub * prop.multiProcessorCount * per_cu;
        const int grid_x = gemvi_part1_grid_x(m);
        const int ny_occ = (grid_x >= sat) ? 1 : ((sat + grid_x - 1) / grid_x);

        return rocsparse::max(1, rocsparse::min(ny_work, ny_occ));
    }

#define LAUNCH_GEMVI_WAVE32(DIM_)                                     \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
        (rocsparse::gemvi_kernel<DIM_, 32>),                          \
        gemvi_blocks,                                                 \
        dim3(DIM_),                                                   \
        0,                                                            \
        handle->stream,                                               \
        m,                                                            \
        n,                                                            \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        A,                                                            \
        lda,                                                          \
        nnz,                                                          \
        x_val,                                                        \
        x_ind,                                                        \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),  \
        y,                                                            \
        idx_base,                                                     \
        handle->pointer_mode == rocsparse_pointer_mode_host)

    template <typename I, typename T>
    rocsparse_status gemvi_dispatch(rocsparse_handle     handle,
                                    rocsparse_operation  trans,
                                    I                    m,
                                    I                    n,
                                    const T*             alpha_device_host,
                                    const T*             A,
                                    int64_t              lda,
                                    I                    nnz,
                                    const T*             x_val,
                                    const I*             x_ind,
                                    const T*             beta_device_host,
                                    T*                   y,
                                    rocsparse_index_base idx_base,
                                    void*                temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // If nnz is zero, only compute beta * y
        if(nnz == 0)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, m, beta_device_host, y));

            return rocsparse_status_success;
        }

        T* workspace = reinterpret_cast<T*>(temp_buffer);

        if(trans == rocsparse_operation_none)
        {
            const int grid_x = gemvi_part1_grid_x(m);
            const int grid_y = gemvi_part1_grid_y(handle->properties, m, nnz);

            dim3 grid(grid_x, grid_y, 1);
            dim3 blocks(gemvi_part1_blocksize, 1, 1);

            // std::cout << "grid_x: " << grid_x << " grid_y: " << grid_y << std::endl;

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (gemvi_kernel_part1<gemvi_part1_blocksize, gemvi_part1_wfsize, gemvi_part1_unroll>),
                grid,
                blocks,
                0,
                handle->stream,
                m,
                n,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                A,
                lda,
                nnz,
                x_val,
                x_ind,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                y,
                workspace,
                idx_base,
                handle->pointer_mode == rocsparse_pointer_mode_host);

            if(grid_y > 1)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (gemvi_kernel_part2<gemvi_part1_blocksize, gemvi_part1_wfsize>),
                    dim3(grid_x),
                    dim3(gemvi_part1_blocksize),
                    0,
                    handle->stream,
                    m,
                    n,
                    grid_y,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    workspace,
                    y,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        // #define GEMVI_DIM 1024
        //         if(trans == rocsparse_operation_none)
        //         {
        //             if(handle->wavefront_size == 32)
        //             {
        //                 dim3 gemvi_blocks((m - 1) / 32 + 1);

        //                 // RDNA4 (gfx1201, wave32) launch tuning.
        //                 //
        //                 // Each block processes WFSIZE(=32) output rows and spreads the
        //                 // sparse-vector dot product across BLOCKSIZE/32 wavefronts,
        //                 // reducing the partial sums through LDS. The baseline always
        //                 // used a 1024-thread block (32 wavefronts). gemvi is memory
        //                 // bound, so when there are already enough row-blocks to saturate
        //                 // the GPU, a 1024-thread block is oversized: it caps occupancy
        //                 // (fewer concurrent blocks per CU) and deepens the LDS reduction.
        //                 //
        //                 // In that regime we shrink the block to raise occupancy and
        //                 // shorten the reduction. We keep the original 1024-thread block
        //                 // whenever the grid is small (few row-blocks), so those shapes
        //                 // launch byte-for-byte identically to the baseline and cannot
        //                 // regress. nnz gates how many wavefronts are actually useful for
        //                 // the reduction (no point spreading a sparse vector shorter than
        //                 // a wavefront over 32 wavefronts).
        //                 //
        //                 // GEMVI_SATURATION_NBLOCKS is the empirically tuned large-grid
        //                 // crossover on gfx1201; below it we reproduce the baseline
        //                 // launch exactly.
        //                 constexpr int64_t GEMVI_SATURATION_NBLOCKS = 1024;
        //                 const int64_t     gemvi_nblocks            = (static_cast<int64_t>(m) - 1) / 32 + 1;
        //                 uint32_t          gemvi_dim                = GEMVI_DIM;
        //                 if(gemvi_nblocks >= GEMVI_SATURATION_NBLOCKS)
        //                 {
        //                     gemvi_dim = (nnz <= static_cast<I>(handle->wavefront_size)) ? 256 : 512;
        //                 }

        //                 switch(gemvi_dim)
        //                 {
        //                 case 256:
        //                     LAUNCH_GEMVI_WAVE32(256);
        //                     break;
        //                 case 512:
        //                     LAUNCH_GEMVI_WAVE32(512);
        //                     break;
        //                 default:
        //                     LAUNCH_GEMVI_WAVE32(1024);
        //                     break;
        //                 }
        //             }
        //             else
        //             {
        //                 rocsparse_host_assert(handle->wavefront_size == 64,
        //                                       "Wrong wavefront size dispatch.");

        //                 dim3 gemvi_blocks((m - 1) / 64 + 1);
        //                 dim3 gemvi_threads(GEMVI_DIM);

        //                 RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        //                     (rocsparse::gemvi_kernel<GEMVI_DIM, 64>),
        //                     gemvi_blocks,
        //                     gemvi_threads,
        //                     0,
        //                     handle->stream,
        //                     m,
        //                     n,
        //                     ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
        //                     A,
        //                     lda,
        //                     nnz,
        //                     x_val,
        //                     x_ind,
        //                     ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
        //                     y,
        //                     idx_base,
        //                     handle->pointer_mode == rocsparse_pointer_mode_host);
        //             }
        // #undef GEMVI_DIM
        //         }
        //         else
        //         {
        //             RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        //         }

        return rocsparse_status_success;
    }

    template <typename I, typename T>
    rocsparse_status gemvi_template(rocsparse_handle     handle, //0
                                    rocsparse_operation  trans, //1
                                    I                    m, //2
                                    I                    n, //3
                                    const T*             alpha_device_host, //4
                                    const T*             A, //5
                                    int64_t              lda, //6
                                    I                    nnz, //7
                                    const T*             x_val, //8
                                    const I*             x_ind, //9
                                    const T*             beta_device_host, //10
                                    T*                   y, //11
                                    rocsparse_index_base idx_base, //12
                                    void*                temp_buffer) //13
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Check for valid handle
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xgemvi"),
                             trans,
                             m,
                             n,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                             (const void*&)A,
                             lda,
                             nnz,
                             (const void*&)x_val,
                             (const void*&)x_ind,
                             LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                             (const void*&)y,
                             idx_base,
                             (const void*&)temp_buffer);

        // Check operation mode
        ROCSPARSE_CHECKARG_ENUM(1, trans);

        // Check index base
        ROCSPARSE_CHECKARG_ENUM(12, idx_base);

        // Check sizes
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, n);
        ROCSPARSE_CHECKARG_SIZE(7, nnz);

        // nnz of sparse vector cannot exceed its size
        ROCSPARSE_CHECKARG(7, nnz, (nnz > n), rocsparse_status_invalid_size);

        // Check leading dimension
        ROCSPARSE_CHECKARG(6,
                           lda,
                           ((lda < m) && (trans == rocsparse_operation_none)),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(6,
                           lda,
                           ((lda < n) && (trans != rocsparse_operation_none)),
                           rocsparse_status_invalid_size);

        // Quick return if possible
        if(m == 0)
        {
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(4, alpha_device_host);

        // Check invalid pointers
        if(m > 0 && n > 0 && nnz > 0)
        {
            ROCSPARSE_CHECKARG_POINTER(5, A);
            ROCSPARSE_CHECKARG_POINTER(8, x_val);
            ROCSPARSE_CHECKARG_POINTER(9, x_ind);
            // Allow temp_buffer to be nullptr
        }

        ROCSPARSE_CHECKARG_POINTER(10, beta_device_host);
        ROCSPARSE_CHECKARG_POINTER(11, y);

        // Quick return if there is no work to do - alpha can be (valid) nullptr!
        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            if(alpha_device_host == nullptr && *beta_device_host == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }

            if(alpha_device_host != nullptr)
            {
                if(*alpha_device_host == static_cast<T>(0)
                   && *beta_device_host == static_cast<T>(1))
                {
                    return rocsparse_status_success;
                }
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemvi_dispatch(handle,
                                                            trans,
                                                            m,
                                                            n,
                                                            alpha_device_host,
                                                            A,
                                                            lda,
                                                            nnz,
                                                            x_val,
                                                            x_ind,
                                                            beta_device_host,
                                                            y,
                                                            idx_base,
                                                            temp_buffer));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

// Definition of the C-implementation

// rocsparse_xgemvi_buffer_size
#define CAPI_IMPL(name_, type_)                                                            \
    rocsparse_status name_(rocsparse_handle    handle,                                     \
                           rocsparse_operation trans,                                      \
                           rocsparse_int       m,                                          \
                           rocsparse_int       n,                                          \
                           rocsparse_int       nnz,                                        \
                           size_t*             buffer_size)                                \
    try                                                                                    \
    {                                                                                      \
        ROCSPARSE_ROUTINE_TRACE;                                                           \
        const int grid_y = (handle != nullptr)                                             \
                               ? rocsparse::gemvi_part1_grid_y(handle->properties, m, nnz) \
                               : rocsparse::gemvi_part1_max_grid_y;                        \
        *buffer_size     = sizeof(type_) * rocsparse::gemvi_part1_wfsize * grid_y          \
                       * rocsparse::gemvi_part1_grid_x(m);                                 \
        return rocsparse_status_success;                                                   \
    }                                                                                      \
    catch(...)                                                                             \
    {                                                                                      \
        RETURN_ROCSPARSE_EXCEPTION();                                                      \
    }

// C-implementations
CAPI_IMPL(rocsparse_sgemvi_buffer_size, float);
CAPI_IMPL(rocsparse_dgemvi_buffer_size, double);
CAPI_IMPL(rocsparse_cgemvi_buffer_size, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zgemvi_buffer_size, rocsparse_double_complex);

// Undefine the CAPI_IMPL macro
#undef CAPI_IMPL

// rocsparse_xgemvi
#define CAPI_IMPL(name_, type_)                                                \
    rocsparse_status name_(rocsparse_handle     handle,                        \
                           rocsparse_operation  trans,                         \
                           rocsparse_int        m,                             \
                           rocsparse_int        n,                             \
                           const type_*         alpha,                         \
                           const type_*         A,                             \
                           rocsparse_int        lda,                           \
                           rocsparse_int        nnz,                           \
                           const type_*         x_val,                         \
                           const rocsparse_int* x_ind,                         \
                           const type_*         beta,                          \
                           type_*               y,                             \
                           rocsparse_index_base idx_base,                      \
                           void*                temp_buffer)                   \
    {                                                                          \
        try                                                                    \
        {                                                                      \
            ROCSPARSE_ROUTINE_TRACE;                                           \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemvi_template(handle,        \
                                                                trans,         \
                                                                m,             \
                                                                n,             \
                                                                alpha,         \
                                                                A,             \
                                                                lda,           \
                                                                nnz,           \
                                                                x_val,         \
                                                                x_ind,         \
                                                                beta,          \
                                                                y,             \
                                                                idx_base,      \
                                                                temp_buffer)); \
            return rocsparse_status_success;                                   \
        }                                                                      \
        catch(...)                                                             \
        {                                                                      \
            RETURN_ROCSPARSE_EXCEPTION();                                      \
        }                                                                      \
    }

// C-implementations
CAPI_IMPL(rocsparse_sgemvi, float);
CAPI_IMPL(rocsparse_dgemvi, double);
CAPI_IMPL(rocsparse_cgemvi, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zgemvi, rocsparse_double_complex);

// Undefine the CAPI_IMPL macro
#undef CAPI_IMPL
}
