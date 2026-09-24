/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/conversion/rocsparse_csrsort.h"
#include "rocsparse_utility.hpp"

#include "csrsort_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsort.hpp"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    // Number of bits needed to represent the column indices, which are at most n.
    static uint32_t csrsort_endbit(int64_t n)
    {
        // __builtin_clzll is undefined for n == 0
        return (n == 0) ? 0 : 64 - __builtin_clzll(static_cast<unsigned long long>(n));
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrsort_buffer_size_template(rocsparse_handle handle,
                                                         int64_t          m,
                                                         int64_t          n,
                                                         int64_t          nnz,
                                                         const void*      csr_row_ptr,
                                                         const void*      csr_col_ind,
                                                         size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0 || nnz == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    const uint32_t startbit = 0;
    const uint32_t endbit   = rocsparse::csrsort_endbit(n);

    // We do not know if sort_pairs or sort_keys will be called, so use the largest buffer between the two
    size_t size1;
    size_t size2;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, I, I>(
            handle, nnz, m, startbit, endbit, &size1)));
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
        handle, nnz, m, startbit, endbit, &size2)));

    *buffer_size = rocsparse::align_size<char>(rocsparse::max(size1, size2));

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays

    // columns buffer
    *buffer_size += rocsparse::align_size<J>(nnz);
    // perm buffer
    *buffer_size += rocsparse::align_size<I>(nnz);
    // segm buffer
    *buffer_size += rocsparse::align_size<I>(m + 1);

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csrsort_template(rocsparse_handle     handle,
                                             int64_t              m,
                                             int64_t              n,
                                             int64_t              nnz,
                                             rocsparse_index_base idx_base,
                                             const void*          csr_row_ptr,
                                             void*                csr_col_ind,
                                             void*                perm,
                                             void*                temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    const I* csr_row_ptr_ = reinterpret_cast<const I*>(csr_row_ptr);
    J*       csr_col_ind_ = reinterpret_cast<J*>(csr_col_ind);
    I*       perm_        = reinterpret_cast<I*>(perm);

    // Stream
    hipStream_t stream = handle->stream;

    const uint32_t startbit = 0;
    const uint32_t endbit   = rocsparse::csrsort_endbit(n);
    size_t         size;

    if(perm_ != nullptr)
    {
        // Sort pairs, if permutation vector is present
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, I, I>(
                handle, nnz, m, startbit, endbit, &size)));
    }
    else
    {
        // Sort keys, if no permutation vector is present
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
                handle, nnz, m, startbit, endbit, &size)));
    }

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // columns buffer
    J* tmp_cols = reinterpret_cast<J*>(ptr);
    ptr += rocsparse::align_size<J>(nnz);

    // perm buffer
    I* tmp_perm = reinterpret_cast<I*>(ptr);
    ptr += rocsparse::align_size<I>(nnz);

    // segm buffer
    I* tmp_segm = reinterpret_cast<I*>(ptr);
    ptr += rocsparse::align_size<I>(m + 1);

    // Index base one requires shift of offset positions
    if(idx_base == rocsparse_index_base_one)
    {
#define CSRSORT_DIM 512
        dim3 csrsort_blocks((m + 1 - 1) / CSRSORT_DIM + 1);
        dim3 csrsort_threads(CSRSORT_DIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrsort_shift_kernel<CSRSORT_DIM>),
                                           csrsort_blocks,
                                           csrsort_threads,
                                           0,
                                           stream,
                                           m + 1,
                                           csr_row_ptr_,
                                           tmp_segm);
#undef CSRSORT_DIM
    }

    // rocprim buffer
    void* tmp_rocprim = reinterpret_cast<void*>(ptr);

    // Switch between offsets
    const I* offsets = (idx_base == rocsparse_index_base_one) ? tmp_segm : csr_row_ptr_;

    // Sort by columns and obtain permutation vector

    if(perm_ != nullptr)
    {
        // Sort by pairs, if permutation vector is present
        rocsparse::primitives::double_buffer<J> keys(csr_col_ind_, tmp_cols);
        rocsparse::primitives::double_buffer<I> vals(perm_, tmp_perm);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_pairs(
            handle, keys, vals, nnz, m, offsets, offsets + 1, startbit, endbit, size, tmp_rocprim));

        if(keys.current() != csr_col_ind_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(csr_col_ind_,
                                                         keys.current(),
                                                         sizeof(J) * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         stream));
        }
        if(vals.current() != perm_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                perm_, vals.current(), sizeof(I) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }
    else
    {
        // Sort by keys, if no permutation vector is present
        rocsparse::primitives::double_buffer<J> keys(csr_col_ind_, tmp_cols);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_keys(
            handle, keys, nnz, m, offsets, offsets + 1, startbit, endbit, size, tmp_rocprim));

        if(keys.current() != csr_col_ind_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(csr_col_ind_,
                                                         keys.current(),
                                                         sizeof(J) * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         stream));
        }
    }
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_csrsort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csrsort_buffer_size",
                         m,
                         n,
                         nnz,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsort_buffer_size_template<rocsparse_int, rocsparse_int>(
        handle, m, n, nnz, csr_row_ptr, csr_col_ind, buffer_size)));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csrsort(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const rocsparse_int*      csr_row_ptr,
                                              rocsparse_int*            csr_col_ind,
                                              rocsparse_int*            perm,
                                              void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csrsort",
                         m,
                         n,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, temp_buffer);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsort_template<rocsparse_int, rocsparse_int>(
        handle, m, n, nnz, descr->base, csr_row_ptr, csr_col_ind, perm, temp_buffer)));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#define INSTANTIATE(I, J)                                                                          \
    template rocsparse_status rocsparse::csrsort_buffer_size_template<I, J>(                       \
        rocsparse_handle handle,                                                                   \
        int64_t          m,                                                                        \
        int64_t          n,                                                                        \
        int64_t          nnz,                                                                      \
        const void*      csr_row_ptr,                                                              \
        const void*      csr_col_ind,                                                              \
        size_t*          buffer_size);                                                             \
    template rocsparse_status rocsparse::csrsort_template<I, J>(rocsparse_handle     handle,       \
                                                                int64_t              m,            \
                                                                int64_t              n,            \
                                                                int64_t              nnz,          \
                                                                rocsparse_index_base idx_base,     \
                                                                const void*          csr_row_ptr,  \
                                                                void*                csr_col_ind,  \
                                                                void*                perm,         \
                                                                void*                temp_buffer)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE
