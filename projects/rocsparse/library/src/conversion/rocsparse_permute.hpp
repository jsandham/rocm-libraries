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

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    template <typename T>
    rocsparse_status permute_buffer_size_template(rocsparse_handle handle,
                                                  int64_t          nnz,
                                                  const void*      values,
                                                  size_t*          buffer_size);

    template <typename T, typename J>
    rocsparse_status permute_template(
        rocsparse_handle handle, int64_t nnz, void* values, const void* perm, void* temp_buffer);

    rocsparse_status permute_buffer_size(rocsparse_handle   handle,
                                         int64_t            nnz,
                                         rocsparse_datatype values_datatype,
                                         const void*        values,
                                         size_t*            buffer_size);

    rocsparse_status permute(rocsparse_handle    handle,
                             int64_t             nnz,
                             rocsparse_datatype  values_datatype,
                             void*               values,
                             rocsparse_indextype perm_indextype,
                             void*               perm,
                             void*               temp_buffer);
}