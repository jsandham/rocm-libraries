/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_SPSORT_H
#define ROCSPARSE_SPSORT_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_spsort_buffer_size returns the size of the required buffer to execute the given stage of the SpSort operation.
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  descr        SpSort descriptor.
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  stage        SpSort stage for the SpSort computation.
*  @param[out]  
*  buffer_size  number of bytes of the temporary storage buffer. \p buffer_size is
*               determined by calling \ref rocsparse_spsort_buffer_size.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if an error descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p mat_A, \p descr, or \p buffer_size pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spsort_buffer_size(rocsparse_handle       handle,
                                              rocsparse_spsort_descr descr,
                                              rocsparse_spmat_descr  mat,
                                              rocsparse_spsort_stage stage,
                                              size_t*                buffer_size,
                                              rocsparse_error*       error);

/*! \ingroup generic_module
*  \brief Sparse matrix sorting.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  descr        SpSort descriptor.
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  stage        SpSort stage for the SpSort computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. \p buffer_size is
*               determined by calling \ref rocsparse_spsort_buffer_size.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if an error descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p mat_A, \p descr, or \p buffer_size pointer is invalid.
*
*  \par Example
*  \snippet example_rocsparse_spsort.cpp doc example
*  \par Example
*  \snippet example_rocsparse_spsort_coo.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spsort(rocsparse_handle       handle,
                                  rocsparse_spsort_descr descr,
                                  rocsparse_spmat_descr  mat_A,
                                  rocsparse_spsort_stage stage,
                                  size_t                 buffer_size,
                                  void*                  temp_buffer,
                                  rocsparse_error*       error);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPGEAM_H */