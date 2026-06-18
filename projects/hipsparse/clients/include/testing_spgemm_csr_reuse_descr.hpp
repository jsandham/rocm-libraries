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
#ifndef TESTING_SPGEMM_CSR_REUSE_DESCR_HPP
#define TESTING_SPGEMM_CSR_REUSE_DESCR_HPP

#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

#include <algorithm>

using namespace hipsparse_test;

template <typename I, typename J, typename T>
void testing_spgemm_csr_reuse_descr_bad_arg(const Arguments& argus)
{
}

// Exercises the per-call buffer-allocation pattern for the standard SpGEMM
// pipeline (workEstimation → compute → allocateC → copy). A fresh
// hipsparseSpGEMMDescr_t and fresh external buffers (for both workEstimation
// and compute) are created and destroyed on every call. The shared
// hipsparseSpMatDescr_t for A and B are reused across all calls. By running
// this repeatedly, we verify that the internal caches on those shared
// sparse-matrix descriptors remain valid despite the SpGEMM descriptor and
// external buffers being re-created on every call.
template <typename I, typename J, typename T>
static void call_spgemm(hipsparseHandle_t&     handle,
                        hipsparseSpMatDescr_t& matA,
                        hipsparseSpMatDescr_t& matB,
                        J                      m,
                        J                      n,
                        J                      k,
                        I                      nnz_A,
                        I                      nnz_B,
                        std::vector<I>&        hcsr_row_ptr_A,
                        std::vector<J>&        hcsr_col_ind_A,
                        std::vector<T>&        hcsr_val_A,
                        std::vector<I>&        hcsr_row_ptr_B,
                        std::vector<J>&        hcsr_col_ind_B,
                        std::vector<T>&        hcsr_val_B,
                        T                      alpha,
                        T                      beta,
                        hipsparseIndexBase_t   idxBaseA,
                        hipsparseIndexBase_t   idxBaseB,
                        hipsparseIndexBase_t   idxBaseC,
                        hipsparseSpGEMMAlg_t   alg)
{
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // Allocate row pointer array for C
    auto dcsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    I* dcsr_row_ptr_C = (I*)dcsr_row_ptr_C_managed.get();

    // Create fresh C sparse matrix descriptor (nnz=0; col_ind and val allocated after compute)
    hipsparseSpMatDescr_t matC;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(
        &matC, m, n, 0, dcsr_row_ptr_C, nullptr, nullptr, typeI, typeJ, idxBaseC, typeT));

    // Create fresh SpGEMM descriptor
    std::unique_ptr<spgemm_struct> unique_ptr_descr(new spgemm_struct);
    hipsparseSpGEMMDescr_t         descr = unique_ptr_descr->descr;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    // Query workEstimation buffer size
    size_t bufferSize1;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_workEstimation(handle,
                                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &alpha,
                                                         matA,
                                                         matB,
                                                         &beta,
                                                         matC,
                                                         typeT,
                                                         alg,
                                                         descr,
                                                         &bufferSize1,
                                                         nullptr));

    void* externalBuffer1;
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer1, bufferSize1));

    // Run workEstimation
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_workEstimation(handle,
                                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &alpha,
                                                         matA,
                                                         matB,
                                                         &beta,
                                                         matC,
                                                         typeT,
                                                         alg,
                                                         descr,
                                                         &bufferSize1,
                                                         externalBuffer1));

    // Query compute buffer size
    size_t bufferSize2;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_compute(handle,
                                                  HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                  HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha,
                                                  matA,
                                                  matB,
                                                  &beta,
                                                  matC,
                                                  typeT,
                                                  alg,
                                                  descr,
                                                  &bufferSize2,
                                                  nullptr));

    void* externalBuffer2;
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer2, bufferSize2));

    // Run compute
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_compute(handle,
                                                  HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                  HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha,
                                                  matA,
                                                  matB,
                                                  &beta,
                                                  matC,
                                                  typeT,
                                                  alg,
                                                  descr,
                                                  &bufferSize2,
                                                  externalBuffer2));

    // Get nnz of C
    int64_t rows_C, cols_C, nnz_C;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMatGetSize(matC, &rows_C, &cols_C, &nnz_C));

    // Allocate C col_ind and val
    auto dcsr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_C), device_free};
    auto dcsr_val_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};

    J* dcsr_col_ind_C = (J*)dcsr_col_ind_C_managed.get();
    T* dcsr_val_C     = (T*)dcsr_val_C_managed.get();

    CHECK_HIP_ERROR(hipMemset(dcsr_val_C, 0, sizeof(T) * nnz_C));

    // Set C pointers
    CHECK_HIPSPARSE_ERROR(
        hipsparseCsrSetPointers(matC, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C));

    // Copy results into C
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_copy(handle,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               typeT,
                                               alg,
                                               descr));

    // Copy output from device to host
    std::vector<I> hcsr_row_ptr_C(m + 1);
    std::vector<J> hcsr_col_ind_C(nnz_C);
    std::vector<T> hcsr_val_C(nnz_C);

    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(I) * (m + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(J) * nnz_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

    // Compute gold reference on host
    std::vector<I> hcsr_row_ptr_C_gold(m + 1);
    int64_t        nnz_C_gold = host_csrgemm2_nnz(m,
                                            n,
                                            k,
                                            &alpha,
                                            hcsr_row_ptr_A.data(),
                                            hcsr_col_ind_A.data(),
                                            hcsr_row_ptr_B.data(),
                                            hcsr_col_ind_B.data(),
                                            (const T*)nullptr,
                                            (const I*)nullptr,
                                            (const J*)nullptr,
                                            hcsr_row_ptr_C_gold.data(),
                                            idxBaseA,
                                            idxBaseB,
                                            idxBaseC,
                                            HIPSPARSE_INDEX_BASE_ZERO);

    std::vector<J> hcsr_col_ind_C_gold(nnz_C_gold);
    std::vector<T> hcsr_val_C_gold(nnz_C_gold);
    host_csrgemm2(m,
                  n,
                  k,
                  &alpha,
                  hcsr_row_ptr_A.data(),
                  hcsr_col_ind_A.data(),
                  hcsr_val_A.data(),
                  hcsr_row_ptr_B.data(),
                  hcsr_col_ind_B.data(),
                  hcsr_val_B.data(),
                  (const T*)nullptr,
                  (const I*)nullptr,
                  (const J*)nullptr,
                  (const T*)nullptr,
                  hcsr_row_ptr_C_gold.data(),
                  hcsr_col_ind_C_gold.data(),
                  hcsr_val_C_gold.data(),
                  idxBaseA,
                  idxBaseB,
                  idxBaseC,
                  HIPSPARSE_INDEX_BASE_ZERO);

    unit_check_general(1, 1, 1, &nnz_C_gold, &nnz_C);
    unit_check_general(1, m + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
    unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C.data());
    unit_check_general(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C.data());

    CHECK_HIP_ERROR(hipFree(externalBuffer1));
    CHECK_HIP_ERROR(hipFree(externalBuffer2));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matC));
}

// Exercises the shared-buffer path: the workEstimation and compute buffer
// sizes are queried once up front and the resulting max-sized buffers are
// reused across every pass. A fresh hipsparseSpGEMMDescr_t and a fresh C
// sparse-matrix descriptor are still created per pass, while matA and matB
// are shared throughout.
template <typename I, typename J, typename T>
static void call_spgemm_shared_buffer(hipsparseHandle_t&   handle,
                                      hipsparseSpMatDescr_t& matA,
                                      hipsparseSpMatDescr_t& matB,
                                      J                      m,
                                      J                      n,
                                      J                      k,
                                      I                      nnz_A,
                                      I                      nnz_B,
                                      std::vector<I>&        hcsr_row_ptr_A,
                                      std::vector<J>&        hcsr_col_ind_A,
                                      std::vector<T>&        hcsr_val_A,
                                      std::vector<I>&        hcsr_row_ptr_B,
                                      std::vector<J>&        hcsr_col_ind_B,
                                      std::vector<T>&        hcsr_val_B,
                                      T                      alpha,
                                      T                      beta,
                                      hipsparseIndexBase_t   idxBaseA,
                                      hipsparseIndexBase_t   idxBaseB,
                                      hipsparseIndexBase_t   idxBaseC,
                                      hipsparseSpGEMMAlg_t   alg,
                                      int                    number_of_passes)
{
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    // Step 1: query buffer sizes once using a temporary C descriptor and a
    // temporary SpGEMM descriptor. The sizes are used to allocate shared
    // buffers that are reused across all passes below.
    auto dcsr_row_ptr_C_tmp_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    I* dcsr_row_ptr_C_tmp = (I*)dcsr_row_ptr_C_tmp_managed.get();

    hipsparseSpMatDescr_t matC_tmp;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matC_tmp,
                                             m,
                                             n,
                                             0,
                                             dcsr_row_ptr_C_tmp,
                                             nullptr,
                                             nullptr,
                                             typeI,
                                             typeJ,
                                             idxBaseC,
                                             typeT));

    std::unique_ptr<spgemm_struct> unique_ptr_descr_tmp(new spgemm_struct);
    hipsparseSpGEMMDescr_t         descr_tmp = unique_ptr_descr_tmp->descr;

    size_t bufferSize1;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_workEstimation(handle,
                                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &alpha,
                                                         matA,
                                                         matB,
                                                         &beta,
                                                         matC_tmp,
                                                         typeT,
                                                         alg,
                                                         descr_tmp,
                                                         &bufferSize1,
                                                         nullptr));

    void* sharedBuffer1;
    CHECK_HIP_ERROR(hipMalloc(&sharedBuffer1, bufferSize1));

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_workEstimation(handle,
                                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &alpha,
                                                         matA,
                                                         matB,
                                                         &beta,
                                                         matC_tmp,
                                                         typeT,
                                                         alg,
                                                         descr_tmp,
                                                         &bufferSize1,
                                                         sharedBuffer1));

    size_t bufferSize2;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_compute(handle,
                                                  HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                  HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha,
                                                  matA,
                                                  matB,
                                                  &beta,
                                                  matC_tmp,
                                                  typeT,
                                                  alg,
                                                  descr_tmp,
                                                  &bufferSize2,
                                                  nullptr));

    void* sharedBuffer2;
    CHECK_HIP_ERROR(hipMalloc(&sharedBuffer2, bufferSize2));

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matC_tmp));

    // Step 2: run the full pipeline for each pass, reusing sharedBuffer1 and
    // sharedBuffer2 but creating fresh SpGEMM and C descriptors each time.
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        auto dcsr_row_ptr_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
        I* dcsr_row_ptr_C = (I*)dcsr_row_ptr_C_managed.get();

        hipsparseSpMatDescr_t matC;
        CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(
            &matC, m, n, 0, dcsr_row_ptr_C, nullptr, nullptr, typeI, typeJ, idxBaseC, typeT));

        std::unique_ptr<spgemm_struct> unique_ptr_descr(new spgemm_struct);
        hipsparseSpGEMMDescr_t         descr = unique_ptr_descr->descr;

        // workEstimation with shared buffer
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpGEMM_workEstimation(handle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           matA,
                                           matB,
                                           &beta,
                                           matC,
                                           typeT,
                                           alg,
                                           descr,
                                           &bufferSize1,
                                           sharedBuffer1));

        // compute with shared buffer
        CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_compute(handle,
                                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha,
                                                      matA,
                                                      matB,
                                                      &beta,
                                                      matC,
                                                      typeT,
                                                      alg,
                                                      descr,
                                                      &bufferSize2,
                                                      sharedBuffer2));

        // Get nnz of C
        int64_t rows_C, cols_C, nnz_C;
        CHECK_HIPSPARSE_ERROR(hipsparseSpMatGetSize(matC, &rows_C, &cols_C, &nnz_C));

        // Allocate C col_ind and val
        auto dcsr_col_ind_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_C), device_free};
        auto dcsr_val_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};

        J* dcsr_col_ind_C = (J*)dcsr_col_ind_C_managed.get();
        T* dcsr_val_C     = (T*)dcsr_val_C_managed.get();

        CHECK_HIP_ERROR(hipMemset(dcsr_val_C, 0, sizeof(T) * nnz_C));

        // Set C pointers
        CHECK_HIPSPARSE_ERROR(
            hipsparseCsrSetPointers(matC, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C));

        // Copy results into C
        CHECK_HIPSPARSE_ERROR(hipsparseSpGEMM_copy(handle,
                                                   HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                   HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   typeT,
                                                   alg,
                                                   descr));

        // Copy output from device to host
        std::vector<I> hcsr_row_ptr_C(m + 1);
        std::vector<J> hcsr_col_ind_C(nnz_C);
        std::vector<T> hcsr_val_C(nnz_C);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(I) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(J) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        // Compute gold reference on host
        std::vector<I> hcsr_row_ptr_C_gold(m + 1);
        int64_t        nnz_C_gold = host_csrgemm2_nnz(m,
                                                n,
                                                k,
                                                &alpha,
                                                hcsr_row_ptr_A.data(),
                                                hcsr_col_ind_A.data(),
                                                hcsr_row_ptr_B.data(),
                                                hcsr_col_ind_B.data(),
                                                (const T*)nullptr,
                                                (const I*)nullptr,
                                                (const J*)nullptr,
                                                hcsr_row_ptr_C_gold.data(),
                                                idxBaseA,
                                                idxBaseB,
                                                idxBaseC,
                                                HIPSPARSE_INDEX_BASE_ZERO);

        std::vector<J> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T> hcsr_val_C_gold(nnz_C_gold);
        host_csrgemm2(m,
                      n,
                      k,
                      &alpha,
                      hcsr_row_ptr_A.data(),
                      hcsr_col_ind_A.data(),
                      hcsr_val_A.data(),
                      hcsr_row_ptr_B.data(),
                      hcsr_col_ind_B.data(),
                      hcsr_val_B.data(),
                      (const T*)nullptr,
                      (const I*)nullptr,
                      (const J*)nullptr,
                      (const T*)nullptr,
                      hcsr_row_ptr_C_gold.data(),
                      hcsr_col_ind_C_gold.data(),
                      hcsr_val_C_gold.data(),
                      idxBaseA,
                      idxBaseB,
                      idxBaseC,
                      HIPSPARSE_INDEX_BASE_ZERO);

        unit_check_general(1, 1, 1, &nnz_C_gold, &nnz_C);
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C.data());

        CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matC));
    }

    CHECK_HIP_ERROR(hipFree(sharedBuffer1));
    CHECK_HIP_ERROR(hipFree(sharedBuffer2));
}

template <typename I, typename J, typename T>
void testing_spgemm_csr_reuse_descr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    J                    m        = argus.M;
    J                    k        = argus.K;
    T                    h_alpha  = argus.get_alpha<T>();
    hipsparseIndexBase_t idxBaseA = argus.baseA;
    hipsparseIndexBase_t idxBaseB = argus.baseB;
    hipsparseIndexBase_t idxBaseC = argus.baseC;
    hipsparseSpGEMMAlg_t alg      = argus.spgemm_alg;
    std::string          filename = argus.filename;

    T h_beta = make_DataType<T>(0);

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<I> hcsr_row_ptr_A;
    std::vector<J> hcsr_col_ind_A;
    std::vector<T> hcsr_val_A;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz_A;
    CHECK_GENERATE_MATRIX_ERROR(generate_csr_matrix(
        filename, m, k, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idxBaseA));

    // Redefine sparse matrix values
    hipsparseInit<T>(hcsr_val_A, hcsr_val_A.size(), 1);

    // Sparse matrix B as the transpose of A: B is k×n (n=m), so C = A*B is m×m
    J n     = m;
    I nnz_B = nnz_A;

    std::vector<I> hcsr_row_ptr_B(k + 1);
    std::vector<J> hcsr_col_ind_B(nnz_B);
    std::vector<T> hcsr_val_B(nnz_B);

    transpose_csr(m,
                  k,
                  nnz_A,
                  hcsr_row_ptr_A.data(),
                  hcsr_col_ind_A.data(),
                  hcsr_val_A.data(),
                  hcsr_row_ptr_B.data(),
                  hcsr_col_ind_B.data(),
                  hcsr_val_B.data(),
                  idxBaseA,
                  idxBaseB);

    // Allocate device memory for A and B — shared across all calls below
    auto dcsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_A), device_free};
    auto dcsr_val_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dcsr_row_ptr_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (k + 1)), device_free};
    auto dcsr_col_ind_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_B), device_free};
    auto dcsr_val_B_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};

    I* dcsr_row_ptr_A = (I*)dcsr_row_ptr_A_managed.get();
    J* dcsr_col_ind_A = (J*)dcsr_col_ind_A_managed.get();
    T* dcsr_val_A     = (T*)dcsr_val_A_managed.get();
    I* dcsr_row_ptr_B = (I*)dcsr_row_ptr_B_managed.get();
    J* dcsr_col_ind_B = (J*)dcsr_col_ind_B_managed.get();
    T* dcsr_val_B     = (T*)dcsr_val_B_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind_A, hcsr_col_ind_A.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B.data(), sizeof(I) * (k + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind_B, hcsr_col_ind_B.data(), sizeof(J) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_B, hcsr_val_B.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));

    // Create shared sparse matrix descriptors for A and B — reused across all
    // scenarios and passes.
    hipsparseSpMatDescr_t matA, matB;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matA,
                                             m,
                                             k,
                                             nnz_A,
                                             dcsr_row_ptr_A,
                                             dcsr_col_ind_A,
                                             dcsr_val_A,
                                             typeI,
                                             typeJ,
                                             idxBaseA,
                                             typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matB,
                                             k,
                                             n,
                                             nnz_B,
                                             dcsr_row_ptr_B,
                                             dcsr_col_ind_B,
                                             dcsr_val_B,
                                             typeI,
                                             typeJ,
                                             idxBaseB,
                                             typeT));

    constexpr int number_of_passes = 3;

    // Scenario 1: per-call buffer allocation. Each call creates a fresh
    // hipsparseSpGEMMDescr_t and freshly allocates (and frees) the external
    // buffers for workEstimation and compute. The shared matA and matB
    // descriptors are reused across all calls.
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        call_spgemm<I, J, T>(handle,
                             matA,
                             matB,
                             m,
                             n,
                             k,
                             nnz_A,
                             nnz_B,
                             hcsr_row_ptr_A,
                             hcsr_col_ind_A,
                             hcsr_val_A,
                             hcsr_row_ptr_B,
                             hcsr_col_ind_B,
                             hcsr_val_B,
                             h_alpha,
                             h_beta,
                             idxBaseA,
                             idxBaseB,
                             idxBaseC,
                             alg);
    }

    // Scenario 2: shared-buffer allocation. The workEstimation and compute
    // buffer sizes are queried once and the buffers are shared across all
    // passes. Each pass still creates a fresh SpGEMM descriptor and C matrix.
    call_spgemm_shared_buffer<I, J, T>(handle,
                                       matA,
                                       matB,
                                       m,
                                       n,
                                       k,
                                       nnz_A,
                                       nnz_B,
                                       hcsr_row_ptr_A,
                                       hcsr_col_ind_A,
                                       hcsr_val_A,
                                       hcsr_row_ptr_B,
                                       hcsr_col_ind_B,
                                       hcsr_val_B,
                                       h_alpha,
                                       h_beta,
                                       idxBaseA,
                                       idxBaseB,
                                       idxBaseC,
                                       alg,
                                       number_of_passes);

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matB));
#endif
}

#endif // TESTING_SPGEMM_CSR_REUSE_DESCR_HPP
