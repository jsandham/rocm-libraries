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
#ifndef TESTING_SPGEMMREUSE_CSR_REUSE_DESCR_HPP
#define TESTING_SPGEMMREUSE_CSR_REUSE_DESCR_HPP

#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

#include <algorithm>
#include <vector>

using namespace hipsparse_test;

template <typename I, typename J, typename T>
void testing_spgemmreuse_csr_reuse_descr_bad_arg(const Arguments& argus)
{
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)

// A single SpGEMM product C = alpha * X * Y where X (m x k) and Y (k x n) are
// sparse matrix descriptors that are created once by the caller and reused by
// every call below, together with the host reference solution for that product
// (computed for the scalar alpha passed to spgemmreuse_init_problem).
template <typename I, typename J, typename T>
struct spgemmreuse_problem
{
    hipsparseSpMatDescr_t X{};
    hipsparseSpMatDescr_t Y{};

    J m{};
    J n{};
    J k{};

    hipsparseIndexBase_t idx_base_C{};

    int64_t        gold_nnz{};
    std::vector<I> gold_row_ptr;
    std::vector<J> gold_col_ind;
    std::vector<T> gold_val;
};

// Computes the host reference solution of C = alpha * X * Y.
template <typename I, typename J, typename T>
static void spgemmreuse_init_problem(spgemmreuse_problem<I, J, T>& problem,
                                     hipsparseSpMatDescr_t         X,
                                     hipsparseSpMatDescr_t         Y,
                                     J                             m,
                                     J                             n,
                                     J                             k,
                                     T                             alpha,
                                     const std::vector<I>&         hcsr_row_ptr_X,
                                     const std::vector<J>&         hcsr_col_ind_X,
                                     const std::vector<T>&         hcsr_val_X,
                                     const std::vector<I>&         hcsr_row_ptr_Y,
                                     const std::vector<J>&         hcsr_col_ind_Y,
                                     const std::vector<T>&         hcsr_val_Y,
                                     hipsparseIndexBase_t          idx_base_X,
                                     hipsparseIndexBase_t          idx_base_Y,
                                     hipsparseIndexBase_t          idx_base_C)
{
    problem.X          = X;
    problem.Y          = Y;
    problem.m          = m;
    problem.n          = n;
    problem.k          = k;
    problem.idx_base_C = idx_base_C;

    problem.gold_row_ptr.resize(m + 1);

    problem.gold_nnz = host_csrgemm2_nnz(m,
                                         n,
                                         k,
                                         &alpha,
                                         hcsr_row_ptr_X.data(),
                                         hcsr_col_ind_X.data(),
                                         hcsr_row_ptr_Y.data(),
                                         hcsr_col_ind_Y.data(),
                                         (const T*)nullptr,
                                         (const I*)nullptr,
                                         (const J*)nullptr,
                                         problem.gold_row_ptr.data(),
                                         idx_base_X,
                                         idx_base_Y,
                                         idx_base_C,
                                         HIPSPARSE_INDEX_BASE_ZERO);

    problem.gold_col_ind.resize(problem.gold_nnz);
    problem.gold_val.resize(problem.gold_nnz);

    host_csrgemm2(m,
                  n,
                  k,
                  &alpha,
                  hcsr_row_ptr_X.data(),
                  hcsr_col_ind_X.data(),
                  hcsr_val_X.data(),
                  hcsr_row_ptr_Y.data(),
                  hcsr_col_ind_Y.data(),
                  hcsr_val_Y.data(),
                  (const T*)nullptr,
                  (const I*)nullptr,
                  (const J*)nullptr,
                  (const T*)nullptr,
                  problem.gold_row_ptr.data(),
                  problem.gold_col_ind.data(),
                  problem.gold_val.data(),
                  idx_base_X,
                  idx_base_Y,
                  idx_base_C,
                  HIPSPARSE_INDEX_BASE_ZERO);
}

// Describes the five external buffers used by the SpGEMM reuse routines:
//   0: work estimation buffer
//   1, 2, 3: nnz buffers
//   4: copy buffer
//
// When no shared buffers are provided every call allocates (and frees) its own
// buffers. In both cases `required` is updated to the maximum size requested
// so far, which is how the sizes for the shared buffers are determined.
struct spgemmreuse_workspace
{
    static constexpr int num_buffers = 5;

    size_t required[num_buffers] = {};
    size_t capacity[num_buffers] = {};
    void*  shared[num_buffers]   = {};

    // Returns the buffer to be used for slot `i` given that `size` bytes are needed.
    void* get(int i, size_t size, hipsparse_unique_ptr& own)
    {
        required[i] = std::max(required[i], size);

        if(shared[i] != nullptr && size <= capacity[i])
        {
            return shared[i];
        }

        // Either no shared buffer was provided or it is too small. The latter
        // is a test failure but keep going with a private buffer so we never
        // write out of bounds.
        EXPECT_TRUE(shared[i] == nullptr) << "shared buffer " << i << " is too small: required "
                                          << size << " but only " << capacity[i] << " available";

        // Ensure the externalBuffer pointer is never null (a null buffer is
        // interpreted as a buffer size query) even when no scratch space is needed.
        own = hipsparse_unique_ptr{device_malloc(std::max(size, size_t(4))), device_free};
        return own.get();
    }
};

static std::vector<hipsparseSpGEMMAlg_t> spgemmreuse_algs()
{
#if(!defined(CUDART_VERSION))
    return {HIPSPARSE_SPGEMM_DEFAULT,
            HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC,
            HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC};
#else
    return {HIPSPARSE_SPGEMM_DEFAULT};
#endif
}

static std::vector<hipsparsePointerMode_t> spgemmreuse_pointer_modes()
{
#if(!defined(CUDART_VERSION))
    return {HIPSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_DEVICE};
#else
    return {HIPSPARSE_POINTER_MODE_HOST};
#endif
}

// Runs the complete work estimation -> nnz -> copy flow for the product described by
// `problem` using the already existing X and Y sparse matrix descriptors, and then
// calls hipsparseSpGEMMreuse_compute once for every entry of `alpha_scales`, using the
// scalar `alpha * alpha_scale`, as is done when the values of the matrices change but
// their sparsity patterns do not. A new C matrix descriptor and a new SpGEMM descriptor
// are created for every call, so the only state carried over between calls is whatever
// hipSPARSE keeps in X and Y. The results are validated against the host reference solution.
template <typename I, typename J, typename T>
static void call_spgemmreuse(hipsparseHandle_t             handle,
                             spgemmreuse_problem<I, J, T>& problem,
                             T                             alpha,
                             const std::vector<double>&    alpha_scales,
                             hipsparseSpGEMMAlg_t          alg,
                             hipsparsePointerMode_t        pointer_mode,
                             spgemmreuse_workspace&        workspace)
{
    const hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    const hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    const hipsparseIndexType_t typeI = getIndexType<I>();
    const hipsparseIndexType_t typeJ = getIndexType<J>();
    const hipDataType          typeT = getDataType<T>();

    const J m = problem.m;
    const J n = problem.n;

    const bool host_mode = (pointer_mode == HIPSPARSE_POINTER_MODE_HOST);

    T beta = make_DataType<T>(0);

    // Scalars for every compute call
    std::vector<T> h_alphas(alpha_scales.size());
    for(size_t i = 0; i < alpha_scales.size(); ++i)
    {
        h_alphas[i] = testing_mult(alpha, make_DataType<T>(alpha_scales[i]));
    }

    std::unique_ptr<spgemm_struct> unique_ptr_descr(new spgemm_struct);
    hipsparseSpGEMMDescr_t         descr = unique_ptr_descr->descr;

    auto dcsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto d_alphas_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * h_alphas.size()), device_free};
    auto d_beta_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dcsr_row_ptr_C = (I*)dcsr_row_ptr_C_managed.get();
    T* d_alphas       = (T*)d_alphas_managed.get();
    T* d_beta         = (T*)d_beta_managed.get();

    CHECK_HIP_ERROR(
        hipMemcpy(d_alphas, h_alphas.data(), sizeof(T) * h_alphas.size(), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &beta, sizeof(T), hipMemcpyHostToDevice));

    const T* p_beta = host_mode ? &beta : d_beta;

    hipsparseSpMatDescr_t C;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(
        &C, m, n, 0, dcsr_row_ptr_C, nullptr, nullptr, typeI, typeJ, problem.idx_base_C, typeT));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, pointer_mode));

    // SpGEMMreuse work estimation
    size_t bufferSize1;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_workEstimation(
        handle, transA, transB, problem.X, problem.Y, C, alg, descr, &bufferSize1, nullptr));

    hipsparse_unique_ptr own_buffer1{nullptr, device_free};
    void*                externalBuffer1 = workspace.get(0, bufferSize1, own_buffer1);

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_workEstimation(handle,
                                                              transA,
                                                              transB,
                                                              problem.X,
                                                              problem.Y,
                                                              C,
                                                              alg,
                                                              descr,
                                                              &bufferSize1,
                                                              externalBuffer1));

    // SpGEMMreuse nnz
    size_t bufferSize2, bufferSize3, bufferSize4;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_nnz(handle,
                                                   transA,
                                                   transB,
                                                   problem.X,
                                                   problem.Y,
                                                   C,
                                                   alg,
                                                   descr,
                                                   &bufferSize2,
                                                   nullptr,
                                                   &bufferSize3,
                                                   nullptr,
                                                   &bufferSize4,
                                                   nullptr));

    hipsparse_unique_ptr own_buffer2{nullptr, device_free};
    hipsparse_unique_ptr own_buffer3{nullptr, device_free};
    hipsparse_unique_ptr own_buffer4{nullptr, device_free};
    void*                externalBuffer2 = workspace.get(1, bufferSize2, own_buffer2);
    void*                externalBuffer3 = workspace.get(2, bufferSize3, own_buffer3);
    void*                externalBuffer4 = workspace.get(3, bufferSize4, own_buffer4);

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_nnz(handle,
                                                   transA,
                                                   transB,
                                                   problem.X,
                                                   problem.Y,
                                                   C,
                                                   alg,
                                                   descr,
                                                   &bufferSize2,
                                                   externalBuffer2,
                                                   &bufferSize3,
                                                   externalBuffer3,
                                                   &bufferSize4,
                                                   externalBuffer4));

    // Get nnz of C
    int64_t rows_C, cols_C, nnz_C;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMatGetSize(C, &rows_C, &cols_C, &nnz_C));

    // Allocate C
    auto dcsr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_C), device_free};
    auto dcsr_val_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};

    J* dcsr_col_ind_C = (J*)dcsr_col_ind_C_managed.get();
    T* dcsr_val_C     = (T*)dcsr_val_C_managed.get();

    CHECK_HIP_ERROR(hipMemset(dcsr_val_C, 0, sizeof(T) * nnz_C));

    // Set C pointers
    CHECK_HIPSPARSE_ERROR(hipsparseCsrSetPointers(C, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C));

    // SpGEMMreuse copy
    size_t bufferSize5;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_copy(
        handle, transA, transB, problem.X, problem.Y, C, alg, descr, &bufferSize5, nullptr));

    hipsparse_unique_ptr own_buffer5{nullptr, device_free};
    void*                externalBuffer5 = workspace.get(4, bufferSize5, own_buffer5);

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_copy(handle,
                                                    transA,
                                                    transB,
                                                    problem.X,
                                                    problem.Y,
                                                    C,
                                                    alg,
                                                    descr,
                                                    &bufferSize5,
                                                    externalBuffer5));

    // Verify nnz, row pointer array and column indices. These do not depend on alpha.
    std::vector<I> hcsr_row_ptr_C(m + 1);
    std::vector<J> hcsr_col_ind_C(nnz_C);
    std::vector<T> hcsr_val_C(nnz_C);

    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(I) * (m + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(J) * nnz_C, hipMemcpyDeviceToHost));

    int64_t nnz_C_gold = problem.gold_nnz;
    unit_check_general(1, 1, 1, &nnz_C_gold, &nnz_C);
    unit_check_general(1, m + 1, 1, problem.gold_row_ptr.data(), hcsr_row_ptr_C.data());

    // The column and value arrays can only be compared when their sizes match
    if(nnz_C == nnz_C_gold)
    {
        unit_check_general(1, nnz_C, 1, problem.gold_col_ind.data(), hcsr_col_ind_C.data());
    }

    // SpGEMMreuse compute. Recompute C = alpha * X * Y for every scalar using the same
    // descriptors and buffers.
    for(size_t i = 0; i < h_alphas.size(); ++i)
    {
        const T* p_alpha = host_mode ? &h_alphas[i] : d_alphas + i;

        CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_compute(
            handle, transA, transB, p_alpha, problem.X, problem.Y, p_beta, C, typeT, alg, descr));

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        if(nnz_C == nnz_C_gold)
        {
            // The reference solution was computed for alpha so scale it by alpha_scales[i]
            std::vector<T> hcsr_val_C_gold(problem.gold_val);
            for(auto& val : hcsr_val_C_gold)
            {
                val = testing_mult(val, make_DataType<T>(alpha_scales[i]));
            }

            unit_check_near(1, nnz_C, 1, hcsr_val_C_gold.data(), hcsr_val_C.data());
        }
    }

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C));
}

// Repeatedly calls the complete SpGEMMreuse flow for every combination of problem,
// algorithm and pointer mode.
template <typename I, typename J, typename T>
static void call_spgemmreuse_all(hipsparseHandle_t                          handle,
                                 std::vector<spgemmreuse_problem<I, J, T>>& problems,
                                 T                                          alpha,
                                 const std::vector<double>&                 alpha_scales,
                                 const std::vector<hipsparseSpGEMMAlg_t>&   algs,
                                 const std::vector<hipsparsePointerMode_t>& pointer_modes,
                                 int                                        number_of_passes,
                                 spgemmreuse_workspace&                     workspace)
{
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(auto& problem : problems)
        {
            for(hipsparseSpGEMMAlg_t alg : algs)
            {
                for(hipsparsePointerMode_t pointer_mode : pointer_modes)
                {
                    call_spgemmreuse<I, J, T>(
                        handle, problem, alpha, alpha_scales, alg, pointer_mode, workspace);
                }
            }
        }
    }
}

template <typename I, typename J, typename T>
void testing_spgemmreuse_csr_reuse_descr(Arguments argus)
{
    J                    m        = argus.M;
    J                    k        = argus.K;
    T                    h_alpha  = argus.get_alpha<T>();
    hipsparseIndexBase_t idxBaseA = argus.baseA;
    hipsparseIndexBase_t idxBaseB = argus.baseB;
    hipsparseIndexBase_t idxBaseC = argus.baseC;
    std::string          filename = argus.filename;

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

    // For sparse matrix B, use the transpose of A
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

    // allocate memory on device
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

    // copy data from CPU to device
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

    // Create the sparse matrices. These descriptors are created once and are reused
    // by every SpGEMMreuse call below.
    hipsparseSpMatDescr_t A, B;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&A,
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
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&B,
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

    // The same two descriptors are used in both roles:
    //   problem 0: C (m x n) = alpha * A * B
    //   problem 1: C (k x k) = alpha * B * A
    std::vector<spgemmreuse_problem<I, J, T>> problems(2);

    spgemmreuse_init_problem<I, J, T>(problems[0],
                                      A,
                                      B,
                                      m,
                                      n,
                                      k,
                                      h_alpha,
                                      hcsr_row_ptr_A,
                                      hcsr_col_ind_A,
                                      hcsr_val_A,
                                      hcsr_row_ptr_B,
                                      hcsr_col_ind_B,
                                      hcsr_val_B,
                                      idxBaseA,
                                      idxBaseB,
                                      idxBaseC);

    spgemmreuse_init_problem<I, J, T>(problems[1],
                                      B,
                                      A,
                                      k,
                                      k,
                                      n,
                                      h_alpha,
                                      hcsr_row_ptr_B,
                                      hcsr_col_ind_B,
                                      hcsr_val_B,
                                      hcsr_row_ptr_A,
                                      hcsr_col_ind_A,
                                      hcsr_val_A,
                                      idxBaseB,
                                      idxBaseA,
                                      idxBaseC);

    const std::vector<hipsparseSpGEMMAlg_t>   algs          = spgemmreuse_algs();
    const std::vector<hipsparsePointerMode_t> pointer_modes = spgemmreuse_pointer_modes();

    // hipsparseSpGEMMreuse_compute is called once for every scalar alpha * alpha_scales[i]
    const std::vector<double> alpha_scales = {1.0, -0.5, 2.0, 1.0};

    constexpr int number_of_passes = 3;

    // Scenario 1: every call allocates its own buffers. Exercises that the same sparse
    // matrix descriptors produce correct results when they are used in a sequence of
    // different SpGEMMreuse computations.
    spgemmreuse_workspace workspace_sizes;
    call_spgemmreuse_all<I, J, T>(handle,
                                  problems,
                                  h_alpha,
                                  alpha_scales,
                                  algs,
                                  pointer_modes,
                                  number_of_passes,
                                  workspace_sizes);

    // Scenario 2: one buffer per slot, each sized to the maximum required across all of the
    // configurations of scenario 1, is shared by every call. Verifies that user provided
    // buffers remain valid when the sparse matrix descriptors are reused across configurations.
    hipsparse_unique_ptr shared_buffers_managed[spgemmreuse_workspace::num_buffers]
        = {{nullptr, device_free},
           {nullptr, device_free},
           {nullptr, device_free},
           {nullptr, device_free},
           {nullptr, device_free}};

    spgemmreuse_workspace shared_workspace;
    for(int i = 0; i < spgemmreuse_workspace::num_buffers; ++i)
    {
        const size_t size = std::max(workspace_sizes.required[i], size_t(4));

        shared_buffers_managed[i]    = hipsparse_unique_ptr{device_malloc(size), device_free};
        shared_workspace.shared[i]   = shared_buffers_managed[i].get();
        shared_workspace.capacity[i] = size;
    }

    call_spgemmreuse_all<I, J, T>(handle,
                                  problems,
                                  h_alpha,
                                  alpha_scales,
                                  algs,
                                  pointer_modes,
                                  number_of_passes,
                                  shared_workspace);

    // Clean up
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(B));
}

#else

template <typename I, typename J, typename T>
void testing_spgemmreuse_csr_reuse_descr(Arguments argus)
{
}

#endif

#endif // TESTING_SPGEMMREUSE_CSR_REUSE_DESCR_HPP
