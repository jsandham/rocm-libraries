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
#ifndef TESTING_SPGEAM_CSR_REUSE_DESCR_HPP
#define TESTING_SPGEAM_CSR_REUSE_DESCR_HPP

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

template <typename T>
void testing_spgeam_csr_reuse_descr_bad_arg(const Arguments& argus)
{
}

// clang-format 19 (math-ci) rewrites "#if(" to "#if (", which clang-format
// 18 (the repo pre-commit hook) reverts; disable so both formatters agree.
// clang-format off
#if(defined(HIPSPARSE_WITH_SPGEAM) && (!defined(CUDART_VERSION) || CUDART_VERSION >= 13030))
// clang-format on

// A single SpGEAM computation C = alpha * X + beta * Y where X and Y (both m x n) are
// sparse matrix descriptors that are created once by the caller and reused by every
// call below, together with the host reference solution for that computation.
template <typename T>
struct spgeam_reuse_case
{
    hipsparseSpMatDescr_t X{};
    hipsparseSpMatDescr_t Y{};

    int m{};
    int n{};

    T alpha{};
    T beta{};

    hipsparseIndexBase_t idx_base_C{};

    int              gold_nnz{};
    std::vector<int> gold_row_ptr;
    std::vector<int> gold_col_ind;
    std::vector<T>   gold_val;
};

// The host matrix data behind a sparse matrix descriptor.
template <typename T>
struct spgeam_reuse_host_matrix
{
    hipsparseSpMatDescr_t   descr{};
    hipsparseIndexBase_t    idx_base{};
    const std::vector<int>* row_ptr{};
    const std::vector<int>* col_ind{};
    const std::vector<T>*   val{};
};

// Computes the host reference solution of C = alpha * X + beta * Y.
//
// Match the cuSPARSE SpGEAM behavior for zero scalars: when alpha is zero the output
// sparsity pattern is that of Y, when beta is zero it is that of X, and when both are
// zero the output is empty. This is emulated by dropping the corresponding operand
// (feeding an empty matrix) to the host csrgeam reference.
template <typename T>
static void spgeam_reuse_init_case(spgeam_reuse_case<T>&              c,
                                   const spgeam_reuse_host_matrix<T>& X,
                                   const spgeam_reuse_host_matrix<T>& Y,
                                   int                                m,
                                   int                                n,
                                   T                                  alpha,
                                   T                                  beta,
                                   bool                               alpha_is_zero,
                                   bool                               beta_is_zero,
                                   hipsparseIndexBase_t               idx_base_C)
{
    c.X          = X.descr;
    c.Y          = Y.descr;
    c.m          = m;
    c.n          = n;
    c.alpha      = alpha;
    c.beta       = beta;
    c.idx_base_C = idx_base_C;

    std::vector<int> empty_row_ptr_X(m + 1, X.idx_base);
    std::vector<int> empty_row_ptr_Y(m + 1, Y.idx_base);
    std::vector<int> empty_col_ind;
    std::vector<T>   empty_val;

    const int* gold_row_ptr_X = alpha_is_zero ? empty_row_ptr_X.data() : X.row_ptr->data();
    const int* gold_col_ind_X = alpha_is_zero ? empty_col_ind.data() : X.col_ind->data();
    const T*   gold_val_X     = alpha_is_zero ? empty_val.data() : X.val->data();
    const int* gold_row_ptr_Y = beta_is_zero ? empty_row_ptr_Y.data() : Y.row_ptr->data();
    const int* gold_col_ind_Y = beta_is_zero ? empty_col_ind.data() : Y.col_ind->data();
    const T*   gold_val_Y     = beta_is_zero ? empty_val.data() : Y.val->data();

    c.gold_row_ptr.resize(m + 1);

    c.gold_nnz = host_csrgeam_nnz(m,
                                  n,
                                  alpha,
                                  gold_row_ptr_X,
                                  gold_col_ind_X,
                                  beta,
                                  gold_row_ptr_Y,
                                  gold_col_ind_Y,
                                  c.gold_row_ptr.data(),
                                  X.idx_base,
                                  Y.idx_base,
                                  idx_base_C);

    c.gold_col_ind.resize(c.gold_nnz);
    c.gold_val.resize(c.gold_nnz);

    host_csrgeam(m,
                 n,
                 alpha,
                 gold_row_ptr_X,
                 gold_col_ind_X,
                 gold_val_X,
                 beta,
                 gold_row_ptr_Y,
                 gold_col_ind_Y,
                 gold_val_Y,
                 c.gold_row_ptr.data(),
                 c.gold_col_ind.data(),
                 c.gold_val.data(),
                 X.idx_base,
                 Y.idx_base,
                 idx_base_C);
}

// Describes the single external buffer used by SpGEAM.
//
// When no shared buffer is provided every call allocates (and frees) its own buffer.
// In both cases `required` is updated to the maximum size requested so far, which is
// how the size of the shared buffer is determined.
struct spgeam_reuse_workspace
{
    size_t required{};
    size_t capacity{};
    void*  shared{};

    // Returns the buffer to be used given that `size` bytes are needed.
    void* get(size_t size, hipsparse_unique_ptr& own)
    {
        required = std::max(required, size);

        if(shared != nullptr && size <= capacity)
        {
            return shared;
        }

        // Either no shared buffer was provided or it is too small. The latter
        // is a test failure but keep going with a private buffer so we never
        // write out of bounds.
        EXPECT_TRUE(shared == nullptr) << "shared buffer is too small: required " << size
                                       << " but only " << capacity << " available";

        // Ensure the externalBuffer pointer is never null even when no scratch space is needed.
        own = hipsparse_unique_ptr{device_malloc(std::max(size, size_t(4))), device_free};
        return own.get();
    }
};

static std::vector<hipsparsePointerMode_t> spgeam_reuse_pointer_modes()
{
    // cusparseSpGEAM only supports host pointer mode scalars so the device pointer mode
    // is only exercised on the rocSPARSE backend.
#if(!defined(CUDART_VERSION))
    return {HIPSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_DEVICE};
#else
    return {HIPSPARSE_POINTER_MODE_HOST};
#endif
}

// Runs the complete buffer size -> nnz -> compute flow for the computation described by
// `c` using the already existing X and Y sparse matrix descriptors. A new C matrix
// descriptor and a new SpGEAM descriptor are created for every call, so the only state
// carried over between calls is whatever hipSPARSE keeps in X and Y. The result is
// validated against the host reference solution.
template <typename T>
static void call_spgeam(hipsparseHandle_t       handle,
                        spgeam_reuse_case<T>&   c,
                        hipsparsePointerMode_t  pointer_mode,
                        spgeam_reuse_workspace& workspace)
{
    const hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    const hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    const hipsparseSpGEAMAlg_t alg    = HIPSPARSE_SPGEAM_ALG1;

    const hipsparseIndexType_t typeI = HIPSPARSE_INDEX_32I;
    const hipsparseIndexType_t typeJ = HIPSPARSE_INDEX_32I;
    const hipDataType          typeT = getDataType<T>();

    const int m = c.m;
    const int n = c.n;

    const bool host_mode = (pointer_mode == HIPSPARSE_POINTER_MODE_HOST);

    T h_alpha = c.alpha;
    T h_beta  = c.beta;

    std::unique_ptr<spgeam_struct> unique_ptr_descr(new spgeam_struct);
    hipsparseSpGEAMDescr_t         descr = unique_ptr_descr->descr;

    auto dcsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dcsr_row_ptr_C = (int*)dcsr_row_ptr_C_managed.get();
    T*   d_alpha        = (T*)d_alpha_managed.get();
    T*   d_beta         = (T*)d_beta_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    const T* p_alpha = host_mode ? &h_alpha : d_alpha;
    const T* p_beta  = host_mode ? &h_beta : d_beta;

    hipsparseSpMatDescr_t C;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(
        &C, m, n, 0, dcsr_row_ptr_C, nullptr, nullptr, typeI, typeJ, c.idx_base_C, typeT));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, pointer_mode));

    // SpGEAM buffer size
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM_bufferSize(
        handle, transA, transB, p_alpha, c.X, p_beta, c.Y, C, typeT, alg, descr, &bufferSize));

    hipsparse_unique_ptr own_buffer{nullptr, device_free};
    void*                externalBuffer = workspace.get(bufferSize, own_buffer);

    // SpGEAM nnz
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM_nnz(
        handle, transA, transB, p_alpha, c.X, p_beta, c.Y, C, typeT, alg, descr, externalBuffer));

    // Get nnz of C
    int64_t rows_C, cols_C, nnz_C;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMatGetSize(C, &rows_C, &cols_C, &nnz_C));

    // Allocate C column indices and values
    auto dcsr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_C), device_free};
    auto dcsr_val_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};

    int* dcsr_col_ind_C = (int*)dcsr_col_ind_C_managed.get();
    T*   dcsr_val_C     = (T*)dcsr_val_C_managed.get();

    CHECK_HIPSPARSE_ERROR(hipsparseCsrSetPointers(C, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C));

    // SpGEAM compute
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM(
        handle, transA, transB, p_alpha, c.X, p_beta, c.Y, C, typeT, alg, descr, externalBuffer));

    // Copy output from device to CPU
    std::vector<int> hcsr_row_ptr_C(m + 1);
    std::vector<int> hcsr_col_ind_C(nnz_C);
    std::vector<T>   hcsr_val_C(nnz_C);

    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int) * nnz_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C));

    // Verify nnz and row pointer array
    int nnz_C_i    = static_cast<int>(nnz_C);
    int nnz_C_gold = c.gold_nnz;
    unit_check_general(1, 1, 1, &nnz_C_gold, &nnz_C_i);
    unit_check_general(1, m + 1, 1, c.gold_row_ptr.data(), hcsr_row_ptr_C.data());

    // The column and value arrays can only be compared when their sizes match
    if(nnz_C_i != nnz_C_gold)
    {
        return;
    }

    // Verify column and value array
    unit_check_general(1, nnz_C_gold, 1, c.gold_col_ind.data(), hcsr_col_ind_C.data());
    unit_check_near(1, nnz_C_gold, 1, c.gold_val.data(), hcsr_val_C.data());
}

// Repeatedly calls the complete SpGEAM flow for every combination of case and pointer mode.
template <typename T>
static void call_spgeam_all(hipsparseHandle_t                          handle,
                            std::vector<spgeam_reuse_case<T>>&         cases,
                            const std::vector<hipsparsePointerMode_t>& pointer_modes,
                            int                                        number_of_passes,
                            spgeam_reuse_workspace&                    workspace)
{
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(auto& c : cases)
        {
            for(hipsparsePointerMode_t pointer_mode : pointer_modes)
            {
                call_spgeam<T>(handle, c, pointer_mode, workspace);
            }
        }
    }
}

template <typename T>
void testing_spgeam_csr_reuse_descr(Arguments argus)
{
    int                  m        = argus.M;
    int                  n        = argus.N;
    T                    h_alpha  = argus.get_alpha<T>();
    T                    h_beta   = argus.get_beta<T>();
    hipsparseIndexBase_t idxBaseA = argus.baseA;
    hipsparseIndexBase_t idxBaseB = argus.baseB;
    hipsparseIndexBase_t idxBaseC = argus.baseC;
    std::string          filename = argus.filename;

    hipsparseIndexType_t typeI = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t typeJ = HIPSPARSE_INDEX_32I;
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures for A
    std::vector<int> hcsr_row_ptr_A;
    std::vector<int> hcsr_col_ind_A;
    std::vector<T>   hcsr_val_A;

    srand(12345ULL);

    int nnz_A = 0;
    CHECK_GENERATE_MATRIX_ERROR(generate_csr_matrix(
        filename, m, n, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idxBaseA));

    // B is generated as an independent m x n matrix so that it has a different sparsity
    // pattern than A.
    std::vector<int> hcsr_row_ptr_B;
    std::vector<int> hcsr_col_ind_B;
    std::vector<T>   hcsr_val_B;

    int nnz_B = 0;
    CHECK_GENERATE_MATRIX_ERROR(generate_csr_matrix(
        "*", m, n, nnz_B, hcsr_row_ptr_B, hcsr_col_ind_B, hcsr_val_B, idxBaseB));

    // Allocate device memory
    auto dcsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_A), device_free};
    auto dcsr_val_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dcsr_row_ptr_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_B), device_free};
    auto dcsr_val_B_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};

    int* dcsr_row_ptr_A = (int*)dcsr_row_ptr_A_managed.get();
    int* dcsr_col_ind_A = (int*)dcsr_col_ind_A_managed.get();
    T*   dcsr_val_A     = (T*)dcsr_val_A_managed.get();
    int* dcsr_row_ptr_B = (int*)dcsr_row_ptr_B_managed.get();
    int* dcsr_col_ind_B = (int*)dcsr_col_ind_B_managed.get();
    T*   dcsr_val_B     = (T*)dcsr_val_B_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_B, hcsr_col_ind_B.data(), sizeof(int) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_B, hcsr_val_B.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));

    // Create the sparse matrices. These descriptors are created once and are reused
    // by every SpGEAM call below.
    hipsparseSpMatDescr_t A, B;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&A,
                                             m,
                                             n,
                                             nnz_A,
                                             dcsr_row_ptr_A,
                                             dcsr_col_ind_A,
                                             dcsr_val_A,
                                             typeI,
                                             typeJ,
                                             idxBaseA,
                                             typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&B,
                                             m,
                                             n,
                                             nnz_B,
                                             dcsr_row_ptr_B,
                                             dcsr_col_ind_B,
                                             dcsr_val_B,
                                             typeI,
                                             typeJ,
                                             idxBaseB,
                                             typeT));

    spgeam_reuse_host_matrix<T> host_A{A, idxBaseA, &hcsr_row_ptr_A, &hcsr_col_ind_A, &hcsr_val_A};
    spgeam_reuse_host_matrix<T> host_B{B, idxBaseB, &hcsr_row_ptr_B, &hcsr_col_ind_B, &hcsr_val_B};

    // The same two descriptors are used in different roles and, in the last case, the same
    // descriptor is passed as both operands:
    //   C = alpha * A + beta * B
    //   C = alpha * B + beta * A
    //   C = alpha * A + beta * A
    const std::vector<
        std::pair<const spgeam_reuse_host_matrix<T>*, const spgeam_reuse_host_matrix<T>*>>
        operands = {{&host_A, &host_B}, {&host_B, &host_A}, {&host_A, &host_A}};

    // Scalars. Besides the scalars provided by the test arguments, also run the cases where
    // alpha and/or beta are zero as these take different code paths in the library.
    const bool base_alpha_is_zero = (argus.alpha == 0.0 && argus.alphai == 0.0);
    const bool base_beta_is_zero  = (argus.beta == 0.0 && argus.betai == 0.0);

    struct scalars_t
    {
        T    alpha;
        T    beta;
        bool alpha_is_zero;
        bool beta_is_zero;
    };

    const T zero = make_DataType<T>(0);

    const std::vector<scalars_t> scalars
        = {{h_alpha, h_beta, base_alpha_is_zero, base_beta_is_zero},
           {h_alpha, zero, base_alpha_is_zero, true},
           {zero, h_beta, true, base_beta_is_zero},
           {zero, zero, true, true}};

    std::vector<spgeam_reuse_case<T>> cases;
    for(const auto& operand : operands)
    {
        const spgeam_reuse_host_matrix<T>& X = *operand.first;
        const spgeam_reuse_host_matrix<T>& Y = *operand.second;

        for(const auto& s : scalars)
        {
            // When exactly one of the scalars is zero C is a scaled copy of the other operand.
            // This copies the sparsity pattern verbatim so it requires the surviving operand
            // and C to use the same index base (base conversion is not supported).
            const bool collapse_to_X = !s.alpha_is_zero && s.beta_is_zero;
            const bool collapse_to_Y = s.alpha_is_zero && !s.beta_is_zero;
            if((collapse_to_X && X.idx_base != idxBaseC)
               || (collapse_to_Y && Y.idx_base != idxBaseC))
            {
                continue;
            }

            cases.emplace_back();
            spgeam_reuse_init_case<T>(cases.back(),
                                      X,
                                      Y,
                                      m,
                                      n,
                                      s.alpha,
                                      s.beta,
                                      s.alpha_is_zero,
                                      s.beta_is_zero,
                                      idxBaseC);
        }
    }

    const std::vector<hipsparsePointerMode_t> pointer_modes = spgeam_reuse_pointer_modes();

    constexpr int number_of_passes = 3;

    // Scenario 1: every call allocates its own buffer. Exercises that the same sparse
    // matrix descriptors produce correct results when they are used in a sequence of
    // different SpGEAM computations.
    spgeam_reuse_workspace workspace_size;
    call_spgeam_all<T>(handle, cases, pointer_modes, number_of_passes, workspace_size);

    // Scenario 2: a single buffer, sized to the maximum required across all of the cases of
    // scenario 1, is shared by every call. Verifies that a user provided buffer remains valid
    // when the sparse matrix descriptors are reused across configurations.
    const size_t shared_size   = std::max(workspace_size.required, size_t(4));
    auto shared_buffer_managed = hipsparse_unique_ptr{device_malloc(shared_size), device_free};

    spgeam_reuse_workspace shared_workspace;
    shared_workspace.shared   = shared_buffer_managed.get();
    shared_workspace.capacity = shared_size;

    call_spgeam_all<T>(handle, cases, pointer_modes, number_of_passes, shared_workspace);

    // Clean up
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(B));
}

#endif

#endif // TESTING_SPGEAM_CSR_REUSE_DESCR_HPP
