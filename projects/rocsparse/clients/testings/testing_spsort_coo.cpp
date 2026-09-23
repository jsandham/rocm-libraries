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

#include "testing.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>

namespace
{
    template <typename I, typename T>
    void host_spsort_coo(rocsparse_direction dir, host_coo_matrix<T, I>& A)
    {
        const int64_t nnz = A.nnz;
        const I*      row = A.row_ind.data();
        const I*      col = A.col_ind.data();
        const T*      val = A.val.data();

        std::vector<int64_t> perm(nnz);
        std::iota(perm.begin(), perm.end(), 0);

        if(dir == rocsparse_direction_row)
        {
            std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                return std::tie(row[a], col[a]) < std::tie(row[b], col[b]);
            });
        }
        else
        {
            std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                return std::tie(col[a], row[a]) < std::tie(col[b], row[b]);
            });
        }

        std::vector<I> sorted_row(nnz);
        std::vector<I> sorted_col(nnz);
        std::vector<T> sorted_val(nnz);
        for(int64_t k = 0; k < nnz; ++k)
        {
            sorted_row[k] = row[perm[k]];
            sorted_col[k] = col[perm[k]];
            sorted_val[k] = val[perm[k]];
        }

        std::copy(sorted_row.begin(), sorted_row.end(), A.row_ind.data());
        std::copy(sorted_col.begin(), sorted_col.end(), A.col_ind.data());
        std::copy(sorted_val.begin(), sorted_val.end(), A.val.data());
    }

    template <typename I, typename T>
    void host_shuffle_coo(host_coo_matrix<T, I>& A)
    {
        const int64_t nnz = A.nnz;
        I*            row = A.row_ind.data();
        I*            col = A.col_ind.data();
        T*            val = A.val.data();

        for(int64_t i = 0; i < nnz; ++i)
        {
            const int64_t j = rand() % nnz;
            std::swap(row[i], row[j]);
            std::swap(col[i], col[j]);
            std::swap(val[i], val[j]);
        }
    }

    void set_spsort_inputs(rocsparse_handle       handle,
                           rocsparse_spsort_descr descr,
                           rocsparse_spsort_alg   alg,
                           rocsparse_direction    dir)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr));
    }
}

template <typename I, typename T>
void testing_spsort_coo_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    // Pointer and enum checks, with dummy descriptors.
    {
        rocsparse_spsort_descr descr   = (rocsparse_spsort_descr)0x4;
        rocsparse_spmat_descr  mat     = (rocsparse_spmat_descr)0x4;
        rocsparse_spsort_stage stage   = rocsparse_spsort_stage_analysis;
        rocsparse_error*       p_error = nullptr;

        {
            size_t* buffer_size = (size_t*)0x4;
#define PARAMS_BUFFER_SIZE handle, descr, mat, stage, buffer_size, p_error
            static constexpr int nex     = 1;
            static const int     ex[nex] = {5};
            select_bad_arg_analysis(rocsparse_spsort_buffer_size, nex, ex, PARAMS_BUFFER_SIZE);
#undef PARAMS_BUFFER_SIZE
        }

        {
            const size_t buffer_size = 10;
            void*        buffer      = (void*)0x4;
#define PARAMS handle, descr, mat, stage, buffer_size, buffer, p_error
            static constexpr int nex     = 2;
            static const int     ex[nex] = {4, 6};
            select_bad_arg_analysis(rocsparse_spsort, nex, ex, PARAMS);
#undef PARAMS
        }

        {
            rocsparse_spsort_input input              = rocsparse_spsort_input_alg;
            const void*            data               = (const void*)0x4;
            const size_t           data_size_in_bytes = sizeof(rocsparse_spsort_alg);
#define PARAMS_SET_INPUT handle, descr, input, data, data_size_in_bytes, p_error
            static constexpr int nex     = 2;
            static const int     ex[nex] = {4, 5};
            select_bad_arg_analysis(rocsparse_spsort_set_input, nex, ex, PARAMS_SET_INPUT);
#undef PARAMS_SET_INPUT
        }
    }

    EXPECT_ROCSPARSE_STATUS(rocsparse_create_spsort_descr(nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spsort_descr(nullptr), rocsparse_status_success);

    // Descriptor input and stage ordering checks, with a real descriptor and matrix.
    {
        device_vector<I> d_coo_row_ind(safe_size);
        device_vector<I> d_coo_col_ind(safe_size);
        device_vector<T> d_coo_val(safe_size);

        rocsparse_local_spmat mat(safe_size,
                                  safe_size,
                                  safe_size,
                                  d_coo_row_ind,
                                  d_coo_col_ind,
                                  d_coo_val,
                                  get_indextype<I>(),
                                  rocsparse_index_base_zero,
                                  get_datatype<T>());

        const rocsparse_spsort_alg alg = rocsparse_spsort_alg_default;
        const rocsparse_direction  dir = rocsparse_direction_row;

        rocsparse_spsort_descr descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_spsort_descr(&descr));

        // Wrong input sizes.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg) + 1, nullptr),
            rocsparse_status_invalid_size);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir) + 1, nullptr),
            rocsparse_status_invalid_size);

        // The algorithm has not been set yet.
        size_t buffer_size;
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_buffer_size(
                handle, descr, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
            rocsparse_status_invalid_value);

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr));

        // The direction has not been set yet.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_buffer_size(
                handle, descr, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
            rocsparse_status_invalid_value);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort(
                handle, descr, mat, rocsparse_spsort_stage_analysis, 0, nullptr, nullptr),
            rocsparse_status_invalid_value);

        // Invalid direction value.
        const rocsparse_direction invalid_dir = (rocsparse_direction)-1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_set_input(handle,
                                                           descr,
                                                           rocsparse_spsort_input_direction,
                                                           &invalid_dir,
                                                           sizeof(invalid_dir),
                                                           nullptr),
                                rocsparse_status_invalid_value);

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr));

        // Compute cannot be executed before analysis.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort(
                handle, descr, mat, rocsparse_spsort_stage_compute, 0, nullptr, nullptr),
            rocsparse_status_invalid_value);

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
            handle, descr, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr));
        void* dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort(
            handle, descr, mat, rocsparse_spsort_stage_analysis, buffer_size, dbuffer, nullptr));

        // Analysis cannot be executed twice.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort(
                handle, descr, mat, rocsparse_spsort_stage_analysis, buffer_size, dbuffer, nullptr),
            rocsparse_status_invalid_value);

        // The algorithm cannot be changed after analysis.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
            rocsparse_status_internal_error);

        // The direction cannot be changed after analysis.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr),
            rocsparse_status_internal_error);

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spsort_descr(descr));
    }
}

template <typename I, typename T>
void testing_spsort_coo(const Arguments& arg)
{
    I                    M    = arg.M;
    I                    N    = arg.N;
    rocsparse_index_base base = arg.baseA;
    rocsparse_direction  dir  = arg.direction;
    rocsparse_spsort_alg alg  = rocsparse_spsort_alg_default;

    rocsparse_local_handle handle(arg);

    rocsparse_matrix_factory<T, I, I> matrix_factory(arg);

    host_coo_matrix<T, I> hA;
    matrix_factory.init_coo(hA, M, N, base);

    host_coo_matrix<T, I> hA_gold(hA);
    host_spsort_coo(dir, hA_gold);

    rocsparse_seedrand();
    host_shuffle_coo(hA);

    device_coo_matrix<T, I> dA(hA);
    rocsparse_local_spmat   matA(dA);

    rocsparse_spsort_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_spsort_descr(&descr));
    set_spsort_inputs(handle, descr, alg, dir);

    // Analysis
    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
        handle, descr, matA, rocsparse_spsort_stage_analysis, &buffer_size, nullptr));

    void* dbuffer = nullptr;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort(
        handle, descr, matA, rocsparse_spsort_stage_analysis, buffer_size, dbuffer, nullptr));
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    dbuffer = nullptr;

    // Compute
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
        handle, descr, matA, rocsparse_spsort_stage_compute, &buffer_size, nullptr));
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort(
            handle, descr, matA, rocsparse_spsort_stage_compute, buffer_size, dbuffer, nullptr));

        hA_gold.unit_check(dA);
    }

    if(arg.timing)
    {
        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_spsort,
                                               handle,
                                               descr,
                                               matA,
                                               rocsparse_spsort_stage_compute,
                                               buffer_size,
                                               dbuffer,
                                               nullptr);

        const double gbyte_count = spsort_coo_gbyte_count<I, T>(dA.nnz);
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            dA.nnz,
                            display_key_t::dir,
                            rocsparse_direction2string(dir),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spsort_descr(descr));
}

#define INSTANTIATE(ITYPE, TTYPE)                                                 \
    template void testing_spsort_coo_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spsort_coo<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
void testing_spsort_coo_extra(const Arguments& arg) {}
