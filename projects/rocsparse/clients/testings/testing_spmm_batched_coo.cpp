/* ************************************************************************
* Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <tuple>
#include <assert.h>

template <typename I, typename A, typename B, typename C, typename T>
void testing_spmm_batched_coo_bad_arg(const Arguments& arg){}

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << " stat: " << stat << std::endl; \
            return;                                                         \
        }                                                                      \
    }

// Simple Matrix Market coordinate format parser (real, general or symmetric)
bool read_mtx(const char* filename,
    int&        m,
    int&        n,
    int64_t&    nnz,
    std::vector<int32_t>& row_ind,
    std::vector<int32_t>& col_ind,
    std::vector<double>&   val)
{
    std::ifstream f(filename);
    if(!f)
    {
        std::cerr << "Cannot open " << filename << std::endl;
        return false;
    }

    std::string line;
    bool        symmetric = false;
    while(std::getline(f, line))
    {
        if(line.empty())
            continue;
        if(line[0] == '%')
        {
            std::string lower = line;
            for(char& ch : lower)
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if(lower.find("symmetric") != std::string::npos)
            symmetric = true;
            continue;
        }
        break;
    }

    std::istringstream iss(line);
    int                nrows, ncols, nentries;
    iss >> nrows >> ncols >> nentries;

    m   = nrows;
    n   = ncols;
    nnz = nentries;

    row_ind.resize(nnz);
    col_ind.resize(nnz);
    val.resize(nnz);

    for(int64_t i = 0; i < nnz; i++)
    {
        if(!std::getline(f, line))
        {
            std::cerr << "Unexpected EOF" << std::endl;
            return false;
        }
        std::istringstream lss(line);
        int                r, c;
        double             v;
        lss >> r >> c >> v;
        row_ind[i] = r - 1; // 1-based to 0-based
        col_ind[i] = c - 1;
        val[i]     = static_cast<double>(v);
    }

    // If symmetric, the file stores only the lower triangle; fill in the upper triangle
    if(symmetric)
    {
        int64_t off_diag = 0;
        for(int64_t i = 0; i < nnz; i++)
            if(row_ind[i] != col_ind[i])
                off_diag++;
        int64_t new_nnz = nnz + off_diag;
        row_ind.resize(new_nnz);
        col_ind.resize(new_nnz);
        val.resize(new_nnz);
        int64_t j = nnz;
        for(int64_t i = 0; i < nnz; i++)
        {
            if(row_ind[i] != col_ind[i])
            {
                row_ind[j] = col_ind[i];
                col_ind[j] = row_ind[i];
                val[j]     = val[i];
                j++;
            }
        }
        nnz = new_nnz;
    }

    // Sort COO by row index, then by column index within each row
    std::vector<int64_t> perm(nnz);
    for(int64_t i = 0; i < nnz; i++)
        perm[i] = i;
    std::sort(perm.begin(), perm.end(), [&](int64_t i, int64_t j) {
        if(row_ind[i] != row_ind[j])
            return row_ind[i] < row_ind[j];
        return col_ind[i] < col_ind[j];
    });
    std::vector<int32_t> row_tmp(row_ind.begin(), row_ind.end());
    std::vector<int32_t> col_tmp(col_ind.begin(), col_ind.end());
    std::vector<double>   val_tmp(val.begin(), val.end());
    for(int64_t i = 0; i < nnz; i++)
    {
        row_ind[i] = row_tmp[perm[i]];
        col_ind[i] = col_tmp[perm[i]];
        val[i]     = val_tmp[perm[i]];
    }

    return true;
}

// shfl
__device__ __forceinline__ double shfl(double var, int src_lane, int width = 32)
{
    return __shfl(var, src_lane, width);
}

__device__ __forceinline__ int32_t shfl(int32_t var, int src_lane, int width = 32)
{
    return __shfl(var, src_lane, width);
}

// coommnn_atomic_remainder_device kernel (BLOCKSIZE=256, WF_SIZE=16, TRANSB=true)
// C = alpha * A * B^T, B and C column-major
__device__ void coommnn_atomic_remainder_device(int32_t              m,
                                                int32_t              n,
                                                int32_t              k,
                                                int64_t              nnz,
                                                const int32_t* __restrict__ coo_row_ind,
                                                const int32_t* __restrict__ coo_col_ind,
                                                const double* __restrict__ coo_val,
                                                const double* __restrict__ dense_B,
                                                int64_t               ldb,
                                                double* __restrict__ dense_C,
                                                int64_t               ldc)
{
    constexpr uint32_t BLOCKSIZE = 256;
    constexpr uint32_t WF_SIZE   = 16;

    const int     tid = hipThreadIdx_x;
    const int     lid = tid & (WF_SIZE - 1);
    const int     wid = tid / WF_SIZE;
    const int64_t gid = BLOCKSIZE * hipBlockIdx_x + tid;

    __shared__ int32_t shared_row[(BLOCKSIZE / WF_SIZE) * WF_SIZE];
    __shared__ double  shared_val[(BLOCKSIZE / WF_SIZE) * WF_SIZE];

    const int32_t row = (gid < nnz) ? coo_row_ind[gid] : -1;
    const int32_t col = (gid < nnz) ? coo_col_ind[gid] : 0;
    const double   val = (gid < nnz) ? coo_val[gid] : 0.0;

    for(int32_t l = 0; l < n; l += WF_SIZE)
    {
        const int32_t colB = l + lid;

        double         sum         = 0.0;
        int32_t       current_row = shfl(row, 0, WF_SIZE);

        for(uint32_t i = 0; i < WF_SIZE; ++i)
        {
            double   v = shfl(val, i, WF_SIZE);
            int32_t c = shfl(col, i, WF_SIZE);
            int32_t r = shfl(row, i, WF_SIZE);

            if(r != current_row)
            {
                if(colB < n)
                {
                    assert((colB * ldc + current_row) >= 0);
                    assert((colB * ldc + current_row) < (ldc * n));
                    atomicAdd(&dense_C[colB * ldc + current_row], sum);
                }
                sum         = 0.0;
                current_row = r;
            }

            if(colB < n)
            {
                assert((c * ldb + colB) >= 0);
                assert((c * ldb + colB) < ldb * k);
                sum = v * dense_B[c * ldb + colB] + sum;
            }
        }

        __syncthreads();
        assert(((BLOCKSIZE / WF_SIZE) * lid + wid) >= 0);
        assert(((BLOCKSIZE / WF_SIZE) * lid + wid) < BLOCKSIZE);
        shared_row[(BLOCKSIZE / WF_SIZE) * lid + wid] = current_row;
        shared_val[(BLOCKSIZE / WF_SIZE) * lid + wid]  = sum;
        __syncthreads();

        current_row = shared_row[tid];
        sum         = shared_val[tid];

        const int slid = tid & ((BLOCKSIZE / WF_SIZE) - 1);
        const int swid = tid / (BLOCKSIZE / WF_SIZE);

        for(uint32_t j = 1; j < (BLOCKSIZE / WF_SIZE); j <<= 1)
        {
            if(slid >= j)
            {
                assert(((BLOCKSIZE / WF_SIZE) * swid + slid - j) >= 0);
                assert(((BLOCKSIZE / WF_SIZE) * swid + slid - j) < BLOCKSIZE);
                if(current_row == shared_row[(BLOCKSIZE / WF_SIZE) * swid + slid - j])
                {
                    sum = sum + shared_val[(BLOCKSIZE / WF_SIZE) * swid + slid - j];
                }
            }
            __syncthreads();
            assert(((BLOCKSIZE / WF_SIZE) * swid + slid) >= 0);
            assert(((BLOCKSIZE / WF_SIZE) * swid + slid) < BLOCKSIZE);
            shared_val[(BLOCKSIZE / WF_SIZE) * swid + slid] = sum;
            __syncthreads();
        }

        if(slid < ((BLOCKSIZE / WF_SIZE) - 1))
        {
            assert(((BLOCKSIZE / WF_SIZE) * swid + slid + 1) >= 0);
            assert(((BLOCKSIZE / WF_SIZE) * swid + slid + 1) < BLOCKSIZE);
            if(current_row != shared_row[(BLOCKSIZE / WF_SIZE) * swid + slid + 1]
               && current_row >= 0)
            {
                if((l + swid) < n)
                {
                    assert(((l + swid) * ldc + current_row) >= 0);
                    assert(((l + swid) * ldc + current_row) < (ldc * n));
                    atomicAdd(&dense_C[(l + swid) * ldc + current_row], sum);
                }
            }
        }

        if(slid == ((BLOCKSIZE / WF_SIZE) - 1))
        {
            if(current_row >= 0)
            {
                if((l + swid) < n)
                {
                    assert(((l + swid) * ldc + current_row) >= 0);
                    assert(((l + swid) * ldc + current_row) < (ldc * n));
                    atomicAdd(&dense_C[(l + swid) * ldc + current_row], sum);
                }
            }
        }
    }
}

// Kernel launcher
__launch_bounds__(256)
__global__ void coommnn_atomic_remainder_kernel(int32_t              m,
                                                int32_t              n,
                                                int32_t              k,
                                                int64_t              nnz,
                                                int64_t              batch_stride_A,
                                                const int32_t* __restrict__ coo_row_ind,
                                                const int32_t* __restrict__ coo_col_ind,
                                                const double* __restrict__ coo_val,
                                                const double* __restrict__ dense_B,
                                                int64_t               ldb,
                                                int64_t               batch_stride_B,
                                                double* __restrict__ dense_C,
                                                int64_t               ldc,
                                                int64_t               batch_stride_C)
{
    coommnn_atomic_remainder_device(m,
                                    n,
                                    k,
                                    nnz,
                                    &coo_row_ind[batch_stride_A * hipBlockIdx_y],
                                    &coo_col_ind[batch_stride_A * hipBlockIdx_y],
                                    &coo_val[batch_stride_A * hipBlockIdx_y],
                                    &dense_B[batch_stride_B * hipBlockIdx_y],
                                    ldb,
                                    &dense_C[batch_stride_C * hipBlockIdx_y],
                                    ldc);
}

template <typename I, typename A, typename B, typename C, typename T>
void testing_spmm_batched_coo(const Arguments& arg)
{
    HIP_CHECK(hipDeviceReset());

    int M               = 1;
    int N               = 15;
    int K               = 1;
    int ld_multiplier_B = 2;
    int ld_multiplier_C = 2;

    int batch_count_A = 1;
    int batch_count_B = 10;
    int batch_count_C = 10;

    std::cout << "M: " << M << " N: " << N << " K: " << K << " ld_multiplier_B: " << ld_multiplier_B << " ld_multiplier_C: " << ld_multiplier_C << std::endl;
    std::cout << "batch_count_A: " << batch_count_A << " batch_count_B: " << batch_count_B << " batch_count_C: " << batch_count_C << std::endl;

    // Allocate host memory for matrix
    std::vector<int> hcoo_row_ind_temp;
    std::vector<int> hcoo_col_ind_temp;
    std::vector<double> hcoo_val_temp;

    int64_t nnz_A;
    if(!read_mtx("nos3/nos3.mtx", M, K, nnz_A, hcoo_row_ind_temp, hcoo_col_ind_temp, hcoo_val_temp))
    {
        std::cerr << "Failed to read " << "nos3/nos3.mtx" << std::endl;
        std::cerr << "Download from: https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/nos3.mtx.gz"
                << std::endl;
        return;
    }

    // Redefine values
    for(size_t i = 0; i < hcoo_val_temp.size(); i++)
    {
        hcoo_val_temp[i] = static_cast<double>(1);
    }

    // Some matrix properties
    int B_m = N;
    int B_n = K;
    int C_m = M;
    int C_n = N;

    int64_t ldb = int64_t(ld_multiplier_B) * N;
    int64_t ldc = int64_t(ld_multiplier_C) * M;

    int64_t nrowB = ldb;
    int64_t ncolB = B_n;
    int64_t nrowC = ldc;
    int64_t ncolC = C_n;

    int64_t nnz_B = nrowB * ncolB;
    int64_t nnz_C = nrowC * ncolC;

    int64_t batch_stride_A = 0;
    int64_t batch_stride_B = nnz_B;
    int64_t batch_stride_C = nnz_C;

    // Allocate host memory for all batches of A matrix
    std::vector<int> hcoo_row_ind(batch_count_A * nnz_A);
    std::vector<int> hcoo_col_ind(batch_count_A * nnz_A);
    std::vector<double> hcoo_val(batch_count_A * nnz_A);

    for(int i = 0; i < batch_count_A; i++)
    {
        for(int64_t j = 0; j < nnz_A; j++)
        {
            hcoo_row_ind[nnz_A * i + j] = hcoo_row_ind_temp[j];
            hcoo_col_ind[nnz_A * i + j] = hcoo_col_ind_temp[j];
            hcoo_val[nnz_A * i + j]     = hcoo_val_temp[j];
        }
    }

    // Allocate host memory for vectors
    std::vector<double> hB(batch_count_B * nnz_B, 1);
    std::vector<double> hC_1(batch_count_C * nnz_C, 0);

    // Allocate device memory
    int*    dcoo_row_ind = nullptr;
    int*    dcoo_col_ind = nullptr;
    double* dcoo_val     = nullptr;
    double* dB           = nullptr;
    double* dC_1         = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcoo_row_ind, sizeof(int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcoo_col_ind, sizeof(int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcoo_val, sizeof(double) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(double) * batch_count_B * nnz_B));
    HIP_CHECK(hipMalloc((void**)&dC_1, sizeof(double) * batch_count_C * nnz_C));

    HIP_CHECK(hipMemcpy(dcoo_row_ind, hcoo_row_ind.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcoo_col_ind, hcoo_col_ind.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcoo_val, hcoo_val.data(), sizeof(double) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(double) * batch_count_B * nnz_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC_1, hC_1.data(), sizeof(double) * batch_count_C * nnz_C, hipMemcpyHostToDevice));

    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl;
    const int     nblocks_x   = (nnz_A - 1) / 256 + 1;
    const int     nblocks_y   = batch_count_C;
    hipLaunchKernelGGL(coommnn_atomic_remainder_kernel,
        dim3(nblocks_x, nblocks_y),
        dim3(256),
        0,
        0,
        M,
        N,
        K,
        nnz_A,
        batch_stride_A,
        dcoo_row_ind,
        dcoo_col_ind,
        dcoo_val,
        dB,
        ldb,
        batch_stride_B,
        dC_1,
        ldc,
        batch_stride_C);
    std::cout << "BBBBBBBBBBBBBBBBBBBBBBBBBBBBB" << std::endl;

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC" << std::endl;

    HIP_CHECK(hipFree(dcoo_row_ind));
    HIP_CHECK(hipFree(dcoo_col_ind));
    HIP_CHECK(hipFree(dcoo_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC_1));
    std::cout << "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD" << std::endl;
}

#define INSTANTIATE(ITYPE, TTYPE)                                                      \
    template void testing_spmm_batched_coo_bad_arg<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                         \
    template void testing_spmm_batched_coo<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)
#define INSTANTIATE_MIXED(ITYPE, ATYPE, XTYPE, YTYPE, TTYPE)                           \
    template void testing_spmm_batched_coo_bad_arg<ITYPE, ATYPE, XTYPE, YTYPE, TTYPE>( \
        const Arguments& arg);                                                         \
    template void testing_spmm_batched_coo<ITYPE, ATYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

INSTANTIATE_MIXED(int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_MIXED(int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_MIXED(int32_t, int8_t, int8_t, float, float);
INSTANTIATE_MIXED(int64_t, int8_t, int8_t, float, float);
INSTANTIATE_MIXED(int32_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int64_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);

void testing_spmm_batched_coo_extra(const Arguments& arg) {}
