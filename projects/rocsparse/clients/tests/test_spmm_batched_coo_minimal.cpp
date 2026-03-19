/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Minimal spmm_batched_coo test that bypasses RocSPARSE_TestData, file iteration,
 * and the full parameterized test framework. Use for debugging sporadic hangs.
 *
 * Run: rocsparse-test.exe --gtest_filter=*SpmmBatchedCooMinimal* --gtest_repeat=100
 * ************************************************************************ */

#include "rocsparse_arguments.hpp"
#include "testing_spmm_batched_coo.hpp"

#include <gtest/gtest.h>

// Single instantiation, no data file, no RocSPARSE_TestData iteration
TEST(SpmmBatchedCooMinimal, SingleRun)
{
    Arguments arg{};
    testing_spmm_batched_coo<int32_t, double, double, double, double>(arg);
}
