/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_clients_envariables.hpp"
#include "rocsparse_data.hpp"
#include "rocsparse_parse_data.hpp"
#include "rocsparse_reproducibility.hpp"
#include "rocsparse_test_listeners.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

#include "test_check.hpp"

bool test_check::s_auto_testing_bad_arg;

bool display_timing_info_is_stdout_disabled()
{
    return false;
}

rocsparse_status rocsparse_record_output(const std::string& s)
{
    return rocsparse_status_success;
}

rocsparse_status rocsparse_record_output_legend(const std::string& s)
{
    return rocsparse_status_success;
}

rocsparse_status rocsparse_record_timing(double msec, double gflops, double gbs)
{
    return rocsparse_status_success;
}

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    std::string datapath = rocsparse_datapath();

    // Print test data path being used
    std::cout << "rocSPARSE data path: " << datapath << std::endl;

    // Set data file path
    rocsparse_parse_data(argc, argv, datapath + "rocsparse_test.data");

    // Initialize google test
    testing::InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    //test_cleanup::cleanup();

    // Run all tests
    int ret = RUN_ALL_TESTS();

    // Reset HIP device
    if(hipDeviceReset() != hipSuccess)
    {
        std::cerr << "Error: cannot reset HIP device" << std::endl;
        return rocsparse_status_internal_error;
    }

    return ret;
}
