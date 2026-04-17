# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Level 1 documentation example programs.
#
# Each documented level 1 routine has a standalone program that is linked
# from the Sphinx documentation (see docs/reference/level1.rst). Two variants
# are built for every routine: a C++ program (example_rocsparse_<name>_cpp)
# and a C program (example_rocsparse_<name>_c). The C and C++ sources are
# annotated with `doc example start` / `doc example end` markers used by
# the `literalinclude` directives in the .rst file.
#
# The parent clients/samples/CMakeLists.txt defines the helper function
# add_rocsparse_example() which is reused here and is inherited via
# add_subdirectory().

# C++ variants (always built)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_axpyi_cpp.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_doti_cpp.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gthr_cpp.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_roti_cpp.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_sctr_cpp.cpp)

# C variants (skipped on Windows because the HIP C toolchain on Windows is
# currently incompatible with plain C sources using the rocsparse headers).
if(NOT WIN32)
  add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_axpyi_c.c)
  add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_doti_c.c)
  add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gthr_c.c)
  add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_roti_c.c)
  add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_sctr_c.c)
endif()
