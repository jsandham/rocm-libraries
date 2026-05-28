/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_pivot_info_t.hpp"
#include "rocsparse_utility.hpp"

#include <cstdio>

rocsparse::position_t::position_t()
    : m_position_indextype((rocsparse_indextype)-1) // set invalid value.
    , m_batch_count(1)
    , m_position(nullptr)
{
}

int64_t rocsparse::position_t::get_stride() const
{
    return 1;
}

rocsparse::position_t::~position_t()
{
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_position));
}

rocsparse_status rocsparse::position_t::free_position_async(hipStream_t stream)
{
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->m_position, stream));
    this->m_position = nullptr;
    return rocsparse_status_success;
}

void* rocsparse::position_t::get_position()
{
    return this->m_position;
}

const void* rocsparse::position_t::get_position() const
{
    return this->m_position;
}

int64_t rocsparse::position_t::get_batch_count() const
{
    return this->m_batch_count;
}

rocsparse_status rocsparse::position_t::set_max_position_async(hipStream_t stream)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_max_async(
        this->m_batch_count, this->m_position_indextype, this->m_position, stream));

    return rocsparse_status_success;
}

rocsparse_indextype rocsparse::position_t::get_indextype() const
{
    return this->m_position_indextype;
}

rocsparse_status rocsparse::position_t::create_position_async(int64_t             batch_count,
                                                              rocsparse_indextype indextype,
                                                              hipStream_t         stream)
{
    //
    // NOTE: we deliberately use the SYNCHRONOUS rocsparse_hipMalloc /
    // rocsparse_hipFree here (rather than the stream-ordered *Async
    // variants). Some callers (e.g. csrsv_solve, bsric0, bsric0_zero_pivot)
    // invoke this from inside a hipGraph capture region (see
    // clients/include/rocsparse_graph.hpp). With hipMallocAsync the
    // allocation becomes a memory-alloc node in the captured graph and the
    // pointer returned at capture time is a pool placeholder. On some
    // HIP/ROCm versions the placeholder is not correctly fixed up at graph
    // launch and downstream kernels (e.g. assign_kernel in
    // set_max_position_async, or markers2position in
    // singularity_get_position_async) fault on what is effectively an
    // unmapped virtual address. A plain hipMalloc is graph-capture safe:
    // it is not stream-ordered, not pool-backed, not recorded as a graph
    // node, and the returned pointer lives until we free it (which the
    // destructor already does via rocsparse_hipFree).
    //
    fprintf(stderr,
            "[position_t::create ENTER] this=%p cur_pos=%p cur_batch=%lld req_batch=%lld "
            "cur_idx=%d req_idx=%d stream=%p\n",
            (void*)this,
            this->m_position,
            (long long)this->m_batch_count,
            (long long)batch_count,
            (int)this->m_position_indextype,
            (int)indextype,
            (void*)stream);
    fflush(stderr);

    if((this->m_position != nullptr) && (this->m_batch_count != batch_count))
    {
        fprintf(stderr,
                "[position_t::create FREE]  this=%p freeing old m_position=%p\n",
                (void*)this,
                this->m_position);
        fflush(stderr);
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(this->m_position));
        this->m_position = nullptr;
    }

    if(this->m_position == nullptr)
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMalloc(&this->m_position, sizeof(int64_t) * batch_count));
        fprintf(stderr,
                "[position_t::create ALLOC] this=%p new m_position=%p bytes=%zu\n",
                (void*)this,
                this->m_position,
                (size_t)(sizeof(int64_t) * batch_count));
        fflush(stderr);
        if(indextype == rocsparse_indextype_i32)
        {
            RETURN_IF_HIP_ERROR(
                hipMemsetAsync(this->m_position, 0, sizeof(int64_t) * batch_count, stream));
        }
        this->m_batch_count        = batch_count;
        this->m_position_indextype = indextype;
    }
    else
    {
        fprintf(stderr,
                "[position_t::create REUSE] this=%p m_position=%p\n",
                (void*)this,
                this->m_position);
        fflush(stderr);
    }
    RETURN_IF_ROCSPARSE_ERROR(this->set_max_position_async(stream));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::position_t::copy_position_async(const position_t* that,
                                                            hipStream_t       stream)
{
    if(that->m_position != nullptr)
    {
        // m position for csrsv, csrsm, csrilu0, csric0
        const size_t J_size = rocsparse::indextype_sizeof(that->m_position_indextype);
        this->create_position_async(that->m_batch_count, this->m_position_indextype, stream);
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(this->m_position,
                                           that->m_position,
                                           J_size * this->m_batch_count,
                                           hipMemcpyDeviceToDevice,
                                           stream));
    }
    return rocsparse_status_success;
}
