/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "internal/generic/rocsparse_spsort.h"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_coosort.hpp"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_alg value_)
{
    switch(value_)
    {
    case rocsparse_spsort_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_input value_)
{
    switch(value_)
    {
    case rocsparse_spsort_input_alg:
    case rocsparse_spsort_input_direction:
    {
        return false;
    }
    }
    return true;
};

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_stage value_)
{
    switch(value_)
    {
    case rocsparse_spsort_stage_analysis:
    case rocsparse_spsort_stage_compute:
    {
        return false;
    }
    }
    return true;
};

struct _rocsparse_spsort_descr
{
protected:
    rocsparse_spsort_stage m_stage;
    rocsparse_spsort_alg   m_alg;
    rocsparse_direction    m_dir;

public:
    ~_rocsparse_spsort_descr() {}

    _rocsparse_spsort_descr()
        : m_stage((rocsparse_spsort_stage)-1)
        , m_alg((rocsparse_spsort_alg)-1)
        , m_dir((rocsparse_direction)-1)
    {
    }

    rocsparse_spsort_stage get_stage() const
    {
        return this->m_stage;
    }
    rocsparse_spsort_alg get_alg() const
    {
        return this->m_alg;
    }
    rocsparse_direction get_dir() const
    {
        return this->m_dir;
    }

    void set_stage(rocsparse_spsort_stage value)
    {
        this->m_stage = value;
    }
    void set_alg(rocsparse_spsort_alg value)
    {
        this->m_alg = value;
    }
    void set_dir(rocsparse_direction value)
    {
        this->m_dir = value;
    }
};

extern "C" rocsparse_status rocsparse_create_spsort_descr(rocsparse_spsort_descr* descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_POINTER(0, descr);
    *descr = new _rocsparse_spsort_descr();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_destroy_spsort_descr(rocsparse_spsort_descr descr)
try
{

    ROCSPARSE_ROUTINE_TRACE;
    if(descr != nullptr)
    {
        delete descr;
    }
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spsort_set_input(rocsparse_handle       handle,
                                                       rocsparse_spsort_descr descr,
                                                       rocsparse_spsort_input input,
                                                       const void*            data,
                                                       size_t                 data_size_in_bytes,
                                                       rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, input);
    ROCSPARSE_CHECKARG_POINTER(3, data);

    switch(input)
    {
    case rocsparse_spsort_input_alg:
    {
        switch(descr->get_stage())
        {
        case rocsparse_spsort_stage_analysis:
        case rocsparse_spsort_stage_compute:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_internal_error,
                "The field 'rocsparse_spsort_input_alg' must be set before the stage "
                "'rocsparse_spsort_stage_analysis' is executed.");
        }
        }

        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_spsort_alg),
                           rocsparse_status_invalid_size);
        const rocsparse_spsort_alg alg = *reinterpret_cast<const rocsparse_spsort_alg*>(data);
        descr->set_alg(alg);
        return rocsparse_status_success;
    }
    case rocsparse_spsort_input_direction:
    {
        switch(descr->get_stage())
        {
        case rocsparse_spsort_stage_analysis:
        case rocsparse_spsort_stage_compute:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_internal_error,
                "The field 'rocsparse_spsort_input_direction' must be set before the stage "
                "'rocsparse_spsort_stage_analysis' is executed.");
        }
        }

        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_direction),
                           rocsparse_status_invalid_size);
        const rocsparse_direction dir = *reinterpret_cast<const rocsparse_direction*>(data);
        ROCSPARSE_CHECKARG(
            3, data, rocsparse::enum_utils::is_invalid(dir), rocsparse_status_invalid_value);
        descr->set_dir(dir);
        return rocsparse_status_success;
    }
        // LCOV_EXCL_START
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    // B receives the sorted A, so both must describe a matrix with the same shape and layout.
    static rocsparse_status spsort_check_matrices(rocsparse_const_spmat_descr mat_A,
                                                  rocsparse_spmat_descr       mat_B)
    {
        ROCSPARSE_CHECKARG(2, mat_A, (mat_A->batch_count != 1), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3, mat_B, (mat_B->batch_count != 1), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->format != mat_A->format), rocsparse_status_invalid_value);
        ROCSPARSE_CHECKARG(3,
                           mat_B,
                           (mat_B->rows != mat_A->rows || mat_B->cols != mat_A->cols
                            || mat_B->nnz != mat_A->nnz),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(3,
                           mat_B,
                           (mat_B->row_type != mat_A->row_type
                            || mat_B->col_type != mat_A->col_type
                            || mat_B->data_type != mat_A->data_type
                            || mat_B->idx_base != mat_A->idx_base),
                           rocsparse_status_invalid_value);

        return rocsparse_status_success;
    }

    rocsparse_status spsort_buffer_size(rocsparse_handle            handle,
                                        rocsparse_spsort_descr      descr,
                                        rocsparse_const_spmat_descr mat,
                                        rocsparse_spsort_stage      stage,
                                        size_t*                     buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_format format = mat->format;

        const rocsparse_spsort_alg alg = descr->get_alg();
        const rocsparse_direction  dir = descr->get_dir();

        switch(stage)
        {
        case rocsparse_spsort_stage_analysis:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }
            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        case rocsparse_spsort_stage_compute:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::coosort_buffer_size(handle,
                                                    rocsparse_coosort_alg_default,
                                                    dir,
                                                    mat->rows,
                                                    mat->cols,
                                                    mat->nnz,
                                                    mat->row_type,
                                                    mat->const_row_data,
                                                    mat->col_type,
                                                    mat->const_col_data,
                                                    mat->data_type,
                                                    mat->const_val_data,
                                                    buffer_size)));
                return rocsparse_status_success;
            }
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }
            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    rocsparse_status spsort(rocsparse_handle            handle,
                            rocsparse_spsort_descr      spsort_descr,
                            rocsparse_const_spmat_descr mat_A,
                            rocsparse_spmat_descr       mat_B,
                            rocsparse_spsort_stage      stage,
                            size_t                      buffer_size_in_bytes,
                            void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_format format = mat_A->format;

        const rocsparse_spsort_alg alg = spsort_descr->get_alg();
        const rocsparse_direction  dir = spsort_descr->get_dir();

        switch(stage)
        {
        case rocsparse_spsort_stage_analysis:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            {
                return rocsparse_status_success;
            }

                // LCOV_EXCL_START
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_sell:
            case rocsparse_format_coo_aos:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
            // LCOV_EXCL_STOP

        case rocsparse_spsort_stage_compute:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::coosort(handle,
                                                              rocsparse_coosort_alg_default,
                                                              dir,
                                                              mat_A->rows,
                                                              mat_A->cols,
                                                              mat_A->nnz,
                                                              mat_A->row_type,
                                                              mat_A->const_row_data,
                                                              mat_A->col_type,
                                                              mat_A->const_col_data,
                                                              mat_A->data_type,
                                                              mat_A->const_val_data,
                                                              mat_B->row_type,
                                                              mat_B->row_data,
                                                              mat_B->col_type,
                                                              mat_B->col_data,
                                                              mat_B->data_type,
                                                              mat_B->val_data,
                                                              buffer)));
                return rocsparse_status_success;
            }

            case rocsparse_format_csr:
            {
                return rocsparse_status_success;
            }
            case rocsparse_format_csc:
            {
                return rocsparse_status_success;
            }

                // LCOV_EXCL_START
            case rocsparse_format_ell:
            case rocsparse_format_coo_aos:
            case rocsparse_format_sell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_spsort_buffer_size(rocsparse_handle            handle,
                                                         rocsparse_spsort_descr      descr,
                                                         rocsparse_const_spmat_descr mat_A,
                                                         rocsparse_spmat_descr       mat_B,
                                                         rocsparse_spsort_stage      stage,
                                                         size_t*                     buffer_size,
                                                         rocsparse_error*            error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, mat_A);
    ROCSPARSE_CHECKARG_POINTER(3, mat_B);
    ROCSPARSE_CHECKARG_ENUM(4, stage);
    ROCSPARSE_CHECKARG_POINTER(5, buffer_size);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsort_check_matrices(mat_A, mat_B));

    // Validate spmv_inputs.
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_alg()),
                       rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_dir()),
                       rocsparse_status_invalid_value);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spsort_buffer_size(handle, descr, mat_A, stage, buffer_size));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spsort(rocsparse_handle            handle, //0
                                             rocsparse_spsort_descr      descr, //1
                                             rocsparse_const_spmat_descr mat_A, //2
                                             rocsparse_spmat_descr       mat_B, //3
                                             rocsparse_spsort_stage      stage, //4
                                             size_t                      buffer_size, //5
                                             void*                       temp_buffer, //6
                                             rocsparse_error*            error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, mat_A);
    ROCSPARSE_CHECKARG_POINTER(3, mat_B);
    ROCSPARSE_CHECKARG_ENUM(4, stage);
    ROCSPARSE_CHECKARG(5,
                       buffer_size,
                       (buffer_size == 0 && temp_buffer != nullptr),
                       rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(6,
                       temp_buffer,
                       (temp_buffer == nullptr && buffer_size > 0),
                       rocsparse_status_invalid_pointer);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsort_check_matrices(mat_A, mat_B));

    // Validate spmv_inputs.
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_alg()),
                       rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_dir()),
                       rocsparse_status_invalid_value);

    // Validate the stage.
    const rocsparse_spsort_stage current_stage = descr->get_stage();
    switch(stage)
    {
    case rocsparse_spsort_stage_analysis:
    {
        if(current_stage == rocsparse_spsort_stage_compute)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_spsort_stage_analysis cannot be called after "
                "the stage rocsparse_spsort_stage_compute");
        }
        else if(current_stage == rocsparse_spsort_stage_analysis)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_spsort_stage_analysis has already been "
                "executed");
        }
        break;
    }
    case rocsparse_spsort_stage_compute:
    {
        if(current_stage == ((rocsparse_spsort_stage)-1))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_spsort_stage_analysis must be executed before "
                "the stage rocsparse_spsort_stage_compute");
        }
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spsort(handle, descr, mat_A, mat_B, stage, buffer_size, temp_buffer));

    // Record the stage that has been executed.
    descr->set_stage(stage);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
