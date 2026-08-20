#undef __OP__
#define __OP__ Expand
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "thinker_status.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/utils.h"

#if THINKER_USE_ARCS || THINKER_USE_VENUSA
#include "arcs/luna/opi_psram_cpy.h"
#endif

#ifdef THINKER_USE_ARCS
#include "arcs/luna/luna_misc_math.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/luna/luna_matrix_math.h"
#include "./venusA/luna/luna_misc_math.h"
#endif

#define EXPAND_MAX_DIMS 7

static int32_t expand_checked_mul_size(size_t lhs, size_t rhs, size_t *out) {
    if (rhs != 0 && lhs > ((size_t)INT_MAX / rhs)) {
        return T_ERR_INVALID_DATA;
    }
    *out = lhs * rhs;
    return T_SUCCESS;
}

static int32_t expand_copy_bytes(uint8_t *dst, const uint8_t *src, size_t size,
                                 bool dst_in_psram, bool src_in_psram) {
    if (size == 0 || dst == src) {
        return T_SUCCESS;
    }

#if THINKER_USE_VENUSA
    if (src_in_psram) {
        return luna_psrammemcpy_i8o8((int8_t *)dst, (int8_t *)src,
                                     (uint32_t)size);
    } else if (dst_in_psram) {
        opi_psram_cpy_out(dst, (void *)src, (int32_t)size);
    } else {
        return luna_memcpy_i8o8((int8_t *)dst, (int8_t *)src, (uint32_t)size);
    }
#elif THINKER_USE_ARCS
    if (dst_in_psram && src_in_psram) {
        return luna_psrammemcpy_i8o8((int8_t *)dst, (int8_t *)src,
                                     (uint32_t)size);
    } else if (dst_in_psram) {
        opi_psram_cpy_out(dst, (void *)src, (int32_t)size);
    } else if (src_in_psram) {
        opi_psram_cpy_in(dst, (void *)src, (int32_t)size);
    } else {
        return luna_memcpy_i8o8((int8_t *)dst, (int8_t *)src, (uint32_t)size);
    }
#else
    memcpy(dst, src, size);
#endif
    return T_SUCCESS;
}

static int32_t expand_shapes_equal_from(int32_t dim, int32_t ndim,
                                        const uint32_t *in_shape,
                                        const uint32_t *out_shape) {
    for (int32_t i = dim; i < ndim; ++i) {
        if (in_shape[i] != out_shape[i]) {
            return 0;
        }
    }
    return 1;
}

static int64_t expand_shape_value(const tTensor *shape, int32_t index) {
    if (shape->dtype_ == Int32) {
        return ((const int32_t *)shape->dptr_)[index];
    }
    return ((const int64_t *)shape->dptr_)[index];
}

static int32_t expand_copy_dim(uint8_t *dst, const uint8_t *src, int32_t dim,
                               int32_t ndim, const uint32_t *in_shape,
                               const uint32_t *out_shape,
                               const size_t *in_stride_bytes,
                                const size_t *out_stride_bytes,
                                size_t elem_bytes, bool dst_in_psram,
                                bool src_in_psram) {
    if (dim == ndim) {
        return expand_copy_bytes(dst, src, elem_bytes, dst_in_psram,
                                 src_in_psram);
    }

    if (expand_shapes_equal_from(dim, ndim, in_shape, out_shape)) {
        return expand_copy_bytes(dst, src, out_stride_bytes[dim] * out_shape[dim],
                                 dst_in_psram, src_in_psram);
    }

    if (in_shape[dim] == out_shape[dim]) {
        for (uint32_t i = 0; i < out_shape[dim]; ++i) {
            int32_t ret = expand_copy_dim(dst + i * out_stride_bytes[dim],
                                          src + i * in_stride_bytes[dim],
                                          dim + 1, ndim, in_shape, out_shape,
                                           in_stride_bytes, out_stride_bytes,
                                           elem_bytes, dst_in_psram,
                                           src_in_psram);
            if (ret != T_SUCCESS) {
                return ret;
            }
        }
    } else {
        for (uint32_t i = 0; i < out_shape[dim]; ++i) {
            int32_t ret = expand_copy_dim(dst + i * out_stride_bytes[dim], src,
                                          dim + 1, ndim, in_shape, out_shape,
                                          in_stride_bytes, out_stride_bytes,
                                          elem_bytes, dst_in_psram,
                                          src_in_psram);
            if (ret != T_SUCCESS) {
                return ret;
            }
        }
    }
    return T_SUCCESS;
}

/**
 * Forward pass implementation for Expand operator
 * Expands input tensor to match target shape by repeating elements
 * @param op: Operator structure containing expansion attributes
 * @param tensors: Array containing data input, shape input, and output
 * @param num_tensor: Total number of tensors (must be 3)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;

#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (op->num_input_ != 2 || op->num_output_ != 1 ||
                        num_tensor != 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    // Get input and output tensors
    tTensor *X = (tTensor *)tensors[0];
    tTensor *Shape = (tTensor *)tensors[1];
    tTensor *Y = (tTensor *)tensors[op->num_input_];
#if THINKER_PARAM_CHECK
    if (X == NULL || Shape == NULL || Y == NULL || X->dptr_ == 0 ||
                        Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }

    if (Y->mem_.type_ != 1 && Y->mem_.type_ != 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    bool dst_in_psram = Y->mem_.type_ == 1;
    bool src_in_psram = X->mem_.type_ == 1;

    // Get shape information
    int32_t xdim = X->shape_.ndim_;
    int32_t ydim = Y->shape_.ndim_;
    const uint32_t *tShape = X->shape_.dims_;
    const uint32_t *yshape = Y->shape_.dims_;

#if THINKER_PARAM_CHECK
    if (xdim < 0 || ydim < xdim || xdim > EXPAND_MAX_DIMS ||
                        ydim > EXPAND_MAX_DIMS) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->dtype_ != Y->dtype_ || X->byte_ != Y->byte_ ||
                        X->byte_ == 0) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (X->layout_ != Y->layout_ || X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }

    if ((Shape->dtype_ != Int32 && Shape->dtype_ != Int64) ||
                        Shape->shape_.ndim_ != 1 ||
                        Shape->shape_.dims_[0] > EXPAND_MAX_DIMS ||
                        (Shape->shape_.dims_[0] > 0 && Shape->dptr_ == 0)) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t shape_dim = (int32_t)Shape->shape_.dims_[0];
#if THINKER_PARAM_CHECK
    if (ydim != (xdim > shape_dim ? xdim : shape_dim)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    for (int32_t i = 0; i < ydim; ++i) {
        int32_t x_index = i - (ydim - xdim);
        int32_t shape_index = i - (ydim - shape_dim);
        uint32_t x_value = x_index < 0 ? 1 : tShape[x_index];
        int64_t shape_value = shape_index < 0 ? 1 :
                              expand_shape_value(Shape, shape_index);
#if THINKER_PARAM_CHECK
        if (x_value == 0 || shape_value <= 0 ||
                            shape_value > UINT32_MAX) {
            return (T_ERR_INVALID_DATA);
        }
#endif
        uint32_t requested = (uint32_t)shape_value;
        uint32_t expected = x_value == 1 ? requested : x_value;
#if THINKER_PARAM_CHECK
        if (requested != 1 && x_value != 1 &&
                            requested != x_value) {
            return (T_ERR_INVALID_DATA);
        }

        if (yshape[i] != expected) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    }

    // Calculate leading dimension multiplier
    int32_t bl = ydim - xdim;
    size_t leading = 1;
    for (int32_t i = 0; i < bl; ++i) {
        if (expand_checked_mul_size(leading, yshape[i], &leading) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
    }

    const uint32_t *expandshape = yshape + bl;
    for (int32_t i = 0; i < xdim; ++i) {
        if (tShape[i] != expandshape[i]) {
            if (tShape[i] != 1 || expandshape[i] == 0) {
                return T_ERR_INVALID_DATA;
            }
        }
    }

    size_t base_elems = 1;
    size_t input_elems = 1;
    for (int32_t i = 0; i < xdim; ++i) {
        if (expand_checked_mul_size(base_elems, expandshape[i], &base_elems) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
        if (expand_checked_mul_size(input_elems, tShape[i], &input_elems) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
    }

    size_t base_bytes = 0;
    if (expand_checked_mul_size(base_elems, X->byte_, &base_bytes) != T_SUCCESS) {
        return T_ERR_INVALID_DATA;
    }

    size_t total_bytes = 0;
    if (expand_checked_mul_size(base_bytes, leading, &total_bytes) != T_SUCCESS) {
        return T_ERR_INVALID_DATA;
    }
    if (total_bytes == 0) {
        return T_SUCCESS;
    }
    size_t input_bytes = 0;
    if (expand_checked_mul_size(input_elems, X->byte_, &input_bytes) != T_SUCCESS) {
        return T_ERR_INVALID_DATA;
    }
#if THINKER_RUNTIME_CHECK
    if (getTensorDataSize(Y) != total_bytes ||
                          getTensorDataSize(X) != input_bytes) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    uint8_t *output = (uint8_t *)Y->dptr_;
    const uint8_t *input = (const uint8_t *)X->dptr_;

    if (xdim == 0) {
        THINKER_RET_CHECK(expand_copy_bytes(output, input, X->byte_, dst_in_psram,
                                            src_in_psram),
                          "expand_copy_bytes");
        for (size_t i = 1; i < leading; ++i) {
            THINKER_RET_CHECK(expand_copy_bytes(output + i * X->byte_, input,
                                                X->byte_, dst_in_psram,
                                                src_in_psram),
                              "expand_copy_bytes");
        }
        return T_SUCCESS;
    }

    size_t input_stride_bytes[EXPAND_MAX_DIMS];
    size_t output_stride_bytes[EXPAND_MAX_DIMS];
    input_stride_bytes[xdim - 1] = X->byte_;
    output_stride_bytes[xdim - 1] = X->byte_;
    for (int32_t i = xdim - 1; i > 0; --i) {
        if (expand_checked_mul_size(input_stride_bytes[i], tShape[i],
                                    &input_stride_bytes[i - 1]) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
        if (expand_checked_mul_size(output_stride_bytes[i], expandshape[i],
                                    &output_stride_bytes[i - 1]) != T_SUCCESS) {
            return T_ERR_INVALID_DATA;
        }
    }

    THINKER_RET_CHECK(expand_copy_dim(output, input, 0, xdim, tShape,
                                      expandshape, input_stride_bytes,
                                      output_stride_bytes, X->byte_,
                                      dst_in_psram, src_in_psram),
                      "expand_copy_dim");
    for (size_t i = 1; i < leading; ++i) {
        THINKER_RET_CHECK(expand_copy_dim(output + i * base_bytes, input, 0,
                                          xdim, tShape, expandshape,
                                          input_stride_bytes,
                                          output_stride_bytes, X->byte_,
                                          dst_in_psram, src_in_psram),
                          "expand_copy_dim");
    }

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
