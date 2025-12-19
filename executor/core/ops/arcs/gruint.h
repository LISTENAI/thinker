#ifndef __GRUINT_H__
#define __GRUINT_H__

#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_matrix_math.h"
#define API_LIB(api) luna_##api
#endif

#define ALIGN(X, T) ((((X) + (T)-1) / (T)) * (T))
#define FAST_MALLOC(buf, size, nsize)              \
    ((buf) = (int8_t *)(buf) + (uint32_t)(size),     \
    (nsize) = (uint32_t)(nsize) + (uint32_t)(size), \
    (void *)((int8_t *)(buf) - (uint32_t)(size)))

/**
 * @brief Calculates the number of splits for matrix multiplication
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @param col2 Number of columns in the second matrix
 * @return int32_t Number of splits
 */
int32_t calc_mat_mul_split_num(int32_t row, int32_t col, int32_t col2) {
    int32_t split = 0;
    int32_t split_size = 0;
    do {
        split++;
        int32_t split_col = col / split;
        split_size = ALIGN(split_col, 8) * ALIGN(col2, 4);
    } while (col % split != 0 || split_size > 32 * 1024);
    return split;
}

/**
 * @brief GRU operation parameters
 */
typedef struct _bigru_param {
    int32_t go_forward;    // Direction of processing
    int32_t hidden_size;   // Size of hidden state
    int32_t input_size;    // Size of input
    int32_t iw_size;       // Size of input-to-hidden weights
    int32_t hw_size;       // Size of hidden-to-hidden weights
    int32_t ib_size;       // Size of input bias
    int32_t hb_size;       // Size of hidden bias
    int32_t q_i;           // Input quantization scale
    int32_t q_iw;          // Input-to-hidden weight quantization scale
    int32_t q_h;           // Hidden state quantization scale
    int32_t q_hw;          // Hidden-to-hidden weight quantization scale
    int32_t q_ib;          // Input bias quantization scale
    int32_t q_hb;          // Hidden bias quantization scale
    int32_t q_o;           // Output quantization scale
    void *p_h_in;          // Pointer to hidden state input
    void *p_iw;            // Pointer to input-to-hidden weights
    void *p_hw;            // Pointer to hidden-to-hidden weights
    void *p_ib;            // Pointer to input bias
    void *p_hb;            // Pointer to hidden bias
} gru_param_t;

/**
 * @brief GRU inner computation function
 * @param params GRU parameters
 * @param t Current time step
 * @param p_input Input data pointer
 * @param p_output Output data pointer
 * @param p_tmp Temporary workspace pointer
 * @param tmp_size Size of temporary workspace
 * @return int32_t Operation status
 */
int32_t gru_luna_inner(gru_param_t *params, int32_t t, int8_t *p_input, int8_t *p_output, int8_t *p_tmp, int32_t tmp_size) {
    int32_t ret = -1;
    const int32_t split_num = 4;
    const int32_t active_q_in = 11;
    const int32_t active_q_out = 7;

    gru_param_t *p_gru_param = params;
    int32_t input_size = p_gru_param->input_size;
    int32_t hidden_size = p_gru_param->hidden_size;

    int32_t iw_size = p_gru_param->iw_size;
    int32_t hw_size = p_gru_param->hw_size;
    int32_t ib_size = p_gru_param->ib_size;
    int32_t hb_size = p_gru_param->hb_size;

    int8_t *p_in = (int8_t *)p_input;
    int8_t *p_out = (int8_t *)p_output;
    int8_t *p_h_in = (int8_t *)p_gru_param->p_h_in;
    int8_t *p_iw_weight = (int8_t *)p_gru_param->p_iw;
    int8_t *p_hw_weight = (int8_t *)p_gru_param->p_hw;
    int32_t *p_ib_bias = (int32_t *)p_gru_param->p_ib;
    int32_t *p_hb_bias = (int32_t *)p_gru_param->p_hb;

    int32_t i_q = p_gru_param->q_i;
    int32_t h_q = p_gru_param->q_h;
    int32_t iw_q = p_gru_param->q_iw;
    int32_t hw_q = p_gru_param->q_hw;
    int32_t ib_q = p_gru_param->q_ib;
    int32_t hb_q = p_gru_param->q_hb;
    int32_t o_q = p_gru_param->q_o;

    int32_t *p_out1 = (int32_t *)p_tmp;
    ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(p_iw_weight, p_in, p_ib_bias, p_out1, hidden_size * 3, input_size, 1, 0);
    if (active_q_in > ib_q) {
        ret = API_LIB(scale_i32i32o32)(p_out1, 1 << (active_q_in - ib_q), p_out1, hidden_size * 3, 0);
    } else {
        ret = API_LIB(scale_i32i32o32)(p_out1, 1, p_out1, hidden_size * 3, (ib_q - active_q_in));
    }

    int32_t *p_out2 = (int32_t *)p_tmp + hidden_size * 3;
    ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(p_hw_weight, p_h_in, p_hb_bias, p_out2, hidden_size * 3, hidden_size, 1, 0);
    if (active_q_in > hb_q) {
        ret = API_LIB(scale_i32i32o32)(p_out2, 1 << (active_q_in - hb_q), p_out2, hidden_size * 3, 0);
    } else {
        ret = API_LIB(scale_i32i32o32)(p_out2, 1, p_out2, hidden_size * 3, (hb_q - active_q_in));
    }

    int32_t *i_n = (int32_t *)p_out1 + hidden_size * 2;
    int32_t *h_n = (int32_t *)p_out2 + hidden_size * 2;
    ret = API_LIB(add_i32i32o32)((const int32_t *)p_out1, (int32_t *)p_out2, (int32_t *)p_out1, hidden_size * 2, 0);
    int32_t *G_r = (int32_t *)p_out1;
    int32_t *G_z = (int32_t *)p_out1 + hidden_size;
    int32_t *G_n = (int32_t *)p_out1 + hidden_size * 2;
    int8_t *g_r = (int8_t *)p_out2;
    int8_t *g_z = (int8_t *)p_out2 + hidden_size;
    int8_t *g_n = (int8_t *)p_out2 + hidden_size * 2;

    ret = API_LIB(sigmoid_i32o8)(G_r, g_r, hidden_size);
    ret = API_LIB(sigmoid_i32o8)(G_z, g_z, hidden_size);

    ret = API_LIB(scale_i8i8o32)(g_r, 1, (int32_t *)G_r, hidden_size, 0);
    ret = API_LIB(mul_i32i32o32)((int32_t *)G_r, h_n, (int32_t *)G_r, hidden_size, active_q_out);
    ret = API_LIB(add_i32i32o32)(i_n, (int32_t *)G_r, G_n, hidden_size, 0);
    ret = API_LIB(tanh_i32o8)(G_n, g_n, hidden_size);

    ret = API_LIB(scale_i8i8o8)(p_h_in, 1, p_h_in, hidden_size, 1);
    ret = API_LIB(mul_i8i8o32)(g_z, p_h_in, G_r, hidden_size, 0);
    ret = API_LIB(scale_i8i8o32)(g_z, (0 - 1), G_z, hidden_size, 0);
    ret = API_LIB(offset_i32i32o32)(G_z, 128, G_z, hidden_size, 0);
    ret = API_LIB(scale_i8i8o32)(g_n, 1, G_n, hidden_size, 0);
    ret = API_LIB(mul_i32i32o32)(G_z, G_n, G_n, hidden_size, 0);
    ret = API_LIB(add_i32i32o8)(G_r, G_n, p_h_in, hidden_size, active_q_out + active_q_out - o_q);

    ret = API_LIB(memcpy_i8o8)(p_out, p_h_in, hidden_size);

    return ret;
}

/**
 * @brief GRU operation implementation
 * @param input Input tensor
 * @param history_h History hidden tensor
 * @param i2h_w Input-to-hidden weight tensor
 * @param h2h_w Hidden-to-hidden weight tensor
 * @param i2h_bias Input bias tensor
 * @param h2h_bias Hidden bias tensor
 * @param mask Mask tensor
 * @param output Output tensor
 * @param hidden_o Hidden output tensor
 * @param params GRU operation attributes
 * @param workspace Workspace tensor
 * @return int32_t Operation status
 */
int32_t gruint_luna(tTensor *input, tTensor *history_h, tTensor *i2h_w, tTensor *h2h_w, tTensor *i2h_bias, tTensor *h2h_bias,
                    tTensor *mask, tTensor *output, tTensor *hidden_o, GRUIntAttrs *params, tTensor *workspace) {
    int32_t ret = -1;
    if (input->dtype_ != Int8) {
        return T_ERR_INVALID_DATATYPE;
    }

    int32_t seq_len = 0, batch_size = 0;
    if (params->layout == 0) {
        seq_len = input->shape_.dims_[0];
        batch_size = input->shape_.dims_[1];
    } else {
        seq_len = input->shape_.dims_[1];
        batch_size = input->shape_.dims_[0];
    }

    gru_param_t gru_param = {0};
    gru_param.go_forward = (params->direction) ^ 1;
    gru_param.input_size = params->input_size;
    gru_param.hidden_size = params->hidden_size;
    gru_param.iw_size = getTensorSize(i2h_w);
    gru_param.hw_size = getTensorSize(h2h_w);
    gru_param.ib_size = getTensorSize(i2h_bias);
    gru_param.hb_size = getTensorSize(h2h_bias);
    gru_param.q_i = (int32_t)input->scale_;
    gru_param.q_h = (int32_t)hidden_o->scale_;
    gru_param.q_iw = (int32_t)i2h_w->scale_;
    gru_param.q_hw = (int32_t)h2h_w->scale_;
    gru_param.q_ib = gru_param.q_i + gru_param.q_iw;
    gru_param.q_hb = gru_param.q_h + gru_param.q_hw;
    gru_param.q_o = (int32_t)output->scale_;
    gru_param.p_h_in = (void *)hidden_o->dptr_;
    gru_param.p_iw = (void *)i2h_w->dptr_;
    gru_param.p_hw = (void *)h2h_w->dptr_;
    gru_param.p_ib = (void *)i2h_bias->dptr_;
    gru_param.p_hb = (void *)h2h_bias->dptr_;

    int32_t go_forward = gru_param.go_forward;
    int32_t step_size = gru_param.input_size;
    int32_t out_step_size = gru_param.hidden_size;
    int8_t *p_input = (int8_t *)input->dptr_;
    int8_t *p_out = (int8_t *)output->dptr_;
    int8_t *p_tmp = (int8_t *)workspace->dptr_;
    int32_t tmp_size = getTensorSize(workspace) * workspace->byte_;

    int32_t t = 0;
    memset(gru_param.p_h_in, 0, gru_param.hidden_size * hidden_o->byte_);
    if (go_forward == 1) {
        for (t = 0; t < seq_len; t++) {
            ret = gru_luna_inner(&gru_param, t, p_input + step_size * t,
                                p_out + out_step_size * t, p_tmp, tmp_size);
        }
    } else {
        for (t = seq_len - 1; t >= 0; t--) {
            ret = gru_luna_inner(&gru_param, seq_len - t - 1, p_input + step_size * t,
                                p_out + out_step_size * t, p_tmp, tmp_size);
        }
    }

    return ret;
}

#endif  // __GRUINT_H__