#ifndef _LOGSOFTMAXINT_LUNA_H_
#define _LOGSOFTMAXINT_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/* Logarithm table for base 2 and natural logarithm */
static const int32_t log2_table[] = {
	-2135405020, 1543054448, -2111388125, 1531138969,
	-2087555976, 1519406104, -2063905754, 1507851685,
	-2040434703, 1496471672, -2017140127, 1485262146,
	-1994019392, 1474219305, -1971069924, 1463339458,
	-1948289204, 1452619022, -1925674768, 1442054520,
	-1903224206, 1431642574, -1880935160, 1421379903,
	-1858805323, 1411263320, -1836832436, 1401289728,
	-1815014290, 1391456116, -1793348719, 1381759558,
	-1771833605, 1372197208, -1750466872, 1362766299,
	-1729246487, 1353464140, -1708170460, 1344288112,
	-1687236840, 1335235667, -1666443715, 1326304325,
	-1645789212, 1317491671, -1625271494, 1308795357,
	-1604888763, 1300213092, -1584639253, 1291742648,
	-1564521234, 1283381855, -1544533010, 1275128595,
	-1524672916, 1266980808, -1504939321, 1258936486,
	-1485330622, 1250993669, -1465845250, 1243150448,
	-1446481662, 1235404963, -1427238345, 1227755396,
	-1408113815, 1220199979, -1389106614, 1212736982,
	-1370215312, 1205364720, -1351438503, 1198081550,
	-1332774808, 1190885865, -1314222872, 1183776099,
	-1295781365, 1176750721, -1277448980, 1169808239,
	-1259224434, 1162947194, -1241106464, 1156166160,
	-1223093832, 1149463748, -1205185319, 1142838597,
	-1187379730, 1136289378, -1169675887, 1129814795,
	-1152072635, 1123413578, -1134568837, 1117084487,
	-1117163376, 1110826311, -1099855152, 1104637864,
	-1082643085, 1098517986, -1065526114, 1092465546,
	-1048503192, 1086479433, -1031573292, 1080558564,
	-1014735402, 1074701878, -997988529,  1068908337,
	-981331693,  1063176925, -964763932,  1057506648,
	-948284297,  1051896533, -931891857,  1046345628,
	-915585693,  1040853000, -899364902,  1035417736,
	-883228594,  1030038943, -867175895,  1024715744,
	-851205942,  1019447283, -835317887,  1014232719,
	-819510893,  1009071229, -803784139,  1003962008,
	-788136812,  998904264,  -772568116,  993897226,
	-757077264,  988940132,  -741663481,  984032241,
	-726326004,  979172822,  -711064081,  974361162,
	-695876973,  969596560,  -680763947,  964878328,
	-665724287,  960205794,  -650757282,  955578296,
	-635862234,  950995187,  -621038454,  946455831,
	-606285264,  941959603,  -591601995,  937505894,
	-576987987,  933094101,  -562442589,  928723637,
	-547965161,  924393923,  -533555070,  920104392,
	-519211692,  915854487,  -504934414,  911643662,
	-490722627,  907471380,  -476575735,  903337114,
	-462493147,  899240347,  -448474281,  895180571,
	-434518564,  891157288,  -420625429,  887170007,
	-406794316,  883218247,  -393024676,  879301537,
	-379315963,  875419411,  -365667642,  871571413,
	-352079181,  867757096,  -338550059,  863976020,
	-325079760,  860227750,  -311667773,  856511864,
	-298313597,  852827942,  -285016736,  849175574,
	-271776698,  845554356,  -258593002,  841963892,
	-245465169,  838403791,  -232392727,  834873669,
	-219375212,  831373151,  -206412164,  827901864,
	-193503129,  824459445,  -180647658,  821045534,
	-167845309,  817659779,  -155095645,  814301834,
	-142398234,  810971356,  -129752649,  807668010,
	-117158469,  804391466,  -104615278,  801141400,
	-92122664,   797917491,  -79680221,   794719425,
	-67287549,   791546892,  -54944250,   788399588,
	-42649932,   785277214,  -30404208,   782179473,
	-18206696,   779106077,  -6057018,    776056738
};

/**
 * @brief Calculate the number of leading zeros in a 32-bit integer
 * @param x Input integer
 * @return Number of leading zeros
 */
static int32_t nsa(int32_t x) {
    uint32_t ux = x < 0 ? -x : x;
    if (ux == 0x80000000) return 0;
    ux &= 0x7FFFFFFF;
    int32_t ix = 0;
    while (!(ux & 0x40000000) && ix < 31) {
        ux <<= 1;
        ix++;
    }
    return ix;
}

/**
 * @brief Determine the sign of a 64-bit integer
 * @param x Input integer
 * @return -1 for negative, 0 for zero, 1 for positive
 */
static int32_t sign_int64(int64_t x) {
    int32_t s = x < 0 ? -1 : 1;
    return x == 0 ? 0 : s;
}

/**
 * @brief Saturated multiplication for 32-bit fixed-point numbers
 * @param z First operand (Q1.63)
 * @param x Second operand (Q1.31)
 * @param y Third operand (Q1.31)
 * @return Result (Q1.63)
 */
static int64_t mula_32_f63(int64_t z, int32_t x, int32_t y) {
    int64_t s = (int64_t)x * y;
    int64_t s0[2] = {s, s < 0 ? -1 : 0};
    int64_t s1[2] = {z, z < 0 ? -1 : 0};
    
    s0[0] <<= 1;
    s0[1] = (s0[1] << 1) | ((s0[0] >> 63) & 1);
    
    int64_t s2[2] = {s0[0] + s1[0], s1[1] + (s0[0] < 0 && s1[0] < 0 ? 1 : 0)};
    int32_t overflow = 0;
    if (sign_int64(s0[0]) * sign_int64(s1[0]) > 0 && sign_int64(s0[0]) * sign_int64(s2[0]) < 0) {
        overflow = 1;
    }
    s2[1] += overflow;
    
    int64_t s3 = s2[0];
    if ((s3 > 0 && s2[1] > 0) || (s3 > 0 && s2[1] == 0 && s3 > 0x7FFFFFFFFFFFFFFF)) {
        s3 = 0x7FFFFFFFFFFFFFFF;
    }
    if ((s3 < 0 && s2[1] < -1) || (s3 < 0 && s2[1] == -1 && s3 < 0x8000000000000000)) {
        s3 = 0x8000000000000000;
    }
    return s3;
}

/**
 * @brief Saturate a 64-bit integer to 32-bit range
 * @param x Input integer
 * @return Saturated 32-bit integer
 */
static int32_t sat32(int64_t x) {
    if (x < 0x80000000) return x;
    if (x > 0x7FFFFFFF) return 0x7FFFFFFF;
    return (int32_t)x;
}

/**
 * @brief Vector logarithm function for 32-bit fixed-point numbers
 * @param Y Output array
 * @param X Input array
 * @param N Length of arrays
 */
static void vec_logn_32x32_sim(int32_t *Y, const int32_t *X, int N) {
    const int32_t min_int32 = 0x80000000;
    const int32_t hx = 1 << 30;  // Q1.30
    const int32_t sx = 1 << 22;  // 0.5 in Q23
    const int32_t mx = (1 << 23) - 1;
    const int32_t ln_2 = 0x58B90BFC;  // Q31
    
    for (int i = 0; i < N; i++) {
        int32_t x = X[i];
        if (x <= 0) {
            Y[i] = min_int32;
            continue;
        }
        
        int32_t x_nsa = nsa(x);
        x = (x * (1 << x_nsa)) - hx;
        int32_t dx = (x & mx) - sx;
        dx <<= 2;
        int32_t offset = (x >> 23) << 3;
        
        int32_t log2_x0 = log2_table[offset];
        int64_t yf = (int64_t)log2_x0 << 32;
        yf = mula_32_f63(yf, log2_table[offset + 1], dx);
        
        int32_t xf = (int32_t)(yf >> 38);
        int32_t nx = (16 - x_nsa) << 25;
        int32_t yx = xf + nx;
        
        int64_t yx_tmp = (int64_t)yx * ln_2;
        yx_tmp = round(yx_tmp * pow(2, -31));
        Y[i] = sat32(yx_tmp);
    }
}

static int32_t logsoftmaxint_dtype_size(uint16_t dtype) {
    if (dtype == Int8) return 1;
    if (dtype == Int16) return 2;
    if (dtype == Int32) return 4;
    return 0;
}

static int32_t logsoftmaxint_scale_i32_to_q25(int32_t *src, int32_t size, int32_t shift) {
    int32_t scalar = 1;
    uint32_t rshift = 0;
    if (shift >= 0) {
#if THINKER_PARAM_CHECK
if (shift > 30) {
    return (T_ERR_INVALID_PARA);
}
#endif
        scalar = 1 << (uint32_t)shift;
    } else {
#if THINKER_PARAM_CHECK
if ((-shift) > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif
        rshift = (uint32_t)(-shift);
    }
    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src, scalar, src, size, rshift),
                      "luna_scale_i32i32o32");
    return T_SUCCESS;
}

static int32_t logsoftmaxint_scale_i32_to_dtype(const int32_t *src, void *dst,
                                                uint16_t dtype, int32_t size,
                                                int32_t shift) {
    int32_t scalar = 1;
    uint32_t rshift = 0;
    if (shift >= 0) {
#if THINKER_PARAM_CHECK
if (shift > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif
        rshift = (uint32_t)shift;
    } else {
#if THINKER_PARAM_CHECK
if ((-shift) > 30) {
    return (T_ERR_INVALID_PARA);
}
#endif
        scalar = 1 << (uint32_t)(-shift);
    }

    if (dtype == Int8) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(src, scalar, (int8_t *)dst, size, rshift),
                          "luna_scale_i32i32o8");
    } else if (dtype == Int16) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o16)(src, scalar, (int16_t *)dst, size, rshift),
                          "luna_scale_i32i32o16");
    } else if (dtype == Int32) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src, scalar, (int32_t *)dst, size, rshift),
                          "luna_scale_i32i32o32");
    } else {
        return T_ERR_INVALID_DATATYPE;
    }
    return T_SUCCESS;
}

static int32_t logsoftmaxint_workspace_bytes(uint16_t dtype, int32_t rows,
                                             int32_t stride) {
    int32_t cur_size = rows * stride;
    if (dtype == Int8) {
        return ALIGN4(cur_size * (int32_t)sizeof(int16_t)) +
               cur_size * (int32_t)sizeof(int32_t) * 2;
    }
    return cur_size * (int32_t)sizeof(int32_t) * 2;
}

static int32_t logsoftmaxint_split_leading(uint16_t dtype, int32_t leading,
                                           int32_t stride,
                                           int32_t workspace_size) {
    int32_t per_row = stride * (dtype == Int8 ? 10 : 8);
    int32_t split = per_row > 0 ? workspace_size / per_row : 0;
    if (split > leading) split = leading;
    while (split > 0 &&
           logsoftmaxint_workspace_bytes(dtype, split, stride) > workspace_size) {
        split--;
    }
    return split;
}

/**
 * @brief LogSoftmax function for integer tensors
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Workspace buffer
 * @param attrs Operation attributes
 * @return Execution status
 */
int32_t logsoftmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, LogSoftmaxIntAttrs *attrs) {
    const int32_t SOFTMAX_Q_IN = 25;
    const int32_t SOFTMAX_Q_OUT = 15;

    if (data == NULL || out == NULL || Workspace == NULL || attrs == NULL ||
        data->dptr_ == 0 || out->dptr_ == 0 || Workspace->dptr_ == 0)
        return T_ERR_INVALID_PARA;

    int32_t axis = attrs->axis < 0 ? ((int32_t)data->shape_.ndim_ + attrs->axis) : attrs->axis;
#if THINKER_PARAM_CHECK
if (axis < 0 || axis >= (int32_t)data->shape_.ndim_) {
    return (T_ERR_INVALID_PARA);
}

if (axis != (int32_t)data->shape_.ndim_ - 1) {
    return (T_ERR_NO_IMPLEMENTED);
}
#endif

    int32_t leading = 1;
    int32_t stride = 1;
    for (int32_t i = 0; i < axis; ++i) leading *= data->shape_.dims_[i];
    for (int32_t i = axis; i < (int32_t)data->shape_.ndim_; ++i) stride *= data->shape_.dims_[i];
#if THINKER_PARAM_CHECK
if (stride <= 0 || stride > 2048) {
    return (T_ERR_INVALID_PARA);
}

if (logsoftmaxint_dtype_size(data->dtype_) == 0 ||
        logsoftmaxint_dtype_size(out->dtype_) == 0) {
    return (T_ERR_INVALID_DATATYPE);
}
#endif
#if THINKER_RUNTIME_CHECK
if (data->mem_.type_ != 2 || out->mem_.type_ != 2) {
    return (T_ERR_INVALID_PLATFROM);
}
#endif

    size_t workspace_bytes = getTensorDataSize(Workspace);
    #if THINKER_RUNTIME_CHECK
    if (workspace_bytes > INT32_MAX) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    int32_t workspace_size = (int32_t)workspace_bytes;
    int32_t split_leading = logsoftmaxint_split_leading(data->dtype_, leading, stride,
                                                        workspace_size);
    if (split_leading <= 0) {
        return T_ERR_NO_WORKSPACE;
    }

    int32_t input_shift = SOFTMAX_Q_IN - (int32_t)data->scale_;
    int32_t output_shift = SOFTMAX_Q_OUT - (int32_t)out->scale_;
    int32_t out_bytes = logsoftmaxint_dtype_size(out->dtype_);
    int32_t paste_leading = 0;

    while (paste_leading < leading) {
        int32_t cur_leading = leading - paste_leading;
        if (cur_leading > split_leading) cur_leading = split_leading;
        int32_t cur_size = cur_leading * stride;
        int8_t *workspace = (int8_t *)Workspace->dptr_;
        int32_t *src_i32 = NULL;
        int32_t *dst_tmp = NULL;

        if (data->dtype_ == Int8) {
            int16_t *src_i16 = (int16_t *)workspace;
            src_i32 = (int32_t *)(workspace + ALIGN4(cur_size * (int32_t)sizeof(int16_t)));
            dst_tmp = src_i32 + cur_size;
            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)((int8_t *)data->dptr_ + paste_leading * stride,
                                                     1, src_i16, cur_size, 0),
                              "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(src_i16, 1, src_i32, cur_size, 0),
                              "luna_scale_i16i16o32");
        } else if (data->dtype_ == Int16) {
            src_i32 = (int32_t *)workspace;
            dst_tmp = src_i32 + cur_size;
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)((int16_t *)data->dptr_ + paste_leading * stride,
                                                       1, src_i32, cur_size, 0),
                              "luna_scale_i16i16o32");
        } else {
            src_i32 = (int32_t *)workspace;
            dst_tmp = src_i32 + cur_size;
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)data->dptr_ + paste_leading * stride,
                                                       1, src_i32, cur_size, 0),
                              "luna_scale_i32i32o32");
        }

        THINKER_RET_CHECK(logsoftmaxint_scale_i32_to_q25(src_i32, cur_size, input_shift),
                          "logsoftmaxint_scale_i32_to_q25");

        for (int32_t l = 0; l < cur_leading; ++l) {
            int32_t offset = l * stride;
            THINKER_RET_CHECK(API_LIB(logsoftmax_i32o32)(src_i32 + offset, dst_tmp + offset, stride),
                              "luna_logsoftmax_i32o32");
        }

        void *out_ptr = (void *)((int8_t *)out->dptr_ + paste_leading * stride * out_bytes);
        THINKER_RET_CHECK(logsoftmaxint_scale_i32_to_dtype(dst_tmp, out_ptr, out->dtype_,
                                                           cur_size, output_shift),
                          "logsoftmaxint_scale_i32_to_dtype");
        paste_leading += cur_leading;
    }

    return T_SUCCESS;
}

#endif
