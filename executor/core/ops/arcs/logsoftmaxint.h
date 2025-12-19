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

// Logarithm lookup table for fixed-point computation
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
 * @brief Calculate number of leading zeros for normalization
 * @param x Input value
 * @return Number of leading zeros
 */
static int32_t nsa(int32_t x)
{
    uint32_t ux = x < 0 ? -x : x;
    if (ux == 0x80000000)
        return 0;
    ux = ux & 0x7FFFFFFF;
    int32_t ix = 0;
    while (!(ux & 0x40000000) && ix < 31)
    {
        ux = ux * 2;
        ix++;
    }
    return ix;
}

/**
 * @brief Get sign of 64-bit integer
 * @param x Input value
 * @return Sign (-1, 0, or 1)
 */
static int32_t sign_int64(int64_t x)
{
    int32_t s = x < 0 ? -1 : 1;
    if (x == 0)
        s = 0;
    return s;
}

/**
 * @brief Multiply and accumulate operation for 64-bit integers
 * @param z First operand
 * @param x Second operand
 * @param y Third operand
 * @return Result of multiplication and accumulation
 */
static int64_t mula_32_f63(int64_t z, int32_t x, int32_t y)
{
    int64_t s3 = 0;
    int64_t s = (int64_t)x * (int64_t)y; // Q1.31*Q1.31=>Q2.62
    int64_t s0[2] = { 0 };    // Q66.62
    int32_t sign_ext = s < 0 ? -1 : 0;
    s0[0] = s;
    s0[1] = (int64_t)sign_ext;

    int64_t s1[2] = { 0 };    // Q65.63
    sign_ext = z < 0 ? -1 : 0;
    s1[0] = z;
    s1[1] = (int64_t)sign_ext;

    int64_t s2[2] = { 0 };

    // s0 * 2; Q2.62 => Q.63
    s0[1] = (s0[1] << 1) | ((s0[0] & (((int64_t)1) << 63)) >> 63);
    s0[0] = s0[0] << 1;

    // s0 + s1
    s2[0] = s0[0] + s1[0];
    int32_t overflow = 0;
    if (sign_int64(s0[0]) * sign_int64(s1[0]) > 0 && sign_int64(s0[0]) * sign_int64(s2[0]) < 0)
    {
        overflow = 1;
    }

    s2[1] = s1[1] + s2[1] + overflow;
    int32_t sign = s2[0] < 0 ? -1 : 1;
    s3 = s2[0];
    if ((sign > 0 && s2[1] > 0) || (sign > 0 && s2[1] == 0 && s2[0] > 0x7FFFFFFFFFFFFFFF))
    {
        s3 = 0x7FFFFFFFFFFFFFFF;
    }
    if ((sign < 0 && s2[1] < -1) || (sign < 0 && s2[1] == -1 && s2[0] < 0x8000000000000000))
    {
        s3 = 0x8000000000000000;
    }
    return s3;
}

/**
 * @brief Saturate 64-bit value to 32-bit range
 * @param x Input 64-bit value
 * @return Saturated 32-bit value
 */
static int32_t sat32(int64_t x)
{
    int32_t y = 0;
    y = x;
    if (x < (int64_t)0xffffffff80000000)
    {
        y = 0x80000000;
    }
    if (x > (int64_t)0x7fffffff)
    {
        y = 0x7fffffff;
    }
    return y;
}

/**
 * @brief Compute natural logarithm using lookup table method
 * @param Y Output array
 * @param X Input array
 * @param N Array length
 */
static void vec_logn_32x32_sim(int32_t *Y, const int32_t *X, int N)
{
    int32_t inf = 0;
    int32_t min_int32 = 0x80000000;
    int32_t x = 3297771; // Q16.15
    int32_t sx = 1 << 22; // 0.5 in Q.23
    int32_t hx = 1 << 30; // 1; Q1.30
    int32_t mx = (1 << 23) - 1;
    int32_t ex = 16;
    int32_t ln_2 = 0x58B90BFC; // Q.31
    
    for (int i = 0; i < N; i++)
    {
        int32_t x = X[i];
        inf = x > 0 ? 0 : 1;
        if (inf == 1)
        {
            Y[i] = min_int32;
            continue;
        }
        int32_t x_nsa = nsa(x);
        x = x * (1 << x_nsa);    // Q0.30
        x = x - hx;        // x[-1 * 2^30, 2^30 - 1]
        int32_t dx = x & mx;    // get low 23 bit
        dx = dx - sx;    // x1 - x0:[0, 0.5] * 2^31,
        dx = dx << 2;    // Q.33
        x = (int32_t)(x >> 23);        // x in Q.7
        x = x << 3;    // x in Q.10
        int32_t offset = (int32_t)(x >> 2);

        x = log2_table[offset]; // load log2(x0)
        int64_t yf = (int64_t)((int64_t)x << 32);   // Q0.31 => Q0.63

        x = log2_table[offset + 1]; // load log2(e) * (x1-x0)
        yf = mula_32_f63(yf, x, dx); // log2(x0) + (1/x0)*log2(e)*(x1-x0); Q.29*Q.33->Q.63
        int32_t xf = (int32_t)(yf >> (38)); // Q.25

        int32_t nx = x_nsa;
        nx = ex - nx;
        nx = nx << 25;    // Q.25
        int32_t yx = xf;
        yx = yx + nx;    // (log2(x0) + (1/x0)*log2(e)*(x1-x0)) + nsa, Q.25
        int64_t yx_tmp = (int64_t)yx * (int64_t)ln_2;
        yx_tmp = round(yx_tmp * pow(2, -31));
        yx = sat32(yx_tmp);

        Y[i] = yx;
    }
    return;
}

/**
 * @brief Log Softmax implementation for integer tensors
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Workspace buffer
 * @param attrs LogSoftmax attributes
 * @return Status code indicating success or failure
 */
int32_t logsoftmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, LogSoftmaxIntAttrs *attrs) 
{
    const int32_t LOG_Q_IN = 25;
    const int32_t LOG_Q_OUT = 25;
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    int32_t leading = 1, stride = 1;
    int32_t i = 0;
    int32_t axis = 1;
    
    // Determine axis for softmax operation
    if (-1 == attrs->axis) {
        axis = data->shape_.ndim_ - 1;
    }

    // Calculate leading dimensions and stride
    for (; i < axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    for (; i < data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }

    // Check memory types
    int32_t input_is_psram = (1 == data->mem_.type_ || 3 == data->mem_.type_) ? 1 : 0;
    int32_t output_is_psram = (1 == out->mem_.type_ || 3 == out->mem_.type_) ? 1 : 0;

    // Process only int8 data type
    if (Int8 == data->dtype_) {
        int8_t *src = (int8_t *)(data->dptr_);
        int8_t *dst = (int8_t *)(out->dptr_);
        int32_t x_scale = (int32_t)data->scale_;
        int32_t y_scale = (int32_t)out->scale_;
        int32_t max_once_size = 1024;

        int32_t *tmp1 = (int32_t *)(Workspace->dptr_);
        int32_t *tmp2 = tmp1 + max_once_size;

        // Process each batch
        for (int32_t l = 0; l < leading; ++l) {
            int8_t *lsrc = src + l * stride;
            int8_t *ldst = dst + l * stride;
            int32_t i_src_size = stride;

            int8_t *lsrc_tmp = lsrc;
            int8_t *ldst_tmp = ldst;

            // Process in chunks
            while (i_src_size > max_once_size) {
                i_src_size -= max_once_size;
                
                // Handle PSRAM input
                if (input_is_psram) {
                    ret = API_LIB(memcpy_i8o8)((int8_t *)tmp2, (int8_t *)lsrc_tmp, max_once_size);
                    lsrc = (int8_t *)tmp2;
                }
                
                // Handle PSRAM output
                if (output_is_psram) {
                    ldst = (int8_t *)tmp1;
                }
                
                // Perform computations
                ret = API_LIB(scale_i8i8o32)(lsrc, 1, tmp1, max_once_size, 0);  // Q4->Q25
                ret |= API_LIB(scale_i32i32o32)(tmp1, (1 << (LOG_Q_IN - x_scale)), tmp2, max_once_size, 0);  // Q4->Q25
                ret |= API_LIB(softmax_i32o32)((int32_t *)tmp1, (int32_t *)tmp2, max_once_size);  // Q25->Q16.15
                vec_logn_32x32_sim((int32_t *)tmp2, (int32_t *)tmp1, max_once_size);  // Q16.15=>Q6.25
                ret |= API_LIB(scale_i32i32o8)(tmp2, 1, ldst, max_once_size, (LOG_Q_OUT - y_scale));
                
                // Copy result back if needed
                if (output_is_psram) {
                    opi_psram_cpy_out(ldst_tmp, tmp1, max_once_size);
                }
                
                lsrc_tmp += max_once_size;
                ldst_tmp += max_once_size;
            }
            
            // Process remaining elements
            if (input_is_psram) {
                ret = API_LIB(memcpy_i8o8)((int8_t *)tmp2, (int8_t *)lsrc_tmp, i_src_size);
                lsrc = (int8_t *)tmp2;
            }
            
            if (output_is_psram) {
                ldst = (int8_t *)tmp1;
            }
            
            ret = API_LIB(scale_i8i8o32)(lsrc, 1, tmp1, i_src_size, 0);  // Q4->Q25
            ret |= API_LIB(scale_i32i32o32)(tmp1, (1 << (LOG_Q_IN - x_scale)), tmp2, i_src_size, 0);  // Q4->Q25
            ret |= API_LIB(softmax_i32o32)((int32_t *)tmp1, (int32_t *)tmp2, i_src_size);  // Q25->Q16.15
            vec_logn_32x32_sim((int32_t *)tmp2, (int32_t *)tmp1, i_src_size);  // Q16.15=>Q6.25
            ret |= API_LIB(scale_i32i32o8)(tmp2, 1, ldst, i_src_size, (LOG_Q_OUT - y_scale));
            
            if (output_is_psram) {
                opi_psram_cpy_out(ldst_tmp, tmp1, i_src_size);
            }
        }
    } 
    else {
        THINKER_LOG_FATAL("LogSoftmaxInt support int8 data type only.");
        ret = T_ERR_INVALID_DATATYPE;
    }
    
    return ret;
}

#endif