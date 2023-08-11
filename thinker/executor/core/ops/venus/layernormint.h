#ifndef _LAYERNORMINT_LUNA_H_
#define _LAYERNORMINT_LUNA_H_

#include <math.h>

#include "c_api/thinker_define.h"
#include "core/operator_attrs.h"
#include "hifi/NatureDSP_Signal.h"
#include "luna/luna_math.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

static const int16_t g_s16Table_sqrt_reciprocal[768] =
    {/*Q15,[0.25-1]的平方根值*/
     32767, 32704, 32640, 32577, 32514, 32452, 32390, 32328, 32267, 32206,
     32146, 32085, 32025, 31966, 31907, 31848, 31789, 31731, 31673, 31615,
     31558, 31501, 31444, 31388, 31332, 31276, 31220, 31165, 31110, 31056,
     31001, 30947, 30893, 30840, 30787, 30734, 30681, 30629, 30577, 30525,
     30473, 30422, 30371, 30320, 30269, 30219, 30169, 30119, 30069, 30020,
     29971, 29922, 29874, 29825, 29777, 29729, 29681, 29634, 29587, 29540,
     29493, 29446, 29400, 29354, 29308, 29262, 29217, 29172, 29127, 29082,
     29037, 28993, 28948, 28904, 28861, 28817, 28774, 28730, 28687, 28644,
     28602, 28559, 28517, 28475, 28433, 28391, 28350, 28308, 28267, 28226,
     28185, 28145, 28104, 28064, 28024, 27984, 27944, 27905, 27865, 27826,
     27787, 27748, 27709, 27670, 27632, 27594, 27555, 27517, 27480, 27442,
     27404, 27367, 27330, 27293, 27256, 27219, 27183, 27146, 27110, 27074,
     27038, 27002, 26966, 26930, 26895, 26860, 26824, 26789, 26754, 26720,
     26685, 26651, 26616, 26582, 26548, 26514, 26480, 26446, 26413, 26379,
     26346, 26313, 26280, 26247, 26214, 26181, 26149, 26116, 26084, 26052,
     26019, 25987, 25956, 25924, 25892, 25861, 25829, 25798, 25767, 25736,
     25705, 25674, 25643, 25613, 25582, 25552, 25521, 25491, 25461, 25431,
     25401, 25372, 25342, 25312, 25283, 25254, 25224, 25195, 25166, 25137,
     25108, 25080, 25051, 25022, 24994, 24966, 24937, 24909, 24881, 24853,
     24825, 24797, 24770, 24742, 24715, 24687, 24660, 24633, 24606, 24579,
     24552, 24525, 24498, 24471, 24445, 24418, 24392, 24365, 24339, 24313,
     24287, 24261, 24235, 24209, 24183, 24157, 24132, 24106, 24081, 24055,
     24030, 24005, 23980, 23955, 23930, 23905, 23880, 23855, 23831, 23806,
     23782, 23757, 23733, 23709, 23684, 23660, 23636, 23612, 23588, 23564,
     23541, 23517, 23493, 23470, 23446, 23423, 23400, 23376, 23353, 23330,
     23307, 23284, 23261, 23238, 23215, 23193, 23170, 23147, 23125, 23102,
     23080, 23058, 23035, 23013, 22991, 22969, 22947, 22925, 22903, 22881,
     22860, 22838, 22816, 22795, 22773, 22752, 22730, 22709, 22688, 22666,
     22645, 22624, 22603, 22582, 22561, 22540, 22520, 22499, 22478, 22458,
     22437, 22416, 22396, 22376, 22355, 22335, 22315, 22294, 22274, 22254,
     22234, 22214, 22194, 22175, 22155, 22135, 22115, 22096, 22076, 22056,
     22037, 22018, 21998, 21979, 21960, 21940, 21921, 21902, 21883, 21864,
     21845, 21826, 21807, 21788, 21769, 21751, 21732, 21713, 21695, 21676,
     21658, 21639, 21621, 21602, 21584, 21566, 21548, 21529, 21511, 21493,
     21475, 21457, 21439, 21421, 21403, 21386, 21368, 21350, 21332, 21315,
     21297, 21280, 21262, 21245, 21227, 21210, 21193, 21175, 21158, 21141,
     21124, 21107, 21089, 21072, 21055, 21038, 21022, 21005, 20988, 20971,
     20954, 20938, 20921, 20904, 20888, 20871, 20855, 20838, 20822, 20805,
     20789, 20773, 20756, 20740, 20724, 20708, 20691, 20675, 20659, 20643,
     20627, 20611, 20595, 20580, 20564, 20548, 20532, 20516, 20501, 20485,
     20470, 20454, 20438, 20423, 20407, 20392, 20377, 20361, 20346, 20331,
     20315, 20300, 20285, 20270, 20255, 20239, 20224, 20209, 20194, 20179,
     20164, 20150, 20135, 20120, 20105, 20090, 20076, 20061, 20046, 20032,
     20017, 20002, 19988, 19973, 19959, 19944, 19930, 19916, 19901, 19887,
     19873, 19858, 19844, 19830, 19816, 19802, 19787, 19773, 19759, 19745,
     19731, 19717, 19703, 19690, 19676, 19662, 19648, 19634, 19620, 19607,
     19593, 19579, 19566, 19552, 19539, 19525, 19511, 19498, 19485, 19471,
     19458, 19444, 19431, 19418, 19404, 19391, 19378, 19365, 19351, 19338,
     19325, 19312, 19299, 19286, 19273, 19260, 19247, 19234, 19221, 19208,
     19195, 19182, 19169, 19157, 19144, 19131, 19118, 19106, 19093, 19080,
     19068, 19055, 19042, 19030, 19017, 19005, 18992, 18980, 18968, 18955,
     18943, 18930, 18918, 18906, 18894, 18881, 18869, 18857, 18845, 18832,
     18820, 18808, 18796, 18784, 18772, 18760, 18748, 18736, 18724, 18712,
     18700, 18688, 18676, 18665, 18653, 18641, 18629, 18618, 18606, 18594,
     18582, 18571, 18559, 18547, 18536, 18524, 18513, 18501, 18490, 18478,
     18467, 18455, 18444, 18432, 18421, 18410, 18398, 18387, 18376, 18365,
     18353, 18342, 18331, 18320, 18308, 18297, 18286, 18275, 18264, 18253,
     18242, 18231, 18220, 18209, 18198, 18187, 18176, 18165, 18154, 18143,
     18132, 18122, 18111, 18100, 18089, 18078, 18068, 18057, 18046, 18036,
     18025, 18014, 18004, 17993, 17982, 17972, 17961, 17951, 17940, 17930,
     17919, 17909, 17898, 17888, 17878, 17867, 17857, 17846, 17836, 17826,
     17816, 17805, 17795, 17785, 17775, 17764, 17754, 17744, 17734, 17724,
     17714, 17703, 17693, 17683, 17673, 17663, 17653, 17643, 17633, 17623,
     17613, 17603, 17593, 17584, 17574, 17564, 17554, 17544, 17534, 17525,
     17515, 17505, 17495, 17485, 17476, 17466, 17456, 17447, 17437, 17427,
     17418, 17408, 17399, 17389, 17379, 17370, 17360, 17351, 17341, 17332,
     17322, 17313, 17304, 17294, 17285, 17275, 17266, 17257, 17247, 17238,
     17229, 17219, 17210, 17201, 17192, 17182, 17173, 17164, 17155, 17146,
     17136, 17127, 17118, 17109, 17100, 17091, 17082, 17073, 17064, 17055,
     17046, 17037, 17028, 17019, 17010, 17001, 16992, 16983, 16974, 16965,
     16956, 16947, 16938, 16930, 16921, 16912, 16903, 16894, 16886, 16877,
     16868, 16859, 16851, 16842, 16833, 16825, 16816, 16807, 16799, 16790,
     16782, 16773, 16764, 16756, 16747, 16739, 16730, 16722, 16713, 16705,
     16696, 16688, 16679, 16671, 16662, 16654, 16646, 16637, 16629, 16621,
     16612, 16604, 16596, 16587, 16579, 16571, 16562, 16554, 16546, 16538,
     16529, 16521, 16513, 16505, 16497, 16489, 16480, 16472, 16464, 16456,
     16448, 16440, 16432, 16424, 16416, 16408, 16400, 16392};

/**
 * -0.5-->0(floor(x+0.5))
 **/
static int32_t shfit_floor_x05_int32(int32_t x, int32_t shift) {
  int32_t val = x;

  if (shift >= 32) {
    return 0;
  }
  if (shift > 0) {
    val = val >> (shift - 1);
    val = (val & 0x1) + (val >> 1);
  }

  return val;
}

static const int16_t calc_sqrt_reciprocal(const int64_t data, int32_t q_x, int32_t *table_shift)
{
	const int q_normal = 10;	//normalize(-32, 32)
	const int q2 = 14;
	int64_t temp;
	int q1;

	if (data & 0xC00000000000)
	{
		temp = data>>38;
		q1 = 24;
	}
	else if (data & 0x300000000000)
	{
		temp = data>>36;
		q1 = 23;
	}
	else if (data & 0xC0000000000)
	{
		temp = data>>34;
		q1 = 22;
	}
	else if (data & 0x30000000000)
	{
		temp = data>>32;
		q1 = 21;
	}
	else if (data & 0xC000000000)
	{
		temp = data>>30;
		q1 = 20;
	}
	else if (data & 0x3000000000)
	{
		temp = data>>28;
		q1 = 19;
	}
	else if (data & 0xC00000000)
	{
		temp = data>>26;
		q1 = 18;
	}
	else if (data & 0x300000000)
	{
		temp = data>>24;
		q1 = 17;
	}

	else if (data & 0xC0000000)
	{
		temp = data>>22;
        q1 = 16;
	}        
    else if (data & 0x30000000)
	{
		temp = data>>20;
        q1 = 15;
	}        
    else if (data & 0xFC000000)
	{
		temp = data>>18;
        q1 = 14;
	}        
    else if (data & 0xF3000000)
	{
        temp = data>>16;
        q1 = 13;
	}
    else if (data & 0xFFC00000)
	{
        temp = data>>14;
        q1 = 12;
	}
    else if (data & 0xFF300000)
	{
        temp = data>>12;
        q1 = 11;
	}
    else if (data & 0xFFFC0000)
	{
        temp = data>>10;
        q1 = 10;
	}
    else if (data & 0xFFF30000)
	{
        temp = data>>8;
        q1 = 9;
	}
    else if (data & 0xFFFFC000)
	{
        temp = data>>6;
        q1 = 8;
	}
    else if (data & 0xFFFF3000)
	{
        temp = data>>4;
        q1 = 7;
	}
    else if (data & 0xFFFFFC00)
	{
        temp = data>>2;
        q1 = 6;
	}
    else if (data & 0xFFFFFF00)
	{
        temp = data;
        q1 = 5;
	}
    else if (data & 0xFFFFFFC0)
	{
        temp = data<<2;
        q1 = 4;
	}
    else if (data & 0xFFFFFFF0)
	{
        temp = data<<4;
        q1 = 3;
	}
    else if (data & 0xFFFFFFFC)
	{
        temp = data<<6;
        q1 = 2;
	}
    else if (data & 0xFFFFFFFF)
	{
        temp = data<<8;
        q1 = 1;
	}
    else
	{
		temp = 256;
		q1 = 0;
	}

	int32_t id = temp - 256;
	int32_t table_out = (int32_t)g_s16Table_sqrt_reciprocal[id];
	int32_t q = q1 + q2 - q_normal;
	*table_shift = q;//(int32_t)powf(2, q);
	return table_out;
}

int32_t layernormalint_venus(const tTensor *X, const tTensor *W,
                             const tTensor *Bias, tTensor *Y,
                             tTensor *workspace, LayerNormIntAttrs *attrs) {
  int32_t ret = T_ERR_FAIL;

  int32_t n_dims = X->shape_.ndim_;
  int32_t size = getTensorSize(W);
  int32_t leading = 1;
  int32_t T = 1;
  if (size == X->shape_.dims_[n_dims - 1]) {
    T = X->shape_.dims_[n_dims - 1] ;
    leading = X->shape_.dims_[n_dims - 3]* X->shape_.dims_[n_dims - 2] ;
  }
  else if (size == X->shape_.dims_[n_dims - 1] * X->shape_.dims_[n_dims - 2]) {
    T = X->shape_.dims_[n_dims - 1] * X->shape_.dims_[n_dims - 2];
    leading = X->shape_.dims_[n_dims - 3];
  }

    int32_t input_size = leading * T;

    // float eps = attrs->eps;
	const float eps = 0.00001;
	int16_t *p_gamma = (int16_t *)W->dptr_;
	int32_t *p_beta = (int32_t *)Bias->dptr_;
    int8_t *p_src = (int8_t *)X->dptr_;
    int8_t *p_dst = (int8_t *)Y->dptr_;
    int8_t *p_tmp = (int8_t *)workspace->dptr_;

	int32_t q_x = (int32_t)X->scale_;
	const int32_t q_normal = 10;
	int32_t q_gamma = (int32_t)W->scale_;
	int32_t q_beta = (int32_t)Bias->scale_;
	int32_t q_y = (int32_t)Y->scale_;
	int32_t shift = q_normal + q_gamma - q_y;

	int32_t *sum_x = (int32_t *)p_tmp;
	int32_t *sum_x2 = (int32_t *)(p_tmp + sizeof(int32_t));
	int16_t *p_y1 = (int16_t *)(p_tmp + 2 * sizeof(int32_t));
	int32_t *p_src2 = (int32_t *)(p_tmp + T * sizeof(int32_t));
	int32_t *p_numerator = p_src2;
	int32_t *p_y2 = p_src2;

	int64_t q_eps = floor(eps * (1 << (q_x * 2)) * T * T + 0.5f);

	//step1: sum(xi) and sum(xi^2)
	for (int i = 0; i < leading; i++)
	{
		int8_t *p_src_once = p_src + i * T;
		int8_t *p_dst_once = p_dst + i * T;
		luna_vector_sum_q7_int32(p_src_once, sum_x, T, 0);
		luna_mul_q7_int32(p_src_once, p_src_once, p_src2, T, 0);
		luna_vector_sum_q31_int32(p_src2, sum_x2, T, 0);

		int32_t sum_x_val = *sum_x;
		int32_t sum_x2_val = *sum_x2;
		int64_t denominator = (int64_t)(T * sum_x2_val) - (int64_t)(sum_x_val * sum_x_val);
		denominator = denominator + q_eps;
		int32_t label_shift = 0;
		denominator = calc_sqrt_reciprocal((const int64_t)denominator, q_x, &label_shift);
		luna_scale_q7_int32(p_src_once, 1, p_numerator, T, 0);
		luna_scale_q31_int32(p_numerator, T, p_numerator, T, 0);
		luna_offset_q31_int32(p_numerator, (0 - sum_x_val), p_numerator, T, 0);
		luna_scale_q31_int16(p_numerator, denominator, (int16_t *)p_y1, T, label_shift);
		luna_mul_q15_int32(p_y1, (int16_t *)p_gamma, p_y2, T, 0);
		luna_add_q31_int8(p_y2, p_beta, p_dst_once, T, shift);
	}

	return 0;
}
#endif