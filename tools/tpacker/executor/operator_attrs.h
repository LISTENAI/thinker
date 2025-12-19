#ifndef _OPERATOR_ATTRS_H_
#define _OPERATOR_ATTRS_H_ ...

#include <stdint.h>

typedef struct _ClipAttrs {
  float max;
  float min;
} ClipAttrs;

typedef struct _iqCatAttrs {
  int8_t axis;
} iqCatAttrs;

typedef struct _GatherAttrs {
  int8_t axis;
} GatherAttrs;

typedef struct _GRUIntAttrs {
  uint16_t direction;
  uint16_t hidden_size;
  uint16_t input_size;
  uint8_t layout;
  uint8_t quant_type;
} GRUIntAttrs;

typedef struct _GluIntAttrs {
  int8_t axis;
} GluIntAttrs;

typedef struct _iqBinaryAttrs {
  uint8_t quant_type;
  uint8_t reserve;
} iqBinaryAttrs;

typedef struct _LayerNormIntAttrs {
  int8_t axis;
  int8_t keepdims;
  float eps;
} LayerNormIntAttrs;

typedef struct _PoolAttrs {
  uint8_t ceil;
  uint8_t kernel[2];
  uint8_t stride[2];
  uint8_t pad[4];
  uint16_t layout;
  uint8_t quant_type;
} PoolAttrs;

typedef struct _LinearIntAttrs {
  float alpha;
  float beta;
  int32_t transA;
  int32_t transB;
  uint8_t quant_type;
} LinearIntAttrs;

typedef struct _FFNIntAttrs {
  uint8_t middle_scale;
  uint8_t quant_type;
} FFNIntAttrs;

typedef struct _LogsoftmaxintAttrs {
  int8_t axis;
} LogSoftmaxIntAttrs;

typedef struct _LogsoftmaxAttrs {
  int8_t axis;
} LogSoftmaxAttrs;

typedef struct _LstmIntAttrs {
  uint16_t direction;
  uint16_t hidden_size;
  uint16_t input_size;
  uint8_t layout;
  uint8_t quant_type;
} LstmIntAttrs;

typedef struct _Conv1dIntAttrs {
  uint16_t kernel;
  uint8_t pad[2];
  uint8_t stride;
  int16_t group;
  int16_t layout;
  uint8_t quant_type;
  uint8_t act_type;
} Conv1dIntAttrs;

typedef struct _Conv2dIntAttrs {
  uint8_t dilation[3];
  uint16_t kernel[3];
  uint8_t pad[6];
  uint8_t stride[3];
  int16_t group;
  int16_t layout;
  uint8_t quant_type;
  uint8_t act_type;
} Conv2dIntAttrs;

typedef struct _ConvTranspose2dIntAttrs {
  uint8_t dilation[3];
  uint16_t kernel[3];
  uint8_t pad[6];
  uint8_t output_padding[6];
  uint8_t stride[3];
  int16_t group;
  int16_t layout;
  uint8_t quant_type;
  uint8_t act_type;
} ConvTranspose2dIntAttrs;

typedef struct _QuantAttrs {
  uint8_t data_bits;
  uint8_t quant_type;
} QuantAttrs;

typedef struct _DequantAttrs {
  uint8_t scale_o;
} DequantAttrs;

typedef struct _RequantAttrs {
  uint8_t data_bits;
  uint8_t o_bits;
  uint8_t quant_type;
} RequantAttrs;

typedef struct _ShuffleChannelAttrs {
  uint8_t num_group;
  int8_t axis;
} ShuffleChannelAttrs;

typedef struct _SoftmaxIntAttrs {
  int8_t axis;
} SoftmaxIntAttrs;

typedef struct _SliceAttrs {
  int32_t axis;
  int32_t dims;
  int32_t split[8];
} SliceAttrs;

typedef struct _SplitAttrs {
  int32_t axis;
  int32_t dims;
  int32_t split[8];
} SplitAttrs;

typedef struct _ResizeAttrs {
  uint32_t coord_trans_mode;
  float cubic_coeff_a;
  uint32_t mode;
  uint32_t nearest_mode;
} ResizeAttrs;

typedef struct _GemmAttrs {
  float alpha;
  float beta;
  int32_t transA;
  int32_t transB;
} GemmAttrs;

typedef struct _iqSumAttrs {
  int32_t axis;
} iqSumAttrs;

typedef struct _PreluAttrs {
  int32_t slope;
  int32_t post_shift;
} PreluAttrs;

typedef struct _ReluxAttrs {
  int32_t threshold;
  int32_t shift;
} ReluxAttrs;

typedef struct _FlattenAttrs {
  int8_t axis;
} FlattenAttrs;

typedef struct _SqueezeAttrs {
  uint8_t ndim;
  int8_t axes[7];
} SqueezeAttrs;

typedef struct _TransposeAttrs {
  uint8_t ndim_;
  int8_t axes_[8];
} TransposeAttrs;

typedef struct _ActivationAttrs {
  int32_t axis;
} ActivationAttrs;

typedef struct _iqvarAttrs {
  uint8_t ndim_;
  int8_t dims;
} iqvarAttrs;

typedef struct _CastAttrs {
  int64_t to;
} CastAttrs;

typedef struct _topNAttrs {
  int8_t dim;
  int8_t max_num;
} topNAttrs;

typedef struct _ArgMaxAttrs {
  int8_t axis;
} ArgMaxAttrs;

typedef struct _iqPadAttrs {
  // int8_t shape;
  int8_t mode;
} iqPadAttrs;

typedef struct _MultiheadAttentionAttrs {
  uint8_t scale_iqmul_x;
  uint8_t scale_iqmul_y;
  uint8_t scale_iqmul_o;
  uint8_t scale_bmm0_y;
  uint8_t scale_bmm0_o;
  uint8_t scale_bmm1_y;
  uint8_t scale_bmm1_o;
  uint8_t scale_bmm2_y;
  uint8_t scale_bmm2_o;
  uint8_t scale_bmm3_y;
  uint8_t scale_bmm3_o;
  uint8_t scale_iqadd1_o;
  uint8_t scale_iqadd2_o;
  int8_t iqmul_scalar;
  uint16_t headers;
  uint16_t head_dim;
} MultiheadAttentionAttrs;

typedef struct _SparifyFFNIntAttrs {
  uint8_t fc1_out_scale;
  uint8_t mask_out_scale;
  uint8_t quant_type;
  uint8_t reserve;
  uint16_t group_num;
} SparifyFFNIntAttrs;

#endif
