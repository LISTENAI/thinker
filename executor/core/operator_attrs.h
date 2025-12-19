#ifndef _OPERATOR_ATTRS_H_
#define _OPERATOR_ATTRS_H_

#include <stdint.h>

// Clip operator attributes - defines min/max clipping values
typedef struct _ClipAttrs {
    float max;  // Maximum value for clipping
    float min;  // Minimum value for clipping
} ClipAttrs;

// Integer concatenation attributes - defines concatenation axis
typedef struct _iqCatAttrs {
    int8_t axis;  // Concatenation axis
} iqCatAttrs;

// Gather operator attributes - defines gathering axis
typedef struct _GatherAttrs {
    int8_t axis;  // Gathering axis
} GatherAttrs;

// Gated Recurrent Unit integer attributes - defines GRU parameters
typedef struct _GRUIntAttrs {
    uint16_t direction;    // Direction (forward/backward)
    uint16_t hidden_size;  // Hidden state size
    uint16_t input_size;   // Input size
    uint8_t layout;        // Data layout
    uint8_t quant_type;    // Quantization type
} GRUIntAttrs;

// Gated Linear Unit integer attributes - defines GLU axis
typedef struct _GluIntAttrs {
    int8_t axis;  // GLU operation axis
} GluIntAttrs;

// Integer binary operation attributes - defines quantization settings
typedef struct _iqBinaryAttrs {
    uint8_t quant_type;  // Quantization type
    uint8_t reserve;     // Reserved field
} iqBinaryAttrs;

// Layer normalization integer attributes - defines normalization parameters
typedef struct _LayerNormIntAttrs {
    int8_t axis;     // Normalization axis
    int8_t keepdims; // Whether to keep dimensions
    float eps;       // Epsilon value for numerical stability
} LayerNormIntAttrs;

// Pooling operation attributes - defines pooling parameters
typedef struct _PoolAttrs {
    uint8_t ceil;        // Ceiling flag for padding
    uint8_t kernel[2];   // Kernel size [height, width]
    uint8_t stride[2];   // Stride [height, width]
    uint8_t pad[4];      // Padding [top, bottom, left, right]
    uint16_t layout;     // Data layout
    uint8_t quant_type;  // Quantization type
} PoolAttrs;

// Linear integer operation attributes - defines linear transformation parameters
typedef struct _LinearIntAttrs {
    float alpha;     // Scaling factor A
    float beta;      // Scaling factor B
    int32_t transA;  // Transpose flag for matrix A
    int32_t transB;  // Transpose flag for matrix B
    uint8_t quant_type; // Quantization type
} LinearIntAttrs;

// Feed-forward network integer attributes - defines FFN parameters
typedef struct _FFNIntAttrs {
    uint8_t middle_scale;  // Middle layer scale
    uint8_t quant_type;    // Quantization type
} FFNIntAttrs;

// Log softmax integer attributes - defines softmax axis
typedef struct _LogsoftmaxintAttrs {
    int8_t axis;  // Softmax axis
} LogSoftmaxIntAttrs;

// Log softmax attributes - defines softmax axis
typedef struct _LogsoftmaxAttrs {
    int8_t axis;  // Softmax axis
} LogSoftmaxAttrs;

// Long Short-Term Memory integer attributes - defines LSTM parameters
typedef struct _LstmIntAttrs {
    uint16_t direction;    // Direction (forward/backward)
    uint16_t hidden_size;  // Hidden state size
    uint16_t input_size;   // Input size
    uint8_t layout;        // Data layout
    uint8_t quant_type;    // Quantization type
} LstmIntAttrs;

// Convolution 1D integer attributes - defines 1D convolution parameters
typedef struct _Conv1dIntAttrs {
    uint16_t kernel;       // Kernel size
    uint8_t pad[2];        // Padding [left, right]
    uint8_t stride;        // Stride
    int16_t group;         // Group count
    int16_t layout;        // Data layout
    uint8_t quant_type;    // Quantization type
    uint8_t act_type;      // Activation type
} Conv1dIntAttrs;

// Convolution 2D integer attributes - defines 2D convolution parameters
typedef struct _Conv2dIntAttrs {
    uint8_t dilation[3];   // Dilation factors
    uint16_t kernel[3];    // Kernel sizes [height, width, depth]
    uint8_t pad[6];        // Padding [top, bottom, left, right]
    uint8_t stride[3];     // Stride [height, width, depth]
    int16_t group;         // Group count
    int16_t layout;        // Data layout
    uint8_t quant_type;    // Quantization type
    uint8_t act_type;      // Activation type
} Conv2dIntAttrs;

// Convolution transpose 2D integer attributes - defines transposed convolution parameters
typedef struct _ConvTranspose2dIntAttrs {
    uint8_t dilation[3];        // Dilation factors
    uint16_t kernel[3];         // Kernel sizes [height, width, depth]
    uint8_t pad[6];             // Padding [top, bottom, left, right]
    uint8_t output_padding[6];  // Output padding
    uint8_t stride[3];          // Stride [height, width, depth]
    int16_t group;              // Group count
    int16_t layout;             // Data layout
    uint8_t quant_type;         // Quantization type
    uint8_t act_type;           // Activation type
} ConvTranspose2dIntAttrs;

// Quantization attributes - defines quantization parameters
typedef struct _QuantAttrs {
    uint8_t data_bits;   // Number of bits for quantization
    uint8_t quant_type;  // Quantization type
} QuantAttrs;

// Dequantization attributes - defines dequantization parameters
typedef struct _DequantAttrs {
    uint8_t scale_o;  // Output scale factor
} DequantAttrs;

// Requantization attributes - defines requantization parameters
typedef struct _RequantAttrs {
  uint8_t data_bits;
  uint8_t o_bits;
  uint8_t quant_type;
} RequantAttrs;

// Shuffle channel attributes - defines channel shuffling parameters
typedef struct _ShuffleChannelAttrs {
    uint8_t num_group;  // Number of groups
    int8_t axis;        // Shuffle axis
} ShuffleChannelAttrs;

// Softmax integer attributes - defines softmax axis
typedef struct _SoftmaxIntAttrs {
    int8_t axis;  // Softmax axis
} SoftmaxIntAttrs;

// Slice operation attributes - defines slicing parameters
typedef struct _SliceAttrs {
    int32_t axis;     // Slice axis
    int32_t dims;     // Number of dimensions
    int32_t split[8]; // Split points
} SliceAttrs;

// Split operation attributes - defines splitting parameters
typedef struct _SplitAttrs {
    int32_t axis;     // Split axis
    int32_t dims;     // Number of dimensions
    int32_t split[8]; // Split points
} SplitAttrs;

// Resize operation attributes - defines resizing parameters
typedef struct _ResizeAttrs {
    uint32_t coord_trans_mode;  // Coordinate transformation mode
    float cubic_coeff_a;        // Cubic interpolation coefficient
    uint32_t mode;              // Resize mode
    uint32_t nearest_mode;      // Nearest neighbor mode
} ResizeAttrs;

// General matrix multiplication attributes - defines GEMM parameters
typedef struct _GemmAttrs {
    float alpha;    // Scaling factor A
    float beta;     // Scaling factor B
    int32_t transA; // Transpose flag for matrix A
    int32_t transB; // Transpose flag for matrix B
} GemmAttrs;

// Integer sum attributes - defines sum axis
typedef struct _iqSumAttrs {
    int32_t axis;  // Sum axis
} iqSumAttrs;

// Parametric ReLU attributes - defines PReLU parameters
typedef struct _PreluAttrs {
    int32_t slope;      // Slope parameter
    int32_t post_shift; // Post-shift value
} PreluAttrs;

// ReLU+ attributes - defines ReLU+ parameters
typedef struct _ReluxAttrs {
    int32_t threshold;  // Threshold value
    int32_t shift;      // Shift value
} ReluxAttrs;

// Flatten operation attributes - defines flatten axis
typedef struct _FlattenAttrs {
    int8_t axis;  // Flatten axis
} FlattenAttrs;

// Squeeze operation attributes - defines squeezing parameters
typedef struct _SqueezeAttrs {
    uint8_t ndim;      // Number of dimensions
    int8_t axes[7];    // Axes to squeeze
} SqueezeAttrs;

// Transpose operation attributes - defines transpose parameters
typedef struct _TransposeAttrs {
    uint8_t ndim_;     // Number of dimensions
    int8_t axes_[8];   // Permutation axes
} TransposeAttrs;

// Activation function attributes - defines activation parameters
typedef struct _ActivationAttrs {
    int32_t axis;  // Activation axis
} ActivationAttrs;

// Integer variance attributes - defines variance parameters
typedef struct _iqvarAttrs {
    uint8_t ndim_;  // Number of dimensions
    int8_t dims;    // Dimension value
} iqvarAttrs;

// Cast operation attributes - defines casting target type
typedef struct _CastAttrs {
    int64_t to;  // Target data type
} CastAttrs;

// Top-N attributes - defines top-N selection parameters
typedef struct _topNAttrs {
    int8_t dim;      // Dimension for selection
    int8_t max_num;  // Maximum number of selections
} topNAttrs;

// ArgMax attributes - defines ArgMax axis
typedef struct _ArgMaxAttrs {
    int8_t axis;  // ArgMax axis
} ArgMaxAttrs;

// Integer padding attributes - defines padding parameters
typedef struct _iqPadAttrs {
    int8_t mode;  // Padding mode
} iqPadAttrs;

// Multi-head attention attributes - defines attention parameters
typedef struct _MultiheadAttentionAttrs {
    uint8_t scale_iqmul_x;    // Scale for input X
    uint8_t scale_iqmul_y;    // Scale for input Y
    uint8_t scale_iqmul_o;    // Scale for output
    uint8_t scale_bmm0_y;     // Scale for BMM0 input Y
    uint8_t scale_bmm0_o;     // Scale for BMM0 output
    uint8_t scale_bmm1_y;     // Scale for BMM1 input Y
    uint8_t scale_bmm1_o;     // Scale for BMM1 output
    uint8_t scale_bmm2_y;     // Scale for BMM2 input Y
    uint8_t scale_bmm2_o;     // Scale for BMM2 output
    uint8_t scale_bmm3_y;     // Scale for BMM3 input Y
    uint8_t scale_bmm3_o;     // Scale for BMM3 output
    uint8_t scale_iqadd1_o;   // Scale for IQADD1 output
    uint8_t scale_iqadd2_o;   // Scale for IQADD2 output
    int8_t iqmul_scalar;      // IQ multiplication scalar
    uint16_t headers;         // Number of attention heads
    uint16_t head_dim;        // Dimension per head
} MultiheadAttentionAttrs;

// Sparse FFN integer attributes - defines sparse FFN parameters
typedef struct _SparifyFFNIntAttrs {
    uint8_t fc1_out_scale;  // FC1 output scale
    uint8_t mask_out_scale; // Mask output scale
    uint8_t quant_type;     // Quantization type
    uint8_t reserve;        // Reserved field
    uint16_t group_num;     // Group number
} SparifyFFNIntAttrs;

#endif