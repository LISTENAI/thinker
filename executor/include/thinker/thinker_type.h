/** @file */
#ifndef __THINKER_TYPE_HPP__
#define __THINKER_TYPE_HPP__ 1
#include <stdint.h>

/**
 * Address type definition for memory management
 */
typedef uint64_t addr_type;

/**
 * Model handle type
 */
typedef addr_type tModelHandle;

/**
 * Executor handle type
 */
typedef addr_type tExecHandle;

/**
 * Data types supported by THINKER
 */
typedef enum __tDType__ {
    DTypeUndefined = 0,
    Float16 = 0x6602,  //'f', 2
    Float32 = 0x6604,  //'f', 4
    Float64 = 0x6608,  //'f', 8
    Int4 = 0x6900,     // i, 0.5
    Int8 = 0x6901,     //'i', 1
    Int16 = 0x6902,    //'i', 2
    Int32 = 0x6904,    //'i', 4
    Int64 = 0x6908,    //'i', 8
    Uint8 = 0x7501,    //'u', 1
    Uint16 = 0x7502,   //'u', 2
    Uint32 = 0x7504,   //'u', 4
    Uint64 = 0x7508,   //'u', 8
    Bool = 0x6201,
} tDType;

/**
 * Memory types
 */
typedef enum __MEM_TYPE__ {
    FLASH = 0,
    PSRAM = 1,
    SHARE_MEM = 2,
    UNCERTAIN = 3,
} MemType;

/**
 * Device types
 */
typedef enum __DEV_TYPE__ {
    VENUS = 0,
    MARS = 1,
    ARCS = 2,
    VENUSA = 3,
} DevType;

/**
 * Shape structure for tensors
 */
typedef struct _t_Shape_ {
    uint32_t ndim_;     // Number of dimensions
    uint32_t dims_[7];  // Dimension sizes
} tShape;

/**
 * Device information structure
 */
typedef struct _t_Device_ {
    uint8_t type_;  // Device type
    uint8_t id_;    // Device ID
} tDevice;

/**
 * Memory descriptor structure
 */
typedef struct _t_Mem_ {
    uint8_t type_;      // Device type
    uint8_t mem_type_;  // Memory type
} tMem;

/**
 * Data structure for tensor data
 */
typedef struct _thinker_Data_ {
    void *dptr_;        // Data pointer
    uint16_t dev_type_; // Device type
    uint16_t dtype_;    // Data type
    uint16_t zero_;     // Zero point
    float scale_;       // Scale factor
    tShape shape_;      // Shape information
} tData;

/**
 * Memory allocation structure
 */
typedef struct _thinker_Memory_ {
    uint32_t size_;     // Size of memory
    uint8_t dev_type_;  // Device type
    uint8_t mem_type_;  // Memory type
    addr_type dptr_;    // Memory address
} tMemory;

/**
 * Memory list structure
 */
typedef struct _t_Memory_List_ {
    uint16_t shared_count_; // Shared count
    uint16_t total_count_;  // Total count
    uint32_t elem_size_;    // Element size
    uint32_t header_size_;  // Header size
    uint32_t offset_;       // Offset
} tMemoryList;

/**
 * Input/Output structure
 */
typedef struct _t_IO_ {
    uint32_t tensor_id; // Tensor ID
    char name[60];      // Name string
} tIO;

/**
 * State structure
 */
typedef struct _t_State_ {
    uint32_t input_tensor_id;   // Input tensor ID
    uint32_t output_tensor_id;  // Output tensor ID
} tState;

/**
 * Tensor structure
 */
typedef struct _t_Tensor_ {
    tMem mem_;          // Memory descriptor
    union {
        uint16_t dtype_; // Data type
        uint8_t byte_;   // Byte representation
    };
    
    int32_t mem_id_;    // Memory ID
    float scale_;       // Scale factor
    int32_t zero_;      // Zero point
    
    tShape shape_;      // Shape information
    union {
        addr_type dptr_; // Data pointer
        addr_type offset_; // Offset
    };
    uint8_t layout_;    // Layout information
    uint32_t reserved;  // Reserved field
} tTensor;

/**
 * Tensor list structure
 */
typedef struct _t_TensorList_ {
    uint32_t count_;        // Count of tensors
    uint32_t elem_size_;    // Element size
    uint32_t header_size_;  // Header size
    uint32_t offset_;       // Offset
} tTensorList;

/**
 * Operator structure
 */
typedef struct _t_Operator_ {
    uint16_t op_id_;        // Operator ID
    uint16_t attr_offset_;  // Attribute offset
    uint16_t tensor_offset_; // Tensor offset
    uint16_t scalar_offset_; // Scalar offset
    uint16_t total_size_;   // Total size
    uint16_t num_input_;    // Number of inputs
    uint16_t num_output_;   // Number of outputs
    uint8_t num_temp_;      // Number of temporary tensors
    uint8_t num_scalar_;    // Number of scalars
} tOperator;

/**
 * Operator list structure
 */
typedef struct _t_OperatorList_ {
    uint16_t op_count_;     // Operator count
    uint16_t type_count_;   // Type count
    uint8_t type_length_;   // Type length
    uint8_t header_size_;   // Header size
    uint16_t type_offset_;  // Type offset
    uint32_t op_offset_;    // Operator offset
    uint32_t op_size_;      // Operator size
} tOperatorList;

/**
 * Parameter structure
 */
typedef struct _t_Parameter_ {
    tMem memory_;   // Memory descriptor
    uint16_t mem_id_; // Memory ID
    uint32_t offset_; // Offset
    uint64_t size_;   // Size
} tParameter;

/**
 * Parameter list structure
 */
typedef struct _t_ParameterList_ {
    uint32_t count_;        // Count
    uint32_t elem_size_;    // Element size
    uint32_t header_size_;  // Header size
    uint32_t offset_;       // Offset
} tParameterList;

/**
 * Tensor-dimension pair structure
 */
typedef struct _t_TenDimPair_ {
    uint16_t tensor_id_;    // Tensor ID
    uint8_t dim_id_;        // Dimension ID
} tTenDimPair;

/**
 * Dynamic axis information structure
 */
typedef struct _t_DyAxisInfo_ {
    uint16_t tensor_id_;        // Tensor ID
    uint8_t dy_dim_id_;         // Dynamic dimension ID
    uint8_t scalar_input_id_;   // Scalar input ID
} tDyAxisInfo;

/**
 * Shape inference header structure
 */
typedef struct _t_ShapeInferHdr_ {
    uint32_t dy_axis_offset_;   // Dynamic axis offset
    uint32_t graph_offset_;     // Graph offset
    uint32_t id_pair_offset_;   // ID pair offset
    
    uint32_t num_dy_axis_;      // Number of dynamic axes
    uint32_t graph_size_;       // Graph size
    uint32_t num_id_pair_;      // Number of ID pairs
} tShapeInferHdr;

/**
 * Unsigned integer structure
 */
typedef struct _t_Uint_ {
    uint32_t val_;  // Value
} tUint;

/**
 * ONNX shape inference header structure
 */
typedef struct _t_ShapeOnnxInferHdr_ {
    uint32_t dy_axis_offset_;   // Dynamic axis offset
    uint32_t graph_offset_;     // Graph offset
    uint32_t id_pair_offset_;   // ID pair offset
    uint32_t tensor_list_offset_; // Tensor list offset
    uint32_t inputs_ids_offset_;  // Inputs IDs offset
    uint32_t outputs_ids_offset_; // Outputs IDs offset
    
    uint32_t num_dy_axis_;      // Number of dynamic axes
    uint32_t graph_size_;       // Graph size
    uint32_t num_id_pair_;      // Number of ID pairs
    uint32_t num_inputs_ids_;   // Number of inputs IDs
    uint32_t num_outputs_ids_;  // Number of outputs IDs
} tShapeOnnxInferHdr;

/**
 * Model header structure
 */
typedef struct _t_Model_Header_ {
    uint8_t label_[16];         // Label
    uint32_t crc32_;            // CRC32 checksum
    uint32_t memory_offset_;    // Memory offset
    uint32_t tensor_offset_;    // Tensor offset
    uint32_t scalar_offset_;    // Scalar offset
    uint32_t op_offset_;        // Operator offset
    uint32_t io_offset_;        // IO offset
    uint32_t state_offset_;     // State offset
    uint32_t debug_offset_;     // Debug offset
    uint32_t shape_infer_offset_; // Shape inference offset
    uint32_t param_offset_;     // Parameter offset
    uint32_t dma_offset_;       // DMA offset
    uint32_t reserved;          // Reserved
    uint64_t total_size_;       // Total size
} tModelHeader;

/**
 * DMA transfer structure
 */
typedef struct _t_DMA_ {
    tDevice src_device_;    // Source device
    tDevice dst_device_;    // Destination device
    uint16_t src_tensor_id_; // Source tensor ID
    uint16_t dst_tensor_id_; // Destination tensor ID
    uint64_t size_;         // Transfer size
} tDMA;

/**
 * DMA list structure
 */
typedef struct _t_DMAList_ {
    uint32_t count_;        // Count
    uint32_t elem_size_;    // Element size
    uint32_t header_size_;  // Header size
    uint32_t offset_;       // Offset
} tDMAList;

/**
 * Tensor name structure (debug info)
 */
typedef struct _t_Tensor_Name_ {
    char name_[64];         // Name string
} tTensorName;

/**
 * Debug list structure
 */
typedef struct _t_DebugList_ {
    uint32_t tensor_name_count_; // Tensor name count
    tTensorName *tensor_name_list_; // Tensor name list
    uint32_t offset_;           // Offset
} tDebugList;

#endif  // __THINKER_TYPE_HPP__