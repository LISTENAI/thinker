#ifndef __OP_LIST__
#define __OP_LIST__

// List of all supported operators - each operator gets its own registry function
#define OP_LIST(func)                                                          \
  func(MaxPool) func(topN) func(Resize) func(BmmInt) func(iqAdd) func(iqCat)   \
      func(iqSigmoid) func(Reshape) func(Tile) func(LayerNormInt) func(Split)  \
          func(Conv2dInt) func(AvgPool2dInt) func(iqTanh) func(iqDiv) func(QGelu) \
              func(Conv1dInt) func(iqVar) func(Relu) func(Transpose) func(     \
                  ConvTranspose2dInt) func(iqSub) func(Quant) func(LogSoftmax) \
                      func(Expand) func(iqSum) func(iqMul) func(GluInt) func(FFNInt)\
                          func(SoftmaxInt) func(Flatten)      \
                              func(LSTMInt) func(LogSoftmaxInt) func(Squeeze)      \
                                  func(Dequant) func(topN2) func(Cast) func(iqPad) \
                                      func(Gather) func(Requant) func(Slice)       \
                                          func(GRUInt) func(Shape) func(QSwish)    \
                                              func(BatchNorm2dInt) func(LinearInt) \
                                                 func(PRelu) func(Relux) func(Clip) func(ArgMax) \
                                                 func(Unsqueeze) func(SparifyFFNInt) \
                                                   func(MultiheadAttention) \
                                                    func(BatchNorm1dInt)

#endif
