#ifndef __OP_LIST__
#define __OP_LIST__
#define OP_LIST(func)                                                          \
  func(MaxPool) func(Clip) func(Resize) func(BmmInt) func(iqAdd) func(iqCat)   \
      func(iqSigmoid) func(Reshape) func(Tile) func(LayerNormInt) func(Split)  \
          func(Conv2dInt) func(AvgPool2dInt) func(iqTanh) func(iqDiv) func(Cast)\
              func(Conv1dInt) func(iqVar) func(Relu) func(Transpose) func(     \
                  ConvTranspose2dInt) func(iqSub) func(Quant) func(LogSoftmax) \
                  func(Expand) func(iqSum) func(iqMul) func(MatMul) func(Dequant)\
                      func(SoftmaxInt) func(ShuffleChannel) func(Flatten)      \
                          func(LSTMInt) func(LogSoftmaxInt) func(Squeeze)      \
                                func(Gather) func(Requant) func(Slice)       \
                                      func(GRUInt) func(UpsampleInt)           \
                                          func(BatchNorm2dInt) func(LinearInt)

#endif