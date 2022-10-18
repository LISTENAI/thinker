#ifndef __OPERATOR_LIST__
#define __OPERATOR_LIST__
#define OP_LIST(func)                                                          \
  func(LogSoftmaxInt) func(Gather) func(MatMul) func(ShuffleChannel)           \
      func(Resize) func(Transpose) func(Split) func(Conv1dInt) func(Conv2dInt) \
          func(LayerNormInt) func(Quant) func(LinearInt) func(Relu)            \
              func(Reshape) func(iqSigmoid) func(Requant) func(iqSub)          \
                  func(iqAdd) func(BmmInt) func(iqDiv) func(Dequant)           \
                      func(PRelu) func(iqCat) func(SoftmaxInt)                 \
                          func(convtranspose2dint) func(MaxPool) func(iqSum)   \
                              func(iqMul) func(GRUInt) func(Clip)              \
                                  func(Squeeze) func(LSTMInt) func(Flatten)    \
                                      func(BatchNorm2dInt) func(AvgPool2dInt)  \
                                          func(UpsampleInt) func(Slice)        \
                                              func(iqVar) func(ShuffleChannel)

#endif
