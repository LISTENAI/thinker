#undef __OP__
#define __OP__ Cast
#include "thinker_status.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/cast.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int num_tensor, tDMA_List*list)
{
	CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
	CastAttrs *attr = (CastAttrs *)((char *)op + op->attr_offset_);
	int ret = T_ERR_NO_IMPLEMENTED;

#ifdef THINKER_USE_VENUS
	ret = cast_luna(tensors[0], tensors[op->num_input_], attr);
#endif
	if (ret != T_SUCCESS)
	{
		return ret;
	}
	return ret;
}

#include "core/operator_template.h"
#undef __OP__
