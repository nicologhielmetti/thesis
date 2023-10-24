#ifndef _FLOAT_QUANTIZER_OP_H_
#define _FLOAT_QUANTIZER_OP_H_
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Quantize")
	.Attr("T: {float, double}")
	.Attr("m_bits: int")
	.Attr("e_bits: int")
	.Input("float: T")
	.Output("qfloat: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

struct QInfo
{
	int m_bits;
	int e_bits;
};

template <typename Device, typename T>
struct Quantizer
{
	void operator()(const Device &d, int64_t size, const T *in, T *out, const QInfo *mask);
};

#endif // _FLOAT_QUANTIZER_OP_H_