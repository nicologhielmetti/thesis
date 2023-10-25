#ifndef _FLOAT_QUANTIZER_OP_H_
#define _FLOAT_QUANTIZER_OP_H_
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Quantize")
	.Attr("T: {float, double}")
	.Attr("m_bits: int")
	.Attr("e_bits: int")
	.Attr("est_bias: int")
	.Attr("use_est_bias: int")
	.Attr("ret_inf_on_ovf: int")
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
	int est_bias;
	int use_est_bias;
	int ret_inf_on_ovf;
};

template <typename Device, typename T>
struct Quantizer
{
	void operator()(const Device &d, int64_t size, const T *in, T *out, const QInfo *mask);
};

#endif // _FLOAT_QUANTIZER_OP_H_