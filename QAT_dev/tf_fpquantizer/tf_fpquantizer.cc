#include "tf_fpquantizer.hpp"

#include "tensorflow/core/framework/op_kernel.h"
#include <cstdint>
#include <type_traits>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct Quantizer<CPUDevice, T>
{
	using view_type = typename std::conditional<std::is_same<T, double>::value, uint64_t, uint32_t>::type;

	void operator()(const CPUDevice &d, int64_t size, const T *in, T *out, const QInfo *mask)
	{
		const uint8_t m_width = std::is_same<T, double>::value ? 52 : 23;
		const uint8_t e_width = std::is_same<T, double>::value ? 11 : 8;
		const uint16_t e_bias = std::is_same<T, double>::value ? 1023 : 127;

		const view_type qm_mask = (1ULL << (e_width + m_width)) | (((1ULL << m_width) - 1) & ~((1ULL << (m_width - mask->m_bits)) - 1));
		const view_type e_mask = (1ULL << e_width) - 1;
		const view_type qe_max = e_bias + (1ULL << (mask->e_bits - 1));
		const view_type qe_min = e_bias - (1ULL << (mask->e_bits - 1)) + 1;

		for (int64_t i = 0; i < size; ++i)
		{
			view_type qm = *((view_type *)&in[i]) & qm_mask;
			view_type e = *((view_type *)&in[i]) >> m_width & e_mask;
			view_type qe = e == 0 ? 0 : (e <= qe_min ? qe_min : (e >= qe_max ? qe_max : e));
			view_type qout = qm | (qe << m_width);
			out[i] = *((T *)&qout);
		}
	}
};

template <typename Device, typename T>
class QuantizeOp : public OpKernel
{
public:
	explicit QuantizeOp(OpKernelConstruction *context) : OpKernel(context)
	{
		// Get the number of fractional bits to preserve
		OP_REQUIRES_OK(context,
					   context->GetAttr("m_bits", &_mask.m_bits));
		// Check that m_bits is positive
		OP_REQUIRES(context, _mask.m_bits >= 0,
					errors::InvalidArgument("Need m_bits >= 0, got ",
											_mask.m_bits));

		// Get the number of exponent bits to preserve
		OP_REQUIRES_OK(context,
					   context->GetAttr("e_bits", &_mask.e_bits));
		// Check that e_bits is positive
		OP_REQUIRES(context, _mask.e_bits > 0,
					errors::InvalidArgument("Need e_bits > 0, got ",
											_mask.e_bits));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &input_tensor = context->input(0);

		// Create an output tensor
		Tensor *output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
														 &output_tensor));

		// Do the computation.
		OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint64max,
					errors::InvalidArgument("Too many elements in tensor"));
		Quantizer<Device, T>()(
			context->eigen_device<Device>(),
			static_cast<int64_t>(input_tensor.NumElements()),
			input_tensor.flat<T>().data(),
			output_tensor->flat<T>().data(),
			&_mask);
	}

private:
	struct QInfo _mask;
};

#define REGISTER_CPU_KERNEL(type)                                      \
	REGISTER_KERNEL_BUILDER(                                           \
		Name("Quantize").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
		QuantizeOp<CPUDevice, type>)

REGISTER_CPU_KERNEL(float);
//REGISTER_CPU_KERNEL(double);

#undef REGISTER_CPU_KERNEL