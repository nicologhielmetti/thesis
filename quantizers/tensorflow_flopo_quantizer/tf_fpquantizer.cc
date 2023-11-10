#include "tf_fpquantizer.hpp"

#include "tensorflow/core/framework/op_kernel.h"
#include <cstdint>
#include <type_traits>
#include <cassert>
#include <cmath>

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

const uint8_t m_width = 23;
const uint8_t e_width = 8;
const uint16_t e_bias = 127;

// CPU specialization of actual computation.
template <typename T>
struct Quantizer<CPUDevice, T>
{
	//using view_type = typename std::conditional<std::is_same<T, double>::value, uint64_t, uint32_t>::type;

    void operator()(const CPUDevice &d, int64_t size, const T *in, T *out, const QInfo *mask)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            if(in[i] == 0.0 || in[i] == -0.0)
            {
                out[i] = in[i];
                continue;
            }
            int input_exp = std::ilogb(in[i]);
            if (input_exp > mask->max_exp)
                out[i] = mask->max_val;
            else if (input_exp < mask->min_exp)
                out[i] = mask->min_val;
            else
            {
                uint32_t qm = *((uint32_t *)&in[i]) & mask->qm_mask;
                uint32_t qe = input_exp + e_bias;
                uint32_t qout = qm | (qe << m_width);
                out[i] = std::copysign(*((T *)&qout), in[i]);
            }
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
		OP_REQUIRES_OK(context,
					   context->GetAttr("exp_offset", &_mask.exp_offset));
		
		// Check that exp_offset is in the fp32 exp range
		OP_REQUIRES(context, _mask.exp_offset >= -126 && _mask.exp_offset < 126,
					errors::InvalidArgument("Need exp_offset >= -126 and < 126, got ",
											_mask.exp_offset));
		OP_REQUIRES_OK(context,
					   context->GetAttr("use_exp_offset", &_mask.use_exp_offset));
		
		// Check that use_exp_offset is 0 or 1
		OP_REQUIRES(context, _mask.use_exp_offset == 0 || _mask.use_exp_offset == 1,
					errors::InvalidArgument("Need use_exp_offset in {0,1}, got ",
											_mask.use_exp_offset));
					   
	    OP_REQUIRES_OK(context,
					   context->GetAttr("ret_inf_on_ovf", &_mask.ret_inf_on_ovf));
		
		// Check that ret_inf_on_ovf is 0 or 1
		OP_REQUIRES(context, _mask.ret_inf_on_ovf == 0 || _mask.ret_inf_on_ovf == 1,
					errors::InvalidArgument("Need ret_inf_on_ovf in {0,1}, got ",
											_mask.ret_inf_on_ovf));
		
		OP_REQUIRES_OK(context,
					   context->GetAttr("debug", &_mask.debug));
		
		// Check that debug is 0 or 1
		OP_REQUIRES(context, _mask.debug == 0 || _mask.debug == 1,
					errors::InvalidArgument("Need debug in {0,1}, got ",
											_mask.debug));
		
		
		_mask.qm_mask = (((1ULL << m_width) - 1) & ~((1ULL << (m_width - _mask.m_bits)) - 1));
		// compute max and minimum exponent for saturation
		// max exclude inf
		// min exclude subnormals
		if(_mask.use_exp_offset)
		{
		    _mask.max_exp =  std::pow(2, _mask.e_bits - 1) - 1 + _mask.exp_offset;
		    _mask.min_exp = -std::pow(2, _mask.e_bits - 1) + 2 + _mask.exp_offset;
		}
		else
		{
		    _mask.max_exp =  std::pow(2, _mask.e_bits - 1) - 1;
		    _mask.min_exp = -std::pow(2, _mask.e_bits - 1) + 2;
		}

		//assertm(_mask.max_exp <=  126, "max_exp <=  126!");
		//assertm(_mask.min_exp >= -126, "min_exp >= -126!");
		
		if(_mask.ret_inf_on_ovf)
		{
		    _mask.max_val =  std::numeric_limits<float>::infinity();
		    _mask.min_val = -std::numeric_limits<float>::infinity();
		}
		else
		{
		    uint32_t biased_max_exp = _mask.max_exp + e_bias;
		    uint32_t biased_min_exp = _mask.min_exp + e_bias;
		    float frac = 0;
		    for(int i = 0; i < _mask.m_bits; ++i)
		        frac += std::pow(2, -(i+1));;
		    _mask.max_val = +1 * std::pow(2, _mask.max_exp) * (1.0 + frac);
		    _mask.min_val = -1 * std::pow(2, _mask.min_exp) * (1.0 + frac);
		}

        //std::cout << "Constructor FP quantizer" << std::endl;
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

//	void operator()(const CPUDevice &d, int64_t size, const T *in, T *out, const QInfo *mask)
//	{
//		const uint8_t m_width = 23;
//		const uint8_t e_width = 8;
//		const uint16_t e_bias = 127;
//
//		const view_type qm_mask = (1ULL << (e_width + m_width)) | (((1ULL << m_width) - 1) & ~((1ULL << (m_width - mask->m_bits)) - 1));
//		const view_type e_mask = (1ULL << e_width) - 1;
//		view_type qe_max;
//		view_type qe_min;
//		uint16_t real_bias = mask->use_exp_offset == 0 ? e_bias : (e_bias + mask->exp_offset);
//		qe_max = real_bias + (1ULL << (mask->e_bits - 1)) - 1;
//		qe_min = real_bias - (1ULL << (mask->e_bits - 1)) + 1;
//		for (int64_t i = 0; i < size; ++i)
//		{
//			view_type qm = *((view_type *)&in[i]) & qm_mask;
//			view_type e = (*((view_type *)&in[i]) >> m_width) & e_mask;
//			view_type qe;			
//            if(e <= qe_min)
//            {
//                if(mask->ret_inf_on_ovf)
//                {
//                    out[i] = std::numeric_limits<float>::infinity();
//                    continue;
//                }
//                else
//                {
//                    qe = qe_min;
//                    if(mask->debug)
//                        std::cout << 'm' << std::endl;
//                }
//            }
//            else if(e >= qe_max)
//            {
//                if(mask->ret_inf_on_ovf)
//                {
//                    out[i] = -std::numeric_limits<float>::infinity();
//                    continue;
//                }
//                else
//                {
//                    qe = qe_max;
//                    if(mask->debug)
//                        std::cout << 'M' << std::endl;
//                }
//            }
//            else
//            {
//                qe = e;
//            }
//            view_type qout = qm | (qe << m_width);
//            out[i] = *((T *)&qout);
//            if(mask->debug)
//            {
//                if(std::isnan(out[i]))
//			        std::cout << 'n' << std::endl;
//			    if(std::isinf(out[i]))
//			        std::cout << 'i' << std::endl;
//            }
//		}
//	}