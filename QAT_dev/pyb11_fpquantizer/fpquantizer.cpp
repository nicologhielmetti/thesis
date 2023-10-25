#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <limits>
  
namespace py = pybind11;

const unsigned short f32_m_bits = 23;
const unsigned short f32_e_bits = 8;
const unsigned short f32_bits = 32;
const u_int32_t ieee_bias = (1 << (f32_e_bits - 1)) - 1;

//void print_f_as_bin(float value) {
//    // Create an integer with the same binary representation as the float
//    uint32_t intValue;
//    std::memcpy(&intValue, &value, sizeof(value));
//
//    std::cout << "0b";
//    // Extract and print each bit
//    for (int i = 31; i >= 0; i--) {
//        int bit = (intValue >> i) & 1;
//        std::cout << bit;
//    }
//
//    std::cout << std::endl;
//}

u_int32_t po2_pow32(const unsigned short &exp)
{
    return (1U << exp);
}

float closest_to_zero(const u_int32_t &est_bias, const unsigned short &e_bits, const unsigned short &m_bits)
{
    u_int32_t sign = 0;
    u_int32_t exp = ieee_bias - po2_pow32(e_bits - 1) + est_bias;
    u_int32_t man = 0;
    exp = exp << f32_m_bits;
    u_int32_t bits = sign | exp | man;
    return (*((float*)(&bits)));
}

float delta(const u_int32_t &est_bias, const unsigned short &e_bits, const unsigned short &m_bits)
{
    float closest = closest_to_zero(est_bias, e_bits, m_bits);
    u_int32_t bits = (*((u_int32_t*)(&closest)));
    u_int32_t man = 1 << (f32_m_bits - m_bits);
    bits = bits | man;
    float closest_man_lsb_1 = (*((float*)(&bits)));
    float delta = closest_man_lsb_1 - closest;
    // print_f_as_bin(delta);
    return delta;
}

float closest_to_zero(const unsigned short &e_bits, const unsigned short &m_bits)
{
    u_int32_t sign = 0;
    u_int32_t exp = ieee_bias - po2_pow32(e_bits - 1);
    u_int32_t man = 0;
    exp = exp << f32_m_bits;
    u_int32_t bits = sign | exp | man;
    return (*((float*)(&bits)));
}

float delta(const unsigned short &e_bits, const unsigned short &m_bits)
{
    float closest = closest_to_zero(e_bits, m_bits);
    u_int32_t bits = (*((u_int32_t*)(&closest)));
    u_int32_t man = 1 << (f32_m_bits - m_bits);
    bits = bits | man;
    float closest_man_lsb_1 = (*((float*)(&bits)));
    float delta = closest_man_lsb_1 - closest;
    // print_f_as_bin(delta);
    return delta;
}

float max(const unsigned short &e_bits, const unsigned short &m_bits)
{
    u_int32_t sign = 0;
    u_int32_t exp = po2_pow32(e_bits - 1) - 1 + ieee_bias;
    u_int32_t man = (1 << m_bits) - 1;
    exp = exp << f32_m_bits;
    man = man << (f32_m_bits - m_bits);
    u_int32_t bits = sign | exp | man;
    return (*((float*)(&bits)));
}

float min(const unsigned short &e_bits, const unsigned short &m_bits)
{
    return -max(e_bits, m_bits);
}

float quantize(const float &input, const unsigned short &e_bits, const unsigned short &m_bits)
{
    u_int32_t bits = (*((u_int32_t*)(&input)));
    u_int32_t exp_mask = (1 << f32_e_bits) - 1;
    u_int32_t sig_mask = (1 << f32_m_bits) - 1;
    u_int32_t sign = bits & (1 << (f32_bits - 1));
    u_int32_t man = bits & sig_mask;
    u_int32_t exp = ((bits >> f32_m_bits) & exp_mask);
    if (exp == 0)
        bits = 0;
    else
        if (exp == exp_mask)
        {
            if (man != 0)
                bits = (exp << f32_m_bits) | (1 << (f32_m_bits - 1));
            // else return +/- infinity as is
        }
        else
        {
            u_int32_t exp_min = -po2_pow32(e_bits - 1) + 1 + ieee_bias;
            u_int32_t exp_max =  po2_pow32(e_bits - 1) - 1 + ieee_bias;
            if (exp > exp_max)
                return std::numeric_limits<float>::infinity(); // exp = exp_max;
            if (exp < exp_min)
                return -std::numeric_limits<float>::infinity(); // exp = exp_min;
            exp = (exp & exp_mask) << f32_m_bits;
            man = man & ~((1 << (f32_m_bits - m_bits)) - 1) & sig_mask;
            bits = sign | exp | man;
        }
    return (*((float*)(&bits)));
}

float quantize(const float &input, const u_int32_t &est_bias, const unsigned short &e_bits, const unsigned short &m_bits)
{
    u_int32_t bits = (*((u_int32_t*)(&input)));
    u_int32_t exp_mask = (1 << f32_e_bits) - 1;
    u_int32_t sig_mask = (1 << f32_m_bits) - 1;
    u_int32_t sign = bits & (1 << (f32_bits - 1));
    u_int32_t man = bits & sig_mask;
    u_int32_t exp = ((bits >> f32_m_bits) & exp_mask);
    if (exp == 0)
        bits = 0;
    else
        if (exp == exp_mask)
        {
            if (man != 0)
                bits = (exp << f32_m_bits) | (1 << (f32_m_bits - 1));
            // else return +/- infinity as is
        }
        else
        {
            u_int32_t exp_min = -po2_pow32(e_bits - 1) + 1 + ieee_bias + est_bias;
            u_int32_t exp_max =  po2_pow32(e_bits - 1) - 1 + ieee_bias + est_bias;
            if (exp > exp_max)
                return std::numeric_limits<float>::infinity(); // exp = exp_max;
            if (exp < exp_min)
                return -std::numeric_limits<float>::infinity(); // exp = exp_min;
            exp = (exp & exp_mask) << f32_m_bits;
            man = man & ~((1 << (f32_m_bits - m_bits)) - 1) & sig_mask;
            bits = sign | exp | man;
        }
    return (*((float*)(&bits)));
}

std::vector<float> compute(std::vector<float> &v, const u_int32_t &est_bias, const unsigned short &e_bits, const unsigned short &m_bits) 
{
    if (!v.empty()) 
    {
        std::vector<float> res;
        std::vector<float>::iterator end = v.end();
        std::vector<float>::iterator it = v.begin();
        for (; it != end; ++it)
        {
            float f = quantize(*(it), est_bias, e_bits, m_bits);
            res.push_back(f);
        }
        return res;
    }
    else
    {
        std::cout << "EMPTY" << std::endl;
        return std::vector<float>();
    }
}

std::vector<float> compute(std::vector<float> &v, const unsigned short &e_bits, const unsigned short &m_bits) 
{
    if (!v.empty()) 
    {
        std::vector<float> res;
        std::vector<float>::iterator end = v.end();
        std::vector<float>::iterator it = v.begin();
        for (; it != end; ++it)
        {
            float f = quantize(*(it), e_bits, m_bits);
            res.push_back(f);
        }
        return res;
    }
    else
    {
        std::cout << "EMPTY" << std::endl;
        return std::vector<float>();
    }
}

PYBIND11_MODULE(fpquantizer, m) {
    m.doc() = R"pbdoc(
        Pybind11 flopo quantize computator. Used for parallel execution
        -----------------------
        .. currentmodule:: pybind11_extension
        .. autosummary::
        :toctree: _generate
    )pbdoc";

    m.def("compute", py::overload_cast<std::vector<float>&, const unsigned short &, const unsigned short &>(&compute));    
    m.def("compute", py::overload_cast<std::vector<float>&, const u_int32_t &, const unsigned short &, const unsigned short &>(&compute));
    m.def("compute_nogil", py::overload_cast<std::vector<float>&, const unsigned short &, const unsigned short &>(&compute), py::call_guard<py::gil_scoped_release>());
    m.def("compute_nogil", py::overload_cast<std::vector<float>&, const u_int32_t &, const unsigned short &, const unsigned short &>(&compute), py::call_guard<py::gil_scoped_release>());
    m.def("max", &max);
    m.def("min", &min);
    m.def("closest_to_zero", py::overload_cast<const u_int32_t &, const unsigned short&, const unsigned short&>(&closest_to_zero));
    m.def("closest_to_zero", py::overload_cast<const unsigned short&, const unsigned short&>(&closest_to_zero));
    m.def("delta", py::overload_cast<const u_int32_t &, const unsigned short&, const unsigned short&>(&delta));
    m.def("delta", py::overload_cast<const unsigned short&, const unsigned short&>(&delta));
}
