#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
  
namespace py = pybind11;

template <class T>
T m_floats_distance(const T& a, const T& b)
{
   //
   // Error handling:
   //
   if(std::isfinite(a) ^ std::isfinite(b))
      return std::numeric_limits<T>::max();

   //
   // Special cases:
   //
   if(a == b)
      return T(0);
   if(std::isinf(a) && std::isinf(b))
      return T(0);
   if(std::isnan(a) && std::isnan(b))
      return T(0);
   if(a > b)
      return m_floats_distance<T>(b, a);
   if(a == 0)
      return 1 + std::fabs(m_floats_distance<T>(static_cast<T>((b < 0) ? T(-std::numeric_limits<T>::epsilon()) :
                                                                         std::numeric_limits<T>::epsilon()),
                                                b));
   if(b == 0)
      return 1 + std::fabs(m_floats_distance<T>(static_cast<T>((a < 0) ? T(-std::numeric_limits<T>::epsilon()) :
                                                                         std::numeric_limits<T>::epsilon()),
                                                a));
   if((a > 0) != (b > 0))
      return 2 +
             std::fabs(m_floats_distance<T>(
                 static_cast<T>((b < 0) ? T(-std::numeric_limits<T>::epsilon()) : std::numeric_limits<T>::epsilon()),
                 b)) +
             std::fabs(m_floats_distance<T>(
                 static_cast<T>((a < 0) ? T(-std::numeric_limits<T>::epsilon()) : std::numeric_limits<T>::epsilon()),
                 a));
   //
   // By the time we get here, both a and b must have the same sign, we want
   // b > a and both positive for the following logic:
   //
   if(a < 0)
      return m_floats_distance<T>(static_cast<T>(-b), static_cast<T>(-a));

   assert(a >= 0);
   assert(b >= a);

   int expon;
   //
   // Note that if a is a denorm then the usual formula fails
   // because we actually have fewer than std::numeric_limits<T>::digits
   // significant bits in the representation:
   //
   (void)std::frexp(((std::fpclassify)(a) == (int)FP_SUBNORMAL) ? std::numeric_limits<T>::min() : a, &expon);
   T upper = std::ldexp(T(1), expon);
   T result = T(0);
   //
   // If b is greater than upper, then we *must* split the calculation
   // as the size of the ULP changes with each order of magnitude change:
   //
   if(b > upper)
   {
      int expon2;
      (void)std::frexp(b, &expon2);
      T upper2 = std::ldexp(T(0.5), expon2);
      result = m_floats_distance<T>(upper2, b);
      result += (expon2 - expon - 1) * std::ldexp(T(1), std::numeric_limits<T>::digits - 1);
   }
   //
   // Use compensated double-double addition to avoid rounding
   // errors in the subtraction:
   //
   expon = std::numeric_limits<T>::digits - expon;
   T mb, x, y, z;
   if(((std::fpclassify)(a) == (int)FP_SUBNORMAL) || (b - a < std::numeric_limits<T>::min()))
   {
      //
      // Special case - either one end of the range is a denormal, or else the difference is.
      // The regular code will fail if we're using the SSE2 registers on Intel and either
      // the FTZ or DAZ flags are set.
      //
      T a2 = std::ldexp(a, std::numeric_limits<T>::digits);
      T b2 = std::ldexp(b, std::numeric_limits<T>::digits);
      mb = -(std::min)(T(std::ldexp(upper, std::numeric_limits<T>::digits)), b2);
      x = a2 + mb;
      z = x - a2;
      y = (a2 - (x - z)) + (mb - z);

      expon -= std::numeric_limits<T>::digits;
   }
   else
   {
      mb = -(std::min)(upper, b);
      x = a + mb;
      z = x - a;
      y = (a - (x - z)) + (mb - z);
   }
   if(x < 0)
   {
      x = -x;
      y = -y;
   }
   result += std::ldexp(x, expon) + std::ldexp(y, expon);
   //
   // Result must be an integer:
   //
   assert(result == std::floor(result));
   return result;
}

template <class T>
T m_float_distance(const T& a, const T& b)
{
   return m_floats_distance<T>((std::fpclassify)(a) == (int)FP_SUBNORMAL ? copysign(T(0), a) : a,
                               (std::fpclassify)(b) == (int)FP_SUBNORMAL ? copysign(T(0), b) : b);
}

void print_f_as_bin(float value) {
    // Create an integer with the same binary representation as the float
    uint32_t intValue;
    std::memcpy(&intValue, &value, sizeof(value));

    // Extract and print each bit
    for (int i = 31; i >= 0; i--) {
        int bit = (intValue >> i) & 1;
        std::cout << bit;
    }

    std::cout << std::endl;
}

std::vector<float> compute(std::vector<float> &v) 
{
    if (!v.empty())
    {
        std::vector<float> res;
        std::sort(v.begin(), v.end());
        float if_emp = v[0];
        v.erase(std::unique(v.begin(), v.end() ), v.end());
        if(v.size() == 1)
        {
            res.push_back(m_floats_distance<float>(if_emp,if_emp));
            return res;
        }
        std::vector<float>::iterator end = v.end() - 1;
        std::vector<float>::iterator it = v.begin();
        for (; it != end; ++it)
        {
            float f = m_floats_distance<float>(*(it), *(it+1));
            res.push_back(f);
          /*  if(f > std::pow(2, 24))
            {
                std::cout << "ULP: " << f << "\n1st val: " << *(it) << " bin: ";
                print_f_as_bin(*(it));
                std::cout << "2nd val: " << *(it+1) << " bin: ";
                print_f_as_bin(*(it+1));
                std::cout << "\n";
            } */
        }
        return res;
    }
    else
    {
        std::cout << "EMPTY" << std::endl;
        return std::vector<float>();
    }
}

PYBIND11_MODULE(ulp, m) {
    m.doc() = R"pbdoc(
        Pybind11 ulp computator. Used for parallel execution
        -----------------------
        .. currentmodule:: pybind11_extension
        .. autosummary::
        :toctree: _generate
    )pbdoc";

    m.def("compute", &compute);
    m.def("compute_nogil", &compute, py::call_guard<py::gil_scoped_release>());
}
