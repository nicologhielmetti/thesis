#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
  
namespace py = pybind11;

//#define FP_ILOGB0 0

std::vector<int> compute(std::vector<float> &v) 
{
    if (!v.empty())
    {
        std::vector<int> res;
        //std::sort(v.begin(), v.end());
        //v.erase(std::unique(v.begin(), v.end() ), v.end());
        for (auto f : v)
            if(f == 0.0)
            {
                res.push_back(0);
            }
            else if (std::isinf(f) || std::isnan(f))
            {
                res.push_back(128);
            }
            else
                res.push_back(std::ilogb(f));
        return res;
    }
    else
    {
        std::cout << "EMPTY" << std::endl;
        return std::vector<int>();
    }
}

PYBIND11_MODULE(exp, m) {
    m.doc() = R"pbdoc(
        Pybind11 exp computator. Used for parallel execution
        -----------------------
        .. currentmodule:: pybind11_extension
        .. autosummary::
        :toctree: _generate
    )pbdoc";

    m.def("compute", &compute);
    m.def("compute_nogil", &compute, py::call_guard<py::gil_scoped_release>());
}
