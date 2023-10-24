#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
  
namespace py = pybind11;

std::vector<int8_t> compute(std::vector<float> &v) 
{
    if (!v.empty())
    {
        std::vector<int8_t> res;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end() ), v.end());
        for (auto f : v)
            res.push_back(((int8_t) ((*((unsigned*)(&f))) >> 23)) - 127);
        return res;
    }
    else
    {
        std::cout << "EMPTY" << std::endl;
        return std::vector<int8_t>();
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
