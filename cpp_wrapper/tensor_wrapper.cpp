#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../cpp/tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tensor, m) {
    py::class_<tensor::Matrix>(m, "Matrix")
            .def(py::init<std::vector<float> &&, int, int>())
            .def("__str__",[](const tensor::Matrix &matrix) {
                return static_cast<std::string>(matrix);
            })
            .def("__repr__",[](const tensor::Matrix &matrix) {
                return static_cast<std::string>(matrix);
            })
            .def("__eq__", [](const tensor::Matrix &lhs, const tensor::Matrix &rhs) {
                return lhs == rhs;
            });

    py::class_<tensor::Mul>(m, "Mul")
            .def(py::init<tensor::Matrix &, tensor::Matrix &>())
            .def("reorder", &tensor::Mul::reorder)
            .def("tile", &tensor::Mul::tile)
            .def("parallel", &tensor::Mul::parallel)
            .def("eval", &tensor::Mul::eval);
}