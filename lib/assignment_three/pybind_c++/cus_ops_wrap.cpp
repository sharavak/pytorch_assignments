#include <iostream>
#include "../src/cus_ops.h"
#include <pybind11/pybind11.h>
#include<torch/torch.h>
#include <torch/extension.h>

using namespace std;


namespace pyth = pybind11;
using namespace pyth;


PYBIND11_MODULE(custom_ops, m) {
    class_<Custom_Ops>(m, "Custom_Ops_Wrapper")
        .def(pyth::init<>())
        .def("cus_logaddexp",&Custom_Ops::cus_logaddexp,pybind11::arg("a"), pybind11::arg("b"))
        .def("cus_addbmm",&Custom_Ops::cus_addbmm,pybind11::arg("a"),pybind11::arg("b"),pybind11::arg("c"));
}