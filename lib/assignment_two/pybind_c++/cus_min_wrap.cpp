using namespace std;
#include <iostream>
#include "../src/cus_min.h"
#include <pybind11/pybind11.h>
#include<torch/torch.h>
#include <torch/extension.h>


namespace pyth = pybind11;
using namespace pyth;


PYBIND11_MODULE(custom_min, m) {
    class_<Cus_Min>(m, "Cus_Min")
        .def(py::init<>())
        .def("cus_min",&Cus_Min::cus_min,pybind11::arg("a"), pybind11::arg("b"));
}



