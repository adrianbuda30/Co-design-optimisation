#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "aerofoil.h"

namespace py = pybind11;

class PyModel : public aerofoil {
public:
    using aerofoil::aerofoil;

    void initialize() {
        aerofoil::initialize();
    }

    void step() {
        aerofoil::step();
    }
    
    py::array_t<double> get_pitch() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = aerofoil::aerofoil_Y.pitch;
        return arr;
    }
    py::array_t<double> get_plunge() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = aerofoil::aerofoil_Y.plunge;
        return arr;
    }
    py::array_t<double> get_delta() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = aerofoil::aerofoil_Y.delta;
        return arr;
    }

    py::array_t<double> get_C_L() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = aerofoil::aerofoil_Y.C_L;
        return arr;
    }

     py::array_t<double> get_plunge_dot() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = aerofoil::aerofoil_Y.plunge_dot;
        return arr;
    }   

     py::array_t<double> get_pitch_dot() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = aerofoil::aerofoil_Y.pitch_dot;
        return arr;
    }   

     py::array_t<double> get_delta_dot() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = aerofoil::aerofoil_Y.delta_dot;
        return arr;
    }   

    void set_MatrixA(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 144) {
            throw std::runtime_error("Input array size must be 144");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 144; ++i) {
            aerofoil::aerofoil_U.MatrixA[i] = ptr[i];
            
        }
    }

    void set_MatrixB(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 12) {
            throw std::runtime_error("Input array size must be 12");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 12; ++i) {
            aerofoil::aerofoil_U.MatrixB[i] = ptr[i];
        }
    }

    void set_MatrixC(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 12) {
            throw std::runtime_error("Input array size must be 12");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 12; ++i) {
            aerofoil::aerofoil_U.MatrixC[i] = ptr[i];
        }
    }

    void set_init_state(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 12) {
            throw std::runtime_error("Input array size must be 12");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 12; ++i) {
            aerofoil::aerofoil_U.init_state[i] = ptr[i];
        }
    }

    void set_delta_ddot(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 1) {
            throw std::runtime_error("Input array size must be 1");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 1; ++i) {
            aerofoil::aerofoil_U.delta_ddot = ptr[i];
        }
    }
};

PYBIND11_MODULE(aerofoil_wrapper, m) {
    py::class_<PyModel>(m, "aerofoil")
        .def(py::init<>())
        .def("initialize", &PyModel::initialize)
        .def("step", &PyModel::step)
        .def("get_pitch", &PyModel::get_pitch)
        .def("get_plunge", &PyModel::get_plunge)
        .def("get_delta", &PyModel::get_delta)
        .def("get_C_L", &PyModel::get_C_L)
        .def("get_plunge_dot", &PyModel::get_plunge_dot)
        .def("get_pitch_dot", &PyModel::get_pitch_dot)
        .def("get_delta_dot", &PyModel::get_delta_dot)
        .def("set_MatrixA", &PyModel::set_MatrixA)
        .def("set_MatrixB", &PyModel::set_MatrixB)
        .def("set_MatrixC", &PyModel::set_MatrixC)
        .def("set_init_state", &PyModel::set_init_state)
        .def("set_delta_ddot", &PyModel::set_delta_ddot);
}
