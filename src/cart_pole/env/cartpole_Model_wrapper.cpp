#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cart_pole_system_dynamics0.h"

namespace py = pybind11;

class PyModel : public cart_pole_system_dynamics0 {
public:
    using cart_pole_system_dynamics0::cart_pole_system_dynamics0;

    // void start() {
    //     multirotor0::start();
    // }    

    void initialize() {
        cart_pole_system_dynamics0::initialize();
    }

    void step() {
        cart_pole_system_dynamics0::step();
    }
    
    py::array_t<double> get_cart_position() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = cart_pole_system_dynamics0::cart_pole_system_dynamics0_Y.cart_position;
        return arr;
    }

    py::array_t<double> get_pole_angle() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = cart_pole_system_dynamics0::cart_pole_system_dynamics0_Y.pole_angle;
        return arr;
    }

    py::array_t<double> get_pole_position() {
        py::array_t<double> arr(2);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        for (int i = 0; i < 2; ++i) {
            ptr[i] = cart_pole_system_dynamics0::cart_pole_system_dynamics0_Y.pole_position[i];
        }
        return arr;
    }

    py::array_t<double> get_effort() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = cart_pole_system_dynamics0::cart_pole_system_dynamics0_Y.effort;
        return arr;
    }

    py::array_t<double> get_pole_angular_velocity() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = cart_pole_system_dynamics0::cart_pole_system_dynamics0_Y.pole_angular_velocity;
        return arr;
    }

    py::array_t<double> get_cart_velocity() {
        py::array_t<double> arr(1);
        auto buffer = arr.request();
        double *ptr = static_cast<double *>(buffer.ptr);
        ptr[0] = cart_pole_system_dynamics0::cart_pole_system_dynamics0_Y.cart_velocity;
        return arr;
    }

    void set_length(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 1) {
            throw std::runtime_error("Input array size must be 1");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 1; ++i) {
            cart_pole_system_dynamics0::cart_pole_system_dynamics0_U.length = ptr[i];
        }
    }

    void set_mass_pole(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 1) {
            throw std::runtime_error("Input array size must be 1");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 1; ++i) {
            cart_pole_system_dynamics0::cart_pole_system_dynamics0_U.mass_pole = ptr[i];
        }
    }

    void set_mass_cart(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 1) {
            throw std::runtime_error("Input array size must be 1");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 1; ++i) {
            cart_pole_system_dynamics0::cart_pole_system_dynamics0_U.mass_cart = ptr[i];
        }
    }

    void set_force_input_cart(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 1) {
            throw std::runtime_error("Input array size must be 1");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 1; ++i) {
            cart_pole_system_dynamics0::cart_pole_system_dynamics0_U.Force = ptr[i];
        }
    }

    void set_init_pole_pos(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 1) {
            throw std::runtime_error("Input array size must be 1");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 1; ++i) {
            cart_pole_system_dynamics0::cart_pole_system_dynamics0_U.init_pole_pos = ptr[i];
        }
    }
};

PYBIND11_MODULE(cartpole_Model_wrapper, m) {
    py::class_<PyModel>(m, "cart_pole_system_dynamics0")
        .def(py::init<>())
        .def("initialize", &PyModel::initialize)
        .def("step", &PyModel::step)
        .def("get_cart_position", &PyModel::get_cart_position)
        .def("get_pole_angle", &PyModel::get_pole_angle)
        .def("get_pole_position", &PyModel::get_pole_position)
        .def("get_effort", &PyModel::get_effort)
        .def("get_pole_angular_velocity", &PyModel::get_pole_angular_velocity)
        .def("get_cart_velocity", &PyModel::get_cart_velocity)
        .def("set_length", &PyModel::set_length)
        .def("set_mass_pole", &PyModel::set_mass_pole)
        .def("set_mass_cart", &PyModel::set_mass_cart)
        .def("set_force_input_cart", &PyModel::set_force_input_cart)
        .def("set_init_pole_pos", &PyModel::set_init_pole_pos);

}
