#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "multirotor0.h"

namespace py = pybind11;

class PyModel : public multirotor0 {
public:
    using multirotor0::multirotor0;

    // void start() {
    //     multirotor0::start();
    // }    

    void initialize() {
        multirotor0::initialize();
    }

    void step0() {
        multirotor0::step0();
    }
    
    void step2() {
        multirotor0::step2();
    }


    py::array_t<double> get_pos_world() {
        return py::array_t<double>(3, multirotor0::multirotor0_Y.X_i);
    }
    py::array_t<double> get_RotationMatrix_world() {
        return py::array_t<double>(9, multirotor0::multirotor0_Y.DCM_ib);
    }
    py::array_t<double> get_velocity_world() {
        return py::array_t<double>(3, multirotor0::multirotor0_Y.V_i);
    }
    py::array_t<double> get_omega_world() {
        return py::array_t<double>(3, multirotor0::multirotor0_Y.omega);
    }
    py::array_t<double> get_motor_rpm() {
        return py::array_t<double>(4, multirotor0::multirotor0_Y.motor_RPM);
    }

    void set_w_0(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.RPMcommands[i] = ptr[i];
        }
    }

    void set_wind_vector(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 3) {
            throw std::runtime_error("Input array size must be 3");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 3; ++i) {
            multirotor0::multirotor0_U.Wind_i[i] = ptr[i];
        }
    }

    void set_force_disturb_vector(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 3) {
            throw std::runtime_error("Input array size must be 3");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 3; ++i) {
            multirotor0::multirotor0_U.Force_disturb[i] = ptr[i];
        }
    }

    void set_moment_disturb_vector(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 3) {
            throw std::runtime_error("Input array size must be 3");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 3; ++i) {
            multirotor0::multirotor0_U.Moment_disturb[i] = ptr[i];
        }
    }

    void set_init_pos(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 3) {
            throw std::runtime_error("Input array size must be 3");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 3; ++i) {
            multirotor0::Model_Init.pos_init[i] = ptr[i];
        }
    }
    void set_init_vel(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 3) {
            throw std::runtime_error("Input array size must be 3");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 3; ++i) {
            multirotor0::Model_Init.vel_init[i] = ptr[i];
        }
    }

    void set_init_omega(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 3) {
            throw std::runtime_error("Input array size must be 3");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 3; ++i) {
            multirotor0::Model_Init.omega_init[i] = ptr[i];
        }
    }

    void set_arm_length(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.arm_length[i] = ptr[i];
        }
    }

    void set_propeller_height(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.prop_height[i] = ptr[i];
        }
    }

    void set_propeller_diameter(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.prop_diameter[i] = ptr[i];
        }
    }

    void set_rotation_direction(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.rotation_direction[i] = ptr[i];
        }
    }    

    void set_max_rpm(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.max_rpm[i] = ptr[i];
        }
    }

    void set_min_rpm(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.min_rpm[i] = ptr[i];
        }
    }
    void set_arm_radius(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.arm_radius[i] = ptr[i];
        }
    }

    void set_motor_arm_angle(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 4) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 4; ++i) {
            multirotor0::multirotor0_U.Motor_arm_angle[i] = ptr[i];
        }
    }

    void set_mass_center(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 1) {
            throw std::runtime_error("Input array size must be 1");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 1; ++i) {
            multirotor0::multirotor0_U.mass_center = ptr[i];
        }
    }

    void set_COM_mass_center(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 3) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 3; ++i) {
            multirotor0::multirotor0_U.COM_mass_center[i] = ptr[i];
        }
    }

    void set_Surface_params(py::array_t<double> input_array) {
        py::buffer_info buf_info = input_array.request();
        if (buf_info.size != 3) {
            throw std::runtime_error("Input array size must be 4");
        }
        double *ptr = static_cast<double *>(buf_info.ptr);
        for (int i = 0; i < 3; ++i) {
            multirotor0::multirotor0_U.Surface_params[i] = ptr[i];
        }
    }

};

PYBIND11_MODULE(Model_wrapper, m) {
    py::class_<PyModel>(m, "multirotor0")
        .def(py::init<>())
        .def("initialize", &PyModel::initialize)
        .def("step0", &PyModel::step0)
        .def("step2", &PyModel::step2)
        .def("get_pos_world", &PyModel::get_pos_world)
        .def("get_RotationMatrix_world", &PyModel::get_RotationMatrix_world)
        .def("get_velocity_world", &PyModel::get_velocity_world)
        .def("get_omega_world", &PyModel::get_omega_world)
        .def("get_motor_rpm", &PyModel::get_motor_rpm)
        .def("set_w_0", &PyModel::set_w_0)
        .def("set_wind_vector", &PyModel::set_wind_vector)
        .def("set_force_disturb_vector", &PyModel::set_force_disturb_vector)
        .def("set_moment_disturb_vector", &PyModel::set_moment_disturb_vector)
        .def("set_init_pos", &PyModel::set_init_pos)
        .def("set_init_vel", &PyModel::set_init_vel)
        .def("set_init_omega", &PyModel::set_init_omega)
        .def("set_arm_length", &PyModel::set_arm_length)
        .def("set_propeller_height", &PyModel::set_propeller_height)
        .def("set_propeller_diameter", &PyModel::set_propeller_diameter)
        .def("set_rotation_direction", &PyModel::set_rotation_direction)
        .def("set_max_rpm", &PyModel::set_max_rpm)
        .def("set_min_rpm", &PyModel::set_min_rpm)
        .def("set_arm_radius", &PyModel::set_arm_radius)
        .def("set_motor_arm_angle", &PyModel::set_motor_arm_angle)
        .def("set_mass_center", &PyModel::set_mass_center)
        .def("set_COM_mass_center", &PyModel::set_COM_mass_center)
        .def("set_Surface_params", &PyModel::set_Surface_params);
}
