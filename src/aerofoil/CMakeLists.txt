cmake_minimum_required(VERSION 3.0.2)
project(aerofoil)

if(DEFINED ENV{CMAKE_EXPORT_COMPILE_COMMANDS} AND NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS $ENV{CMAKE_EXPORT_COMPILE_COMMANDS})
endif()
## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS roscpp rospy tf)

catkin_package(
)

find_library(FOUND_LIBM m NO_SYSTEM_ENVIRONMENT_PATH PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

include_directories(${catkin_INCLUDE_DIRS})

set(pybind11_DIR "/home/divij/anaconda3/envs/cs285/lib/python3.7/site-packages/pybind11/share/cmake/pybind11")

set(PYBIND11_PYTHON_VERSION 3.7)
find_package(pybind11 REQUIRED)

pybind11_add_module(aerofoil_wrapper env/aerofoil_wrapper.cpp
    env/aerofoil.cpp
    )

