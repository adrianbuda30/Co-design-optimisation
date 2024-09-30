######################################################################
# CMakeLists.txt generated for component aerofoil
# Product type: executable
######################################################################
cmake_minimum_required(VERSION 3.12)
project(aerofoil)
if(DEFINED ENV{CMAKE_EXPORT_COMPILE_COMMANDS} AND NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS $ENV{CMAKE_EXPORT_COMPILE_COMMANDS})
endif()


######################################################################
## Path variables
######################################################################
get_filename_component(START_DIR .. ABSOLUTE)
set(MATLAB_ROOT /usr/local/MATLAB/R2023a CACHE PATH "")
if(${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(BUILDING_IN_SOURCE_DIR TRUE)
    set(BINARY_START_DIR "${START_DIR}")
else()
    set(BUILDING_IN_SOURCE_DIR FALSE)
    set(BINARY_START_DIR "${CMAKE_BINARY_DIR}")
endif()
if(WIN32 AND NOT MINGW)
    set(SYSLIB_PREFIX "lib")
else()
    set(SYSLIB_PREFIX "")
endif()

######################################################################
## System libraries
######################################################################
find_library(FOUND_LIBM m NO_SYSTEM_ENVIRONMENT_PATH PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
    ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

######################################################################
## Target definition and commands
######################################################################
add_executable(aerofoil  ${MATLAB_ROOT}/rtw/c/src/rt_logging.c
    ${START_DIR}/aerofoil_grt_rtw/aerofoil.cpp
    ${START_DIR}/aerofoil_grt_rtw/aerofoil_data.cpp
    ${START_DIR}/aerofoil_grt_rtw/rtGetInf.cpp
    ${START_DIR}/aerofoil_grt_rtw/rtGetNaN.cpp
    ${START_DIR}/aerofoil_grt_rtw/rt_nonfinite.cpp
    ${MATLAB_ROOT}/rtw/c/src/common/rt_cppclass_main.cpp)
set_target_properties(aerofoil PROPERTIES
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY "${BINARY_START_DIR}/./$<0:>"
    LIBRARY_OUTPUT_DIRECTORY "${BINARY_START_DIR}/./$<0:>"
    ARCHIVE_OUTPUT_DIRECTORY "${BINARY_START_DIR}/./$<0:>")
target_compile_features(aerofoil PUBLIC cxx_std_11)
target_compile_definitions(aerofoil PRIVATE -DMODEL=aerofoil
    -DNUMST=2
    -DNCSTATES=14
    -DHAVESTDIO
    -DRT
    -DUSE_RTMODEL
    -DCLASSIC_INTERFACE=0
    -DALLOCATIONFCN=0
    -DTID01EQ=1
    -DMAT_FILE=1
    -DONESTEPFCN=1
    -DTERMFCN=1
    -DMULTI_INSTANCE_CODE=1
    -DINTEGER_CODE=0
    -DMT=0)
target_include_directories(aerofoil PRIVATE ${START_DIR}
    ${START_DIR}/aerofoil_grt_rtw
    ${MATLAB_ROOT}/extern/include
    ${MATLAB_ROOT}/simulink/include
    ${MATLAB_ROOT}/rtw/c/src
    ${MATLAB_ROOT}/rtw/c/src/ext_mode/common)
target_link_libraries(aerofoil PRIVATE $<$<BOOL:${FOUND_LIBM}>:m>)
if(WIN32)
   target_link_libraries(aerofoil PRIVATE kernel32
       ws2_32
       mswsock
       advapi32)
endif()

######################################################################
## Extract debug symbols
######################################################################
if(APPLE AND (CMAKE_C_COMPILER_ID STREQUAL "AppleClang"))
    string(APPEND _dsymutilcmd "$<IF:$<CONFIG:DEBUG,RELWITHDEBINFO>,"
                "xcrun;dsymutil;$<TARGET_FILE:aerofoil>;--flat,"
                "true"
        >)
    add_custom_command(TARGET aerofoil
            POST_BUILD
            COMMAND "${_dsymutilcmd}"
            COMMAND_EXPAND_LISTS
            BYPRODUCTS aerofoil.dwarf)
endif()


######################################################################
## Build success message
######################################################################
add_custom_command(TARGET aerofoil POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "\\#\\#\\# Created executable: $<TARGET_FILE:aerofoil>")

######################################################################
## Call toolchain hook function if defined
######################################################################
if(COMMAND toolchain_target_hook)
    toolchain_target_hook(aerofoil)
endif()
