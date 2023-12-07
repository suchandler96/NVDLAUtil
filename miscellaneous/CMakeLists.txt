cmake_minimum_required(VERSION 3.25)
project(gem5-NVDLA)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)
include_directories(include)
include_directories(build/ARM)
include_directories(ext/rtl/model_nvdla)
include_directories(ext/rtl/model_nvdla/verilator_nvdla)
include_directories(~/Software/gurobi1003/linux64/include)

file( GLOB_RECURSE APP_SOURCES src/*.c* src/*.cpp src/*.c ext/rtl/model_nvdla/*.c*)
add_executable( gem5-NVDLA ${APP_SOURCES} )
