cmake_minimum_required(VERSION 3.12)
project("transvoxelpp")

add_library(transvoxelpp STATIC "src/march.cpp")
target_include_directories(transvoxelpp PUBLIC "include")
set_property(TARGET transvoxelpp PROPERTY CXX_STANDARD 20)
target_compile_options(transvoxelpp PUBLIC -ffast-math -fno-rtti -mfma -mavx2 -O3)
