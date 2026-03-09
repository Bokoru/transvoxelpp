cmake_minimum_required(VERSION 3.12)
project("transvoxelpp")

add_library(transvoxelpp STATIC "src/transvoxel.cpp" "include/transvoxel.hpp")
target_include_directories(transvoxelpp PUBLIC "include")
set_property(TARGET transvoxelpp PROPERTY CXX_STANDARD 20)
