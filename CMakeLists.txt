cmake_minimum_required(VERSION 3.23)
project(MatrixLib)

set(CMAKE_CXX_STANDARD 17)


FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    link_libraries(-fopenmp)
    add_definitions(-D__OPENMP__)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    message("AVX2 FOUND")
    add_definitions(-D__AVX2__)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

CHECK_CXX_COMPILER_FLAG("-mfpu=neon" COMPILER_SUPPORTS_NEON)
if(COMPILER_SUPPORTS_NEON)
    message("NEON FOUND")
    add_definitions(-D__NEON__)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
endif()


add_executable(main main.cpp Matrix.cpp)
target_include_directories(main PUBLIC ./include)

