cmake_minimum_required(VERSION 2.8)

project(EccentricFD C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c99")

add_executable(EccentricFD_main main.c InspiralEccentricFD.c InspiralOptimizedCoefficientsEccentricityFD.c InspiralEccentricFDBasic.c)

target_link_libraries(EccentricFD_main m gsl gslcblas)

# If you have different gsl path, please change the following three line
#set_target_properties(EccentricFD_main PROPERTIES LINK_FLAGS "-Wl,-rpath,/public/software/gsl-2.7.1/lib")

#include_directories("/public/software/gsl-2.7.1/include")
#link_directories("/public/software/gsl-2.7.1/lib")

set_target_properties(EccentricFD_main PROPERTIES LINK_FLAGS "-Wl,-rpath,/opt/gsl-2.7.1/lib")

include_directories("/opt/gsl-2.7.1/include")
link_directories("/opt/gsl-2.7.1/lib")

add_library(EccFD SHARED InspiralEccentricFD.c InspiralOptimizedCoefficientsEccentricityFD.c InspiralEccentricFDBasic.c)

target_link_libraries(EccFD m gsl gslcblas)
