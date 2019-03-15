cmake_minimum_required(VERSION 3.8)

project(jetnet-wrapper)

# Request C++11 standard, using new CMake variables.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(jetnet REQUIRED)

include_directories(${TARGET_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} run_yolo.cpp)

target_link_libraries(
    ${PROJECT_NAME}
    jetnet
    jetnet_static
    ${TARGET_LIBS}
)
