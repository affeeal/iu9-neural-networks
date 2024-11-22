cmake_minimum_required(VERSION 3.28.3)
project(hw3)

find_package(spdlog REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(Eigen3 3.4.90 REQUIRED)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen spdlog::spdlog Matplot++::matplot)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
