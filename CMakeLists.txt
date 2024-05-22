cmake_minimum_required(VERSION 3.10)
project(cudaLearn LANGUAGES CUDA CXX)
# find_package(CUDA 12.2 EXACT REQUIRED)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

add_subdirectory(3rd)
set(CMAKE_CUDA_ARCHITECTURES 75)

file(GLOB TEST_SOURCES "*.cu" "test/*.cu")
foreach(TEST_SOURCE ${TEST_SOURCES})
    # 获取文件名
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # 添加可执行文件
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PUBLIC cutlass)
    target_link_libraries(${TEST_NAME} PUBLIC cublas)

    # target_compile_options(gelu PRIVATE --run)
    target_compile_options(${TEST_NAME} PRIVATE -lineinfo)
    target_compile_options(${TEST_NAME} PRIVATE --expt-relaxed-constexpr)
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_17)

    # 指定可执行文件生成的路径
    # set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dist)
    set_target_properties(${TEST_NAME} PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/dist
                      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/dist)

endforeach()
