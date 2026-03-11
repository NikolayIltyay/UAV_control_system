cmake_minimum_required(VERSION 3.16)

project(UAV_control_system
    VERSION 0.1.0
    DESCRIPTION "Autonomy systems experiments"
    LANGUAGES CXX
)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)


set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)


file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

find_package(OpenCV REQUIRED)


add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_compile_definitions(UAV_control_system PRIVATE CAMERA_CAPTURE_BACKEND=V4L2)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-aarch64-1.24.3/include
        ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.17.1/include
)


target_link_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-aarch64-1.24.3/lib
    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.17.1/lib
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
        onnxruntime
)

