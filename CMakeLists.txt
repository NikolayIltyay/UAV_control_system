cmake_minimum_required(VERSION 3.16)

# ===== Project =====
project(UAV_control_system
    VERSION 0.1.0
    DESCRIPTION "Autonomy systems experiments"
    LANGUAGES CXX
)

# ===== C++ standard =====
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ===== Options =====
option(AP_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(AP_ENABLE_SANITIZERS "Enable sanitizers" OFF)

# ===== Compiler warnings =====
if(AP_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
    elseif(MSVC)
        add_compile_options(/W4)
    endif()
endif()

# ===== Sanitizers =====
if(AP_ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ===== Include directories =====
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# ===== Automatically collect sources =====
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

find_package(OpenCV REQUIRED)

# ===== Executable =====
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# ===== Include paths =====
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${PROJECT_INCLUDE_DIR}
)

# ===== Preprocessor definitions =====
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        PROJECT_VERSION="${PROJECT_VERSION}"
        $<$<CONFIG:Debug>:AP_DEBUG_BUILD>
)

# ===== Libraries (add later) =====
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
        # onnxruntime
)

# ===== Install =====
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
