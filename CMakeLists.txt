cmake_minimum_required(VERSION 3.15)
project(OptixProjectCMake)

# ── OptiX SDK location ────────────────────────────────────────────────────────
# Resolved in this order, first hit wins:
#   1. -DOptiX_INSTALL_DIR=...        (what CMakePresets.json passes)
#   2. the OPTIX_INSTALL_DIR environment variable
#   3. the SDK's default Windows install path
# Keeping the fallback here rather than in pyproject.toml is what makes
# `pip install .` work on a machine where the SDK lives somewhere else.
#
# It has to end up as a CACHE entry, not a plain variable: FindOptiX.cmake opens
# with its own `set(OptiX_INSTALL_DIR ... CACHE PATH ...)` defaulting to
# ${CMAKE_SOURCE_DIR}/.., and creating a cache entry that did not exist before
# discards the normal variable of the same name. A plain set() here would
# therefore be silently thrown away, and the SDK headers would not be found.
if(DEFINED ENV{OPTIX_INSTALL_DIR})
  set(_optix_default "$ENV{OPTIX_INSTALL_DIR}")
else()
  set(_optix_default "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
endif()

# Normalise the path: the default install directory contains a space, and an
# environment variable may well arrive with backslashes.
file(TO_CMAKE_PATH "${_optix_default}" _optix_default)

# No FORCE: an explicit -DOptiX_INSTALL_DIR wins.
set(OptiX_INSTALL_DIR "${_optix_default}" CACHE PATH "Path to the OptiX SDK")

if(NOT EXISTS "${OptiX_INSTALL_DIR}")
  message(FATAL_ERROR
    "OptiX SDK not found at '${OptiX_INSTALL_DIR}'.\n"
    "Install the OptiX SDK 9.0.0 and point CMake at it with either\n"
    "  -DOptiX_INSTALL_DIR=<path>\n"
    "or the OPTIX_INSTALL_DIR environment variable.")
endif()

# FindOptiX.cmake ships with the SDK.
list(APPEND CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")

find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${OptiX_INCLUDE}
)

link_directories(${CUDA_LIBRARY_DIRS})

# ── GLFW (vendored under common/) ─────────────────────────────────────────────
include_directories("${PROJECT_SOURCE_DIR}/common/glfw/include")
find_library(GLFW_LIB NAMES glfw3 PATHS "${PROJECT_SOURCE_DIR}/common/glfw/lib-vc2022" NO_DEFAULT_PATH)
link_directories("${PROJECT_SOURCE_DIR}/common/glfw/lib-vc2022")

# The CUDA driver API (cuLaunch*, cuMemAlloc*) is a separate library from the
# runtime pulled in by ${CUDA_LIBRARIES}.
find_library(CUDA_LIB NAMES cuda PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" NO_DEFAULT_PATH)

link_libraries(
  ${CUDA_LIBRARIES}
  ${OptiX_LIBRARIES}
  ${CUDA_LIB}
  ${GLFW_LIB}
)

# ── Third-party support code ──────────────────────────────────────────────────
set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

include_directories(common)
add_subdirectory(common/glfWindow EXCLUDE_FROM_ALL)

# Everything else under common/ is vendored into the repository, but pybind11 is a git
# submodule, so a clone made without --recurse-submodules leaves it empty. Say so, instead
# of letting add_subdirectory fail with a generic "does not contain a CMakeLists.txt".
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/common/pybind11/CMakeLists.txt")
  message(FATAL_ERROR
    "common/pybind11 is empty: it is a git submodule.\n"
    "Run:  git submodule update --init --recursive")
endif()
add_subdirectory(common/pybind11)

# ── This project ──────────────────────────────────────────────────────────────
add_subdirectory(depthMap)
