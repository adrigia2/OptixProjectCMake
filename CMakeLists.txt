cmake_minimum_required(VERSION 3.15)
project(OptixProjectCMake)

#set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
set(OptiX_INSTALL_DIR)

# ✅ Converti il path (gestisce spazi in 'ProgramData')
#file(TO_CMAKE_PATH "${OptiX_INSTALL_DIR}" OptiX_INSTALL_DIR)

# ✅ Aggiungi la cartella con FindOptiX.cmake
list(APPEND CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")

# ✅ Trova CUDA e OptiX
find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)

# ✅ Includi header
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${OptiX_INCLUDE}
)

# ✅ Collega librerie CUDA
link_directories(${CUDA_LIBRARY_DIRS})





# add GLFW
# project path + /common/glfw/include
include_directories("${PROJECT_SOURCE_DIR}/common/glfw/include")
find_library(GLFW_LIB NAMES glfw3 PATHS "${PROJECT_SOURCE_DIR}/common/glfw/lib-vc2022" NO_DEFAULT_PATH)
link_directories("${PROJECT_SOURCE_DIR}/common/glfw/lib-vc2022")
find_library(CUDA_LIB NAMES cuda PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" NO_DEFAULT_PATH)

link_libraries(
  ${CUDA_LIBRARIES}
  ${OptiX_LIBRARIES}
  ${CUDA_LIB}
  ${GLFW_LIB}
)

message(${CUDA_LIB})


set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

include_directories(common)
add_subdirectory(common/glfWindow EXCLUDE_FROM_ALL)
add_subdirectory(common/pybind11)



add_subdirectory(depthMap) 
