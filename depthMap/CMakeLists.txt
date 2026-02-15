# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

find_package(OpenGL REQUIRED)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(depthMap
${embedded_ptx_code}
Camera.h
optix7.h
CUDABuffer.h
LaunchParams.h
TransformReader.h
LogManager.h
LogManager.cpp
IOManager.h
IOManager.cpp
OptixManager.h
OptixManager.cpp
TriangleMesh.h
TriangleMesh.cpp
main.cpp
ImageResultType.h
IUM_Generator.h
IUM_Generator.cpp
Depth_Generator.h
Depth_Generator.cpp)

target_link_libraries(depthMap
  gdt
  glfWindow
  ${OPENGL_gl_LIBRARY}
  OpenEXR::OpenEXR
)

pybind11_add_module(depthMapModule
  ${embedded_ptx_code}
  pybind11_wrapper.cpp
  
  # Core classes
  OptixManager.cpp
  OptixManager.h
  IUM_Generator.cpp
  IUM_Generator.h
  Depth_Generator.cpp
  Depth_Generator.h
  TriangleMesh.cpp
  TriangleMesh.h
  
  # Utility classes
  LogManager.cpp
  LogManager.h
  IOManager.cpp
  IOManager.h
  
  # Headers
  Camera.h
  CUDABuffer.h
  LaunchParams.h
  optix7.h
  ImageResultType.h
  TransformReader.h
)

target_link_libraries(depthMapModule PRIVATE
  gdt
  glfWindow
  ${CUDA_LIBRARIES}
  ${OptiX_LIBRARY}
  ${OPENGL_gl_LIBRARY}
  OpenEXR::OpenEXR
)

target_include_directories(depthMapModule PRIVATE
  ${OptiX_INCLUDE}
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)
