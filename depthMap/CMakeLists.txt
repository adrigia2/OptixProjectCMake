# depthMap/CMakeLists.txt
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")


# PTX embed: ora il .cu sta in depthMap/cuda/
cuda_compile_and_embed(embedded_ptx_code
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/devicePrograms.cu
)

# PTX embed: ora il .cu sta in depthMap/cuda/
cuda_compile_and_embed(embedded_ptx_code_ium
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/deviceProgramsIUM.cu
)

# ----- Executable -----
add_executable(depthMap
  ${embedded_ptx_code}
  ${embedded_ptx_code_ium}

  # src/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/LogManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/IOManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/OptixManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TriangleMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/IUM_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Depth_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/OptixActor.cpp

  # headers (opzionale elencarli, ma utile in Visual Studio)
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Camera.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/optix7.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/CUDABuffer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_IUM.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_DPN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/TransformReader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LogManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/IOManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/OptixManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/TriangleMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImageResultType.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/IUM_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Depth_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/OptixActor.h
)

target_include_directories(depthMap PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${OptiX_INCLUDE}
)

target_link_libraries(depthMap PRIVATE
  gdt
  glfWindow
  ${OPENGL_gl_LIBRARY}
  #OpenEXR::OpenEXR
  ${CUDA_LIBRARIES}
  ${OptiX_LIBRARIES}
  ${CUDA_LIB}
  ${GLFW_LIB}
)

# ----- Pybind module -----
pybind11_add_module(depthMapModule
  ${embedded_ptx_code}

  ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind11_wrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/OptixManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/IUM_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Depth_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TriangleMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/LogManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/IOManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/OptixActor.cpp


  ${CMAKE_CURRENT_SOURCE_DIR}/include/OptixManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/IUM_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Depth_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/TriangleMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LogManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/IOManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Camera.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/CUDABuffer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_IUM.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_DPN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/optix7.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImageResultType.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/TransformReader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/OptixActor.h
)

target_include_directories(depthMapModule PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${OptiX_INCLUDE}
)

target_link_libraries(depthMapModule PRIVATE
  gdt
  glfWindow
  ${OPENGL_gl_LIBRARY}
  #OpenEXR::OpenEXR
  ${CUDA_LIBRARIES}
  ${OptiX_LIBRARIES}   # <--- meglio usare questo, non OptiX_LIBRARY singolo
  ${CUDA_LIB}
  ${GLFW_LIB}
)
