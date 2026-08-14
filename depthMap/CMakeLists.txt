# depthMap/CMakeLists.txt
#
# Two targets are built from (almost) the same sources:
#   depthMap                  - standalone CLI smoke test (adds src/main.cpp)
#   OptixProgrammablePasses   - the Python extension (adds src/pybind11_wrapper.cpp)
# The shared sources are listed ONCE below, so adding a pass is a single edit.

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")

# ── PTX embedding ─────────────────────────────────────────────────────────────
# Each call compiles a .cu to PTX and embeds it in a generated .cpp as a char
# array. The variable name is load-bearing: the matching generator declares it
# with `extern "C" char <name>[]` and returns it from getPtxCode(), so keep the
# names greppable rather than generating them in a loop.
cuda_compile_and_embed(embedded_ptx_code
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/devicePrograms.cu)
cuda_compile_and_embed(embedded_ptx_code_ium
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/deviceProgramsIUM.cu)
cuda_compile_and_embed(embedded_ptx_code_vis
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/deviceProgramsVis.cu)
cuda_compile_and_embed(embedded_ptx_code_colortex
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/deviceProgramsColorTex.cu)
cuda_compile_and_embed(embedded_ptx_code_irradiance
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/deviceProgramsIrradiance.cu)
cuda_compile_and_embed(embedded_ptx_code_indirect
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/deviceProgramsIndirect.cu)
cuda_compile_and_embed(embedded_ptx_code_speccone
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/deviceProgramsSpecCone.cu)
cuda_compile_and_embed(embedded_ptx_code_hemivis
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/deviceProgramsHemiVis.cu)

set(OPTIX_PTX_EMBEDS
  ${embedded_ptx_code}
  ${embedded_ptx_code_ium}
  ${embedded_ptx_code_vis}
  ${embedded_ptx_code_colortex}
  ${embedded_ptx_code_irradiance}
  ${embedded_ptx_code_indirect}
  ${embedded_ptx_code_speccone}
  ${embedded_ptx_code_hemivis}
)

# ── Sources shared by both targets ────────────────────────────────────────────
set(OPTIX_PASS_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/OptixActor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/OptixManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TriangleMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/LogManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/IOManager.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/src/Depth_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/IUM_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Visibility_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ColorTex_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Irradiance_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Indirect_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/SpecCone_Generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/HemiVis_Generator.cpp
)

# Headers do not need listing to compile; they are here so they show up in the
# Visual Studio solution explorer.
set(OPTIX_PASS_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/OptixActor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/OptixManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/optix7.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/CUDABuffer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/TriangleMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/TransformReader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LogManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/IOManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Camera.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Frame.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImageResultType.h

  ${CMAKE_CURRENT_SOURCE_DIR}/include/Depth_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/IUM_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Visibility_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ColorTex_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Irradiance_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Indirect_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/SpecCone_Generator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/HemiVis_Generator.h

  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_DPN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_IUM.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_Vis.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_ColorTex.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_Irradiance.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_Indirect.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_SpecCone.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/LaunchParams_HemiVis.h
)

set(OPTIX_PASS_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${OptiX_INCLUDE}
)

set(OPTIX_PASS_LIBS
  gdt
  glfWindow
  ${OPENGL_gl_LIBRARY}
  ${CUDA_LIBRARIES}
  ${OptiX_LIBRARIES}
  ${CUDA_LIB}
  ${GLFW_LIB}
)

# ── Standalone CLI ────────────────────────────────────────────────────────────
add_executable(depthMap
  ${OPTIX_PTX_EMBEDS}
  ${OPTIX_PASS_SOURCES}
  ${OPTIX_PASS_HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
target_include_directories(depthMap PRIVATE ${OPTIX_PASS_INCLUDE_DIRS})
target_link_libraries(depthMap PRIVATE ${OPTIX_PASS_LIBS})

# ── Python extension ──────────────────────────────────────────────────────────
pybind11_add_module(OptixProgrammablePasses
  ${OPTIX_PTX_EMBEDS}
  ${OPTIX_PASS_SOURCES}
  ${OPTIX_PASS_HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind11_wrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/vec3f_caster.h
)
target_include_directories(OptixProgrammablePasses PRIVATE ${OPTIX_PASS_INCLUDE_DIRS})
target_link_libraries(OptixProgrammablePasses PRIVATE ${OPTIX_PASS_LIBS})

install(TARGETS OptixProgrammablePasses DESTINATION .)
