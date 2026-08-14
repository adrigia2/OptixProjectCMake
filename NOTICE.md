# Third-party code

This repository redistributes third-party source code. Everything under `common/` is
third-party; the project's own code lives in `depthMap/` and in the build files at the root.

The list below records provenance and license for each. Full license texts stay with the
sources (`common/glfw/LICENSE.md`, the headers of `nlohmann/json.hpp`, `tiny_obj_loader.h`,
`stb_image.h`, `ply.h`, and the Apache-2.0 headers of the gdt/glfWindow files).

| Path | Project | Copyright | License |
|---|---|---|---|
| `common/gdt/` | gdt, from the OptiX 7 SIGGRAPH course | Ingo Wald, 2018–2019 | Apache-2.0 |
| `common/glfWindow/` | GLFWindow, same course | Ingo Wald, 2018–2019 | Apache-2.0 |
| `common/glfw/` | GLFW | Marcus Geelnard 2002–2006; Camilla Löwy 2006–2019 | zlib/libpng |
| `common/nlohmann/` | nlohmann/json | Niels Lohmann, 2013–2025 | MIT |
| `common/objLoader/tiny_obj_loader.h`, `common/3rdParty/tiny_obj_loader.h` | tinyobjloader | Syoyo Fujita and contributors, 2012–2016 | MIT |
| `common/3rdParty/stb_image.h`, `common/3rdParty/stb_image_write.h` | stb | Sean Barrett | public domain (or MIT) |
| `common/3rdParty/ply.h`, `common/3rdParty/ply.cpp` | Stanford PLY reader | The Board of Trustees of The Leland Stanford Junior University, 1994 | Stanford license |
| `common/pybind11/` | pybind11 (git submodule, not vendored) | Wenzel Jakob and contributors | BSD-3-Clause |

The following files under `depthMap/` are derivatives of the OptiX 7 SIGGRAPH course and keep
its Apache-2.0 header:

- `depthMap/include/optix7.h`
- `depthMap/include/CUDABuffer.h`
- `depthMap/include/TransformReader.h` (partly)
- `depthMap/cuda/devicePrograms.cu`
- `depthMap/src/main.cpp`

The NVIDIA OptiX SDK and the CUDA Toolkit are **not** redistributed here. They have to be
installed separately and are governed by their own NVIDIA license agreements.

Scene data under `Scenes/` (meshes, textures, HDR environment maps) comes from various
sources and is included for reproducing the thesis results; it is not covered by the licenses
above.
