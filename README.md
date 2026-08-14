# OptixProjectCMake — programmable OptiX passes

C++/CUDA half of a thesis project that reconstructs PBR material maps by combining GPU ray
tracing with neural rendering. It builds **eight OptiX ray tracing passes** and exposes them
to Python as the extension module **`OptixProgrammablePasses`**, which the sibling
[`NeRF_Pytorch/`](../NeRF_Pytorch/README.md) pipeline drives.

There is also a small standalone binary, `depthMap`, that runs two of the passes from the
command line — useful for checking the ray tracing side without involving Python at all.

---

## 1. Build

### Prerequisites

- **Visual Studio 2022** (MSVC toolset, x64)
- **NVIDIA CUDA Toolkit**
- **NVIDIA OptiX SDK 9.0.0**
- the pybind11 submodule:

```bash
git submodule update --init --recursive
```

The OptiX SDK is looked up in this order, first hit wins:

1. `-DOptiX_INSTALL_DIR=<path>` (what `CMakePresets.json` passes)
2. the `OPTIX_INSTALL_DIR` environment variable
3. `C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0`, the default install path

If none of them exists, CMake stops with a message saying so rather than failing later on a
missing header.

### The Python module (what the pipeline uses)

```bash
pip install -e .
python -c "import OptixProgrammablePasses; print('ok')"
```

On a machine where the SDK lives elsewhere:

```bash
OPTIX_INSTALL_DIR="D:/SDKs/OptiX SDK 9.0.0" pip install -e .
# or, for a single build:
pip install -e . -C cmake.define.OptiX_INSTALL_DIR="D:/SDKs/OptiX SDK 9.0.0"
```

Build it into the environment that will run the pipeline — see `NeRF_Pytorch/environment.yml`,
which creates `tesi-nerf`.

### The C++ targets

```bash
cmake --preset x64-release
cmake --build out/build/x64-release --config Release
```

`x64-debug` is the other preset. **Pass `--config` explicitly**: the Visual Studio generator is
multi-config, so `CMAKE_BUILD_TYPE` from the preset does not select the configuration on its
own, and without it the binaries land in `out/build/x64-release/Debug/` whatever the preset
says.

Then, as a smoke test:

```bash
out/build/x64-release/Release/depthMap.exe Scenes/SwordShield/Models/SwordShield.obj 512
```

It loads the mesh, runs the IUM and depth passes, and prints UV coverage, hit rate and depth
range. Exit codes: `0` ok, `1` failure, `2` usage.

---

## 2. The eight passes

Every pass is a subclass of `OptixActor` with its own `.cu` kernel and its own
`LaunchParams_*` struct.

| Class | What it computes | Kernel | Launch params |
|---|---|---|---|
| `Depth_Generator` | Per-pixel depth, world position, normal and mask, in camera space | `devicePrograms.cu` | `LaunchParams_DPN.h` |
| `IUM_Generator` | Inverse UV mapping: per-texel world position, **face normal**, mask | `deviceProgramsIUM.cu` | `LaunchParams_IUM.h` |
| `Visibility_Generator` | Per-texel visibility from a set of cameras (occlusion only) | `deviceProgramsVis.cu` | `LaunchParams_Vis.h` |
| `ColorTex_Generator` | Bakes observed colour into texture space; also emits the authoritative per-camera masks | `deviceProgramsColorTex.cu` | `LaunchParams_ColorTex.h` |
| `Irradiance_Generator` | Per-texel HDR irradiance from the skybox | `deviceProgramsIrradiance.cu` | `LaunchParams_Irradiance.h` |
| `Indirect_Generator` | Occluded rays for the indirect irradiance, tile by tile, for NeRF radiance queries | `deviceProgramsIndirect.cu` | `LaunchParams_Indirect.h` |
| `SpecCone_Generator` | Per-texel per-camera specular cones: concentric rings around `R = reflect(v, n)` | `deviceProgramsSpecCone.cu` | `LaunchParams_SpecCone.h` |
| `HemiVis_Generator` | **Camera-agnostic** hemisphere visibility oracle: one shared Fibonacci ray set per texel | `deviceProgramsHemiVis.cu` | `LaunchParams_HemiVis.h` |

`SpecCone_Generator` and `HemiVis_Generator` compute the same quantity two ways. The spec-cone
pass aims rays at each camera's reflected direction and has to be relaunched per camera; the
HemiVis pass traces one set per texel that every camera reuses, because incident radiance
along a direction does not depend on the camera. Which one runs is chosen on the Python side
(`spec_cone_scheme`).

**Sampling is deterministic quadrature, never Monte Carlo.** Directions come from Fibonacci /
golden-angle sets with a per-texel rotation; no direction is ever drawn at random anywhere in
this code.

---

## 3. Architecture

```
depthMap/
├── cuda/      one .cu per pass (device code, compiled to PTX and embedded)
├── include/   generators, LaunchParams_*, and the support headers
└── src/       generator implementations, the pybind wrapper, the CLI
common/        third-party code only, see §7
```

**`OptixActor`** (`include/OptixActor.h`) is the abstract base. It provides `createModule()`,
`createPipeline()`, `createSBT()`, `cleanup()` and the protected helper `createGAS()`, and
leaves four hooks to each pass: `createRaygenPrograms()`, `createMissPrograms()`,
`createHitgroupPrograms()`, `getPtxCode()`, plus `setTraversable(const TriangleMesh&)`.

**`OptixManager`** holds the OptiX context and device context shared by every pass.

**`TriangleMesh`** loads OBJ files through tiny_obj_loader. **`TransformReader`** parses
NeRF-format `transforms.json`. **`LogManager`** is a severity-levelled logger, also exposed to
Python. **`IOManager`** does raw binary file I/O only — despite the name it writes no images;
every EXR and PNG in this project is written by the Python side.

**`CUDABuffer`** (`include/CUDABuffer.h`) is the RAII device-allocation wrapper: single owner,
move-only, and its destructor calls `cudaFree` directly rather than through `CUDA_CHECK`,
because a throwing destructor would call `std::terminate` and at process shutdown the CUDA
context may already be gone.

### Adding a new pass

1. Write `cuda/deviceProgramsX.cu`, `include/LaunchParams_X.h`, and
   `include/X_Generator.h` + `src/X_Generator.cpp` deriving from `OptixActor`.
2. Add one `cuda_compile_and_embed(embedded_ptx_code_x cuda/deviceProgramsX.cu)` in
   `depthMap/CMakeLists.txt`, and add `${embedded_ptx_code_x}` to `OPTIX_PTX_EMBEDS`.
3. Declare `extern "C" char embedded_ptx_code_x[];` in `X_Generator.cpp` and return it from
   `getPtxCode()`.
4. Add the sources to `OPTIX_PASS_SOURCES` / `OPTIX_PASS_HEADERS`.

Those two `set()` lists feed **both** the `depthMap` executable and the Python module, so a
pass is registered once. (Before this cleanup the list was duplicated per target, and
forgetting one half was an easy mistake.)

---

## 4. The Python module

`src/pybind11_wrapper.cpp` exports:

- generators — `DepthGenerator`, `IUMGenerator`, `VisibilityGenerator`, `ColorTexGenerator`,
  `IrradianceGenerator`, `IndirectGenerator`, `SpecConeGenerator`, `HemiVisGenerator`
- their results — `DepthResult`, `IUMResult`, `ColorTexResult`, `IrradianceResult`,
  `IndirectTileResult`, `SpecConeTileResult`, `HemiVisTileResult`
- support types — `TriangleMesh`, `Camera`, `Frame`, `OptixManager`, `LogManager`, `LogLevel`,
  `vec2i`, `vec3f`

Method names are snake_case on the Python side (`set_inputs`, `render_tile`,
`download_camera_colors`, …), and the result objects expose their buffers as `*_np` NumPy
arrays.

> **Ownership rule.** Results are *moved* into Python and the `*_np` arrays are **zero-copy
> views into C++ memory**. The caller must keep the result object alive for as long as it
> uses the views. The Python pipeline does this by holding `ium_res`, `ct_result`, `irr_res`
> in scope; a bare `gen.get_result().positions_np` would dangle.

`stubs/OptixProgrammablePasses.pyi` carries the type stubs. Regenerate it after changing the
bindings:

```bash
pip install pybind11-stubgen
pybind11-stubgen OptixProgrammablePasses -o stubs
```

---

## 5. Conventions and gotchas

**World axes: Z-up, Y-forward (Blender-native).** OBJ vertices are loaded as-is, with no axis
swap, and the equirectangular envmap lookup uses +Z as zenith. Normal maps baked in Blender in
object/world space need no remapping.

**OBJ vertex normals are not loaded.** `TriangleMesh` reads positions, indices and UVs only,
so `IUM_Generator` computes *face* normals from the cross product of the triangle's world-space
edges, assuming CCW winding. Smooth normals have to be supplied as a baked normal map from the
Python side.

**Depth is a Euclidean distance.** The depth kernel normalises the ray direction before
tracing, so what comes back is a world-space distance, not a camera-axis Z depth. The Python
side depends on this: its ray directions are unit vectors to match.

**Golden-angle arithmetic is done in double precision** and reduced modulo 2π *before* the
trigonometry, in both `deviceProgramsHemiVis.cu` and `deviceProgramsIrradiance.cu`. In float32
`s * goldenAngle` reaches ~6e5 rad at large sample counts, where one ULP is already 3.58°, and
the sequence quietly loses its low discrepancy. The two kernels use opposite azimuthal
directions (`(3-√5)/2` vs `1/φ`) and must keep them, or the patterns they already produced
would change.

**The HemiVis directions are replicated bit for bit in Python.** The reconstruction is indexed
by position, so a divergence between the kernel formula and the torch one has no symptom other
than a silently wrong result. `NeRF_Pytorch/scripts/test_hemivis_shared.py` exists to catch
exactly that; run it after touching that kernel.

**Exit codes from Python are unreliable.** A process that has used OptiX exits non-zero at
interpreter shutdown regardless of outcome. Judge the Python smoke tests by their final `✓`
line. The `depthMap` binary is not affected and returns a meaningful code.

---

## 6. Scenes

`Scenes/<Name>/` holds the capture data. A scene folder typically contains:

```
Scenes/SwordShield/
├── Nerf<Variant>/     transforms.json + images/   (the capture)
├── Models<Variant>/   the OBJ the passes trace
├── Blender*/          .blend sources and HDR environment maps
└── BlenderBaked*/     reference bakes (BakedMaterial_{base_color,metallic,normal,roughness}.exr)
```

`transforms.json` is the standard NeRF/instant-ngp format; the fields read here are
`camera_angle_x`, `fl_x`/`fl_y`, `cx`/`cy`, `w`/`h`, and per frame `file_path` (resolved
relative to the json) and `transform_matrix`.

> `Scenes/` is about **1.6 GB of tracked capture data** (60 EXR frames at ~23 MB each, plus a
> 93 MB HDR). A clone is correspondingly large.

---

## 7. Repository layout and third-party code

Only `depthMap/` and the build files at the root are this project's own code. Everything under
`common/` is vendored third-party code, unmodified except where noted, and is not covered by
the same authorship:

| Path | Project | License |
|---|---|---|
| `common/gdt/` | gdt, from Ingo Wald's OptiX 7 SIGGRAPH course | Apache-2.0 |
| `common/glfWindow/` | GLFWindow, same course | Apache-2.0 |
| `common/glfw/` | GLFW | zlib/libpng |
| `common/nlohmann/` | nlohmann/json | MIT |
| `common/objLoader/`, `common/3rdParty/tiny_obj_loader.h` | tinyobjloader | MIT |
| `common/3rdParty/stb_image*.h` | stb | public domain / MIT |
| `common/3rdParty/ply.*` | Stanford PLY reader | Stanford license |
| `common/pybind11/` | pybind11 (git submodule) | BSD-3-Clause |

A few files under `depthMap/` are derivatives of the same OptiX 7 course and keep its
Apache-2.0 header: `include/optix7.h`, `include/CUDABuffer.h`, parts of
`include/TransformReader.h`, `cuda/devicePrograms.cu` and `src/main.cpp`.

Build output (`out/`, `.vs/`, `cmake-build-debug/`, `dist/`, `wheelhouse/`) is ignored, not
tracked.
