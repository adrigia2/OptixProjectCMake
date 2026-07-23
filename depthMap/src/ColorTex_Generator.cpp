#include "ColorTex_Generator.h"
#include <optix_stubs.h>
#include "LogManager.h"
#include "OptixManager.h"
#include <cmath>
#include <stdexcept>

extern "C" char embedded_ptx_code_colortex[];

namespace osc {

char* ColorTex_Generator::getPtxCode() {
    return embedded_ptx_code_colortex;
}

void ColorTex_Generator::createRaygenPrograms() {
    raygenPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__colorTex";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &raygenPGs[0]));
}

void ColorTex_Generator::createMissPrograms() {
    missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__colorTex";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &missPGs[0]));
}

void ColorTex_Generator::createHitgroupPrograms() {
    hitgroupPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__colorTex";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPGs[0]));
}

void ColorTex_Generator::setInputs(
    const IUM_Generator::Result& ium,
    const std::vector<uint8_t>&  visibility,
    const std::vector<Frame>&    frames,
    float                        grazingMaxDeg)
{
    int num_pixels  = (int)ium.positions.size();
    int num_cameras = (int)frames.size();

    if (num_pixels == 0 || num_cameras == 0)
        throw std::runtime_error("ColorTex_Generator::setInputs: empty IUM or frames");

    if ((int)visibility.size() != num_pixels * num_cameras)
        throw std::runtime_error("ColorTex_Generator::setInputs: visibility size mismatch");

    iumPositionsBuffer.alloc_and_upload(ium.positions);

    // Normals: upload if available, otherwise disable the grazing filter
    if (ium.hasNormals()) {
        iumNormalsBuffer.alloc_and_upload(ium.normals);
        launchParams.ium_normals = (vec3f*)iumNormalsBuffer.d_pointer();
    } else {
        launchParams.ium_normals   = nullptr;
        grazingMaxDeg              = 90.f;  // force-disable: no normals available
    }

    // Convert threshold to cosine once on CPU; sentinel -1 disables the check in the kernel
    constexpr float kPi = 3.14159265358979323846f;
    launchParams.grazing_min_cos = (grazingMaxDeg >= 90.f)
        ? -1.f
        : std::cos(grazingMaxDeg * kPi / 180.f);

    iumMasksBuffer.alloc_and_upload(ium.masks);
    visibilityBuffer.alloc_and_upload(visibility);

    // Build per-camera GPU structs and upload images
    imageBuffers.clear();
    imageBuffers.resize(num_cameras);
    std::vector<ColorCameraDef> camDefs(num_cameras);

    for (int i = 0; i < num_cameras; ++i) {
        const Camera& cam  = frames[i].camera;
        const float   fovY = cam.getFovY();
        const vec2i   fs   = cam.getFrameSize();
        const float   aspect     = fs.x / float(fs.y);
        const float   halfHeight = tanf(0.5f * fovY);
        const float   halfWidth  = halfHeight * aspect;

        vec3f fwd = normalize(cam.getForward());
        // Same convention as Depth_Generator::setCamera
        vec3f right_unit = normalize(cross(fwd, cam.getUp()));
        vec3f up_unit    = normalize(cross(right_unit, fwd));

        // Pre-scale: dot(d, right)/t in [-0.5, +0.5] over the horizontal span
        camDefs[i].position   = cam.getPos();
        camDefs[i].forward    = fwd;
        camDefs[i].right      = right_unit / (2.0f * halfWidth);
        camDefs[i].up_vec     = up_unit    / (2.0f * halfHeight);
        camDefs[i].frame_size = fs;
        camDefs[i].peak       = frames[i].peak;

        imageBuffers[i].alloc_and_upload(frames[i].image);
        camDefs[i].image_ptr = (vec3f*)imageBuffers[i].d_pointer();
    }

    camerasBuffer.alloc_and_upload(camDefs);

    colorOutputBuffer.alloc(num_pixels * sizeof(vec3f));
    cudaMemset((void*)colorOutputBuffer.d_pointer(), 0, num_pixels * sizeof(vec3f));

    colorMinOutputBuffer.alloc_and_upload(std::vector<vec3f>(num_pixels, {0.f, 0.f, 0.f}));
    colorMaxOutputBuffer.alloc_and_upload(std::vector<vec3f>(num_pixels, {0.f, 0.f, 0.f}));
    colorVarianceOutputBuffer.alloc(num_pixels * sizeof(vec3f));
    cudaMemset((void*)colorVarianceOutputBuffer.d_pointer(), 0, num_pixels * sizeof(vec3f));

    // size_t fin dal primo prodotto: int*int overflowa già a 128 camere con texture 4096²
    const size_t camColorBytes = size_t(num_pixels) * num_cameras * sizeof(vec3f);
    cameraColorOutputBuffer.alloc(camColorBytes);
    cudaMemset((void*)cameraColorOutputBuffer.d_pointer(), 0, camColorBytes);

    const size_t camMaskBytes = size_t(num_pixels) * num_cameras * sizeof(uint8_t);
    cameraMaskOutputBuffer.alloc(camMaskBytes);
    cudaMemset((void*)cameraMaskOutputBuffer.d_pointer(), 0, camMaskBytes);

    launchParams.ium_positions        = (vec3f*)iumPositionsBuffer.d_pointer();
    // ium_normals and grazing_min_cos already set above
    launchParams.ium_masks            = (uint8_t*)iumMasksBuffer.d_pointer();
    launchParams.num_pixels           = num_pixels;
    launchParams.visibility           = (uint8_t*)visibilityBuffer.d_pointer();
    launchParams.cameras              = (ColorCameraDef*)camerasBuffer.d_pointer();
    launchParams.num_cameras          = num_cameras;
    launchParams.color_output         = (vec3f*)colorOutputBuffer.d_pointer();
    launchParams.color_min_output      = (vec3f*)colorMinOutputBuffer.d_pointer();
    launchParams.color_max_output      = (vec3f*)colorMaxOutputBuffer.d_pointer();
    launchParams.color_variance_output = (vec3f*)colorVarianceOutputBuffer.d_pointer();
    launchParams.camera_color_output   = (vec3f*)cameraColorOutputBuffer.d_pointer();
    launchParams.camera_mask_output    = (uint8_t*)cameraMaskOutputBuffer.d_pointer();
}

void ColorTex_Generator::render() {
    if (launchParams.num_pixels == 0)
        throw std::runtime_error("ColorTex_Generator::render: call setInputs first");

    launchParamsBuffer.alloc(sizeof(launchParams));
    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(pipeline, 0,
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        launchParams.num_pixels, // width
        1,                       // height
        1                        // depth
    ));

    CUDA_SYNC_CHECK();

    result.colors.resize(launchParams.num_pixels);
    colorOutputBuffer.download(result.colors.data(), result.colors.size());

    result.color_min.resize(launchParams.num_pixels);
    colorMinOutputBuffer.download(result.color_min.data(), launchParams.num_pixels);

    result.color_max.resize(launchParams.num_pixels);
    colorMaxOutputBuffer.download(result.color_max.data(), launchParams.num_pixels);

    result.color_variance.resize(launchParams.num_pixels);
    colorVarianceOutputBuffer.download(result.color_variance.data(), launchParams.num_pixels);

    // I colori per-camera restano solo su GPU (num_pixels × num_cameras non scala
    // su host: ~12 GB con texture 4096² e 60 camere); vanno letti una camera alla
    // volta con downloadCameraColors().
    result.num_cameras = launchParams.num_cameras;
}

void ColorTex_Generator::downloadCameraColors(int cam, vec3f* dst) const {
    if (launchParams.num_pixels == 0)
        throw std::runtime_error("ColorTex_Generator::downloadCameraColors: call setInputs first");
    if (cam < 0 || cam >= launchParams.num_cameras)
        throw std::out_of_range("ColorTex_Generator::downloadCameraColors: camera index out of range");

    const size_t sliceBytes = size_t(launchParams.num_pixels) * sizeof(vec3f);
    const CUdeviceptr src = cameraColorOutputBuffer.d_pointer() + size_t(cam) * sliceBytes;
    CUDA_CHECK(Memcpy((void*)dst, (void*)src, sliceBytes, cudaMemcpyDeviceToHost));
}

void ColorTex_Generator::downloadCameraMask(int cam, uint8_t* dst) const {
    if (launchParams.num_pixels == 0)
        throw std::runtime_error("ColorTex_Generator::downloadCameraMask: call setInputs first");
    if (cam < 0 || cam >= launchParams.num_cameras)
        throw std::out_of_range("ColorTex_Generator::downloadCameraMask: camera index out of range");

    const size_t sliceBytes = size_t(launchParams.num_pixels) * sizeof(uint8_t);
    const CUdeviceptr src = cameraMaskOutputBuffer.d_pointer() + size_t(cam) * sliceBytes;
    CUDA_CHECK(Memcpy((void*)dst, (void*)src, sliceBytes, cudaMemcpyDeviceToHost));
}

void ColorTex_Generator::cleanup() {
    OptixActor::cleanup();
    iumPositionsBuffer.free();
    iumNormalsBuffer.free();
    iumMasksBuffer.free();
    visibilityBuffer.free();
    camerasBuffer.free();
    for (auto& buf : imageBuffers)
        buf.free();
    imageBuffers.clear();
    colorOutputBuffer.free();
    colorMinOutputBuffer.free();
    colorMaxOutputBuffer.free();
    colorVarianceOutputBuffer.free();
    cameraColorOutputBuffer.free();
    cameraMaskOutputBuffer.free();
}

} // namespace osc
