#include "Irradiance_Generator.h"
#include <optix_stubs.h>
#include "LogManager.h"
#include "OptixManager.h"
#include <stdexcept>

extern "C" char embedded_ptx_code_irradiance[];

namespace osc {

char* Irradiance_Generator::getPtxCode() {
    return embedded_ptx_code_irradiance;
}

void Irradiance_Generator::createRaygenPrograms() {
    raygenPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderIrradiance";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &raygenPGs[0]));
}

void Irradiance_Generator::createMissPrograms() {
    missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__shadow";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &missPGs[0]));
}

void Irradiance_Generator::createHitgroupPrograms() {
    hitgroupPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPGs[0]));
}

OptixTraversableHandle Irradiance_Generator::createGAS(const TriangleMesh& model) {
    OptixTraversableHandle asHandle{ 0 };

    vertexBuffer.alloc_and_upload(model.vertex);
    indexBuffer.alloc_and_upload(model.index);

    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr d_vertices = vertexBuffer.d_pointer();
    CUdeviceptr d_indices = indexBuffer.d_pointer();

    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput.triangleArray.numVertices = (int)model.vertex.size();
    triangleInput.triangleArray.vertexBuffers = &d_vertices;

    triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes = sizeof(vec3i);
    triangleInput.triangleArray.numIndexTriplets = (int)model.index.size();
    triangleInput.triangleArray.indexBuffer = d_indices;

    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    triangleInput.triangleArray.flags = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        OptixManager::instance().getContext(),
        &accelOptions,
        &triangleInput,
        1,
        &blasBufferSizes));

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        OptixManager::instance().getContext(),
        0,
        &accelOptions,
        &triangleInput,
        1,
        tempBuffer.d_pointer(),
        tempBuffer.sizeInBytes,
        outputBuffer.d_pointer(),
        outputBuffer.sizeInBytes,
        &asHandle,
        &emitDesc,
        1));

    CUDA_SYNC_CHECK();

    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(OptixManager::instance().getContext(),
        0, asHandle,
        asBuffer.d_pointer(), asBuffer.sizeInBytes,
        &asHandle));

    CUDA_SYNC_CHECK();

    outputBuffer.free();
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

void Irradiance_Generator::setTraversable(const TriangleMesh& model) {
    if (asBuffer.d_ptr != 0) return;
    auto handle = createGAS(model);
    launchParams.traversable = handle;
}

void Irradiance_Generator::cleanup() {
    OptixActor::cleanup();
    vertexBuffer.free();
    indexBuffer.free();
    asBuffer.free();

    iumPositionsBuffer.free();
    iumNormalsBuffer.free();
    iumMasksBuffer.free();
    skyboxBuffer.free();
    irradianceBuffer.free();
    launchParamsBuffer.free();
}

void Irradiance_Generator::setInputs(const IUM_Generator::Result& ium_result,
                                     const std::vector<vec3f>& skybox,
                                     vec2i skyboxSize,
                                     int sampleSide,
                                     float skyboxYawDegrees)
{
    numPixels = (int)ium_result.positions.size();

    if (numPixels == 0) {
        throw std::runtime_error("Irradiance_Generator: empty IUM positions");
    }
    if (ium_result.normals.size() != (size_t)numPixels) {
        throw std::runtime_error("Irradiance_Generator: IUM normals size mismatch");
    }
    if (ium_result.masks.size() != (size_t)numPixels) {
        throw std::runtime_error("Irradiance_Generator: IUM masks size mismatch");
    }
    if ((int)skybox.size() != skyboxSize.x * skyboxSize.y) {
        throw std::runtime_error("Irradiance_Generator: skybox size does not match width*height");
    }
    if (sampleSide <= 0) {
        throw std::runtime_error("Irradiance_Generator: sample_side must be > 0");
    }

    // Upload IUM
    if (iumPositionsBuffer.sizeInBytes != numPixels * sizeof(vec3f)) {
        iumPositionsBuffer.resize(numPixels * sizeof(vec3f));
    }
    iumPositionsBuffer.upload(ium_result.positions.data(), numPixels);

    if (iumNormalsBuffer.sizeInBytes != numPixels * sizeof(vec3f)) {
        iumNormalsBuffer.resize(numPixels * sizeof(vec3f));
    }
    iumNormalsBuffer.upload(ium_result.normals.data(), numPixels);

    if (iumMasksBuffer.sizeInBytes != numPixels * sizeof(uint8_t)) {
        iumMasksBuffer.resize(numPixels * sizeof(uint8_t));
    }
    iumMasksBuffer.upload(ium_result.masks.data(), numPixels);

    // Upload skybox
    const size_t skyBytes = skybox.size() * sizeof(vec3f);
    if (skyboxBuffer.sizeInBytes != skyBytes) {
        skyboxBuffer.resize(skyBytes);
    }
    skyboxBuffer.upload(skybox.data(), skybox.size());

    // Output
    const size_t outBytes = numPixels * sizeof(vec3f);
    if (irradianceBuffer.sizeInBytes != outBytes) {
        irradianceBuffer.resize(outBytes);
    }
    cudaMemset((void*)irradianceBuffer.d_pointer(), 0, outBytes);

    // Riempi launch params
    launchParams.results.irradiance      = (vec3f*)irradianceBuffer.d_pointer();

    launchParams.ium_data.ium_positions  = (vec3f*)iumPositionsBuffer.d_pointer();
    launchParams.ium_data.ium_normals    = (vec3f*)iumNormalsBuffer.d_pointer();
    launchParams.ium_data.ium_masks      = (uint8_t*)iumMasksBuffer.d_pointer();
    launchParams.ium_data.num_pixels     = numPixels;

    launchParams.skybox.envmap           = (vec3f*)skyboxBuffer.d_pointer();
    launchParams.skybox.skybox_size      = skyboxSize;

    constexpr float PI_F = 3.14159265358979323846f;
    const float yaw_rad = skyboxYawDegrees * (PI_F / 180.0f);
    float yaw_offset_u = yaw_rad / (2.0f * PI_F);
    yaw_offset_u -= floorf(yaw_offset_u);  // normalizza in [0, 1)
    launchParams.skybox.yaw_offset_u     = yaw_offset_u;

    launchParams.sample_side             = sampleSide;
    launchParams.epsilon                 = 1e-4f;

    LogManager::Log("Irradiance inputs ready: %d pixels, skybox %dx%d, %dx%d samples, yaw=%.1f deg",
                    numPixels, skyboxSize.x, skyboxSize.y, sampleSide, sampleSide, skyboxYawDegrees);
}

void Irradiance_Generator::render() {
    if (numPixels == 0) {
        LogManager::LogError("Irradiance_Generator::render called before setInputs");
        return;
    }

    if (launchParamsBuffer.sizeInBytes != sizeof(LaunchParams_Irradiance)) {
        launchParamsBuffer.resize(sizeof(LaunchParams_Irradiance));
    }
    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(pipeline, 0,
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        numPixels, // width
        1,         // height
        1          // depth
    ));
    CUDA_SYNC_CHECK();

    result.irradiance.clear();
    result.irradiance.resize(numPixels);
    irradianceBuffer.download(result.irradiance.data(), numPixels);

    LogManager::Log("Irradiance rendering completed: %d pixels", numPixels);
}

} // namespace osc
