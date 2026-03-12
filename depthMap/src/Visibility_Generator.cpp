#include "Visibility_Generator.h"
#include <optix_stubs.h>
#include "LogManager.h"
#include "OptixManager.h"
#include <iostream>

extern "C" char embedded_ptx_code_vis[];

namespace osc {

char* Visibility_Generator::getPtxCode() {
    return embedded_ptx_code_vis;
}

void Visibility_Generator::createRaygenPrograms() {
    raygenPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &raygenPGs[0]));
}

void Visibility_Generator::createMissPrograms() {
    missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &missPGs[0]));
}

void Visibility_Generator::createHitgroupPrograms() {
    hitgroupPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPGs[0]));
}

OptixTraversableHandle Visibility_Generator::createGAS(const TriangleMesh& model) {
    // Normal 3D AS creation
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

void Visibility_Generator::setTraversable(const TriangleMesh& model) {
    if (asBuffer.d_ptr != 0) return;
    auto handle = createGAS(model);
    launchParams.traversable = handle;
}

void Visibility_Generator::cleanup() {
    OptixActor::cleanup();
    vertexBuffer.free();
    indexBuffer.free();
    asBuffer.free();

    camBuffer.free();
    iumPositionsBuffer.free();
    iumMasksBuffer.free();
    resultsBuffer.free();
}

std::vector<uint8_t> Visibility_Generator::checkVisibility(const IUM_Generator::Result& ium_result, int width, int height, const std::vector<Camera>& cameras) {
    int num_pixels = width * height;
    int num_cameras = (int)cameras.size();

    // Verify IUM sizes
    if (ium_result.positions.size() != num_pixels || ium_result.masks.size() != num_pixels) {
        throw std::runtime_error("IUM result size does not match requested dimensions");
    }

    if(num_cameras == 0 || num_pixels == 0) {
        return std::vector<uint8_t>();
    }

    std::vector<CameraDef> camDefs(num_cameras);
    for(int i=0; i<num_cameras; ++i) {
        camDefs[i].position = cameras[i].getPos();
        camDefs[i].forward = cameras[i].getForward();
        camDefs[i].up = cameras[i].getUp();
        camDefs[i].fovY = cameras[i].getFovY();
        camDefs[i].frameSize = cameras[i].getFrameSize();
    }

    // Allocate buffers if needed
    if(camBuffer.sizeInBytes != (num_cameras * sizeof(CameraDef))) {
        camBuffer.alloc(num_cameras * sizeof(CameraDef));
    }
    camBuffer.upload(camDefs.data(), camDefs.size());
    
    if(iumPositionsBuffer.sizeInBytes != (num_pixels * sizeof(vec3f))) {
        iumPositionsBuffer.alloc(num_pixels * sizeof(vec3f));
    }
    iumPositionsBuffer.upload(ium_result.positions.data(), ium_result.positions.size());

    if(iumMasksBuffer.sizeInBytes != (num_pixels * sizeof(uint8_t))) {
        iumMasksBuffer.alloc(num_pixels * sizeof(uint8_t));
    }
    iumMasksBuffer.upload(ium_result.masks.data(), ium_result.masks.size());

    size_t result_size = num_pixels * num_cameras * sizeof(uint8_t);
    if(resultsBuffer.sizeInBytes != result_size) {
        resultsBuffer.alloc(result_size);
    }
    
    // Clear the output to 0 initially
    cudaMemset((void*)resultsBuffer.d_pointer(), 0, result_size);

    launchParams.camera_data.cameras = (CameraDef*)camBuffer.d_pointer();
    launchParams.camera_data.num_cameras = num_cameras;

    launchParams.ium_data.ium_positions = (vec3f*)iumPositionsBuffer.d_pointer();
    launchParams.ium_data.ium_masks = (uint8_t*)iumMasksBuffer.d_pointer();
    launchParams.ium_data.num_pixels = num_pixels;

    launchParams.results.visibility_results = (uint8_t*)resultsBuffer.d_pointer();

    launchParamsBuffer.alloc(sizeof(launchParams));
    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(pipeline, 0,
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        num_pixels, // width
        num_cameras, // height
        1           // depth
    ));

    CUDA_SYNC_CHECK();

    std::vector<uint8_t> h_results(num_pixels * num_cameras);
    resultsBuffer.download(h_results.data(), h_results.size());

    return h_results;
}

} // namespace osc
