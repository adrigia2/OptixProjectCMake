#include "HemiVis_Generator.h"
#include <optix_stubs.h>
#include "LogManager.h"
#include "OptixManager.h"
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <string>

extern "C" char embedded_ptx_code_hemivis[];

namespace osc {

char* HemiVis_Generator::getPtxCode() {
    return embedded_ptx_code_hemivis;
}

void HemiVis_Generator::createRaygenPrograms() {
    raygenPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__hemiVis";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &raygenPGs[0]));
}

void HemiVis_Generator::createMissPrograms() {
    missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__hemiVis";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &missPGs[0]));
}

void HemiVis_Generator::createHitgroupPrograms() {
    hitgroupPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__hemiVis";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPGs[0]));
}

OptixTraversableHandle HemiVis_Generator::createGAS(const TriangleMesh& model) {
    OptixTraversableHandle asHandle{ 0 };

    vertexBuffer.alloc_and_upload(model.vertex);
    indexBuffer.alloc_and_upload(model.index);

    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr d_vertices = vertexBuffer.d_pointer();
    CUdeviceptr d_indices  = indexBuffer.d_pointer();

    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput.triangleArray.numVertices         = (int)model.vertex.size();
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;

    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = sizeof(vec3i);
    triangleInput.triangleArray.numIndexTriplets    = (int)model.index.size();
    triangleInput.triangleArray.indexBuffer         = d_indices;

    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    triangleInput.triangleArray.flags      = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        OptixManager::instance().getContext(),
        &accelOptions, &triangleInput, 1, &blasBufferSizes));

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        OptixManager::instance().getContext(), 0,
        &accelOptions, &triangleInput, 1,
        tempBuffer.d_pointer(), tempBuffer.sizeInBytes,
        outputBuffer.d_pointer(), outputBuffer.sizeInBytes,
        &asHandle, &emitDesc, 1));
    CUDA_SYNC_CHECK();

    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(OptixManager::instance().getContext(),
        0, asHandle, asBuffer.d_pointer(), asBuffer.sizeInBytes, &asHandle));
    CUDA_SYNC_CHECK();

    outputBuffer.free();
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

void HemiVis_Generator::setTraversable(const TriangleMesh& model) {
    if (asBuffer.d_ptr != 0) return;
    auto handle = createGAS(model);
    launchParams.traversable = handle;
}

void HemiVis_Generator::cleanup() {
    OptixActor::cleanup();
    vertexBuffer.free();
    indexBuffer.free();
    asBuffer.free();

    iumPositionsBuffer.free();
    iumNormalsBuffer.free();
    iumMasksBuffer.free();
    camPosBuffer.free();

    tHitSharedBuffer.free();
    tHitMirrorBuffer.free();
    dbgDirsBuffer.free();

    launchParamsBuffer.free();
}

void HemiVis_Generator::setInputs(const IUM_Generator::Result& ium_result,
                                  int numSamples,
                                  int tileSize)
{
    numPix = (int)ium_result.positions.size();
    tileSz = tileSize;
    numSmp = numSamples;

    if (numPix == 0)
        throw std::runtime_error("HemiVis_Generator: empty IUM positions");
    if (ium_result.normals.size() != (size_t)numPix)
        throw std::runtime_error("HemiVis_Generator: IUM normals size mismatch");
    if (ium_result.masks.size() != (size_t)numPix)
        throw std::runtime_error("HemiVis_Generator: IUM masks size mismatch");
    if (tileSize <= 0)
        throw std::runtime_error("HemiVis_Generator: tile_size must be > 0");
    if (numSamples <= 0)
        throw std::runtime_error("HemiVis_Generator: num_samples must be > 0");

    iumPositionsBuffer.resize(numPix * sizeof(vec3f));
    iumPositionsBuffer.upload(ium_result.positions.data(), numPix);

    iumNormalsBuffer.resize(numPix * sizeof(vec3f));
    iumNormalsBuffer.upload(ium_result.normals.data(), numPix);

    iumMasksBuffer.resize(numPix * sizeof(uint8_t));
    iumMasksBuffer.upload(ium_result.masks.data(), numPix);

    // The count has to be done in size_t: large tiles with a high S overflow INT_MAX fast
    // (at S=16384, 131k texels per tile are enough), and a negative size
    // would silently zero the buffer.
    const size_t sharedRays = (size_t)tileSz * (size_t)numSmp;
    if (sharedRays > (size_t)std::numeric_limits<int>::max())
        throw std::runtime_error(
            "HemiVis_Generator: tile_size x num_samples overflow (" +
            std::to_string(sharedRays) + " rays); reduce tile_size (" +
            std::to_string(tileSz) + ") or num_samples (" +
            std::to_string(numSmp) + ")");
    tHitSharedBuffer.resize(sharedRays * sizeof(float));

    launchParams.ium_data.ium_positions = (vec3f*)iumPositionsBuffer.d_pointer();
    launchParams.ium_data.ium_normals   = (vec3f*)iumNormalsBuffer.d_pointer();
    launchParams.ium_data.ium_masks     = (uint8_t*)iumMasksBuffer.d_pointer();
    launchParams.ium_data.num_pixels    = numPix;

    launchParams.num_samples  = numSmp;
    launchParams.t_hit_shared = (float*)tHitSharedBuffer.d_pointer();
    launchParams.epsilon      = 1e-4f;

    // setCameras may have been called before setInputs: the mirror buffer depends
    // on tileSz, so it has to be reallocated here.
    if (!camPos.empty()) {
        const std::vector<vec3f> cams = camPos;
        setCameras(cams);
    }

    LogManager::Log("HemiVis inputs ready: %d pixels, %d shared samples/texel, "
                    "tile_size=%d, %zu rays/tile (%.0f MB device)",
                    numPix, numSmp, tileSz, sharedRays,
                    sharedRays * 4.0 / 1048576.0);
}

void HemiVis_Generator::setCameras(const std::vector<vec3f>& camPositions)
{
    camPos = camPositions;

    if (camPos.empty()) {
        launchParams.cam_pos      = nullptr;
        launchParams.num_cams     = 0;
        launchParams.t_hit_mirror = nullptr;
        return;
    }

    camPosBuffer.resize(camPos.size() * sizeof(vec3f));
    camPosBuffer.upload(camPos.data(), camPos.size());

    launchParams.cam_pos  = (vec3f*)camPosBuffer.d_pointer();
    launchParams.num_cams = (int)camPos.size();

    if (tileSz > 0) {
        const size_t mirrorRays = (size_t)tileSz * camPos.size();
        tHitMirrorBuffer.resize(mirrorRays * sizeof(float));
        launchParams.t_hit_mirror = (float*)tHitMirrorBuffer.d_pointer();
        if (debugDirs)                 // the lane count may have changed
            setDebugDirections(true);
    }
}

void HemiVis_Generator::setDebugDirections(bool enabled)
{
    debugDirs = enabled;
    if (!enabled) {
        dbgDirsBuffer.free();
        launchParams.dbg_dirs = nullptr;
        return;
    }
    if (tileSz <= 0)
        throw std::runtime_error("HemiVis_Generator::setDebugDirections called before setInputs");

    const size_t lanes = std::max((size_t)numSmp, camPos.size());
    dbgDirsBuffer.resize((size_t)tileSz * lanes * sizeof(vec3f));
    launchParams.dbg_dirs = (vec3f*)dbgDirsBuffer.d_pointer();
}

int HemiVis_Generator::numTiles() const {
    if (tileSz <= 0 || numPix <= 0) return 0;
    return (numPix + tileSz - 1) / tileSz;
}

HemiVis_Generator::TileResult HemiVis_Generator::renderTile(int tileIdx) {
    if (numPix == 0)
        throw std::runtime_error("HemiVis_Generator::renderTile called before setInputs");

    const int tileOffset     = tileIdx * tileSz;
    const int actualTileSize = std::min(tileSz, numPix - tileOffset);

    if (actualTileSize <= 0)
        throw std::runtime_error("HemiVis_Generator::renderTile: tileIdx out of range");

    launchParams.tile_offset = tileOffset;
    launchParams.tile_size   = actualTileSize;

    if (launchParamsBuffer.sizeInBytes != sizeof(LaunchParams_HemiVis))
        launchParamsBuffer.resize(sizeof(LaunchParams_HemiVis));

    TileResult res;
    res.tile_texels = actualTileSize;
    res.num_samples = numSmp;
    res.num_cams    = (int)camPos.size();

    // Shared pass: (texel, sample). Every thread writes its own slot, so there is
    // no need to clear the buffer between tiles.
    launchParams.mode = LaunchParams_HemiVis::MODE_SHARED;
    launchParamsBuffer.upload(&launchParams, 1);
    OPTIX_CHECK(optixLaunch(pipeline, 0,
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        actualTileSize, numSmp, 1));
    CUDA_SYNC_CHECK();

    res.t_hit_shared.resize((size_t)actualTileSize * numSmp);
    tHitSharedBuffer.download(res.t_hit_shared.data(), res.t_hit_shared.size());
    if (debugDirs) {
        res.dirs_shared.resize((size_t)actualTileSize * numSmp);
        dbgDirsBuffer.download(res.dirs_shared.data(), res.dirs_shared.size());
    }

    // Mirror pass: (texel, camera)
    if (!camPos.empty()) {
        launchParams.mode = LaunchParams_HemiVis::MODE_MIRROR;
        launchParamsBuffer.upload(&launchParams, 1);
        OPTIX_CHECK(optixLaunch(pipeline, 0,
            launchParamsBuffer.d_pointer(),
            launchParamsBuffer.sizeInBytes,
            &sbt,
            actualTileSize, (int)camPos.size(), 1));
        CUDA_SYNC_CHECK();

        res.t_hit_mirror.resize((size_t)actualTileSize * camPos.size());
        tHitMirrorBuffer.download(res.t_hit_mirror.data(), res.t_hit_mirror.size());
        if (debugDirs) {
            res.dirs_mirror.resize((size_t)actualTileSize * camPos.size());
            dbgDirsBuffer.download(res.dirs_mirror.data(), res.dirs_mirror.size());
        }
    }

    return res;
}

} // namespace osc
