#include "Indirect_Generator.h"
#include <optix_stubs.h>
#include "LogManager.h"
#include "OptixManager.h"
#include <stdexcept>
#include <algorithm>

extern "C" char embedded_ptx_code_indirect[];

namespace osc {

char* Indirect_Generator::getPtxCode() {
    return embedded_ptx_code_indirect;
}

void Indirect_Generator::createRaygenPrograms() {
    raygenPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__collectOccluded";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &raygenPGs[0]));
}

void Indirect_Generator::createMissPrograms() {
    missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__indirect";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &missPGs[0]));
}

void Indirect_Generator::createHitgroupPrograms() {
    hitgroupPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__indirect";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPGs[0]));
}

OptixTraversableHandle Indirect_Generator::createGAS(const TriangleMesh& model) {
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

void Indirect_Generator::setTraversable(const TriangleMesh& model) {
    if (asBuffer.d_ptr != 0) return;
    auto handle = createGAS(model);
    launchParams.traversable = handle;
}

void Indirect_Generator::cleanup() {
    OptixActor::cleanup();
    vertexBuffer.free();
    indexBuffer.free();
    asBuffer.free();

    iumPositionsBuffer.free();
    iumNormalsBuffer.free();
    iumMasksBuffer.free();

    tileRaysDirBuffer.free();
    tileRaysCosBuffer.free();
    tileRaysTHitBuffer.free();
    tileRaysLocalIdxBuffer.free();
    tileCounterBuffer.free();

    launchParamsBuffer.free();
}

void Indirect_Generator::setInputs(const IUM_Generator::Result& ium_result,
                                    int sampleSide,
                                    int tileSize)
{
    numPix   = (int)ium_result.positions.size();
    tileSz   = tileSize;
    sampleSz = sampleSide;

    if (numPix == 0)
        throw std::runtime_error("Indirect_Generator: empty IUM positions");
    if (ium_result.normals.size() != (size_t)numPix)
        throw std::runtime_error("Indirect_Generator: IUM normals size mismatch");
    if (ium_result.masks.size() != (size_t)numPix)
        throw std::runtime_error("Indirect_Generator: IUM masks size mismatch");
    if (sampleSide <= 0)
        throw std::runtime_error("Indirect_Generator: sample_side must be > 0");
    if (tileSize <= 0)
        throw std::runtime_error("Indirect_Generator: tile_size must be > 0");

    // Upload IUM buffers
    iumPositionsBuffer.resize(numPix * sizeof(vec3f));
    iumPositionsBuffer.upload(ium_result.positions.data(), numPix);

    iumNormalsBuffer.resize(numPix * sizeof(vec3f));
    iumNormalsBuffer.upload(ium_result.normals.data(), numPix);

    iumMasksBuffer.resize(numPix * sizeof(uint8_t));
    iumMasksBuffer.upload(ium_result.masks.data(), numPix);

    // Alloca buffer di tile (worst case: tutti i campioni occlusi)
    tileCapacity = tileSz * sampleSide * sampleSide;
    tileRaysDirBuffer.resize(tileCapacity * sizeof(vec3f));
    tileRaysCosBuffer.resize(tileCapacity * sizeof(float));
    tileRaysTHitBuffer.resize(tileCapacity * sizeof(float));
    tileRaysLocalIdxBuffer.resize(tileCapacity * sizeof(int));
    tileCounterBuffer.alloc(sizeof(unsigned int));

    // Fill launch params (campi fissi)
    launchParams.ium_data.ium_positions = (vec3f*)iumPositionsBuffer.d_pointer();
    launchParams.ium_data.ium_normals   = (vec3f*)iumNormalsBuffer.d_pointer();
    launchParams.ium_data.ium_masks     = (uint8_t*)iumMasksBuffer.d_pointer();
    launchParams.ium_data.num_pixels    = numPix;

    launchParams.tile_rays_dir       = (vec3f*)tileRaysDirBuffer.d_pointer();
    launchParams.tile_rays_cos       = (float*)tileRaysCosBuffer.d_pointer();
    launchParams.tile_rays_t_hit     = (float*)tileRaysTHitBuffer.d_pointer();
    launchParams.tile_rays_local_idx = (int*)tileRaysLocalIdxBuffer.d_pointer();
    launchParams.tile_counter        = (unsigned int*)tileCounterBuffer.d_pointer();
    launchParams.tile_capacity       = tileCapacity;

    launchParams.sample_side = sampleSide;
    launchParams.epsilon     = 1e-4f;

    LogManager::Log("Indirect inputs ready: %d pixels, %dx%d samples/texel, tile_size=%d",
                    numPix, sampleSide, sampleSide, tileSize);
}

int Indirect_Generator::numTiles() const {
    if (tileSz <= 0 || numPix <= 0) return 0;
    return (numPix + tileSz - 1) / tileSz;
}

Indirect_Generator::TileResult Indirect_Generator::renderTile(int tileIdx) {
    if (numPix == 0)
        throw std::runtime_error("Indirect_Generator::renderTile called before setInputs");

    const int tileOffset      = tileIdx * tileSz;
    const int actualTileSize  = std::min(tileSz, numPix - tileOffset);

    if (actualTileSize <= 0)
        throw std::runtime_error("Indirect_Generator::renderTile: tileIdx out of range");

    // Reset counter
    const unsigned int zero = 0u;
    tileCounterBuffer.upload(&zero, 1);

    // Aggiorna i campi variabili del tile
    launchParams.tile_offset = tileOffset;
    launchParams.tile_size   = actualTileSize;

    if (launchParamsBuffer.sizeInBytes != sizeof(LaunchParams_Indirect))
        launchParamsBuffer.resize(sizeof(LaunchParams_Indirect));
    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(pipeline, 0,
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        actualTileSize, 1, 1));
    CUDA_SYNC_CHECK();

    // Scarica il contatore
    unsigned int count = 0;
    tileCounterBuffer.download(&count, 1);

    // Clamp per sicurezza
    const int clampedCount = std::min((int)count, tileCapacity);

    TileResult res;
    res.count = clampedCount;

    if (clampedCount > 0) {
        res.dirs.resize(clampedCount);
        res.cos_weights.resize(clampedCount);
        res.t_hits.resize(clampedCount);
        res.local_indices.resize(clampedCount);

        tileRaysDirBuffer.download(res.dirs.data(), clampedCount);
        tileRaysCosBuffer.download(res.cos_weights.data(), clampedCount);
        tileRaysTHitBuffer.download(res.t_hits.data(), clampedCount);
        tileRaysLocalIdxBuffer.download(res.local_indices.data(), clampedCount);
    }

    return res;
}

} // namespace osc
