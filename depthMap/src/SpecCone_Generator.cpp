#include "SpecCone_Generator.h"
#include <optix_stubs.h>
#include "LogManager.h"
#include "OptixManager.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

extern "C" char embedded_ptx_code_speccone[];

namespace osc {

char* SpecCone_Generator::getPtxCode() {
    return embedded_ptx_code_speccone;
}

void SpecCone_Generator::createRaygenPrograms() {
    raygenPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__specCone";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &raygenPGs[0]));
}

void SpecCone_Generator::createMissPrograms() {
    missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__specCone";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &missPGs[0]));
}

void SpecCone_Generator::createHitgroupPrograms() {
    hitgroupPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__specCone";
    OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
        &pgDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPGs[0]));
}

OptixTraversableHandle SpecCone_Generator::createGAS(const TriangleMesh& model) {
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

void SpecCone_Generator::setTraversable(const TriangleMesh& model) {
    if (asBuffer.d_ptr != 0) return;
    auto handle = createGAS(model);
    launchParams.traversable = handle;
}

void SpecCone_Generator::cleanup() {
    OptixActor::cleanup();
    vertexBuffer.free();
    indexBuffer.free();
    asBuffer.free();

    iumPositionsBuffer.free();
    iumNormalsBuffer.free();
    iumMasksBuffer.free();
    ringCosBuffer.free();
    ringSamplesBuffer.free();
    skyboxBuffer.free();
    visibilityBuffer.free();

    tileRaysDirBuffer.free();
    tileRaysTHitBuffer.free();
    tileRaysLocalIdxBuffer.free();
    tileRaysRingIdxBuffer.free();
    tileCounterBuffer.free();
    skySumBuffer.free();
    validCountBuffer.free();

    launchParamsBuffer.free();
}

void SpecCone_Generator::setInputs(const IUM_Generator::Result& ium_result,
                                   const std::vector<float>& coneAperturesDeg,
                                   int samplesPerRing,
                                   int tileSize)
{
    if (coneAperturesDeg.size() < 2)
        throw std::runtime_error("SpecCone_Generator: need at least 2 cone apertures");
    setInputs(ium_result, coneAperturesDeg,
              std::vector<int>(coneAperturesDeg.size() - 1, samplesPerRing),
              tileSize);
}

void SpecCone_Generator::setInputs(const IUM_Generator::Result& ium_result,
                                   const std::vector<float>& coneAperturesDeg,
                                   const std::vector<int>& samplesPerRing,
                                   int tileSize)
{
    numPix = (int)ium_result.positions.size();
    tileSz = tileSize;

    if (numPix == 0)
        throw std::runtime_error("SpecCone_Generator: empty IUM positions");
    if (ium_result.normals.size() != (size_t)numPix)
        throw std::runtime_error("SpecCone_Generator: IUM normals size mismatch");
    if (ium_result.masks.size() != (size_t)numPix)
        throw std::runtime_error("SpecCone_Generator: IUM masks size mismatch");
    if (tileSize <= 0)
        throw std::runtime_error("SpecCone_Generator: tile_size must be > 0");
    if (coneAperturesDeg.size() < 2)
        throw std::runtime_error("SpecCone_Generator: need at least 2 cone apertures");
    if (coneAperturesDeg[0] != 0.0f)
        throw std::runtime_error("SpecCone_Generator: first cone aperture must be 0 (mirror ray)");
    for (size_t i = 1; i < coneAperturesDeg.size(); ++i) {
        if (coneAperturesDeg[i] <= coneAperturesDeg[i - 1])
            throw std::runtime_error("SpecCone_Generator: cone apertures must be strictly ascending");
        if (coneAperturesDeg[i] > 360.0f)
            throw std::runtime_error("SpecCone_Generator: cone aperture exceeds 360 degrees");
    }

    numRings = (int)coneAperturesDeg.size() - 1;

    if ((int)samplesPerRing.size() != numRings)
        throw std::runtime_error(
            "SpecCone_Generator: samples_per_ring has " +
            std::to_string(samplesPerRing.size()) + " entries, expected " +
            std::to_string(numRings) + " (one per ring = apertures - 1)");
    for (int i = 0; i < numRings; ++i)
        if (samplesPerRing[i] <= 0)
            throw std::runtime_error("SpecCone_Generator: samples_per_ring[" +
                                     std::to_string(i) + "] must be > 0");

    // Samples per level: [0] = 1 (the mirror ray), then one value per ring
    ringSamples.assign(numRings + 1, 1);
    for (int i = 0; i < numRings; ++i)
        ringSamples[i + 1] = samplesPerRing[i];

    // Ring edges: cosines of the half-apertures (total aperture / 2)
    std::vector<float> ringCos(coneAperturesDeg.size());
    constexpr float DEG2RAD = 3.14159265358979323846f / 180.0f;
    for (size_t i = 0; i < coneAperturesDeg.size(); ++i)
        ringCos[i] = std::cos(0.5f * coneAperturesDeg[i] * DEG2RAD);

    // Upload the persistent inputs
    iumPositionsBuffer.resize(numPix * sizeof(vec3f));
    iumPositionsBuffer.upload(ium_result.positions.data(), numPix);

    iumNormalsBuffer.resize(numPix * sizeof(vec3f));
    iumNormalsBuffer.upload(ium_result.normals.data(), numPix);

    iumMasksBuffer.resize(numPix * sizeof(uint8_t));
    iumMasksBuffer.upload(ium_result.masks.data(), numPix);

    ringCosBuffer.resize(ringCos.size() * sizeof(float));
    ringCosBuffer.upload(ringCos.data(), ringCos.size());

    ringSamplesBuffer.resize(ringSamples.size() * sizeof(int));
    ringSamplesBuffer.upload(ringSamples.data(), ringSamples.size());

    // Tile buffers (worst case: every sample hits geometry).
    // The count has to be done in size_t: with large tiles and many samples the
    // product overflows INT_MAX, and a negative capacity would zero the rays read back.
    const int numLevels = numRings + 1;
    size_t raysPerTexel = 1;                       // level 0 = the mirror ray
    for (int i = 0; i < numRings; ++i)
        raysPerTexel += (size_t)samplesPerRing[i];
    const size_t capacity = (size_t)tileSz * raysPerTexel;
    if (capacity > (size_t)std::numeric_limits<int>::max())
        throw std::runtime_error(
            "SpecCone_Generator: tile_capacity overflow (" +
            std::to_string(capacity) + " rays); reduce tile_size (" +
            std::to_string(tileSz) + ") or samples per ring");
    tileCapacity = (int)capacity;
    tileRaysDirBuffer.resize(tileCapacity * sizeof(vec3f));
    tileRaysTHitBuffer.resize(tileCapacity * sizeof(float));
    tileRaysLocalIdxBuffer.resize(tileCapacity * sizeof(int));
    tileRaysRingIdxBuffer.resize(tileCapacity * sizeof(int));
    tileCounterBuffer.resize(sizeof(unsigned int));
    skySumBuffer.resize((size_t)tileSz * numLevels * sizeof(vec3f));
    validCountBuffer.resize((size_t)tileSz * numLevels * sizeof(int));

    launchParams.ium_data.ium_positions = (vec3f*)iumPositionsBuffer.d_pointer();
    launchParams.ium_data.ium_normals   = (vec3f*)iumNormalsBuffer.d_pointer();
    launchParams.ium_data.ium_masks     = (uint8_t*)iumMasksBuffer.d_pointer();
    launchParams.ium_data.num_pixels    = numPix;

    launchParams.ring_cos     = (float*)ringCosBuffer.d_pointer();
    launchParams.num_rings    = numRings;
    launchParams.ring_samples = (int*)ringSamplesBuffer.d_pointer();

    launchParams.tile_rays_dir       = (vec3f*)tileRaysDirBuffer.d_pointer();
    launchParams.tile_rays_t_hit     = (float*)tileRaysTHitBuffer.d_pointer();
    launchParams.tile_rays_local_idx = (int*)tileRaysLocalIdxBuffer.d_pointer();
    launchParams.tile_rays_ring_idx  = (int*)tileRaysRingIdxBuffer.d_pointer();
    launchParams.tile_counter        = (unsigned int*)tileCounterBuffer.d_pointer();
    launchParams.tile_capacity       = tileCapacity;

    launchParams.sky_sum     = (vec3f*)skySumBuffer.d_pointer();
    launchParams.valid_count = (int*)validCountBuffer.d_pointer();

    launchParams.epsilon = 1e-4f;

    LogManager::Log("SpecCone inputs ready: %d pixels, %d rings + mirror, "
                    "%zu rays/texel, tile_size=%d, tile_capacity=%d (%.0f MB device)",
                    numPix, numRings, raysPerTexel, tileSize, tileCapacity,
                    tileCapacity * 24.0 / 1048576.0);
}

void SpecCone_Generator::setEnvmap(const std::vector<vec3f>& skybox,
                                   vec2i skyboxSize,
                                   float skyboxYawDegrees)
{
    if (skybox.empty() || (size_t)(skyboxSize.x * skyboxSize.y) != skybox.size())
        throw std::runtime_error("SpecCone_Generator: skybox size mismatch");

    skyboxBuffer.resize(skybox.size() * sizeof(vec3f));
    skyboxBuffer.upload(skybox.data(), skybox.size());

    launchParams.skybox.envmap       = (vec3f*)skyboxBuffer.d_pointer();
    launchParams.skybox.skybox_size  = skyboxSize;
    launchParams.skybox.yaw_offset_u = skyboxYawDegrees / 360.0f;
}

void SpecCone_Generator::setCamera(const vec3f& camPos,
                                   const std::vector<uint8_t>& visibility)
{
    if (numPix == 0)
        throw std::runtime_error("SpecCone_Generator::setCamera called before setInputs");

    launchParams.cam_pos = camPos;

    if (visibility.empty()) {
        launchParams.visibility = nullptr;
    } else {
        if (visibility.size() != (size_t)numPix)
            throw std::runtime_error("SpecCone_Generator: visibility size mismatch");
        visibilityBuffer.resize(numPix * sizeof(uint8_t));
        visibilityBuffer.upload(visibility.data(), numPix);
        launchParams.visibility = (uint8_t*)visibilityBuffer.d_pointer();
    }
    cameraSet = true;
}

int SpecCone_Generator::numTiles() const {
    if (tileSz <= 0 || numPix <= 0) return 0;
    return (numPix + tileSz - 1) / tileSz;
}

SpecCone_Generator::TileResult SpecCone_Generator::renderTile(int tileIdx) {
    if (numPix == 0)
        throw std::runtime_error("SpecCone_Generator::renderTile called before setInputs");
    if (!cameraSet)
        throw std::runtime_error("SpecCone_Generator::renderTile called before setCamera");

    const int tileOffset     = tileIdx * tileSz;
    const int actualTileSize = std::min(tileSz, numPix - tileOffset);

    if (actualTileSize <= 0)
        throw std::runtime_error("SpecCone_Generator::renderTile: tileIdx out of range");

    const int numLevels = numRings + 1;

    // Reset the counter and the per-texel accumulators
    const unsigned int zero = 0u;
    tileCounterBuffer.upload(&zero, 1);
    CUDA_CHECK(Memset((void*)skySumBuffer.d_pointer(), 0, skySumBuffer.sizeInBytes));
    CUDA_CHECK(Memset((void*)validCountBuffer.d_pointer(), 0, validCountBuffer.sizeInBytes));

    launchParams.tile_offset = tileOffset;
    launchParams.tile_size   = actualTileSize;

    if (launchParamsBuffer.sizeInBytes != sizeof(LaunchParams_SpecCone))
        launchParamsBuffer.resize(sizeof(LaunchParams_SpecCone));
    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(pipeline, 0,
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        actualTileSize, 1, 1));
    CUDA_SYNC_CHECK();

    unsigned int count = 0;
    tileCounterBuffer.download(&count, 1);
    const bool overflow = (count > (unsigned int)tileCapacity);
    if (overflow)
        LogManager::LogWarning(
            "SpecCone tile %d: compact buffer OVERFLOW (%u rays requested > "
            "capacity %d), %u samples dropped. Capacity is the exact worst "
            "case, so this means tile_capacity and the kernel disagree.",
            tileIdx, count, tileCapacity, count - (unsigned int)tileCapacity);
    const int clampedCount = std::min((int)count, tileCapacity);

    TileResult res;
    res.count       = clampedCount;
    res.tile_texels = actualTileSize;
    res.num_levels  = numLevels;
    res.overflow    = overflow;
    res.requested   = (int)std::min<unsigned int>(count, (unsigned int)std::numeric_limits<int>::max());

    if (clampedCount > 0) {
        res.dirs.resize(clampedCount);
        res.t_hits.resize(clampedCount);
        res.local_indices.resize(clampedCount);
        res.ring_indices.resize(clampedCount);

        tileRaysDirBuffer.download(res.dirs.data(), clampedCount);
        tileRaysTHitBuffer.download(res.t_hits.data(), clampedCount);
        tileRaysLocalIdxBuffer.download(res.local_indices.data(), clampedCount);
        tileRaysRingIdxBuffer.download(res.ring_indices.data(), clampedCount);
    }

    const size_t accumCount = (size_t)actualTileSize * numLevels;
    res.sky_sum.resize(accumCount);
    res.valid_count.resize(accumCount);
    skySumBuffer.download(res.sky_sum.data(), accumCount);
    validCountBuffer.download(res.valid_count.data(), accumCount);

    return res;
}

} // namespace osc