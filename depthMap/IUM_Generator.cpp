#include "IUM_Generator.h"



void IUM_Generator::createRaygenPrograms()
{
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = optixManager.getModule();
    pgDesc.raygen.entryFunctionName = "__raygen__renderIUM";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixManager.getContext(),
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &raygenPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void IUM_Generator::createMissPrograms()
{
    // we do a single ray gen program in this example:
    missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = optixManager.getModule();
    pgDesc.miss.entryFunctionName = "__miss__renderIUM";
    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixManager.getContext(),
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void IUM_Generator::createHitgroupPrograms()
{
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = optixManager.getModule();
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__renderIUM";
    pgDesc.hitgroup.moduleAH = optixManager.getModule();
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__renderIUM";
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixManager.getContext(),
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void IUM_Generator::addProgramsInOptixManager()
{
    for (auto pg : raygenPGs)
		optixManager.addRaygenProgram(pg);
    for (auto pg : missPGs)
		optixManager.addMissProgram(pg);
    for (auto pg : hitgroupPGs)
		optixManager.addHitgroupProgram(pg);
}

void IUM_Generator::render()
{
    auto launchParams = optixManager.getLaunchParams();
    if (launchParams.ium.size.width == 0 || launchParams.ium.size.height == 0)
    {
        LogManager::LogError("Invalid launch size: %u x %u", launchParams.ium.size.width, launchParams.ium.size.height);
        return;
    }

    optixManager.render(launchParams.ium.size.width, launchParams.ium.size.height);
}

void IUM_Generator::setTraversable(TriangleMesh& model)
{
    auto& launchParams = optixManager.getLaunchParams();
	launchParams.traversable = buildAccel(model);
}

void IUM_Generator::setTextureSize(uint32_t width, uint32_t height)
{
	positionsBuffer.resize(width * height * sizeof(vec3f));
	masksBuffer.resize(width * height * sizeof(uint8_t));
	LaunchParams& launchParams = optixManager.getLaunchParams();

    launchParams.ium.size.width = width;
    launchParams.ium.size.height = height;
	launchParams.ium.positions = (vec3f*)positionsBuffer.d_pointer();
	launchParams.ium.masks = (uint8_t*)masksBuffer.d_pointer();

	LogManager::LogDebug("IUM texture size set to %u x %u", width, height);
}

void IUM_Generator::printStatus()
{
    LogManager::LogDebug("IUM_Generator status:");
    LogManager::LogDebug("  Raygen programs: %zu", raygenPGs.size());
    LogManager::LogDebug("  Miss programs: %zu", missPGs.size());
    LogManager::LogDebug("  Hitgroup programs: %zu", hitgroupPGs.size());
    LogManager::LogDebug("  UV vertex buffer size: %zu bytes", uvVertexBuffer.sizeInBytes);
    LogManager::LogDebug("  Index buffer size: %zu bytes", indexBuffer.sizeInBytes);
    LogManager::LogDebug("  Acceleration structure buffer size: %zu bytes", asBuffer.sizeInBytes);
    LogManager::LogDebug("  Positions buffer size: %zu bytes", positionsBuffer.sizeInBytes);
    LogManager::LogDebug("  Masks buffer size: %zu bytes", masksBuffer.sizeInBytes);
}

OptixTraversableHandle IUM_Generator::buildAccel(const TriangleMesh &model)
{
    // 1) Upload indici (uguali)
    indexBuffer.alloc_and_upload(model.index);

    // 2) Costruisci un buffer di "posizioni" in UV space: (u,v,0)
    std::vector<vec3f> uvVerts;
    uvVerts.resize(model.vertex.size());
    for (size_t i = 0; i < uvVerts.size(); ++i) {
        const vec2f uv = model.texcoord[i];          // <-- serve nel model
        uvVerts[i] = vec3f(uv.x, uv.y, 0.f);
    }
    uvVertexBuffer.alloc_and_upload(uvVerts);

    OptixTraversableHandle asHandle{ 0 };

    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr d_vertices = uvVertexBuffer.d_pointer();
    CUdeviceptr d_indices = indexBuffer.d_pointer();

    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput.triangleArray.numVertices = (int)uvVerts.size();
    triangleInput.triangleArray.vertexBuffers = &d_vertices;

    triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes = sizeof(vec3i);
    triangleInput.triangleArray.numIndexTriplets = (int)model.index.size();
    triangleInput.triangleArray.indexBuffer = d_indices;

    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    triangleInput.triangleArray.flags = triangleInputFlags;

    // tipico: 1 SBT record per tutta la mesh
    triangleInput.triangleArray.numSbtRecords = 1;
    triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
        | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
        ;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
    (optixManager.getContext(),
        &accelOptions,
        &triangleInput,
        1,  // num_build_inputs
        &blasBufferSizes
    ));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixManager.getContext(),
        /* stream */0,
        &accelOptions,
        &triangleInput,
        1,
        tempBuffer.d_pointer(),
        tempBuffer.sizeInBytes,

        outputBuffer.d_pointer(),
        outputBuffer.sizeInBytes,

        &asHandle,

        &emitDesc, 1
    ));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixManager.getContext(),
        /*stream:*/0,
        asHandle,
        asBuffer.d_pointer(),
        asBuffer.sizeInBytes,
        &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;

}



