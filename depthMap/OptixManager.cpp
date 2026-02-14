#include "optixManager.h"
#include "LogManager.h"
#include <optix_function_table_definition.h>

/// <summary>
/// compiled ptx code from the .cu file, embedded as a string literal in the executable.
/// </summary>
extern "C" char embedded_ptx_code[];

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
};

static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*)
{
	LogManager::Log("[%2d][%12s]: %s", (int)level, tag, message);
}


void OptixManager::initOptix()
{
    std::cout << "#osc: initializing optix..." << std::endl;

    // -------------------------------------------------------
    // check for available optix capable devices
    // -------------------------------------------------------
    cudaFree(0);

	// check for CUDA capable devices
    int numDevices;
    cudaGetDeviceCount(&numDevices);

	// if no devices found, bail out with error message
    if (numDevices == 0)
        throw std::runtime_error("No CUDA capable devices found!");
	LogManager::LogInfo("Found %d CUDA devices", numDevices);

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
	LogManager::LogInfo("Successfully initialized Optix");
}

void OptixManager::createContext()
{
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
	LogManager::LogInfo("Running on device: %s", deviceProps.name);

    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
		LogManager::LogError("Error querying current context: error code %d", cuRes);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
	(optixContext, context_log_cb, nullptr, (int)LogManager::Level::Default));
}

void OptixManager::createModule()
{
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
#if OPTIX_VERSION >= 70700
    OPTIX_CHECK(optixModuleCreate(optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log, &sizeof_log,
        &module
    ));
#else
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log,      // Log string
        &sizeof_log,// Log string sizse
        &module
    ));
#endif
    if (sizeof_log > 1) PRINT(log);
}

void OptixManager::addRaygenProgram(const OptixProgramGroup& program)
{
	raygenPGs.push_back(program);
}

void OptixManager::addMissProgram(const OptixProgramGroup& program)
{
    missPGs.push_back(program);
}


void OptixManager::addHitgroupProgram(const OptixProgramGroup& program)
{
    hitgroupPGs.push_back(program);
}

void OptixManager::createPipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log, &sizeof_log,
        &pipeline
    ));
    if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
    (/* [in] The pipeline to configure the stack size for */
        pipeline,
        /* [in] The direct stack size requirement for direct
           callables invoked from IS or AH. */
        2 * 1024,
        /* [in] The direct stack size requirement for direct
           callables invoked from RG, MS, or CH.  */
        2 * 1024,
        /* [in] The continuation stack requirement. */
        2 * 1024,
        /* [in] The maximum depth of a traversable graph
           passed to trace. */
        1));
    if (sizeof_log > 1) PRINT(log);
}

void OptixManager::buildSBT()
{
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    int numObjects = 1;
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i = 0; i < numObjects; i++) {
        int objectType = 0;
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
        rec.objectID = i;
        hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void OptixManager::render(int launchWidth, int launchHeight)
{
    if(launchWidth == 0 || launchHeight == 0)
		return;

    launchParamsBuffer.resize(sizeof(launchParams));
    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        pipeline, stream,
        /*! parameters and SBT */
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        /*! dimensions of the launch: */
        launchWidth,
        launchHeight,
        1
    ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
}

void OptixManager::printStatus()
{
    LogManager::LogDebug("OptixManager status:");
    LogManager::LogDebug("  Raygen programs: %zu", raygenPGs.size());
    LogManager::LogDebug("  Miss programs: %zu", missPGs.size());
    LogManager::LogDebug("  Hitgroup programs: %zu", hitgroupPGs.size());
    LogManager::LogDebug("  Launch params buffer size: %zu bytes", launchParamsBuffer.sizeInBytes);

	// print launch params info
    LogManager::LogDebug("  Launch params - depth frame size: %u x %u", launchParams.depth.frame.size.x, launchParams.depth.frame.size.y);
    LogManager::LogDebug("  Launch params - depth camera position: (%f, %f, %f)", launchParams.depth.camera.position.x, launchParams.depth.camera.position.y, launchParams.depth.camera.position.z);
    LogManager::LogDebug("  Launch params - depth camera direction: (%f, %f, %f)", launchParams.depth.camera.direction.x, launchParams.depth.camera.direction.y, launchParams.depth.camera.direction.z);
    LogManager::LogDebug("  Launch params - depth camera horizontal: (%f, %f, %f)", launchParams.depth.camera.horizontal.x, launchParams.depth.camera.horizontal.y, launchParams.depth.camera.horizontal.z);
    LogManager::LogDebug("  Launch params - depth camera vertical: (%f, %f, %f)", launchParams.depth.camera.vertical.x, launchParams.depth.camera.vertical.y, launchParams.depth.camera.vertical.z);
	LogManager::LogDebug("  Launch params - ium size: %u x %u", launchParams.ium.size.width, launchParams.ium.size.height);
	LogManager::LogDebug("  Launch params - ium positions pointer: %p", launchParams.ium.positions);
	LogManager::LogDebug("  Launch params - ium masks pointer: %p", launchParams.ium.masks);
	LogManager::LogDebug("  Launch params - traversable handle: %llu", launchParams.traversable);
}

