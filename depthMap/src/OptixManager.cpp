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



void OptixManager::cleanup()
{
    //OPTIX_CHECK(optixPipelineDestroy(pipeline));
    //OPTIX_CHECK(optixModuleDestroy(module));
	OPTIX_CHECK(optixDeviceContextDestroy(optixContext));
	LogManager::LogDebug("Cleaned up Optix");
}


void OptixManager::render(int launchWidth, int launchHeight, int launchDepth, CUDABuffer& launchParams, OptixShaderBindingTable& sbt)
{
    if(launchWidth == 0 || launchHeight == 0)
		return;
	LogManager::LogInfo("Launching Optix with dimensions: %u x %u", launchWidth, launchHeight);

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        pipeline, stream,
        /*! parameters and SBT */
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        /*! dimensions of the launch: */
        launchWidth,
        launchHeight,
        launchDepth
    ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
}

void OptixManager::printStatus()
{

}

