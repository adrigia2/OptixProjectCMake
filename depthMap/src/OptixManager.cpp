#include "optixManager.h"
#include "LogManager.h"
#include <optix_function_table_definition.h>


static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*)
{
	LogManager::Log("[%2d][%12s]: %s", (int)level, tag, message);
}


void OptixManager::initOptix()
{
	LogManager::Log("Initializing Optix...");

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
	LogManager::Log("Found %d CUDA devices", numDevices);

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
	LogManager::Log("Successfully initialized Optix");
}

void OptixManager::createContext()
{
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
	LogManager::Log("Running on device: %s", deviceProps.name);

    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
		LogManager::LogError("Error querying current context: error code %d", cuRes);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	setLogLevel(LogManager::Level::Verbose);
}





void OptixManager::cleanup()
{
    //OPTIX_CHECK(optixPipelineDestroy(pipeline));
    //OPTIX_CHECK(optixModuleDestroy(module));
	OPTIX_CHECK(optixDeviceContextDestroy(optixContext));
	LogManager::Log("Cleaned up Optix");
}

void OptixManager::setLogLevel(LogManager::Level level)
{
    OPTIX_CHECK(optixDeviceContextSetLogCallback
    (optixContext, context_log_cb, nullptr, (int)level));
}


void OptixManager::render(int launchWidth, int launchHeight, int launchDepth, OptixPipeline& pipeline, CUDABuffer& launchParamsBuffer, OptixShaderBindingTable& sbt)
{
    if(launchWidth == 0 || launchHeight == 0)
		return;
	LogManager::Log("Launching Optix with dimensions: %u x %u", launchWidth, launchHeight);

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

