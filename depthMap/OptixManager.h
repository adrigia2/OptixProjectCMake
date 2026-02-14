#pragma once

#include "LaunchParams.h"
#include "gdt/math/AffineSpace.h"
#include "objLoader/tiny_obj_loader.h"
#include "SampleRenderer.h"
#include "CUDABuffer.h"

using namespace osc;

class OptixManager {

protected:

    /// <summary>
    /// context of cuda
    /// </summary>
    CUcontext          cudaContext;

	// set when the context is created, used for error checking
    CUstream           stream;

    /// <summary>
	/// info of the cuda device, e.g. name, memory size, etc.
    /// </summary>
    cudaDeviceProp     deviceProps;


	////////////////////// optix related members /////////////////////
    OptixDeviceContext optixContext;

	OptixPipeline               pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions    pipelineLinkOptions = {};

	OptixModule                 module;
	OptixModuleCompileOptions   moduleCompileOptions = {};

    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};


	LaunchParams launchParams;
	CUDABuffer launchParamsBuffer;


    void initOptix();
    void createContext();
    void createModule();

    void createPipeline();
    void buildSBT();


public:
    void render(int launchWidth, int launchHeight);


	void addRaygenProgram(const OptixProgramGroup& program);
	void addMissProgram(const OptixProgramGroup& program);
	void addHitgroupProgram(const OptixProgramGroup& program);
	void printStatus();

	OptixModule& getModule() { return module; }
	OptixDeviceContext& getContext() { return optixContext; }
	LaunchParams& getLaunchParams() { return launchParams; }

    OptixManager()
    {
		launchParams = {};
        initOptix();
        createContext();
        createModule();
    }

   
};