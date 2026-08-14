#pragma once

#include <TriangleMesh.h>
#include <optix7.h>
#include <CUDABuffer.h>
using namespace osc;
class OptixActor {

public:
	virtual void createModule();
	virtual void createPipeline();
	virtual void createRaygenPrograms() = 0;
	virtual void createMissPrograms() = 0;
	virtual void createHitgroupPrograms() = 0;
	virtual void createSBT();
	virtual char* getPtxCode() = 0;
	virtual void cleanup();

	virtual void setTraversable(const TriangleMesh& model) = 0;
protected:

	virtual OptixTraversableHandle createGAS(const TriangleMesh& model);
	std::vector<OptixProgramGroup> raygenPGs;
	CUDABuffer raygenRecordsBuffer;
	std::vector<OptixProgramGroup> missPGs;
	CUDABuffer missRecordsBuffer;
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUDABuffer hitgroupRecordsBuffer;
	OptixShaderBindingTable sbt = {};

	// module
	OptixModule                 module;
	OptixModuleCompileOptions   moduleCompileOptions = {};


	// pipeline
	OptixPipeline               pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions    pipelineLinkOptions = {};

	// mesh buffers
	CUDABuffer vertexBuffer;
	CUDABuffer indexBuffer;

	// as output buffer
	CUDABuffer asBuffer;

	CUDABuffer launchParamsBuffer; // buffer holding the launch parameters
};
