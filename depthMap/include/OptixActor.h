#include <TriangleMesh.h>
#include <optix7.h>
#include <CUDABuffer.h>
using namespace osc;
namespace pipeline {
	class OptixActor {

	public:
		virtual void render(int launchWidth, int launchHeight) = 0;
		virtual void createModule();
		virtual void createPipeline();
		virtual void createRaygenPrograms() = 0;
		virtual void createMissPrograms() = 0;
		virtual void createHitgroupPrograms() = 0;
		virtual void createSBT();
		virtual char* getPtxCode() = 0;

		virtual OptixTraversableHandle createGAS(const TriangleMesh& model);
	protected:

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
	};
}