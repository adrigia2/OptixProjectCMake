#include "SampleRenderer.h"
#include "OptixManager.h"
#include "optix7.h"
#include "LogManager.h"


using namespace osc;

struct PositionUv {
	vec3f position;
	uint1 mask = { 0 };
};


class IUM_Generator
{

public:

	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();

	void addProgramsInOptixManager();
	void render();

	IUM_Generator(OptixManager& optixManager)
		: optixManager(optixManager)
	{
		createRaygenPrograms();
		createMissPrograms();
		createHitgroupPrograms();
		addProgramsInOptixManager();
	}

	void setTraversable(TriangleMesh& model);
	void setTextureSize(uint32_t width, uint32_t height);
	void printStatus();
	
protected:
	std::vector<OptixProgramGroup> raygenPGs;
	std::vector<OptixProgramGroup> missPGs;
	std::vector<OptixProgramGroup> hitgroupPGs;

	OptixTraversableHandle buildAccel(const TriangleMesh& model);


	CUDABuffer uvVertexBuffer;
	CUDABuffer indexBuffer;

	// final acceleration structure buffer
	CUDABuffer asBuffer;

	CUDABuffer positionsBuffer;
	CUDABuffer masksBuffer;

private:
	OptixManager& optixManager;

	
};
