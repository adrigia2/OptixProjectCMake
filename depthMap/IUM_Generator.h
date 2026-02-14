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


	CUDABuffer uvVertexBuffer;     // Già esistente - per la GAS in UV space
	CUDABuffer indexBuffer;        // Già esistente
	CUDABuffer asBuffer;           // Già esistente
	CUDABuffer positionsBuffer;    // Già esistente
	CUDABuffer masksBuffer;        // Già esistente
	
	// NUOVI BUFFERS per inverse UV mapping
	CUDABuffer worldVertexBuffer;  // Vertici 3D reali del modello
	CUDABuffer uvCoordBuffer;      // Coordinate UV per il dispositivo

private:
	OptixManager& optixManager;

	
};
