#include "OptixManager.h"
#include "Camera.h"
#include "optix7.h"
#include "LogManager.h"

using namespace osc;

class Depth_Generator
{
public:
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();

	void addProgramsInOptixManager();
	void renderTransforms(const std::string& transformFile, const std::string& outputDir);
	void render();
	
	Depth_Generator(OptixManager& optixManager)
		: optixManager(optixManager)
	{
		createRaygenPrograms();
		createMissPrograms();
		createHitgroupPrograms();
		addProgramsInOptixManager();
	}

	void setTraversable(TriangleMesh& model);

	void saveIUMTextureToBitmap(const std::string& filename);

protected:
	std::vector<OptixProgramGroup> raygenPGs;
	std::vector<OptixProgramGroup> missPGs;
	std::vector<OptixProgramGroup> hitgroupPGs;

	OptixTraversableHandle buildAccel(const TriangleMesh& model);


	CUDABuffer vertexBuffer;     
	CUDABuffer indexBuffer;
	CUDABuffer asBuffer;          

	// result buffers per la raygen program
    CUDABuffer depthBuffer;  // Aggiunto buffer per depth map


private:
	OptixManager& optixManager;

	void setCamera(const Camera& camera, float fovY);
};