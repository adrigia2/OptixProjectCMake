#include "OptixManager.h"
#include "Camera.h"
#include "optix7.h"
#include "LogManager.h"

using namespace osc;

struct FrameResult {
	std::vector<float> depthData; // Dati della depth map
	std::string depthFileName; // Nome del file in cui è salvata la depth map
};
	

class Depth_Generator
{
public:
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();

	void addProgramsInOptixManager();
	void renderTransforms(const std::string& transformFile, const std::string& outputDir);
	void render();

	void saveIUMTextureToBitmapAll(const std::string& outDir);

	
	Depth_Generator(OptixManager& optixManager)
		: optixManager(optixManager)
	{
		createRaygenPrograms();
		createMissPrograms();
		createHitgroupPrograms();
		addProgramsInOptixManager();
	}

	void setTraversable(TriangleMesh& model);


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
	void saveIUMTextureToBitmap(const std::string& outDir, FrameResult& frame);

	std::vector<FrameResult> frameResults; // Per memorizzare i risultati di ogni frame
};