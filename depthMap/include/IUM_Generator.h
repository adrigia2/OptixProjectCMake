#include "SampleRenderer.h"
#include "OptixManager.h"
#include "optix7.h"
#include "LogManager.h"


using namespace osc;
struct IUMResult {
	std::vector<vec3f> positions; // Posizioni 3D reali dei vertici
	std::vector<uint8_t> masks;    // Maschere per i pixel validi
};

class IUM_Generator : public OptixActor
{

public:

	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();

	void addProgramsInOptixManager();
	void render();

	IUM_Generator()
	{
		createRaygenPrograms();
		createMissPrograms();
		createHitgroupPrograms();
	}

	void setTraversable(TriangleMesh& model);
	void setTextureSize(uint32_t width, uint32_t height);
	void printStatus();
	
	// Salva la texture IUM in un file bitmap
	void saveIUMTextureToBitmap(const std::string& filename);
	
	// Salva la texture IUM in un file OpenEXR
	void saveIUMTextureToOpenExr(const std::string& filename);
	
protected:
	std::vector<OptixProgramGroup> raygenPGs;
	std::vector<OptixProgramGroup> missPGs;
	std::vector<OptixProgramGroup> hitgroupPGs;

	OptixTraversableHandle buildAccel(const TriangleMesh& model);


	CUDABuffer uvVertexBuffer;     // Già esistente - per la GAS in UV space
	CUDABuffer indexBuffer;        // Già esistente
	CUDABuffer asBuffer;           // Già esistente

	// result buffers per la raygen program
	CUDABuffer positionsBuffer;    // Già esistente
	CUDABuffer masksBuffer;        // Già esistente
	
	// NUOVI BUFFERS per inverse UV mapping
	CUDABuffer worldVertexBuffer;  // Vertici 3D reali del modello
	CUDABuffer uvCoordBuffer;      // Coordinate UV per il dispositivo

private:
	OptixManager& optixManager;
	IUMResult result; // Per memorizzare i risultati della generazione IUM

	
};
