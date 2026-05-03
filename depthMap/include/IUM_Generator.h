#pragma once
#include "OptixManager.h"
#include "optix7.h"
#include "LogManager.h"
#include "LaunchParams_IUM.h"
#include <TriangleMesh.h>
#include "OptixActor.h"


using namespace osc;

class IUM_Generator : public OptixActor
{
	

public:

	struct Result
	{
		std::vector<vec3f> positions; // Posizioni 3D reali dei vertici
		std::vector<vec3f> normals;   // Normali di faccia per texel
		std::vector<uint8_t> masks;    // Maschere per i pixel validi
		bool hasPositions() const { return !positions.empty(); }
		bool hasNormals() const { return !normals.empty(); }
		bool hasMasks() const { return !masks.empty(); }
	};

	void createRaygenPrograms() override;
	void createMissPrograms() override;
	void createHitgroupPrograms() override;
	char* getPtxCode() override;

	IUM_Generator()
	{
		createModule();
		createRaygenPrograms();
		createMissPrograms();
		createHitgroupPrograms();
		createSBT();
		createPipeline();
	}

	void setTraversable(const TriangleMesh& model) override;
	void setTextureSize(vec2i size);
	void printStatus();
	void render();
	void cleanup() override;

	Result getResult() const { return result; }

protected:
	OptixTraversableHandle createGAS(const TriangleMesh& model) override;

	CUDABuffer vertexBuffer;     // Gi� esistente - per la GAS in UV space
	CUDABuffer indexBuffer;        // Gi� esistente
	CUDABuffer asBuffer;           // Gi� esistente

	// result buffers per la raygen program
	CUDABuffer positionsBuffer;    // Gi� esistente
	CUDABuffer normalsBuffer;      // Normali di faccia per texel
	CUDABuffer masksBuffer;        // Gi� esistente
	
	// NUOVI BUFFERS per inverse UV mapping
	CUDABuffer worldVertexBuffer;  // Vertici 3D reali del modello
	CUDABuffer uvCoordBuffer;      // Coordinate UV per il dispositivo

	LaunchParams_IUM launchParams; // Parametri di lancio specifici per IUM

private:
	Result result; // Per memorizzare i risultati della generazione IUM
};
