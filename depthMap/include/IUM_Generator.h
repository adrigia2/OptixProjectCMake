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
	struct Result
	{
		std::vector<vec3f> positions; // Posizioni 3D reali dei vertici
		std::vector<uint8_t> masks;    // Maschere per i pixel validi
		bool hasPositions() const { return !positions.empty(); }
		bool hasMasks() const { return !masks.empty(); }
	};

public:

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

protected:
	OptixTraversableHandle createGAS(const TriangleMesh& model) override;

	CUDABuffer vertexBuffer;     // Già esistente - per la GAS in UV space
	CUDABuffer indexBuffer;        // Già esistente
	CUDABuffer asBuffer;           // Già esistente

	// result buffers per la raygen program
	CUDABuffer positionsBuffer;    // Già esistente
	CUDABuffer masksBuffer;        // Già esistente
	
	// NUOVI BUFFERS per inverse UV mapping
	CUDABuffer worldVertexBuffer;  // Vertici 3D reali del modello
	CUDABuffer uvCoordBuffer;      // Coordinate UV per il dispositivo

	LaunchParams_IUM launchParams; // Parametri di lancio specifici per IUM

private:
	Result result; // Per memorizzare i risultati della generazione IUM
};
