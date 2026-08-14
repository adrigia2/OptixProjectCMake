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
		std::vector<vec3f> positions; // real 3D positions behind each texel
		std::vector<vec3f> normals;   // face normals per texel
		std::vector<uint8_t> masks;   // mask of the valid texels
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

	CUDABuffer vertexBuffer;       // for the GAS built in UV space
	CUDABuffer indexBuffer;
	CUDABuffer asBuffer;

	// result buffers for the raygen program
	CUDABuffer positionsBuffer;
	CUDABuffer normalsBuffer;      // face normals per texel
	CUDABuffer masksBuffer;
	
	// Buffers specific to the inverse UV mapping
	CUDABuffer worldVertexBuffer;  // the model's real 3D vertices
	CUDABuffer uvCoordBuffer;      // UV coordinates, on the device

	LaunchParams_IUM launchParams; // launch parameters of the IUM pass

private:
	Result result;                 // results of the IUM pass
};
