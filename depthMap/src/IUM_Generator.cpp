#include "IUM_Generator.h"
#include "IOManager.h"
#include <fstream>
#include <algorithm>
#include <limits>

/// <summary>
/// compiled ptx code from the .cu file, embedded as a string literal in the executable.
/// </summary>
extern "C" char embedded_ptx_code_ium[];

char* IUM_Generator::getPtxCode()
{
	return embedded_ptx_code_ium;
}

void IUM_Generator::createRaygenPrograms()
{
	// we do a single ray gen program in this example:
	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderIUM";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&raygenPGs[0]
	));
	if (sizeof_log > 1) PRINT(log);
}

void IUM_Generator::createMissPrograms()
{
	// we do a single ray gen program in this example:
	missPGs.resize(1);
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;
	pgDesc.miss.entryFunctionName = "__miss__renderIUM";
	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&missPGs[0]
	));
	if (sizeof_log > 1) PRINT(log);
}

void IUM_Generator::createHitgroupPrograms()
{
	// for this simple example, we set up a single hit group
	hitgroupPGs.resize(1);
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__renderIUM";
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__renderIUM";
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(OptixManager::instance().getContext(),
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&hitgroupPGs[0]
	));
	if (sizeof_log > 1) PRINT(log);
}

void IUM_Generator::render()
{
	if (launchParams.size.x == 0 || launchParams.size.y == 0)
	{
		LogManager::LogError("Invalid launch size: %u x %u", launchParams.size.x, launchParams.size.y);
		return;
	}

	launchParamsBuffer.resize(sizeof(LaunchParams_IUM));
	launchParamsBuffer.upload(&launchParams, 1);
	OptixManager::instance().render(launchParams.size.x, launchParams.size.y, 1, pipeline, launchParamsBuffer, sbt);

	// Download the results from the GPU
	int dimension = launchParams.size.x * launchParams.size.y;
	// Size the vectors to hold the results

	LogManager::Log("IUM generation completed: %u pixels processed",
		dimension);

	result.positions.clear();
	result.normals.clear();
	result.masks.clear();

	result.positions.resize(dimension);
	result.normals.resize(dimension);
	result.masks.resize(dimension);

	positionsBuffer.download(result.positions.data(), dimension);
	normalsBuffer.download(result.normals.data(), dimension);
	masksBuffer.download(result.masks.data(), dimension);
}

void IUM_Generator::cleanup()
{
	OptixActor::cleanup();
	positionsBuffer.free();
	normalsBuffer.free();
	masksBuffer.free();
	worldVertexBuffer.free();
	uvCoordBuffer.free();
	launchParamsBuffer.free();
}

void IUM_Generator::setTraversable(const TriangleMesh& model)
{
	// Build the GAS in UV space
	launchParams.traversable = createGAS(model);

	// Upload the real 3D vertices, for the mapping back to world space
	worldVertexBuffer.alloc_and_upload(model.vertex);
	launchParams.data.worldVertices = (vec3f*)worldVertexBuffer.d_pointer();

	// Upload the UV coordinates
	uvCoordBuffer.alloc_and_upload(model.texcoord);
	launchParams.data.uvVertices = (vec2f*)uvCoordBuffer.d_pointer();

	// Pass the indices along (already uploaded in buildAccel)
	launchParams.data.indices = (vec3i*)indexBuffer.d_pointer();
	launchParams.data.numTriangles = (uint32_t)model.index.size();

	LogManager::Log("IUM geometry data uploaded: %zu vertices, %zu triangles",
		model.vertex.size(), model.index.size());
}

void IUM_Generator::setTextureSize(vec2i size)
{
	int dimension = size.x * size.y;
	positionsBuffer.resize(dimension * sizeof(vec3f));
	normalsBuffer.resize(dimension * sizeof(vec3f));
	masksBuffer.resize(dimension * sizeof(uint8_t));

	launchParams.size = size;
	launchParams.results.positions = (vec3f*)positionsBuffer.d_pointer();
	launchParams.results.normals = (vec3f*)normalsBuffer.d_pointer();
	launchParams.results.masks = (uint8_t*)masksBuffer.d_pointer();

	LogManager::Log("IUM texture size set to %u x %u", size.x, size.y);
}

void IUM_Generator::printStatus()
{
	LogManager::Log("IUM_Generator status:");
	LogManager::Log("  Raygen programs: %zu", raygenPGs.size());
	LogManager::Log("  Miss programs: %zu", missPGs.size());
	LogManager::Log("  Hitgroup programs: %zu", hitgroupPGs.size());
	LogManager::Log("  UV vertex buffer size: %zu bytes", vertexBuffer.sizeInBytes);
	LogManager::Log("  Index buffer size: %zu bytes", indexBuffer.sizeInBytes);
	LogManager::Log("  Acceleration structure buffer size: %zu bytes", asBuffer.sizeInBytes);
	LogManager::Log("  Positions buffer size: %zu bytes", positionsBuffer.sizeInBytes);
	LogManager::Log("  Masks buffer size: %zu bytes", masksBuffer.sizeInBytes);
}

OptixTraversableHandle IUM_Generator::createGAS(const TriangleMesh& model)
{
	// 1) Upload the indices (unchanged)
	indexBuffer.alloc_and_upload(model.index);

	// 2) Build a buffer of "positions" in UV space: (u, v, 0)
	std::vector<vec3f> uvVerts;
	uvVerts.resize(model.vertex.size());
	for (size_t i = 0; i < uvVerts.size(); ++i) {
		const vec2f uv = model.texcoord[i];          // <-- the model has to carry UVs
		uvVerts[i] = vec3f(uv.x, uv.y, 0.f);
	}
	vertexBuffer.alloc_and_upload(uvVerts);

	OptixTraversableHandle asHandle{ 0 };

	OptixBuildInput triangleInput = {};
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	CUdeviceptr d_vertices = vertexBuffer.d_pointer();
	CUdeviceptr d_indices = indexBuffer.d_pointer();

	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
	triangleInput.triangleArray.numVertices = (int)uvVerts.size();
	triangleInput.triangleArray.vertexBuffers = &d_vertices;

	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexStrideInBytes = sizeof(vec3i);
	triangleInput.triangleArray.numIndexTriplets = (int)model.index.size();
	triangleInput.triangleArray.indexBuffer = d_indices;

	uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
	triangleInput.triangleArray.flags = triangleInputFlags;

	// tipico: 1 SBT record per tutta la mesh
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
	triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	// ==================================================================
	// BLAS setup
	// ==================================================================

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
		| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
		;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage
	(OptixManager::instance().getContext(),
		&accelOptions,
		&triangleInput,
		1,  // num_build_inputs
		&blasBufferSizes
	));

	// ==================================================================
	// prepare compaction
	// ==================================================================

	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	// ==================================================================
	// execute build (main stage)
	// ==================================================================

	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(OptixManager::instance().getContext(),
		/* stream */0,
		&accelOptions,
		&triangleInput,
		1,
		tempBuffer.d_pointer(),
		tempBuffer.sizeInBytes,

		outputBuffer.d_pointer(),
		outputBuffer.sizeInBytes,

		&asHandle,

		&emitDesc, 1
	));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// perform compaction
	// ==================================================================
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	asBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(OptixManager::instance().getContext(),
		/*stream:*/0,
		asHandle,
		asBuffer.d_pointer(),
		asBuffer.sizeInBytes,
		&asHandle));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// aaaaaand .... clean up
	// ==================================================================
	outputBuffer.free(); // << the UNcompacted, temporary output buffer
	tempBuffer.free();
	compactedSizeBuffer.free();

	return asHandle;
}
