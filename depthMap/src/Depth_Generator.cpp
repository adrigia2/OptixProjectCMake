#include "Depth_Generator.h"
#include "TransformReader.h"
#include "IOManager.h"
#include <algorithm>
#include <limits>
#include <cmath>

/// <summary>
/// compiled ptx code from the .cu file, embedded as a string literal in the executable.
/// </summary>
extern "C" char embedded_ptx_code[];

char* Depth_Generator::getPtxCode()
{
	return embedded_ptx_code;
}

void Depth_Generator::createRaygenPrograms()
{
	// we do a single ray gen program in this example:
	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

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

/*! does all setup for the miss program(s) we are going to use */
void Depth_Generator::createMissPrograms()
{
	// we do a single ray gen program in this example:
	missPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;
	pgDesc.miss.entryFunctionName = "__miss__radiance";

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

/*! does all setup for the hitgroup program(s) we are going to use */
void Depth_Generator::createHitgroupPrograms()
{
	// for this simple example, we set up a single hit group
	hitgroupPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

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

void Depth_Generator::setCamera(const Camera& camera)
{
	const float fovY = camera.getFovY();
	const vec2i frameSize = camera.getFrameSize();

	launchParams.camera.position = camera.getPos();
	launchParams.camera.direction = normalize(camera.getForward());
	launchParams.size = frameSize;

	const float aspect = frameSize.x / float(frameSize.y);

	// Convert FOV -> image plane half-size (pinhole model)
	const float halfHeight = tanf(0.5f * fovY);
	const float halfWidth = halfHeight * aspect;

	// Build a robust orthonormal basis (right, up, forward)
	// (If your handedness is flipped, swap cross order as noted below.)
	auto right = cross(launchParams.camera.direction, camera.getUp());
	const float rightLen2 = dot(right, right);
	if (rightLen2 < 1e-12f) {
		// Fallback if up is (almost) parallel to direction
		const auto worldUp = vec3f(0.f, 0.f, 1.f);
		right = cross(launchParams.camera.direction, worldUp);
	}
	right = normalize(right);

	vec3f up = normalize(cross(right, launchParams.camera.direction));

	// These represent half-spans of the image plane in world units at z=1
	launchParams.camera.horizontal = halfWidth * right;
	launchParams.camera.vertical = halfHeight * up;

	maskBuffer.resize(frameSize.x * frameSize.y);
	launchParams.results.maskBuffer = (uint8_t*)maskBuffer.d_ptr;
}

void Depth_Generator::render()
{
	auto needDepth = launchParams.flags.computeDepth;
	auto needPosition = launchParams.flags.computePositional;
	auto needNormal = launchParams.flags.computeNormal;

	auto frameSize = launchParams.size;
	if (needDepth) {
		launchParams.flags.computeDepth = true;
		depthBuffer.resize(frameSize.x * frameSize.y * sizeof(float));
		launchParams.results.depthBuffer = reinterpret_cast<float*>(depthBuffer.d_pointer());
	}
	else {
		launchParams.flags.computeDepth = false;
		launchParams.results.depthBuffer = nullptr;
	}

	if (needPosition) {
		launchParams.flags.computePositional = true;
		positionBuffer.resize(frameSize.x * frameSize.y * sizeof(vec3f));
		launchParams.results.positionalBuffer = reinterpret_cast<vec3f*>(positionBuffer.d_pointer());
	}
	else {
		launchParams.flags.computePositional = false;
		launchParams.results.positionalBuffer = nullptr;
	}
	if (needNormal) {
		launchParams.flags.computeNormal = true;
		normalBuffer.resize(frameSize.x * frameSize.y * sizeof(vec3f));
		launchParams.results.normalBuffer = reinterpret_cast<vec3f*>(normalBuffer.d_pointer());
	}
	else {
		launchParams.flags.computeNormal = false;
		launchParams.results.normalBuffer = nullptr;
	}

	//throw std::runtime_error("Depth_Generator::render() not implemented yet");
	//LaunchParams& launchParams = optixManager.getLaunchParams();

	launchParamsBuffer.resize(sizeof(LaunchParams_DPN));
	launchParamsBuffer.upload(&launchParams, 1);
	OptixManager::instance().render(
		launchParams.size.x,
		launchParams.size.y,
		1,
		pipeline,
		launchParamsBuffer,
		sbt);

	result.depthData.clear();
	if (needDepth) {
		result.depthData.resize(frameSize.x * frameSize.y);
		depthBuffer.download(result.depthData.data(), frameSize.x * frameSize.y);
	}
	result.positionalData.clear();
	if (needPosition) {
		result.positionalData.resize(frameSize.x * frameSize.y);
		positionBuffer.download(result.positionalData.data(), frameSize.x * frameSize.y);
	}
	result.normalData.clear();
	if (needNormal) {
		result.normalData.resize(frameSize.x * frameSize.y);
		normalBuffer.download(result.normalData.data(), frameSize.x * frameSize.y);
	}

	result.maskData.clear();
	result.maskData.resize(frameSize.x * frameSize.y);
	maskBuffer.download(result.maskData.data(), frameSize.x * frameSize.y);
}

void Depth_Generator::needRenderDepth(bool isNeeded)
{
	launchParams.flags.computeDepth = isNeeded;
}

void Depth_Generator::needRenderPosition(bool isNeeded)
{
	launchParams.flags.computePositional = isNeeded;
}

void Depth_Generator::needRenderNormal(bool isNeeded)
{
	launchParams.flags.computeNormal = isNeeded;
}

void Depth_Generator::setTraversable(const TriangleMesh& model)
{
	auto gasHandle = createGAS(model);
	launchParams.traversable = gasHandle;
}

void Depth_Generator::cleanup()
{
	OptixActor::cleanup();
	// Free the buffers owned by Depth_Generator
	depthBuffer.free();
	positionBuffer.free();
	normalBuffer.free();
	launchParamsBuffer.free();
}
