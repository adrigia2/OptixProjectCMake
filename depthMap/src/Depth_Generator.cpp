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

void Depth_Generator::setCamera(const Camera& camera, float fovY, vec2i frameSize)
{
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

//void Depth_Generator::saveIUMTextureToBitmap(const std::string& outDir, FrameResult& frame)
//{
//    LaunchParams& launchParams = optixManager.getLaunchParams();
//    const uint32_t width = launchParams.depth.frame.size.x;
//    const uint32_t height = launchParams.depth.frame.size.y;
//
//    if (width == 0 || height == 0) {
//        LogManager::LogError("Cannot save depth map: invalid size %u x %u", width, height);
//        return;
//    }
//
//    if (!launchParams.depth.frame.depthBuffer) {
//        LogManager::LogError("Cannot save depth map: depth buffer is null");
//        return;
//    }
//
//    // Scarica i dati depth dalla GPU
//	auto& depths = frame.depthData;
//	auto& fileName = frame.depthFileName;
//
//    LogManager::LogInfo("Downloaded %u depth values from GPU", width * height);
//
//    // Trova i valori min/max per la normalizzazione (escludendo valori infiniti)
//    float minDepth = std::numeric_limits<float>::max();
//    float maxDepth = -std::numeric_limits<float>::max();
//
//    for (uint32_t i = 0; i < width * height; i++) {
//        const float depth = depths[i];
//        // Ignora valori infiniti o invalidi (miss rays)
//        if (std::isfinite(depth) && depth < 1e10f) {
//            minDepth = std::min(minDepth, depth);
//            maxDepth = std::max(maxDepth, depth);
//        }
//    }
//
//    LogManager::LogInfo("Depth range: min=%.3f, max=%.3f", minDepth, maxDepth);
//
//    // Prepara i dati RGB per BMP (24 bit per pixel)
//    // Depth viene convertito in scala di grigi
//    std::vector<uint8_t> pixels(width * height * 3);
//
//    const float depthRange = maxDepth - minDepth;
//    const bool hasValidRange = depthRange > 1e-6f;
//
//    for (uint32_t y = 0; y < height; y++) {
//        for (uint32_t x = 0; x < width; x++) {
//            const uint32_t srcIdx = x + y * width;
//            const uint32_t dstIdx = (x + y * width) * 3;
//
//            float depth = depths[srcIdx];
//            uint8_t grayValue;
//
//            // Normalizza depth in [0, 255]
//            if (std::isfinite(depth) && depth < 1e10f && hasValidRange) {
//                // Normalizza in [0, 1] e poi converti in [0, 255]
//                float normalized = (depth - minDepth) / depthRange;
//                // Inverti per far sì che vicino = bianco, lontano = nero
//                normalized = 1.0f - normalized;
//                grayValue = static_cast<uint8_t>(normalized * 255.0f);
//            }
//            else {
//                // Pixel con miss (depth infinito) = nero
//                grayValue = 0;
//            }
//
//            // Grayscale: R = G = B
//            pixels[dstIdx + 0] = grayValue;  // B
//            pixels[dstIdx + 1] = grayValue;  // G
//            pixels[dstIdx + 2] = grayValue;  // R
//        }
//    }
//
//    // Calcola padding per allineamento a 4 byte
//    const uint32_t rowSize = ((width * 3 + 3) / 4) * 4;
//    const uint32_t paddingSize = rowSize - width * 3;
//
//    // Strutture header BMP
//#pragma pack(push, 1)
//    struct BMPHeader {
//        uint16_t fileType{ 0x4D42 };     // "BM"
//        uint32_t fileSize{ 0 };
//        uint16_t reserved1{ 0 };
//        uint16_t reserved2{ 0 };
//        uint32_t offsetData{ 54 };
//    };
//
//    struct BMPInfoHeader {
//        uint32_t size{ 40 };
//        int32_t  width{ 0 };
//        int32_t  height{ 0 };
//        uint16_t planes{ 1 };
//        uint16_t bitCount{ 24 };
//        uint32_t compression{ 0 };
//        uint32_t sizeImage{ 0 };
//        int32_t  xPixelsPerMeter{ 0 };
//        int32_t  yPixelsPerMeter{ 0 };
//        uint32_t colorsUsed{ 0 };
//        uint32_t colorsImportant{ 0 };
//    };
//#pragma pack(pop)
//
//    // Prepara header BMP
//    BMPHeader fileHeader;
//    fileHeader.fileSize = 54 + rowSize * height;
//
//    BMPInfoHeader infoHeader;
//    infoHeader.width = width;
//    infoHeader.height = height;
//    infoHeader.sizeImage = rowSize * height;
//
//    // Costruisci il buffer completo: header + infoHeader + pixel data
//    std::vector<uint8_t> fileData;
//    fileData.reserve(54 + rowSize * height);
//
//    // Aggiungi gli header
//    const uint8_t* headerPtr = reinterpret_cast<const uint8_t*>(&fileHeader);
//    fileData.insert(fileData.end(), headerPtr, headerPtr + sizeof(fileHeader));
//
//    const uint8_t* infoHeaderPtr = reinterpret_cast<const uint8_t*>(&infoHeader);
//    fileData.insert(fileData.end(), infoHeaderPtr, infoHeaderPtr + sizeof(infoHeader));
//
//    // Aggiungi i dati pixel con padding (dal basso verso l'alto per il formato BMP)
//    std::vector<uint8_t> padding(paddingSize, 0);
//    for (int32_t y = height - 1; y >= 0; y--) {
//        const uint8_t* rowPtr = &pixels[y * width * 3];
//        fileData.insert(fileData.end(), rowPtr, rowPtr + width * 3);
//        if (paddingSize > 0) {
//            fileData.insert(fileData.end(), padding.begin(), padding.end());
//        }
//    }
//
//    // Usa IOManager per salvare il file
//	IOManager::IOResult result = IOManager::saveBinaryFile(outDir + fileName + ".bmp"
//        , fileData);
//
//    if (result.success) {
//        LogManager::LogInfo("Depth map saved to: %s (%u x %u pixels, %zu bytes)",
//            fileName.c_str(), width, height, result.bytesWritten);
//    }
//    else {
//        LogManager::LogError("Failed to save depth map: %s", result.errorMessage.c_str());
//    }
//}

//void Depth_Generator::renderTransforms(const std::string& transformFile, const std::string& outputDir)
//{
//    TransformData transforms;
//    if (!transforms.loadFromFile(transformFile)) {
//		LogManager::LogError("Failed to load transforms from file: %s", transformFile.c_str());
//        return;
//    }
//
//	depthBuffer.resize(transforms.w * transforms.h * sizeof(float));
//	LaunchParams& launchParams = optixManager.getLaunchParams();
//
//	launchParams.depth.frame.depthBuffer = (float*)depthBuffer.d_pointer();
//	launchParams.depth.frame.size = vec2i(transforms.w, transforms.h);
//
//    for (size_t i = 0; i < transforms.frames.size(); i++) {
//        const auto& frame = transforms.frames[i];
//        const auto filename = frame.getFileName(osc::IMAGE_TYPE_RGB, false);
//
//        // Crea la camera dal transform
//        Camera camera;
//        camera.pos = frame.getPosition();
//        camera.forward = frame.getForward();
//        camera.up = frame.getUp();
//
//		LogManager::LogDebug("Rendering frame %zu: %s", i, filename.c_str());
//		setCamera(camera, transforms.camera_angle_x);
//		render();
//
//		std::vector<float> depths(transforms.w * transforms.h);
//		// Salva la depth map come bitmap per debug
//		std::string depthFilename = "depth_" + filename;
//
//		LogManager::LogDebug("Downloading depth buffer for frame %zu: %s", i, depthFilename.c_str());
//		depthBuffer.download(depths.data(), transforms.w * transforms.h);
//		frameResults.push_back({ depths, depthFilename });
//
//    }
//}

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

void Depth_Generator::meedRenderPosition(bool isNeeded)
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
	// Pulizia dei buffer specifici di Depth_Generator
	depthBuffer.free();
	positionBuffer.free();
	normalBuffer.free();
	launchParamsBuffer.free();
}

//void Depth_Generator::saveDepthMapToOpenExr(const std::string& outDir, FrameResult& frame)
//{
//    LaunchParams& launchParams = optixManager.getLaunchParams();
//    const uint32_t width = launchParams.depth.frame.size.x;
//    const uint32_t height = launchParams.depth.frame.size.y;
//
//    if (width == 0 || height == 0) {
//        LogManager::LogError("Cannot save depth map: invalid size %u x %u", width, height);
//        return;
//    }
//
//    auto& depths = frame.depthData;
//    auto& fileName = frame.depthFileName;
//
//    LogManager::LogInfo("Saving %u depth values to OpenEXR", width * height);
//
//    try {
//        // Crea l'header con un singolo canale float "Z"
//        Imf::Header header(width, height);
//        header.channels().insert("Z", Imf::Channel(Imf::FLOAT));
//
//        // Crea il file di output
//        const std::string fullPath = outDir + "/" + fileName + ".exr";
//        Imf::OutputFile file(fullPath.c_str(), header);
//
//        // Prepara il frame buffer
//        Imf::FrameBuffer frameBuffer;
//        frameBuffer.insert("Z",
//            Imf::Slice(Imf::FLOAT,
//                (char*)depths.data(),
//                sizeof(float),      // xStride: distanza tra pixel sulla stessa riga
//                sizeof(float) * width)); // yStride: distanza tra righe
//
//        file.setFrameBuffer(frameBuffer);
//        file.writePixels(height);
//
//        LogManager::LogInfo("Depth map saved to OpenEXR: %s (%u x %u pixels)",
//            fullPath.c_str(), width, height);
//    }
//    catch (const std::exception& e) {
//        LogManager::LogError("Failed to save depth map to OpenEXR: %s", e.what());
//    }
//}
//
//void Depth_Generator::saveIUMTextureToBitmapAll(const std::string& outDir)
//{
//    if(frameResults.empty()) {
//        LogManager::LogWarning("No frames rendered, skipping saving depth maps.");
//        return;
//	}
//
//    for (auto& frameResult : frameResults) {
//        const std::string depthFilename = outDir + "/depth_" + frameResult.depthFileName + ".bmp";
//        LogManager::LogInfo("Saving depth map to: %s", depthFilename.c_str());
//        saveIUMTextureToBitmap(outDir, frameResult);
//	}
//}
//
//void Depth_Generator::saveDepthMapsToOpenExrAll(const std::string& outDir)
//{
//    if (frameResults.empty()) {
//        LogManager::LogWarning("No frames rendered, skipping saving depth maps.");
//        return;
//    }
//
//    for (auto& frameResult : frameResults) {
//        LogManager::LogInfo("Saving depth map to OpenEXR: %s", frameResult.depthFileName.c_str());
//        saveDepthMapToOpenExr(outDir, frameResult);
//    }
//}

