#include "IUM_Generator.h"
#include "IOManager.h"
#include <fstream>
#include <algorithm>
#include <limits>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfArray.h>

// Struttura header BMP
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{ 0x4D42 };     // "BM"
    uint32_t fileSize{ 0 };
    uint16_t reserved1{ 0 };
    uint16_t reserved2{ 0 };
    uint32_t offsetData{ 54 };       // Offset dove iniziano i dati pixel
};

struct BMPInfoHeader {
    uint32_t size{ 40 };             // Dimensione di questo header
    int32_t  width{ 0 };
    int32_t  height{ 0 };
    uint16_t planes{ 1 };
    uint16_t bitCount{ 24 };         // 24 bit per pixel (RGB)
    uint32_t compression{ 0 };       // Nessuna compressione
    uint32_t sizeImage{ 0 };
    int32_t  xPixelsPerMeter{ 0 };
    int32_t  yPixelsPerMeter{ 0 };
    uint32_t colorsUsed{ 0 };
    uint32_t colorsImportant{ 0 };
};
#pragma pack(pop)

void IUM_Generator::createRaygenPrograms()
{
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = optixManager.getModule();
    pgDesc.raygen.entryFunctionName = "__raygen__renderIUM";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixManager.getContext(),
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
    pgDesc.miss.module = optixManager.getModule();
    pgDesc.miss.entryFunctionName = "__miss__renderIUM";
    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixManager.getContext(),
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
    pgDesc.hitgroup.moduleCH = optixManager.getModule();
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__renderIUM";
    pgDesc.hitgroup.moduleAH = optixManager.getModule();
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__renderIUM";
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixManager.getContext(),
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void IUM_Generator::addProgramsInOptixManager()
{
    for (auto pg : raygenPGs)
		optixManager.addRaygenProgram(pg);
    for (auto pg : missPGs)
		optixManager.addMissProgram(pg);
    for (auto pg : hitgroupPGs)
		optixManager.addHitgroupProgram(pg);
}

void IUM_Generator::render()
{
    auto launchParams = optixManager.getLaunchParams();
    if (launchParams.ium.size.width == 0 || launchParams.ium.size.height == 0)
    {
        LogManager::LogError("Invalid launch size: %u x %u", launchParams.ium.size.width, launchParams.ium.size.height);
        return;
    }

    optixManager.render(launchParams.ium.size.width, launchParams.ium.size.height);
	// Scarica i risultati dalla GPU
    std::vector<vec3f> positions(launchParams.ium.size.width * launchParams.ium.size.height);
    std::vector<uint8_t> masks(launchParams.ium.size.width * launchParams.ium.size.height);
    
    positionsBuffer.download(positions.data(), launchParams.ium.size.width * launchParams.ium.size.height);
    masksBuffer.download(masks.data(), launchParams.ium.size.width * launchParams.ium.size.height);
    
    // Salva i risultati nella struttura result
    result.positions = std::move(positions);
    result.masks = std::move(masks);
    
    LogManager::LogInfo("IUM generation completed: %u pixels processed", 
		launchParams.ium.size.width * launchParams.ium.size.height);
}

void IUM_Generator::setTraversable(TriangleMesh& model)
{
    auto& launchParams = optixManager.getLaunchParams();
    
    // Costruisci la GAS in UV space (come prima)
    launchParams.traversable = buildAccel(model);
    
    // NUOVO: Carica i vertici 3D reali per il mapping
    worldVertexBuffer.alloc_and_upload(model.vertex);
    launchParams.ium.worldVertices = (vec3f*)worldVertexBuffer.d_pointer();
    
    // NUOVO: Carica le coordinate UV
    uvCoordBuffer.alloc_and_upload(model.texcoord);
    launchParams.ium.uvVertices = (vec2f*)uvCoordBuffer.d_pointer();
    
    // NUOVO: Passa gli indici (già caricati in buildAccel)
    launchParams.ium.indices = (vec3i*)indexBuffer.d_pointer();
    launchParams.ium.numTriangles = (uint32_t)model.index.size();
    
    LogManager::LogInfo("IUM geometry data uploaded: %zu vertices, %zu triangles", 
                        model.vertex.size(), model.index.size());
}

void IUM_Generator::setTextureSize(uint32_t width, uint32_t height)
{
	positionsBuffer.resize(width * height * sizeof(vec3f));
	masksBuffer.resize(width * height * sizeof(uint8_t));
	LaunchParams& launchParams = optixManager.getLaunchParams();

    launchParams.ium.size.width = width;
    launchParams.ium.size.height = height;
	launchParams.ium.positions = (vec3f*)positionsBuffer.d_pointer();
	launchParams.ium.masks = (uint8_t*)masksBuffer.d_pointer();

	LogManager::LogDebug("IUM texture size set to %u x %u", width, height);
}

void IUM_Generator::printStatus()
{
    LogManager::LogDebug("IUM_Generator status:");
    LogManager::LogDebug("  Raygen programs: %zu", raygenPGs.size());
    LogManager::LogDebug("  Miss programs: %zu", missPGs.size());
    LogManager::LogDebug("  Hitgroup programs: %zu", hitgroupPGs.size());
    LogManager::LogDebug("  UV vertex buffer size: %zu bytes", uvVertexBuffer.sizeInBytes);
    LogManager::LogDebug("  Index buffer size: %zu bytes", indexBuffer.sizeInBytes);
    LogManager::LogDebug("  Acceleration structure buffer size: %zu bytes", asBuffer.sizeInBytes);
    LogManager::LogDebug("  Positions buffer size: %zu bytes", positionsBuffer.sizeInBytes);
    LogManager::LogDebug("  Masks buffer size: %zu bytes", masksBuffer.sizeInBytes);
}

void IUM_Generator::saveIUMTextureToBitmap(const std::string& filename)
{
    auto& launchParams = optixManager.getLaunchParams();
    const uint32_t width = launchParams.ium.size.width;
    const uint32_t height = launchParams.ium.size.height;
    
    if (width == 0 || height == 0) {
        LogManager::LogError("Cannot save IUM texture: invalid size %u x %u", width, height);
        return;
    }
    
	auto& positions = result.positions;
	auto& masks = result.masks;
    
    
    LogManager::LogInfo("Downloaded %u pixels from GPU", width * height);
    
    // Trova i limiti delle posizioni per normalizzarle
    vec3f minPos(std::numeric_limits<float>::max());
    vec3f maxPos(std::numeric_limits<float>::lowest());
    
    for (uint32_t i = 0; i < width * height; i++) {
        if (masks[i] == 1) {  // Solo pixel validi
            const vec3f& pos = positions[i];
            minPos.x = std::min(minPos.x, pos.x);
            minPos.y = std::min(minPos.y, pos.y);
            minPos.z = std::min(minPos.z, pos.z);
            maxPos.x = std::max(maxPos.x, pos.x);
            maxPos.y = std::max(maxPos.y, pos.y);
            maxPos.z = std::max(maxPos.z, pos.z);
        }
    }
    
    vec3f range = maxPos - minPos;
    LogManager::LogInfo("Position range: min=(%.3f, %.3f, %.3f), max=(%.3f, %.3f, %.3f)",
                        minPos.x, minPos.y, minPos.z, maxPos.x, maxPos.y, maxPos.z);
    
    // Prepara i dati RGB per BMP (24 bit per pixel)
    std::vector<uint8_t> pixels(width * height * 3);
    
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            const uint32_t srcIdx = x + y * width;
            // BMP è bottom-up, ma scriviamo riga per riga quindi non invertiamo qui
            const uint32_t dstIdx = (x + y * width) * 3;
            
            if (masks[srcIdx] == 1) {
                // Normalizza la posizione in [0, 1]
                vec3f normalizedPos;
                if (range.x > 1e-6f) normalizedPos.x = (positions[srcIdx].x - minPos.x) / range.x;
                else normalizedPos.x = 0.5f;
                
                if (range.y > 1e-6f) normalizedPos.y = (positions[srcIdx].y - minPos.y) / range.y;
                else normalizedPos.y = 0.5f;
                
                if (range.z > 1e-6f) normalizedPos.z = (positions[srcIdx].z - minPos.z) / range.z;
                else normalizedPos.z = 0.5f;
                
                // Converti in RGB (0-255)
                pixels[dstIdx + 2] = static_cast<uint8_t>(normalizedPos.x * 255.0f);  // R
                pixels[dstIdx + 1] = static_cast<uint8_t>(normalizedPos.y * 255.0f);  // G
                pixels[dstIdx + 0] = static_cast<uint8_t>(normalizedPos.z * 255.0f);  // B
            } else {
                // Pixel non valido: nero
                pixels[dstIdx + 0] = 0;
                pixels[dstIdx + 1] = 0;
                pixels[dstIdx + 2] = 0;
            }
        }
    }
    
    // Calcola padding per allineamento a 4 byte
    const uint32_t rowSize = ((width * 3 + 3) / 4) * 4;
    const uint32_t paddingSize = rowSize - width * 3;
    
    // Prepara header BMP
    BMPHeader fileHeader;
    fileHeader.fileSize = 54 + rowSize * height;
    
    BMPInfoHeader infoHeader;
    infoHeader.width = width;
    infoHeader.height = height;
    infoHeader.sizeImage = rowSize * height;
    
    // Costruisci il buffer completo: header + infoHeader + pixel data
    std::vector<uint8_t> fileData;
    fileData.reserve(54 + rowSize * height);
    
    // Aggiungi gli header
    const uint8_t* headerPtr = reinterpret_cast<const uint8_t*>(&fileHeader);
    fileData.insert(fileData.end(), headerPtr, headerPtr + sizeof(fileHeader));
    
    const uint8_t* infoHeaderPtr = reinterpret_cast<const uint8_t*>(&infoHeader);
    fileData.insert(fileData.end(), infoHeaderPtr, infoHeaderPtr + sizeof(infoHeader));
    
    // Aggiungi i dati pixel con padding (dal basso verso l'alto per il formato BMP)
    std::vector<uint8_t> padding(paddingSize, 0);
    for (int32_t y = height - 1; y >= 0; y--) {
        const uint8_t* rowPtr = &pixels[y * width * 3];
        fileData.insert(fileData.end(), rowPtr, rowPtr + width * 3);
        if (paddingSize > 0) {
            fileData.insert(fileData.end(), padding.begin(), padding.end());
        }
    }
    
    // Usa IOManager per salvare il file
    IOManager::IOResult result = IOManager::saveBinaryFile(filename, fileData);
    
    if (result.success) {
        LogManager::LogInfo("IUM texture saved to: %s (%u x %u pixels, %zu bytes)", 
                           filename.c_str(), width, height, result.bytesWritten);
    } else {
        LogManager::LogError("Failed to save IUM texture: %s", result.errorMessage.c_str());
    }
}

OptixTraversableHandle IUM_Generator::buildAccel(const TriangleMesh &model)
{
    // 1) Upload indici (uguali)
    indexBuffer.alloc_and_upload(model.index);

    // 2) Costruisci un buffer di "posizioni" in UV space: (u,v,0)
    std::vector<vec3f> uvVerts;
    uvVerts.resize(model.vertex.size());
    for (size_t i = 0; i < uvVerts.size(); ++i) {
        const vec2f uv = model.texcoord[i];          // <-- serve nel model
        uvVerts[i] = vec3f(uv.x, uv.y, 0.f);
    }
    uvVertexBuffer.alloc_and_upload(uvVerts);

    OptixTraversableHandle asHandle{ 0 };

    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr d_vertices = uvVertexBuffer.d_pointer();
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
    (optixManager.getContext(),
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

    OPTIX_CHECK(optixAccelBuild(optixManager.getContext(),
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
    OPTIX_CHECK(optixAccelCompact(optixManager.getContext(),
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

void IUM_Generator::saveIUMTextureToOpenExr(const std::string& filename)
{
    auto& launchParams = optixManager.getLaunchParams();
    const uint32_t width = launchParams.ium.size.width;
    const uint32_t height = launchParams.ium.size.height;
    
    if (width == 0 || height == 0) {
        LogManager::LogError("Cannot save IUM texture: invalid size %u x %u", width, height);
        return;
    }
    
    auto& positions = result.positions;
    auto& masks = result.masks;
    
    LogManager::LogInfo("Downloaded %u pixels from GPU", width * height);
    
    try {
        // Crea un array di pixel RGBA per OpenEXR
        Imf::Array2D<Imf::Rgba> pixels(height, width);
        
        // Riempi l'array di pixel con i valori grezzi
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                const uint32_t idx = x + y * width;
                
                // Scrivi direttamente i valori grezzi delle posizioni
                pixels[y][x].r = positions[idx].x;
                pixels[y][x].g = positions[idx].y;
                pixels[y][x].b = positions[idx].z;
                // Salva la mask nel canale alpha (convertito da uint8_t a float)
                pixels[y][x].a = static_cast<float>(masks[idx]);
            }
        }
        
        // Scrivi il file OpenEXR
        Imf::RgbaOutputFile file(filename.c_str(), width, height, Imf::WRITE_RGBA);
        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.writePixels(height);
        
        LogManager::LogInfo("IUM texture saved to OpenEXR: %s (%u x %u pixels)", 
                           filename.c_str(), width, height);
    }
    catch (const std::exception& e) {
        LogManager::LogError("Failed to save IUM texture to OpenEXR: %s", e.what());
    }
}



