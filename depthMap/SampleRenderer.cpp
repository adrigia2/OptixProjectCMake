// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#define TINYOBJLOADER_IMPLEMENTATION
#include "SampleRenderer.h"
#include "TransformReader.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  extern "C" char embedded_ptx_code[];

  /*! SBT record for a raygen program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a miss program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
  };


  //! add aligned cube with front-lower-left corner and size
  void TriangleMesh::addCube(const vec3f &center, const vec3f &size)
  {
    affine3f xfm;
    xfm.p = center - 0.5f*size;
    xfm.l.vx = vec3f(size.x,0.f,0.f);
    xfm.l.vy = vec3f(0.f,size.y,0.f);
    xfm.l.vz = vec3f(0.f,0.f,size.z);
    addUnitCube(xfm);
  }

  void TriangleMesh::addFromObjFile(const std::string& filename)
  {
      tinyobj::attrib_t attrib;
	  std::vector<tinyobj::shape_t> shapes;
	  std::vector<tinyobj::material_t> materials;
	  std::string warn;
      

	  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, filename.c_str());
      if (!warn.empty()) {
          std::cout << "WARN: " << warn << std::endl;
      }
      if (!ret) {
          std::cerr << "Failed to load/parse .obj file: " << filename << std::endl;
          return;
      }
      // Loop over shapes
      for (size_t s = 0; s < shapes.size(); s++) {
          // Loop over faces(polygon)
          size_t index_offset = 0;
          for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
              int fv = shapes[s].mesh.num_face_vertices[f];
              // Only process triangles
              if (fv != 3) {
                  index_offset += fv;
                  continue;
              }
              vec3i face_idx;
              // Loop over vertices in the face.
              for (size_t v = 0; v < fv; v++) {
                  // access to vertex
                  tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                  tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                  tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                  tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                  vertex.push_back(vec3f(vx, vy, vz));
                  face_idx[v] = static_cast<int>(vertex.size() - 1);
              }
              index.push_back(face_idx);
              index_offset += fv;
          }
	  }
  }

  /*! add a unit cube (subject to given xfm matrix) to the current
      triangleMesh */
  void TriangleMesh::addUnitCube(const affine3f &xfm)
  {
    int firstVertexID = (int)vertex.size();
    vertex.push_back(xfmPoint(xfm,vec3f(0.f,0.f,0.f)));
    vertex.push_back(xfmPoint(xfm,vec3f(1.f,0.f,0.f)));
    vertex.push_back(xfmPoint(xfm,vec3f(0.f,1.f,0.f)));
    vertex.push_back(xfmPoint(xfm,vec3f(1.f,1.f,0.f)));
    vertex.push_back(xfmPoint(xfm,vec3f(0.f,0.f,1.f)));
    vertex.push_back(xfmPoint(xfm,vec3f(1.f,0.f,1.f)));
    vertex.push_back(xfmPoint(xfm,vec3f(0.f,1.f,1.f)));
    vertex.push_back(xfmPoint(xfm,vec3f(1.f,1.f,1.f)));


    int indices[] = {0,1,3, 2,0,3,
                     5,7,6, 5,6,4,
                     0,4,5, 0,5,1,
                     2,3,7, 2,7,6,
                     1,5,7, 1,7,3,
                     4,0,2, 4,2,6
                     };
    for (int i=0;i<12;i++)
      index.push_back(firstVertexID+vec3i(indices[3*i+0],
                                          indices[3*i+1],
                                          indices[3*i+2]));
  }


  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer::SampleRenderer(const TriangleMesh &model)
  {
    initOptix();

    std::cout << "#osc: creating optix context ..." << std::endl;
    createContext();

    std::cout << "#osc: setting up module ..." << std::endl;
    createModule();

    std::cout << "#osc: creating raygen programs ..." << std::endl;
    createRaygenPrograms();
    std::cout << "#osc: creating miss programs ..." << std::endl;
    createMissPrograms();
    std::cout << "#osc: creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    launchParams.traversable = buildAccel(model);

    std::cout << "#osc: setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "#osc: building SBT ..." << std::endl;
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;
  }

  OptixTraversableHandle SampleRenderer::buildAccel(const TriangleMesh &model)
  {
    // upload the model to the device: the builder
    vertexBuffer.alloc_and_upload(model.vertex);
    indexBuffer.alloc_and_upload(model.index);

    OptixTraversableHandle asHandle { 0 };

    // ==================================================================
    // triangle inputs
    // ==================================================================
    OptixBuildInput triangleInput = {};
    triangleInput.type
      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_vertices = vertexBuffer.d_pointer();
    CUdeviceptr d_indices  = indexBuffer.d_pointer();

    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput.triangleArray.numVertices         = (int)model.vertex.size();
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;

    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = sizeof(vec3i);
    triangleInput.triangleArray.numIndexTriplets    = (int)model.index.size();
    triangleInput.triangleArray.indexBuffer         = d_indices;

    uint32_t triangleInputFlags[1] = { 0 };

    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput.triangleArray.flags               = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords               = 1;
    triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
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
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accelOptions,
                                &triangleInput,
                                1,
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,

                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,

                                &asHandle,

                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
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

  /*! helper function that initializes optix and checks for errors */
  void SampleRenderer::initOptix()
  {
    std::cout << "#osc: initializing optix..." << std::endl;

    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
  void SampleRenderer::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS )
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }



  /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
  void SampleRenderer::createModule()
  {
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth          = 2;

    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof( log );
#if OPTIX_VERSION >= 70700
    OPTIX_CHECK(optixModuleCreate(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,&sizeof_log,
                                         &module
                                         ));
#else
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,      // Log string
                                         &sizeof_log,// Log string sizse
                                         &module
                                         ));
#endif
    if (sizeof_log > 1) PRINT(log);
  }



  /*! does all setup for the raygen program(s) we are going to use */
  void SampleRenderer::createRaygenPrograms()
  {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &raygenPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }

  /*! does all setup for the miss program(s) we are going to use */
  void SampleRenderer::createMissPrograms()
  {
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }

  /*! does all setup for the hitgroup program(s) we are going to use */
  void SampleRenderer::createHitgroupPrograms()
  {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH            = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }


  /*! assembles the full pipeline of all programs */
  void SampleRenderer::createPipeline()
  {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));
    if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
                (/* [in] The pipeline to configure the stack size for */
                 pipeline,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 2*1024,
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 1));
    if (sizeof_log > 1) PRINT(log);
  }


  /*! constructs the shader binding table */
  void SampleRenderer::buildSBT()
  {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i=0;i<raygenPGs.size();i++) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i=0;i<missPGs.size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    int numObjects = 1;
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i=0;i<numObjects;i++) {
      int objectType = 0;
      HitgroupRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType],&rec));
      rec.objectID = i;
      hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }



  /*! render one frame */
  void SampleRenderer::render()
  {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.frame.size.x == 0) return;

    launchParamsBuffer.upload(&launchParams,1);

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline,stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.frame.size.x,
                            launchParams.frame.size.y,
                            1
                            ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
  }

  /*! set camera to render with */
  void SampleRenderer::setCamera(const Camera &camera, float fovY_radians)
  {
    lastSetCamera = camera;
    launchParams.camera.position  = camera.pos;
    launchParams.camera.direction = normalize(camera.forward);
    
    const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
    
    // Convert FOV -> image plane half-size (pinhole model)
    const float halfHeight = tanf(0.5f * fovY_radians);
    const float halfWidth = halfHeight * aspect;

    // Build a robust orthonormal basis (right, up, forward)
    // (If your handedness is flipped, swap cross order as noted below.)
    auto right = cross(launchParams.camera.direction, camera.up);
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
  }

  /*! resize frame buffer to given resolution */
  void SampleRenderer::resize(const vec2i &newSize)
  {
    // if window minimized
    if (newSize.x == 0 | newSize.y == 0) return;

    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));
    depthBuffer.resize(newSize.x*newSize.y*sizeof(float));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size  = newSize;
    launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
    launchParams.frame.depthBuffer = (float*)depthBuffer.d_pointer();

    // and re-set the camera, since aspect may have changed
	setCamera(lastSetCamera, 0.66f); // 0.66 radians ~= 37.8 degrees
  }

  /*! download the rendered color buffer */
  void SampleRenderer::downloadPixels(uint32_t h_pixels[])
  {
    colorBuffer.download(h_pixels,
                         launchParams.frame.size.x*launchParams.frame.size.y);
  }

  /*! download the rendered depth buffer */
  void SampleRenderer::downloadDepthMap(float h_depths[])
  {
    depthBuffer.download(h_depths,
                       launchParams.frame.size.x*launchParams.frame.size.y);
  }

  /*! generate depth maps for all cameras in transforms.json */
  void SampleRenderer::generateDepthMapsFromTransform(const std::string& transformFile, 
                                                   const std::string& outputDir)
  {
    TransformData transforms;
    if (!transforms.loadFromFile(transformFile)) {
      std::cerr << "Errore nel caricamento del file transforms.json" << std::endl;
      return;
    }
    
    // Ridimensiona i buffer per la risoluzione specificata
    // resize() allocherà automaticamente sia colorBuffer che depthBuffer
    resize(vec2i(transforms.w, transforms.h));
    
    std::vector<float> depthData(transforms.w * transforms.h);
    
    std::cout << "Inizio generazione depth maps..." << std::endl;
    
    for (size_t i = 0; i < transforms.frames.size(); i++) {
      const auto& frame = transforms.frames[i];
	  const auto filename = frame.getFileName(osc::IMAGE_TYPE_RGB, false);
      
      // Crea la camera dal transform
      Camera camera;
      camera.pos = frame.getPosition();
      camera.forward = frame.getForward();
      camera.up = frame.getUp();
      
      std::cout << "Rendering depth map " << (i+1) << "/" << transforms.frames.size() 
                << " - Camera position: " << camera.pos << std::endl;
      
      // Imposta la camera e renderizza
	  setCamera(camera, transforms.camera_angle_y);
      render();
      
      // Scarica i dati depth
      downloadDepthMap(depthData.data());
      
      // Salva su file binario
      std::string outputFile = outputDir + "\\depth_" + filename + ".bin";
      transforms.frames[i].setDepthPath(outputFile); // Aggiorna il percorso del file depth nel frame

      std::ofstream outFile(outputFile, std::ios::binary);
      if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(depthData.data()), 
                      depthData.size() * sizeof(float));
        outFile.close();
        std::cout << "  Salvata: " << outputFile << std::endl;
      } else {
        std::cerr << "  ERRORE: Impossibile salvare " << outputFile << std::endl;
      }
      
      // Opzionale: salva anche un file di testo con metadati
      if (i == 0) {
        std::string metaFile = outputDir + "/depth_metadata.txt";
        std::ofstream metaOut(metaFile);
        if (metaOut.is_open()) {
          metaOut << "Width: " << transforms.w << std::endl;
          metaOut << "Height: " << transforms.h << std::endl;
          metaOut << "NumFrames: " << transforms.frames.size() << std::endl;
          metaOut << "CameraAngleX: " << transforms.camera_angle_x << std::endl;
          metaOut.close();
        }
      }
    }
    
    std::cout << std::endl;
    std::cout << GDT_TERMINAL_GREEN 
              << "Generazione depth maps completata! Totale: " << transforms.frames.size() 
              << GDT_TERMINAL_DEFAULT << std::endl;
    std::cout << "File salvati in: " << outputDir << std::endl;

    // Salva transformDepth.json nella stessa directory delle depth map
    std::string transformDepthFile = outputDir + "\\transformDepth.json";
    std::cout << "\nCreazione file transformDepth.json..." << std::endl;

    if (transforms.saveToFileWithDepth(transformDepthFile, "depth")) {
        std::cout << GDT_TERMINAL_GREEN 
                  << "File transformDepth.json creato con successo!" 
                  << GDT_TERMINAL_DEFAULT << std::endl;
    } else {
        std::cerr << GDT_TERMINAL_RED 
                  << "ERRORE: Impossibile creare transformDepth.json" 
                  << GDT_TERMINAL_DEFAULT << std::endl;
    }
  }

} // ::osc
