#pragma once

#include "LaunchParams_Irradiance.h"
#include <TriangleMesh.h>
#include "OptixActor.h"
#include "IUM_Generator.h"
#include <vector>

namespace osc {

class Irradiance_Generator : public OptixActor
{
public:
    struct Result
    {
        std::vector<vec3f> irradiance; // size = num_pixels
        bool hasIrradiance() const { return !irradiance.empty(); }
    };

    Irradiance_Generator()
    {
        createModule();
        createRaygenPrograms();
        createMissPrograms();
        createHitgroupPrograms();
        createSBT();
        createPipeline();
    }

    void createRaygenPrograms() override;
    void createMissPrograms() override;
    void createHitgroupPrograms() override;
    char* getPtxCode() override;

    void setTraversable(const TriangleMesh& model) override;
    void cleanup() override;

    // Load the inputs. skyboxSize = (width, height) in pixels of the equirectangular envmap.
    // sampleSide = N -> N*N samples are used per texel.
    // skyboxYawDegrees = rotation around the vertical Z axis (up in Blender space).
    //   Default 0 degrees: -Y (Blender's camera forward) at the centre of the envmap.
    //   Adjust it when the HDRI is oriented differently.
    void setInputs(const IUM_Generator::Result& ium_result,
                   const std::vector<vec3f>& skybox,
                   vec2i skyboxSize,
                   int sampleSide,
                   float skyboxYawDegrees = 0.0f);

    void render();

    Result getResult() const { return result; }

protected:
    OptixTraversableHandle createGAS(const TriangleMesh& model) override;

    // World-space GAS (reuses the Visibility_Generator pattern)
    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    CUDABuffer asBuffer;

    // Input buffers
    CUDABuffer iumPositionsBuffer;
    CUDABuffer iumNormalsBuffer;
    CUDABuffer iumMasksBuffer;
    CUDABuffer skyboxBuffer;

    // Output
    CUDABuffer irradianceBuffer;

    int numPixels = 0;
    LaunchParams_Irradiance launchParams;

private:
    Result result;
};

} // namespace osc
