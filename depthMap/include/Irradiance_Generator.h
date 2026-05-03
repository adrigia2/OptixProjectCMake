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

    // Carica gli input. skyboxSize = (width, height) in pixel della envmap equirettangolare.
    // sampleSide = N → vengono usati N*N campioni per texel.
    void setInputs(const IUM_Generator::Result& ium_result,
                   const std::vector<vec3f>& skybox,
                   vec2i skyboxSize,
                   int sampleSide);

    void render();

    Result getResult() const { return result; }

protected:
    OptixTraversableHandle createGAS(const TriangleMesh& model) override;

    // GAS world-space (riusa pattern di Visibility_Generator)
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
