#pragma once

#include "OptixActor.h"
#include "IUM_Generator.h"
#include "Frame.h"
#include "LaunchParams_ColorTex.h"
#include <vector>

namespace osc {

class ColorTex_Generator : public OptixActor {
public:
    struct Result {
        std::vector<vec3f> colors;         // size = num_pixels
        std::vector<vec3f> camera_colors;  // size = num_pixels * num_cameras
        int                num_cameras = 0;
    };

    ColorTex_Generator() {
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

    // Not used: ColorTex_Generator does not trace rays against geometry.
    void setTraversable(const TriangleMesh& /*model*/) override {}

    void cleanup() override;

    void setInputs(const IUM_Generator::Result& ium,
                   const std::vector<uint8_t>& visibility,
                   const std::vector<Frame>& frames);

    void render();

    const Result& getResult() const { return result; }

private:
    LaunchParams_ColorTex   launchParams = {};
    CUDABuffer              iumPositionsBuffer;
    CUDABuffer              iumMasksBuffer;
    CUDABuffer              visibilityBuffer;
    CUDABuffer              camerasBuffer;
    std::vector<CUDABuffer> imageBuffers;
    CUDABuffer              colorOutputBuffer;
    CUDABuffer              cameraColorOutputBuffer;
    Result                  result;
};

} // namespace osc
