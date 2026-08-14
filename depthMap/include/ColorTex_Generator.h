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
        std::vector<vec3f> colors;          // size = num_pixels
        std::vector<vec3f> color_min;       // size = num_pixels
        std::vector<vec3f> color_max;       // size = num_pixels
        std::vector<vec3f> color_variance;  // size = num_pixels, per-channel variance across cameras
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

    // grazingMaxDeg: angle (degrees) from the surface normal beyond which a camera
    // contribution is discarded. 90.f (default) disables the filter entirely.
    void setInputs(const IUM_Generator::Result& ium,
                   const std::vector<uint8_t>& visibility,
                   const std::vector<Frame>& frames,
                   float grazingMaxDeg = 90.f);

    void render();

    // Per-camera colors are not mirrored on the host (num_pixels x num_cameras
    // would not scale); the GPU buffer is camera-major, so each camera slice is
    // contiguous and downloaded on demand. Valid after render(), until cleanup().
    void downloadCameraColors(int cam, vec3f* dst) const;

    // Per-camera mask (uint8, 1 = the camera contributes), in the same camera-major
    // layout as the colours. Written before the peak filter, hence source-independent
    // and reusable as shared visibility. Valid after render(), until cleanup().
    void downloadCameraMask(int cam, uint8_t* dst) const;

    int numPixels()  const { return launchParams.num_pixels; }
    int numCameras() const { return launchParams.num_cameras; }

    const Result& getResult() const { return result; }

private:
    LaunchParams_ColorTex   launchParams = {};
    CUDABuffer              iumPositionsBuffer;
    CUDABuffer              iumNormalsBuffer;
    CUDABuffer              iumMasksBuffer;
    CUDABuffer              visibilityBuffer;
    CUDABuffer              camerasBuffer;
    std::vector<CUDABuffer> imageBuffers;
    CUDABuffer              colorOutputBuffer;
    CUDABuffer              colorMinOutputBuffer;
    CUDABuffer              colorMaxOutputBuffer;
    CUDABuffer              colorVarianceOutputBuffer;
    CUDABuffer              cameraColorOutputBuffer;
    CUDABuffer              cameraMaskOutputBuffer;
    Result                  result;
};

} // namespace osc
