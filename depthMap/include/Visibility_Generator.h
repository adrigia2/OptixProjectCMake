#pragma once

#include "LaunchParams_Vis.h"
#include <TriangleMesh.h>
#include "OptixActor.h"
#include "IUM_Generator.h"
#include "Camera.h"
#include <vector>

namespace osc {

class Visibility_Generator : public OptixActor
{
public:
    Visibility_Generator()
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

    // Esegue il controllo visibilità e ritorna un array di shape (num_pixels * num_cameras)
    // di bool (espressi come uint8_t 1 o 0).
    std::vector<uint8_t> checkVisibility(const IUM_Generator::Result& ium_result, int width, int height, const std::vector<Camera>& cameras);

protected:
    OptixTraversableHandle createGAS(const TriangleMesh& model) override;

    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    CUDABuffer asBuffer;

    CUDABuffer camBuffer;
    CUDABuffer iumPositionsBuffer;
    CUDABuffer iumMasksBuffer;
    CUDABuffer resultsBuffer;

    LaunchParams_Vis launchParams;
};

} // namespace osc
