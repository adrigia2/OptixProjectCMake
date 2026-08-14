#pragma once

#include "LaunchParams_Indirect.h"
#include <TriangleMesh.h>
#include "OptixActor.h"
#include "IUM_Generator.h"
#include <vector>

namespace osc {

class Indirect_Generator : public OptixActor
{
public:
    // Data for a single tile, as returned by renderTile().
    // The vectors are already copied to the host; Python reads them as NumPy views
    // through the pybind11 wrapper.
    struct TileResult
    {
        int              count;          // occluded rays actually in this tile
        std::vector<vec3f>  dirs;        // [count] world directions of the occluded rays
        std::vector<float>  cos_weights; // [count] cos(theta) for each ray
        std::vector<float>  t_hits;      // [count] distance from the origin to the hit point
        std::vector<int>    local_indices; // [count] local index within the tile, [0, tile_size)
    };

    Indirect_Generator()
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

    // Load the IUM inputs and allocate the tile buffers.
    // sampleSide = N -> N*N samples per texel.
    // tileSize   = texels per tile (default 1024).
    void setInputs(const IUM_Generator::Result& ium_result,
                   int sampleSide,
                   int tileSize = 1024);

    int numTiles() const;
    int tileSize() const { return tileSz; }
    int numPixels() const { return numPix; }

    // Launch OptiX on tile tileIdx, download the results and return them.
    TileResult renderTile(int tileIdx);

protected:
    OptixTraversableHandle createGAS(const TriangleMesh& model) override;

    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    CUDABuffer asBuffer;

    // IUM buffers on the device (persistent for the whole session)
    CUDABuffer iumPositionsBuffer;
    CUDABuffer iumNormalsBuffer;
    CUDABuffer iumMasksBuffer;

    // Tile buffers (reallocated only when the size changes)
    CUDABuffer tileRaysDirBuffer;
    CUDABuffer tileRaysCosBuffer;
    CUDABuffer tileRaysTHitBuffer;
    CUDABuffer tileRaysLocalIdxBuffer;
    CUDABuffer tileCounterBuffer;

    int numPix    = 0;
    int tileSz    = 0;
    int sampleSz  = 0;
    int tileCapacity = 0;

    LaunchParams_Indirect launchParams;
};

} // namespace osc
