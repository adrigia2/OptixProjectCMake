#pragma once

#include "LaunchParams_SpecCone.h"
#include <TriangleMesh.h>
#include "OptixActor.h"
#include "IUM_Generator.h"
#include <vector>

namespace osc {

// Specular cone pass: for every IUM texel, and for the current camera, it samples
// concentric rings around the reflected ray R = reflect(v, n).
// Miss rays accumulate the envmap on the GPU (sky_sum); hit rays are returned in a
// compact buffer for the NeRF queries on the Python side.
// The mean cone radiances L(r_k) are reconstructed by a cumulative sum weighted by
// the solid angles of the rings (see pbr_solver.py).
class SpecCone_Generator : public OptixActor
{
public:
    struct TileResult
    {
        int count;        // hit rays in the compact buffer (clamped to tile_capacity)
        int tile_texels;  // texels actually in this tile
        int num_levels;   // rings + 1 (level 0 = mirror ray)

        // Compact-buffer overflow: rays past capacity are dropped (without bias,
        // they leave valid_count too) but the tile is incomplete.
        bool overflow;    // true when requested > capacity
        int  requested;   // hit rays requested before the clamp

        // Compact buffer of the hit rays [count]
        std::vector<vec3f> dirs;
        std::vector<float> t_hits;
        std::vector<int>   local_indices; // local index within the tile
        std::vector<int>   ring_indices;  // level the ray belongs to

        // Per-texel per-level accumulators [tile_texels * num_levels]
        std::vector<vec3f> sky_sum;
        std::vector<int>   valid_count;
    };

    SpecCone_Generator()
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

    // Load the IUM inputs and the aperture grid (in degrees, TOTAL cone aperture,
    // increasing, first element = 0 -> mirror ray).
    // samplesPerRing = one value per ring, length coneAperturesDeg.size()-1
    // (rings 1..K-1; level 0, the mirror ray, is always a single sample).
    // tileSize = texels per tile.
    void setInputs(const IUM_Generator::Result& ium_result,
                   const std::vector<float>& coneAperturesDeg,
                   const std::vector<int>& samplesPerRing,
                   int tileSize = 1024);

    // Convenience overload: the same number of samples on every ring.
    void setInputs(const IUM_Generator::Result& ium_result,
                   const std::vector<float>& coneAperturesDeg,
                   int samplesPerRing,
                   int tileSize = 1024);

    // Equirectangular skybox for the miss rays (same convention as
    // Irradiance_Generator). Optional: without it, misses contribute 0.
    void setEnvmap(const std::vector<vec3f>& skybox,
                   vec2i skyboxSize,
                   float skyboxYawDegrees = 0.0f);

    // Current camera: world position + per-texel visibility mask
    // (empty vector -> every texel counts as visible).
    void setCamera(const vec3f& camPos,
                   const std::vector<uint8_t>& visibility);

    int numTiles() const;
    int tileSize() const { return tileSz; }
    int numPixels() const { return numPix; }
    int numLevels() const { return numRings + 1; }

    // Launch OptiX on tile tileIdx, download the results and return them.
    TileResult renderTile(int tileIdx);

protected:
    OptixTraversableHandle createGAS(const TriangleMesh& model) override;

    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    CUDABuffer asBuffer;

    // Persistent inputs
    CUDABuffer iumPositionsBuffer;
    CUDABuffer iumNormalsBuffer;
    CUDABuffer iumMasksBuffer;
    CUDABuffer ringCosBuffer;
    CUDABuffer ringSamplesBuffer;
    CUDABuffer skyboxBuffer;

    // Per-camera inputs
    CUDABuffer visibilityBuffer;

    // Tile buffers
    CUDABuffer tileRaysDirBuffer;
    CUDABuffer tileRaysTHitBuffer;
    CUDABuffer tileRaysLocalIdxBuffer;
    CUDABuffer tileRaysRingIdxBuffer;
    CUDABuffer tileCounterBuffer;
    CUDABuffer skySumBuffer;
    CUDABuffer validCountBuffer;

    int numPix       = 0;
    int tileSz       = 0;
    int numRings     = 0;
    std::vector<int> ringSamples;   // [numRings + 1], [0] = 1 (mirror)
    int tileCapacity = 0;
    bool cameraSet   = false;

    LaunchParams_SpecCone launchParams;
};

} // namespace osc