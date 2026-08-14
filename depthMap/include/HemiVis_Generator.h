#pragma once

#include "LaunchParams_HemiVis.h"
#include <TriangleMesh.h>
#include "OptixActor.h"
#include "IUM_Generator.h"
#include <vector>

namespace osc {

// Hemispherical visibility oracle, shared between cameras.
//
// For every IUM texel it traces a fixed Fibonacci set of `numSamples` rays uniform
// in solid angle over the hemisphere above the normal and returns only the t_hit of
// each one; the directions are deterministic and are rebuilt on the Python side,
// where the envmap lookup, the NeRF query and the per-camera classification happen.
// One ray (and its NeRF query) therefore serves every camera that sees the texel,
// because the incident radiance along a direction does not depend on the camera.
//
// The mirror rays R_j = reflect(v_j, n) stay per-camera and come from a second
// pass over the same tile (t_hit_mirror).
//
// Alternative to SpecCone_Generator, which samples in rings around R_j and so has to
// be relaunched for every camera; both remain available.
class HemiVis_Generator : public OptixActor
{
public:
    struct TileResult
    {
        int tile_texels;   // texels actually in this tile
        int num_samples;   // S, shared samples per texel
        int num_cams;      // cameras for the mirror rays

        // t_hit per ray, dense (no compaction, no counters):
        //   > 0 -> hit, distance;  = 0 -> miss (sky);  < 0 -> ray not launched
        //   (texel outside the mask, degenerate normal, camera behind the surface)
        std::vector<float> t_hit_shared;   // [tile_texels * num_samples]
        std::vector<float> t_hit_mirror;   // [tile_texels * num_cams]

        // Traced directions, non-empty only with setDebugDirections(true)
        std::vector<vec3f> dirs_shared;    // [tile_texels * num_samples]
        std::vector<vec3f> dirs_mirror;    // [tile_texels * num_cams]
    };

    HemiVis_Generator()
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

    // IUM input + number of shared samples per texel + texels per tile.
    void setInputs(const IUM_Generator::Result& ium_result,
                   int numSamples,
                   int tileSize = 1024);

    // World positions of the cameras for the mirror rays (empty -> no mirror ray,
    // t_hit_mirror stays empty).
    void setCameras(const std::vector<vec3f>& camPositions);

    // Also returns the traced directions. This is what the kernel<->torch parity
    // test needs: the Python-side reconstruction is indexed by position, so a
    // divergence between the two formulas has no symptom other than a wrong L_j.
    // Costs 12 B/ray of device memory and bandwidth: do not use in production.
    void setDebugDirections(bool enabled);

    int numTiles() const;
    int tileSize() const { return tileSz; }
    int numPixels() const { return numPix; }
    int numSamples() const { return numSmp; }
    int numCameras() const { return (int)camPos.size(); }

    // Launch OptiX on the tile (two passes: shared + mirror), download the t_hits.
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
    CUDABuffer camPosBuffer;

    // Tile buffers
    CUDABuffer tHitSharedBuffer;
    CUDABuffer tHitMirrorBuffer;
    CUDABuffer dbgDirsBuffer;

    int numPix = 0;
    int tileSz = 0;
    int numSmp = 0;
    bool debugDirs = false;
    std::vector<vec3f> camPos;

    // zero-init: dbg_dirs/cam_pos/t_hit_mirror stay nullptr until they are needed,
    // and the kernel tests dbg_dirs to decide whether to write the directions
    LaunchParams_HemiVis launchParams{};
};

} // namespace osc
