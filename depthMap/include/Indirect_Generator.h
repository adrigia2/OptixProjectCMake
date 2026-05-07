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
    // Dati di un singolo tile restituiti da renderTile().
    // I vettori sono già copiati su host; Python li legge come NumPy views
    // tramite il wrapper pybind11.
    struct TileResult
    {
        int              count;          // numero effettivo di raggi occlusi in questo tile
        std::vector<vec3f>  dirs;        // [count] direzioni mondo dei raggi occlusi
        std::vector<float>  cos_weights; // [count] cosθ per ogni raggio
        std::vector<float>  t_hits;      // [count] distanza dall'origine al punto di hit
        std::vector<int>    local_indices; // [count] indice locale nel tile [0, tile_size)
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

    // Carica gli input IUM e alloca i buffer di tile.
    // sampleSide = N → N*N campioni per texel.
    // tileSize   = numero di texel per tile (default 1024).
    void setInputs(const IUM_Generator::Result& ium_result,
                   int sampleSide,
                   int tileSize = 1024);

    int numTiles() const;
    int tileSize() const { return tileSz; }
    int numPixels() const { return numPix; }

    // Lancia OptiX sul tile tileIdx, scarica i risultati e li restituisce.
    TileResult renderTile(int tileIdx);

protected:
    OptixTraversableHandle createGAS(const TriangleMesh& model) override;

    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    CUDABuffer asBuffer;

    // Buffer IUM su device (permanenti per tutta la sessione)
    CUDABuffer iumPositionsBuffer;
    CUDABuffer iumNormalsBuffer;
    CUDABuffer iumMasksBuffer;

    // Buffer di tile (riallocati solo se la dimensione cambia)
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
