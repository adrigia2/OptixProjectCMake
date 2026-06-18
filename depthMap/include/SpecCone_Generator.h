#pragma once

#include "LaunchParams_SpecCone.h"
#include <TriangleMesh.h>
#include "OptixActor.h"
#include "IUM_Generator.h"
#include <vector>

namespace osc {

// Pass speculare a cono: per ogni texel IUM e per la camera corrente campiona
// anelli concentrici attorno al raggio riflesso R = reflect(v, n).
// I raggi miss accumulano l'envmap su GPU (sky_sum); i raggi hit vengono
// restituiti in un buffer compatto per le query NeRF lato Python.
// Le radianze medie dei coni L(r_k) si ricostruiscono per somma cumulativa
// pesata sugli angoli solidi degli anelli (vedi pbr_solver.py).
class SpecCone_Generator : public OptixActor
{
public:
    struct TileResult
    {
        int count;        // raggi hit nel buffer compatto
        int tile_texels;  // texel effettivi di questo tile
        int num_levels;   // anelli + 1 (livello 0 = raggio specchio)

        // Buffer compatto dei raggi hit [count]
        std::vector<vec3f> dirs;
        std::vector<float> t_hits;
        std::vector<int>   local_indices; // indice locale nel tile
        std::vector<int>   ring_indices;  // livello di appartenenza

        // Accumuli per texel per livello [tile_texels * num_levels]
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

    // Carica gli input IUM e la griglia di aperture (in gradi, apertura TOTALE
    // del cono, crescente, primo elemento = 0 → raggio specchio).
    // samplesPerRing = campioni per anello; tileSize = texel per tile.
    void setInputs(const IUM_Generator::Result& ium_result,
                   const std::vector<float>& coneAperturesDeg,
                   int samplesPerRing,
                   int tileSize = 1024);

    // Skybox equirettangolare per i raggi miss (stessa convenzione di
    // Irradiance_Generator). Facoltativa: senza, i miss contribuiscono 0.
    void setEnvmap(const std::vector<vec3f>& skybox,
                   vec2i skyboxSize,
                   float skyboxYawDegrees = 0.0f);

    // Camera corrente: posizione mondo + maschera di visibilità per texel
    // (vettore vuoto → tutti i texel considerati visibili).
    void setCamera(const vec3f& camPos,
                   const std::vector<uint8_t>& visibility);

    int numTiles() const;
    int tileSize() const { return tileSz; }
    int numPixels() const { return numPix; }
    int numLevels() const { return numRings + 1; }

    // Lancia OptiX sul tile tileIdx, scarica i risultati e li restituisce.
    TileResult renderTile(int tileIdx);

protected:
    OptixTraversableHandle createGAS(const TriangleMesh& model) override;

    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    CUDABuffer asBuffer;

    // Input permanenti
    CUDABuffer iumPositionsBuffer;
    CUDABuffer iumNormalsBuffer;
    CUDABuffer iumMasksBuffer;
    CUDABuffer ringCosBuffer;
    CUDABuffer skyboxBuffer;

    // Input per camera
    CUDABuffer visibilityBuffer;

    // Buffer di tile
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
    int samplesRing  = 0;
    int tileCapacity = 0;
    bool cameraSet   = false;

    LaunchParams_SpecCone launchParams;
};

} // namespace osc