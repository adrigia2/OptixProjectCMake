#pragma once

#include "LaunchParams_HemiVis.h"
#include <TriangleMesh.h>
#include "OptixActor.h"
#include "IUM_Generator.h"
#include <vector>

namespace osc {

// Oracolo di visibilità emisferico condiviso tra camere.
//
// Per ogni texel IUM traccia un set Fibonacci fisso di `numSamples` raggi uniformi
// in angolo solido sull'emisfero sopra la normale e restituisce solo il t_hit di
// ciascuno; le direzioni sono deterministiche e vengono ricostruite lato Python,
// dove avvengono lookup envmap, query NeRF e classificazione per camera. Lo stesso
// raggio (e la sua query NeRF) serve quindi tutte le camere che vedono il texel,
// perché la radianza incidente lungo una direzione non dipende dalla camera.
//
// I raggi specchio R_j = reflect(v_j, n) restano per-camera e si ottengono con una
// seconda passata sullo stesso tile (t_hit_mirror).
//
// Alternativa a SpecCone_Generator, che campiona ad anelli attorno a R_j e va
// quindi rilanciato per ogni camera; entrambi restano disponibili.
class HemiVis_Generator : public OptixActor
{
public:
    struct TileResult
    {
        int tile_texels;   // texel effettivi di questo tile
        int num_samples;   // S, campioni condivisi per texel
        int num_cams;      // camere per i raggi specchio

        // t_hit per raggio, denso (nessuna compattazione, nessun contatore):
        //   > 0 → hit, distanza;  = 0 → miss (cielo);  < 0 → raggio non lanciato
        //   (texel fuori maschera, normale degenere, camera dietro la superficie)
        std::vector<float> t_hit_shared;   // [tile_texels * num_samples]
        std::vector<float> t_hit_mirror;   // [tile_texels * num_cams]

        // Direzioni tracciate, non vuote solo con setDebugDirections(true)
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

    // Input IUM + numero di campioni condivisi per texel + texel per tile.
    void setInputs(const IUM_Generator::Result& ium_result,
                   int numSamples,
                   int tileSize = 1024);

    // Posizioni mondo delle camere per i raggi specchio (vuoto → nessun raggio
    // specchio, t_hit_mirror resta vuoto).
    void setCameras(const std::vector<vec3f>& camPositions);

    // Restituisce anche le direzioni tracciate. Serve al test di parità
    // kernel↔torch: la ricostruzione lato Python è indicizzata per posizione,
    // quindi una divergenza delle formule non ha altri sintomi che una L_j
    // sbagliata. Costa 12 B/raggio di device e di banda: non usare in produzione.
    void setDebugDirections(bool enabled);

    int numTiles() const;
    int tileSize() const { return tileSz; }
    int numPixels() const { return numPix; }
    int numSamples() const { return numSmp; }
    int numCameras() const { return (int)camPos.size(); }

    // Lancia OptiX sul tile (due passate: condivisa + specchio), scarica i t_hit.
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
    CUDABuffer camPosBuffer;

    // Buffer di tile
    CUDABuffer tHitSharedBuffer;
    CUDABuffer tHitMirrorBuffer;
    CUDABuffer dbgDirsBuffer;

    int numPix = 0;
    int tileSz = 0;
    int numSmp = 0;
    bool debugDirs = false;
    std::vector<vec3f> camPos;

    // zero-init: dbg_dirs/cam_pos/t_hit_mirror restano nullptr finché non servono,
    // e il kernel testa dbg_dirs per decidere se scrivere le direzioni
    LaunchParams_HemiVis launchParams{};
};

} // namespace osc
