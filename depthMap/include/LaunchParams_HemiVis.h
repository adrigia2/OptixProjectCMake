#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    // Oracolo di visibilità emisferico, camera-agnostico.
    //
    // A differenza di LaunchParams_SpecCone questo pass non conosce né camere né
    // anelli: per ogni texel traccia un set Fibonacci FISSO di num_samples raggi
    // uniformi in angolo solido sull'emisfero sopra la normale, e restituisce solo
    // il t_hit di ciascuno. Direzioni, envmap, query NeRF e classificazione per
    // camera vivono lato Python (torch), dove lo stesso raggio serve tutte le
    // camere che vedono il texel.
    //
    // I raggi specchio R_j = reflect(v_j, n) restano per-camera (il livello 0 è una
    // direzione delta, non condivisibile) e si lanciano in una seconda passata con
    // mode = MODE_MIRROR.
    struct LaunchParams_HemiVis
    {
        enum Mode { MODE_SHARED = 0, MODE_MIRROR = 1 };

        // Dati IUM (mirror di LaunchParams_SpecCone.ium_data)
        struct {
            vec3f*   ium_positions;
            vec3f*   ium_normals;
            uint8_t* ium_masks;
            int      num_pixels;
        } ium_data;

        // Quale delle due passate sta girando (vedi Mode). Le dimensioni di lancio
        // sono (tile_size, num_samples) per MODE_SHARED e (tile_size, num_cams)
        // per MODE_MIRROR.
        int mode;

        // Campioni condivisi per texel (S). La sequenza è deterministica e viene
        // ricostruita identica lato Python: cosθ_s = 1 − (s + 0.5)/S,
        // φ_s = s·goldenAngle + rot(global_idx), nella ONB di Frisvad attorno a n.
        int num_samples;

        // Posizioni mondo delle camere, per i raggi specchio
        vec3f* cam_pos;          // [num_cams]
        int    num_cams;

        // Tile corrente
        int tile_offset;
        int tile_size;

        // Output densi, un valore per raggio, senza compattazione né contatori:
        //   > 0  → hit, distanza dalla superficie (serve alla finestra NeRF)
        //   = 0  → miss (cielo)
        //   < 0  → raggio non lanciato (texel fuori maschera, normale degenere,
        //          oppure camera dietro la superficie in MODE_MIRROR)
        // Ogni thread scrive sempre il proprio slot, quindi non serve azzerarli.
        float* t_hit_shared;     // [tile_size * num_samples]
        float* t_hit_mirror;     // [tile_size * num_cams]

        // Direzioni effettivamente tracciate, stessa indicizzazione dei t_hit.
        // nullptr = disattivato: serve solo al test di parità kernel↔torch
        // (test_hemivis_shared.py), che è l'unico modo di accorgersi se le due
        // formule divergono — il pass di produzione non le trasferisce mai.
        vec3f* dbg_dirs;         // [tile_size * max(num_samples, num_cams)]

        float epsilon;           // offset self-intersection lungo la normale

        OptixTraversableHandle traversable;
    };

} // namespace osc
