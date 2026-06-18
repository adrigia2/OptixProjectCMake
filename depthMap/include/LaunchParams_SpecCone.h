#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    struct LaunchParams_SpecCone
    {
        // Dati IUM (mirror di LaunchParams_Indirect.ium_data)
        struct {
            vec3f*   ium_positions;
            vec3f*   ium_normals;
            uint8_t* ium_masks;
            int      num_pixels;
        } ium_data;

        // Camera corrente: posizione mondo + maschera di visibilità per texel
        // (nullptr → tutti i texel considerati visibili)
        vec3f    cam_pos;
        uint8_t* visibility;     // [num_pixels], 1 = texel visto dalla camera

        // Griglia di anelli concentrici attorno al raggio riflesso R.
        // ring_cos[i] = cos(semi-apertura_i), decrescente, ring_cos[0] = 1
        // (apertura 0 = raggio specchio). L'anello i (1-based) copre
        // [ring_cos[i], ring_cos[i-1]] in cosθ. Livello 0 = raggio specchio.
        float* ring_cos;         // [num_rings + 1]
        int    num_rings;
        int    samples_per_ring;

        // Skybox HDR equirettangolare per i raggi miss
        // (stessa convenzione di LaunchParams_Irradiance.skybox)
        struct {
            vec3f* envmap;       // nullptr → contributo cielo = 0
            vec2i  skybox_size;
            float  yaw_offset_u;
        } skybox;

        // Tile corrente
        int tile_offset;
        int tile_size;

        // Buffer compatto di output per i raggi che colpiscono la geometria
        // (le radianze verranno interrogate sul NeRF lato Python)
        vec3f* tile_rays_dir;
        float* tile_rays_t_hit;
        int*   tile_rays_local_idx; // indice locale nel tile [0, tile_size)
        int*   tile_rays_ring_idx;  // livello/anello di appartenenza [0, num_rings]

        unsigned int* tile_counter;
        int           tile_capacity; // tile_size * (1 + num_rings * samples_per_ring)

        // Accumuli per texel per livello (num_levels = num_rings + 1):
        // sky_sum     = somma envmap dei raggi miss
        // valid_count = campioni validi (sopra l'orizzonte), hit + miss
        vec3f* sky_sum;       // [tile_size * (num_rings + 1)]
        int*   valid_count;   // [tile_size * (num_rings + 1)]

        float epsilon;        // offset self-intersection lungo la normale

        OptixTraversableHandle traversable;
    };

} // namespace osc
