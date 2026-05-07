#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    struct LaunchParams_Indirect
    {
        // Dati IUM (mirror di LaunchParams_Irradiance.ium_data)
        struct {
            vec3f*   ium_positions;
            vec3f*   ium_normals;
            uint8_t* ium_masks;
            int      num_pixels;
        } ium_data;

        // Tile corrente
        int tile_offset;   // indice globale del primo texel del tile
        int tile_size;     // numero di texel nel tile (<= ium_data.num_pixels - tile_offset)

        // Buffer compatto di output per i raggi occlusi (dimensione = tile_capacity)
        vec3f* tile_rays_dir;
        float* tile_rays_cos;
        float* tile_rays_t_hit;     // distanza dal punto di emissione all'intersezione
        int*   tile_rays_local_idx; // indice locale nel tile [0, tile_size)

        unsigned int* tile_counter;  // atomic counter su device
        int           tile_capacity; // tile_size * sample_side * sample_side

        int   sample_side; // N → N*N campioni per texel
        float epsilon;     // offset self-intersection lungo la normale

        OptixTraversableHandle traversable;
    };

} // namespace osc
