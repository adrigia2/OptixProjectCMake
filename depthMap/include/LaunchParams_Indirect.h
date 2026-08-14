#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    struct LaunchParams_Indirect
    {
        // IUM data (mirrors LaunchParams_Irradiance.ium_data)
        struct {
            vec3f*   ium_positions;
            vec3f*   ium_normals;
            uint8_t* ium_masks;
            int      num_pixels;
        } ium_data;

        // Current tile
        int tile_offset;   // global index of the tile's first texel
        int tile_size;     // texels in the tile (<= ium_data.num_pixels - tile_offset)

        // Compact output buffer for the occluded rays (size = tile_capacity)
        vec3f* tile_rays_dir;
        float* tile_rays_cos;
        float* tile_rays_t_hit;     // distance from the emission point to the intersection
        int*   tile_rays_local_idx; // local index within the tile, [0, tile_size)

        unsigned int* tile_counter;  // atomic counter, on the device
        int           tile_capacity; // tile_size * sample_side * sample_side

        int   sample_side; // N -> N*N samples per texel
        float epsilon;     // self-intersection offset along the normal

        OptixTraversableHandle traversable;
    };

} // namespace osc
