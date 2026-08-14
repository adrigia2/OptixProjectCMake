#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    struct LaunchParams_SpecCone
    {
        // IUM data (mirrors LaunchParams_Indirect.ium_data)
        struct {
            vec3f*   ium_positions;
            vec3f*   ium_normals;
            uint8_t* ium_masks;
            int      num_pixels;
        } ium_data;

        // Current camera: world position + per-texel visibility mask
        // (nullptr -> every texel counts as visible)
        vec3f    cam_pos;
        uint8_t* visibility;     // [num_pixels], 1 = texel seen by the camera

        // Grid of concentric rings around the reflected ray R.
        // ring_cos[i] = cos(half-aperture_i), decreasing, ring_cos[0] = 1
        // (aperture 0 = mirror ray). Ring i (1-based) covers
        // [ring_cos[i], ring_cos[i-1]] in cos(theta). Level 0 = mirror ray.
        float* ring_cos;         // [num_rings + 1]
        int    num_rings;
        // Samples LAUNCHED per ring (not valid_count, which counts only those above
        // the horizon). ring_samples[0] = 1: level 0 is the mirror ray, a degenerate
        // ring of zero width.
        int*   ring_samples;     // [num_rings + 1]

        // Equirectangular HDR skybox for the miss rays
        // (same convention as LaunchParams_Irradiance.skybox)
        struct {
            vec3f* envmap;       // nullptr -> sky contribution = 0
            vec2i  skybox_size;
            float  yaw_offset_u;
        } skybox;

        // Current tile
        int tile_offset;
        int tile_size;

        // Compact output buffer for the rays that hit geometry
        // (their radiances are queried from the NeRF on the Python side)
        vec3f* tile_rays_dir;
        float* tile_rays_t_hit;
        int*   tile_rays_local_idx; // local index within the tile, [0, tile_size)
        int*   tile_rays_ring_idx;  // level/ring the ray belongs to, [0, num_rings]

        unsigned int* tile_counter;
        int           tile_capacity; // tile_size * (1 + sum_i ring_samples[i])

        // Per-texel per-level accumulators (num_levels = num_rings + 1):
        // sky_sum     = envmap sum over the miss rays
        // valid_count = valid samples (above the horizon), hits and misses alike
        vec3f* sky_sum;       // [tile_size * (num_rings + 1)]
        int*   valid_count;   // [tile_size * (num_rings + 1)]

        float epsilon;        // self-intersection offset along the normal

        OptixTraversableHandle traversable;
    };

} // namespace osc
