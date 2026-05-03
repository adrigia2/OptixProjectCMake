#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    struct LaunchParams_Irradiance
    {
        struct {
            // Buffer di output [num_pixels] — irradiance HDR per texel
            vec3f* irradiance;
        } results;

        // Dati ricevuti da IUM (esteso con normali)
        struct {
            vec3f*   ium_positions; // posizioni 3D world per texel
            vec3f*   ium_normals;   // normali di faccia per texel
            uint8_t* ium_masks;     // 1 = texel valido
            int      num_pixels;    // width * height della texture
        } ium_data;

        // Skybox HDR equirettangolare
        struct {
            vec3f* envmap;       // [skybox_size.x * skybox_size.y] RGB float
            vec2i  skybox_size;
            float  yaw_offset_u; // = yaw_radians / (2π) — shift su asse U per rotazione yaw
        } skybox;

        int   sample_side;       // N → N×N campioni Monte Carlo per texel
        float epsilon;           // offset self-intersection lungo la normale

        OptixTraversableHandle traversable;
    };

} // namespace osc
