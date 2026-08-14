#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    struct LaunchParams_Irradiance
    {
        struct {
            // Output buffer [num_pixels] -- HDR irradiance per texel
            vec3f* irradiance;
        } results;

        // Data coming from IUM (extended with the normals)
        struct {
            vec3f*   ium_positions; // world 3D position per texel
            vec3f*   ium_normals;   // face normal per texel
            uint8_t* ium_masks;     // 1 = valid texel
            int      num_pixels;    // width * height of the texture
        } ium_data;

        // Equirectangular HDR skybox
        struct {
            vec3f* envmap;       // [skybox_size.x * skybox_size.y] RGB float
            vec2i  skybox_size;
            float  yaw_offset_u; // = yaw_radians / (2*pi) -- U-axis shift for the yaw
        } skybox;

        int   sample_side;       // N -> N*N directions per texel, on a deterministic
                                 // Fibonacci spiral (this is not random sampling)
        float epsilon;           // self-intersection offset along the normal

        OptixTraversableHandle traversable;
    };

} // namespace osc
