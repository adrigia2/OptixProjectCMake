#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"
#include "Camera.h"

namespace osc {
    using namespace gdt;

    struct CameraDef {
        vec3f position;
        vec3f forward;
        vec3f up;
        float fovY;
        vec2i frameSize;
    };

    struct LaunchParams_Vis
    {
        struct {
            // Buffer in cui scriveremo l'output (size = num_pixels * num_cameras)
            // 1 = visibile (nessuna occlusione tra camera e punto)
            // 0 = occluso o invalid
            uint8_t* visibility_results; 
        } results;

        // Dati ricevuti da IUM
        struct {
            vec3f* ium_positions;  // Posizioni 3D per ciascun pixel
            uint8_t* ium_masks;    // 1 = valid texel, 0 = background/invalid
            int num_pixels;        // total pixels (width * height of the texture)
        } ium_data;

        // Dati delle videocamere
        struct {
            CameraDef* cameras;
            int num_cameras;
        } camera_data;

        OptixTraversableHandle traversable;
    };
} // ::osc
