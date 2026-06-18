#pragma once

#include "gdt/math/vec.h"

namespace osc {
    using namespace gdt;

    // Per-camera data uploaded to GPU.
    // 'right' and 'up_vec' are pre-scaled unit vectors such that:
    //   dot(d, right)  / t  in [-0.5, +0.5]  spans the full horizontal extent
    //   dot(d, up_vec) / t  in [-0.5, +0.5]  spans the full vertical extent
    // Adding 0.5 gives UV in [0, 1] for pixels inside the frame.
    struct ColorCameraDef {
        vec3f  position;
        vec3f  forward;
        vec3f  right;    // normalize(cross(forward, up)) / (2 * halfWidth)
        vec3f  up_vec;   // normalize(cross(right_unit, forward)) / (2 * halfHeight)
        vec2i  frame_size;
        float  peak;
        vec3f* image_ptr; // device pointer to flat image (frame_size.x * frame_size.y)
    };

    struct LaunchParams_ColorTex {
        // IUM input
        vec3f*   ium_positions;
        vec3f*   ium_normals;        // face normals per texel (size = num_pixels)
        uint8_t* ium_masks;
        int      num_pixels;

        // Grazing-angle cull: discard camera contributions where n·v < grazing_min_cos.
        // Set to -1.f (or any value <= -1) to disable the filter entirely.
        float    grazing_min_cos;

        // Visibility: shape [num_pixels * num_cameras]
        uint8_t* visibility;

        // Cameras + images
        ColorCameraDef* cameras;
        int             num_cameras;

        // Output: shape [num_pixels]
        vec3f* color_output;
        vec3f* color_min_output;      // per-texel minimum color across cameras
        vec3f* color_max_output;      // per-texel maximum color across cameras
        vec3f* color_variance_output; // per-texel variance (E[X²]-E[X]²) across cameras
        // Output: per-camera, shape [num_pixels * num_cameras]
        vec3f* camera_color_output;
    };

} // namespace osc
