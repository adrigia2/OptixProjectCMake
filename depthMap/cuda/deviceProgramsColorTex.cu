#include <optix_device.h>
#include "LaunchParams_ColorTex.h"

using namespace osc;

namespace colortex {

extern "C" __constant__ LaunchParams_ColorTex optixLaunchParams;

//------------------------------------------------------------------------------
// Raygen — no ray tracing: pure projection + color sampling
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__colorTex()
{
    const int idx = (int)optixGetLaunchIndex().x;

    if (idx >= optixLaunchParams.num_pixels)
        return;

    // Skip IUM pixels not on the mesh
    if (optixLaunchParams.ium_masks[idx] == 0) {
        optixLaunchParams.color_output[idx] = vec3f(0.f, 0.f, 0.f);
        return;
    }

    const vec3f pos         = optixLaunchParams.ium_positions[idx];
    const int   num_cameras = optixLaunchParams.num_cameras;

    vec3f sum  = vec3f(0.f, 0.f, 0.f);
    int   count = 0;

    for (int k = 0; k < num_cameras; ++k) {
        // Skip occluded cameras
        if (optixLaunchParams.visibility[idx * num_cameras + k] == 0)
            continue;

        const ColorCameraDef& cam = optixLaunchParams.cameras[k];

        // Perspective projection
        const vec3f d = pos - cam.position;
        const float t = dot(d, cam.forward);
        if (t <= 0.f) continue;

        // UV in [0, 1] when inside the frustum
        const float uv_x = dot(d, cam.right)  / t + 0.5f;
        const float uv_y = dot(d, cam.up_vec) / t + 0.5f;

        if (uv_x < 0.f || uv_x >= 1.f || uv_y < 0.f || uv_y >= 1.f) continue;

        const int px = (int)(uv_x * cam.frame_size.x);
        const int py = (int)(uv_y * cam.frame_size.y);

        if (px < 0 || px >= cam.frame_size.x || py < 0 || py >= cam.frame_size.y) continue;

        const vec3f color = cam.image_ptr[py * cam.frame_size.x + px];

        // Discard overexposed pixels
        const float max_val = fmaxf(color.x, fmaxf(color.y, color.z));
        if (max_val >= cam.peak) continue;

        sum.x += color.x;
        sum.y += color.y;
        sum.z += color.z;
        ++count;
    }

    if (count > 0) {
        const float inv = 1.0f / float(count);
        optixLaunchParams.color_output[idx] = vec3f(sum.x * inv, sum.y * inv, sum.z * inv);
    } else {
        optixLaunchParams.color_output[idx] = vec3f(0.f, 0.f, 0.f);
    }
}

//------------------------------------------------------------------------------
// Stub miss — never called, satisfies SBT requirements
//------------------------------------------------------------------------------
extern "C" __global__ void __miss__colorTex() {}

//------------------------------------------------------------------------------
// Stub closest hit — never called, satisfies SBT requirements
//------------------------------------------------------------------------------
extern "C" __global__ void __closesthit__colorTex() {}

} // namespace colortex
