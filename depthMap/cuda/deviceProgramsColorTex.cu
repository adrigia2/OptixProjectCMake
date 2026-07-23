#include <optix_device.h>
#include <float.h>
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
        optixLaunchParams.color_output[idx]          = vec3f(0.f, 0.f, 0.f);
        optixLaunchParams.color_min_output[idx]      = vec3f(0.f, 0.f, 0.f);
        optixLaunchParams.color_max_output[idx]      = vec3f(0.f, 0.f, 0.f);
        optixLaunchParams.color_variance_output[idx] = vec3f(0.f, 0.f, 0.f);
        const int num_cameras_early = optixLaunchParams.num_cameras;
        for (int k = 0; k < num_cameras_early; ++k) {
            optixLaunchParams.camera_color_output[size_t(k) * optixLaunchParams.num_pixels + idx] = vec3f(0.f, 0.f, 0.f);
            optixLaunchParams.camera_mask_output[size_t(k) * optixLaunchParams.num_pixels + idx] = 0;
        }
        return;
    }

    const vec3f pos         = optixLaunchParams.ium_positions[idx];
    const int   num_cameras = optixLaunchParams.num_cameras;

    vec3f sum    = vec3f(0.f, 0.f, 0.f);
    vec3f sum_sq = vec3f(0.f, 0.f, 0.f);
    int   count = 0;
    vec3f local_min = {FLT_MAX, FLT_MAX, FLT_MAX};
    vec3f local_max = {0.f, 0.f, 0.f};

    for (int k = 0; k < num_cameras; ++k) {
        vec3f   cam_color = vec3f(0.f, 0.f, 0.f);
        uint8_t cam_valid = 0;   // 1 = la camera contribuisce (pre-peak)

        // Skip occluded cameras
        if (optixLaunchParams.visibility[idx * num_cameras + k] != 0) {
            const ColorCameraDef& cam = optixLaunchParams.cameras[k];

            // Perspective projection
            const vec3f d = pos - cam.position;

            // Grazing-angle cull: se la camera vede il texel troppo di taglio
            // (n·v < soglia) viene trattata come se non vedesse il punto.
            // La direzione texel->camera è -d, quindi n·v = -dot(n,d)/|d|.
            // Per evitare la divisione confrontiamo: -dot(n,d) < grazing_min_cos * |d|
            if (optixLaunchParams.grazing_min_cos > -1.f) {
                const vec3f n = optixLaunchParams.ium_normals[idx];
                if (-dot(n, d) < optixLaunchParams.grazing_min_cos * length(d)) {
                    optixLaunchParams.camera_color_output[size_t(k) * optixLaunchParams.num_pixels + idx] = vec3f(0.f, 0.f, 0.f);
                    optixLaunchParams.camera_mask_output[size_t(k) * optixLaunchParams.num_pixels + idx] = 0;
                    continue;
                }
            }

            const float t = dot(d, cam.forward);

            if (t > 0.f) {
                // UV in [0, 1] when inside the frustum
                const float uv_x = dot(d, cam.right)  / t + 0.5f;
                const float uv_y = 0.5f - dot(d, cam.up_vec) / t;

                if (uv_x >= 0.f && uv_x < 1.f && uv_y >= 0.f && uv_y < 1.f) {
                    const int px = (int)(uv_x * cam.frame_size.x);
                    const int py = (int)(uv_y * cam.frame_size.y);

                    if (px >= 0 && px < cam.frame_size.x && py >= 0 && py < cam.frame_size.y) {
                        // Il texel proietta in un pixel valido, non occluso, non di
                        // taglio: la camera lo "vede" (indipendente dal peak, quindi
                        // source-indipendente → usabile come visibility condivisa).
                        cam_valid = 1;

                        const vec3f color = cam.image_ptr[py * cam.frame_size.x + px];

                        // Discard overexposed pixels
                        const float max_val = fmaxf(color.x, fmaxf(color.y, color.z));
                        if (max_val < cam.peak) {
                            cam_color = color;
                            sum.x += color.x;
                            sum.y += color.y;
                            sum.z += color.z;
                            sum_sq.x += color.x * color.x;
                            sum_sq.y += color.y * color.y;
                            sum_sq.z += color.z * color.z;
                            local_min.x = fminf(local_min.x, color.x);
                            local_min.y = fminf(local_min.y, color.y);
                            local_min.z = fminf(local_min.z, color.z);
                            local_max.x = fmaxf(local_max.x, color.x);
                            local_max.y = fmaxf(local_max.y, color.y);
                            local_max.z = fmaxf(local_max.z, color.z);
                            ++count;
                        }
                    }
                }
            }
        }

        optixLaunchParams.camera_color_output[size_t(k) * optixLaunchParams.num_pixels + idx] = cam_color;
        optixLaunchParams.camera_mask_output[size_t(k) * optixLaunchParams.num_pixels + idx] = cam_valid;
    }

    if (count > 0) {
        const float inv  = 1.0f / float(count);
        const vec3f mean = vec3f(sum.x * inv, sum.y * inv, sum.z * inv);
        optixLaunchParams.color_output[idx]          = mean;
        optixLaunchParams.color_min_output[idx]      = local_min;
        optixLaunchParams.color_max_output[idx]      = local_max;
        optixLaunchParams.color_variance_output[idx] = vec3f(
            sum_sq.x * inv - mean.x * mean.x,
            sum_sq.y * inv - mean.y * mean.y,
            sum_sq.z * inv - mean.z * mean.z
        );
    } else {
        optixLaunchParams.color_output[idx]          = vec3f(0.f, 0.f, 0.f);
        optixLaunchParams.color_min_output[idx]      = vec3f(0.f, 0.f, 0.f);
        optixLaunchParams.color_max_output[idx]      = vec3f(0.f, 0.f, 0.f);
        optixLaunchParams.color_variance_output[idx] = vec3f(0.f, 0.f, 0.f);
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
