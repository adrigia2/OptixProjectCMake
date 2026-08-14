#include <optix_device.h>
#include "LaunchParams_Indirect.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

using namespace osc;

namespace indirect {

extern "C" __constant__ LaunchParams_Indirect optixLaunchParams;

// Frisvad 2012 branchless ONB (identical to deviceProgramsIrradiance.cu)
static __forceinline__ __device__
void buildONB(const vec3f& n, vec3f& T, vec3f& B)
{
    const float sgn = copysignf(1.0f, n.z);
    const float a   = -1.0f / (sgn + n.z);
    const float b   = n.x * n.y * a;
    T = vec3f(1.0f + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    B = vec3f(b, sgn + n.y * n.y * a, -n.y);
}

// ----------------------------------------------------------------------------
// Raygen -- one thread per texel of the tile. It writes into the compact buffer
// the rays that hit geometry (occluded == 1). Rays towards the sky
// (occluded == 0) are ignored: the skybox contribution is already handled
// by Irradiance_Generator.
// ----------------------------------------------------------------------------
extern "C" __global__ void __raygen__collectOccluded()
{
    const int local_idx = optixGetLaunchIndex().x;
    if (local_idx >= optixLaunchParams.tile_size) return;

    const int global_idx = optixLaunchParams.tile_offset + local_idx;
    if (global_idx >= optixLaunchParams.ium_data.num_pixels) return;

    if (optixLaunchParams.ium_data.ium_masks[global_idx] == 0) return;

    const vec3f pos = optixLaunchParams.ium_data.ium_positions[global_idx];

    vec3f n = optixLaunchParams.ium_data.ium_normals[global_idx];
    const float nLen = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (nLen < 1e-8f) return;
    n = n * (1.0f / nLen);

    vec3f T, B;
    buildONB(n, T, B);

    const int   N           = optixLaunchParams.sample_side;
    const int   numSamples  = N * N;
    const float invNS       = 1.0f / (float)numSamples;
    const float eps         = optixLaunchParams.epsilon;
    const vec3f origin      = pos + n * eps;
    const float goldenAngle = M_PIf * (3.0f - sqrtf(5.0f));

    for (int i = 0; i < numSamples; ++i) {
        const float z    = 1.0f - ((float)i + 0.5f) * invNS;
        const float r    = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float phi  = (float)i * goldenAngle;
        const vec3f local(r * cosf(phi), r * sinf(phi), z);
        const vec3f worldDir = T * local.x + B * local.y + n * local.z;

        unsigned int occluded = 1u;
        unsigned int t_hit_bits = 0u;
        optixTrace(optixLaunchParams.traversable,
                   origin,
                   worldDir,
                   0.0f,
                   1e16f,
                   0.0f,
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   0, 1, 0,
                   occluded, t_hit_bits);

        if (occluded == 1u) {
            unsigned int slot = atomicAdd(optixLaunchParams.tile_counter, 1u);
            if (slot < (unsigned int)optixLaunchParams.tile_capacity) {
                optixLaunchParams.tile_rays_dir[slot]       = worldDir;
                optixLaunchParams.tile_rays_cos[slot]       = z;
                optixLaunchParams.tile_rays_t_hit[slot]     = __uint_as_float(t_hit_bits);
                optixLaunchParams.tile_rays_local_idx[slot] = local_idx;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Miss -- the ray hit nothing (sky visible) -> occluded = 0
// ----------------------------------------------------------------------------
extern "C" __global__ void __miss__indirect()
{
    optixSetPayload_0(0u);
}

// ----------------------------------------------------------------------------
// Closest hit -- ray blocked by geometry: occluded = 1, t_hit = tmax
// ----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__indirect()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

} // namespace indirect
