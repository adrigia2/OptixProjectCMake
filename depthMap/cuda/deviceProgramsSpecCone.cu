#include <optix_device.h>
#include "LaunchParams_SpecCone.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

using namespace osc;

namespace speccone {

extern "C" __constant__ LaunchParams_SpecCone optixLaunchParams;

// Frisvad 2012 branchless ONB (identico a deviceProgramsIrradiance.cu)
static __forceinline__ __device__
void buildONB(const vec3f& n, vec3f& T, vec3f& B)
{
    const float sgn = copysignf(1.0f, n.z);
    const float a   = -1.0f / (sgn + n.z);
    const float b   = n.x * n.y * a;
    T = vec3f(1.0f + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    B = vec3f(b, sgn + n.y * n.y * a, -n.y);
}

// Equirectangular envmap lookup — identico a deviceProgramsIrradiance.cu
// (world space Z-up, Y-forward, Blender native)
static __forceinline__ __device__
vec3f sampleEnvmap(const vec3f& d, const vec3f* env, vec2i sz, float yaw_offset_u)
{
    const float dx = d.x;
    const float dy = d.y;
    const float dz = fmaxf(-1.0f, fminf(1.0f, d.z));

    float u = 0.5f - atan2f(dy, dx) * (1.0f / (2.0f * M_PIf));  // azimuth: Blender equirectangular convention
    u += yaw_offset_u;
    u -= floorf(u);
    const float v = 0.5f - asinf(dz) * (1.0f / M_PIf);

    int px = (int)(u * (float)sz.x);
    int py = (int)(v * (float)sz.y);
    if (px < 0) px = 0; else if (px >= sz.x) px = sz.x - 1;
    if (py < 0) py = 0; else if (py >= sz.y) py = sz.y - 1;
    return env[py * sz.x + px];
}

// Traccia un singolo campione del cono: miss → accumula envmap in sky_sum,
// hit → slot nel buffer compatto per la query NeRF lato Python.
// In entrambi i casi il campione conta in valid_count (la media per livello
// deve includere sia cielo che geometria).
static __forceinline__ __device__
void traceSample(const vec3f& origin, const vec3f& dir, int level,
                 int local_idx, vec3f* skySum, int* validCount)
{
    unsigned int occluded   = 1u;
    unsigned int t_hit_bits = 0u;
    optixTrace(optixLaunchParams.traversable,
               origin,
               dir,
               0.0f,
               1e16f,
               0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0, 1, 0,
               occluded, t_hit_bits);

    if (occluded == 0u) {
        if (optixLaunchParams.skybox.envmap) {
            const vec3f L = sampleEnvmap(dir,
                                         optixLaunchParams.skybox.envmap,
                                         optixLaunchParams.skybox.skybox_size,
                                         optixLaunchParams.skybox.yaw_offset_u);
            skySum[level] = skySum[level] + L;
        }
    } else {
        const unsigned int slot = atomicAdd(optixLaunchParams.tile_counter, 1u);
        if (slot < (unsigned int)optixLaunchParams.tile_capacity) {
            optixLaunchParams.tile_rays_dir[slot]       = dir;
            optixLaunchParams.tile_rays_t_hit[slot]     = __uint_as_float(t_hit_bits);
            optixLaunchParams.tile_rays_local_idx[slot] = local_idx;
            optixLaunchParams.tile_rays_ring_idx[slot]  = level;
        }
    }
    validCount[level] += 1;
}

// ----------------------------------------------------------------------------
// Raygen — un thread per texel del tile. Per la camera corrente costruisce il
// raggio riflesso R = reflect(v, n) e campiona anelli concentrici attorno a R
// (cosθ uniforme per anello = uniforme in angolo solido). Livello 0 = raggio
// specchio puro. I coni L(r_k) si ricostruiscono lato Python per somma
// cumulativa pesata sugli angoli solidi degli anelli.
// ----------------------------------------------------------------------------
extern "C" __global__ void __raygen__specCone()
{
    const int local_idx = optixGetLaunchIndex().x;
    if (local_idx >= optixLaunchParams.tile_size) return;

    const int global_idx = optixLaunchParams.tile_offset + local_idx;
    if (global_idx >= optixLaunchParams.ium_data.num_pixels) return;

    if (optixLaunchParams.ium_data.ium_masks[global_idx] == 0) return;
    if (optixLaunchParams.visibility &&
        optixLaunchParams.visibility[global_idx] == 0) return;

    const vec3f pos = optixLaunchParams.ium_data.ium_positions[global_idx];

    vec3f n = optixLaunchParams.ium_data.ium_normals[global_idx];
    const float nLen = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (nLen < 1e-8f) return;
    n = n * (1.0f / nLen);

    vec3f v = optixLaunchParams.cam_pos - pos;
    const float vLen = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (vLen < 1e-8f) return;
    v = v * (1.0f / vLen);

    const float nv = n.x*v.x + n.y*v.y + n.z*v.z;
    if (nv <= 0.0f) return;   // camera dietro la superficie (backface)

    const vec3f R = n * (2.0f * nv) - v;

    vec3f T, B;
    buildONB(R, T, B);

    const int numLevels = optixLaunchParams.num_rings + 1;
    vec3f* skySum     = optixLaunchParams.sky_sum     + (size_t)local_idx * numLevels;
    int*   validCount = optixLaunchParams.valid_count + (size_t)local_idx * numLevels;

    const float eps         = optixLaunchParams.epsilon;
    const vec3f origin      = pos + n * eps;
    const float goldenAngle = M_PIf * (3.0f - sqrtf(5.0f));

    // Livello 0: raggio specchio puro (anello degenere di apertura 0)
    traceSample(origin, R, 0, local_idx, skySum, validCount);

    const int M = optixLaunchParams.samples_per_ring;
    int sGlobal = 1; // progressivo per decorrelare φ tra anelli

    for (int ring = 1; ring <= optixLaunchParams.num_rings; ++ring) {
        const float cosHi = optixLaunchParams.ring_cos[ring - 1]; // bordo interno
        const float cosLo = optixLaunchParams.ring_cos[ring];     // bordo esterno

        for (int s = 0; s < M; ++s, ++sGlobal) {
            const float u    = ((float)s + 0.5f) / (float)M;
            const float cosT = cosLo + (cosHi - cosLo) * u;
            const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
            const float phi  = (float)sGlobal * goldenAngle;

            const vec3f dir = T * (sinT * cosf(phi))
                            + B * (sinT * sinf(phi))
                            + R * cosT;

            // Scarta i campioni sotto l'orizzonte della superficie
            if (dir.x*n.x + dir.y*n.y + dir.z*n.z <= 0.0f) continue;

            traceSample(origin, dir, ring, local_idx, skySum, validCount);
        }
    }
}

// ----------------------------------------------------------------------------
// Miss — raggio non ha colpito nulla (cielo visibile) → occluded = 0
// ----------------------------------------------------------------------------
extern "C" __global__ void __miss__specCone()
{
    optixSetPayload_0(0u);
}

// ----------------------------------------------------------------------------
// Closest hit — raggio bloccato dalla geometria: occluded = 1, t_hit = tmax
// ----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__specCone()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

} // namespace speccone