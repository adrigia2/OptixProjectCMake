#include <optix_device.h>
#include "LaunchParams_Irradiance.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

using namespace osc;

namespace irr {

extern "C" __constant__ LaunchParams_Irradiance optixLaunchParams;

// ----------------------------------------------------------------------------
// Frame ortonormale tangent-space dato il normale (Frisvad 2012, branchless ONB)
// ----------------------------------------------------------------------------
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
// Equirectangular envmap lookup (assume world Y-up — convenzione tipica HDR Blender)
// ----------------------------------------------------------------------------
static __forceinline__ __device__
vec3f sampleEnvmap(const vec3f& d, const vec3f* env, vec2i sz, float yaw_offset_u)
{
    const float dx = d.x;
    const float dy = fmaxf(-1.0f, fminf(1.0f, d.y));
    const float dz = d.z;

    float u = 0.5f + atan2f(dx, -dz) * (1.0f / (2.0f * M_PIf));
    u += yaw_offset_u;
    u -= floorf(u);                       // wrap a [0, 1)
    const float v = 0.5f - asinf(dy) * (1.0f / M_PIf);

    int px = (int)(u * (float)sz.x);
    int py = (int)(v * (float)sz.y);
    if (px < 0) px = 0; else if (px >= sz.x) px = sz.x - 1;
    if (py < 0) py = 0; else if (py >= sz.y) py = sz.y - 1;
    return env[py * sz.x + px];
}

// ----------------------------------------------------------------------------
// Raygen — un thread per texel; loop interno di N×N campioni sull'emisfero
// ----------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderIrradiance()
{
    const int idx = optixGetLaunchIndex().x;
    if (idx >= optixLaunchParams.ium_data.num_pixels) return;

    if (optixLaunchParams.ium_data.ium_masks[idx] == 0) {
        optixLaunchParams.results.irradiance[idx] = vec3f(0.f);
        return;
    }

    const vec3f pos = optixLaunchParams.ium_data.ium_positions[idx];

    vec3f n = optixLaunchParams.ium_data.ium_normals[idx];
    const float nLen = length(n);
    if (nLen < 1e-8f) {
        // Normale degenere — non possiamo costruire un emisfero
        optixLaunchParams.results.irradiance[idx] = vec3f(0.f);
        return;
    }
    n = n * (1.0f / nLen);

    vec3f T, B;
    buildONB(n, T, B);

    const int   N     = optixLaunchParams.sample_side;
    const float invN  = 1.0f / (float)N;
    const float eps   = optixLaunchParams.epsilon;
    const vec3f origin = pos + n * eps;

    vec3f accum(0.f);

    for (int i = 0; i < N; ++i) {
        const float u = ((float)i + 0.5f) * invN;       // = cosθ ∈ (0,1)
        const float z = u;
        const float sinT = sqrtf(fmaxf(0.0f, 1.0f - u * u));

        for (int j = 0; j < N; ++j) {
            const float v   = ((float)j + 0.5f) * invN; // = φ / (2π) ∈ (0,1)
            const float phi = 2.0f * M_PIf * v;
            const float cp  = cosf(phi);
            const float sp  = sinf(phi);

            // direzione in tangent space (z = cosθ)
            const vec3f local(sinT * cp, sinT * sp, z);
            const vec3f worldDir = T * local.x + B * local.y + n * local.z;

            // Shadow ray: payload[0] = 1 (occluso) di default; miss lo mette a 0
            unsigned int occluded = 1u;
            optixTrace(optixLaunchParams.traversable,
                       origin,
                       worldDir,
                       0.0f,        // tmin
                       1e16f,       // tmax (sky distance)
                       0.0f,        // ray time
                       OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT
                         | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                       0,           // SBT offset
                       1,           // SBT stride
                       0,           // missSBTIndex
                       occluded);   // payload_0 (in/out)

            if (occluded == 0u) {
                const vec3f L = sampleEnvmap(worldDir,
                                             optixLaunchParams.skybox.envmap,
                                             optixLaunchParams.skybox.skybox_size,
                                             optixLaunchParams.skybox.yaw_offset_u);
                // cosTerm = dot(worldDir, n) = local.z = z (per costruzione)
                accum = accum + L * z;
            }
        }
    }

    // Scaling Monte Carlo: (2π / |S|), con |S| = N*N
    const float scale = (2.0f * M_PIf) * invN * invN;
    optixLaunchParams.results.irradiance[idx] = accum * scale;
}

// ----------------------------------------------------------------------------
// Miss — ray non ha colpito nulla → cielo visibile
// ----------------------------------------------------------------------------
extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(0u);
}

// ----------------------------------------------------------------------------
// Closest hit — ray bloccato dalla geometria (con TERMINATE_ON_FIRST_HIT
// questo viene saltato; presente per completezza dell'SBT)
// ----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__shadow()
{
    optixSetPayload_0(1u);
}

} // namespace irr
