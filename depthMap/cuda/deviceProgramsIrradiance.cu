#include <optix_device.h>
#include "LaunchParams_Irradiance.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

// Angolo aureo espresso come frazione di giro: PI*(3-sqrt(5)) / (2*PI) = (3-sqrt(5))/2.
// E' il complemento di HEMIVIS_INV_GOLDEN (1/phi) usato in deviceProgramsHemiVis.cu:
// stessa sequenza a bassa discrepanza, verso azimutale opposto. Qui si tiene questo
// verso per non alterare il pattern gia' prodotto dal kernel.
#define IRR_GOLDEN_TURN 0.3819660112501051518   // (3 - sqrt(5))/2
#define IRR_TWO_PI      6.283185307179586477

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
// Equirectangular envmap lookup — world space is Z-up, Y-forward (Blender native).
// Azimuth: -atan2(dy, dx) → +X direction at u=0.5 matches Blender equirectangular convention.
// Elevation: asinf(dz)    → +Z is zenith (v=0), -Z is nadir (v=1).
// ----------------------------------------------------------------------------
static __forceinline__ __device__
vec3f sampleEnvmap(const vec3f& d, const vec3f* env, vec2i sz, float yaw_offset_u)
{
    const float dx = d.x;
    const float dy = d.y;
    const float dz = fmaxf(-1.0f, fminf(1.0f, d.z));   // elevation axis (Z-up)

    float u = 0.5f - atan2f(dy, dx) * (1.0f / (2.0f * M_PIf));  // azimuth: Blender equirectangular convention
    u += yaw_offset_u;
    u -= floorf(u);                       // wrap a [0, 1)
    const float v = 0.5f - asinf(dz) * (1.0f / M_PIf);  // Z is up

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
        optixLaunchParams.results.irradiance[idx] = vec3f(0.f);
        return;
    }
    n = n * (1.0f / nLen);

    vec3f T, B;
    buildONB(n, T, B);

    // Manteniamo il numero totale di campioni uguale a prima per coerenza (N * N)
    const int N = optixLaunchParams.sample_side;
    const int numSamples = N * N;
    const float invNumSamples = 1.0f / (float)numSamples;
    const float eps = optixLaunchParams.epsilon;
    const vec3f origin = pos + n * eps;

    vec3f accum(0.f);

    // Un singolo loop basato sul numero totale di campioni
    for (int i = 0; i < numSamples; ++i) {

        // z va da ~1 (zenit) fino a ~0 (orizzonte) per coprire l'emisfero positivo
        const float z = 1.0f - ((float)i + 0.5f) * invNumSamples;

        // raggio nel piano XY locale
        const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));

        // Azimut sulla sequenza aurea: aritmetica in doppia precisione, ridotta in
        // [0,1) PRIMA della trigonometria. Calcolato come (float)i * goldenAngle,
        // a sample_side=512 (S=262144) phi arriva a 6.3e5 rad, dove un solo ULP
        // float32 vale 0.0625 rad (3.58 gradi): la sequenza perde la bassa
        // discrepanza (gap azimutale massimo 5.0x l'ideale contro 1.56x in doppia).
        // L'errore e' invisibile ai default (0.014 gradi a S=256) e cresce con S,
        // quindi alzare i campioni ne restituiva indietro una parte.
        // Stesso accorgimento di sharedDirection() in deviceProgramsHemiVis.cu.
        double xphi = (double)i * IRR_GOLDEN_TURN;
        xphi -= floor(xphi);
        const float phi = (float)(xphi * IRR_TWO_PI);

        // Direzione nello spazio tangente
        const vec3f local(r * cosf(phi), r * sinf(phi), z);

        // Trasformazione nello spazio mondo tramite l'ONB
        const vec3f worldDir = T * local.x + B * local.y + n * local.z;

        unsigned int occluded = 1u;
        optixTrace(optixLaunchParams.traversable,
                   origin,
                   worldDir,
                   0.0f,
                   1e16f,
                   0.0f,
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   0,
                   1,
                   0,
                   occluded);

        if (occluded == 0u) {
            const vec3f L = sampleEnvmap(worldDir,
                                         optixLaunchParams.skybox.envmap,
                                         optixLaunchParams.skybox.skybox_size,
                                         optixLaunchParams.skybox.yaw_offset_u);

            // z è esattamente il coseno dell'angolo tra la normale e il raggio
            accum = accum + L * z;
        }
    }

    // Scaling finale esattamente identico al paper: 2 * PI / |S_L|
    const float scale = (2.0f * M_PIf) * invNumSamples;
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
