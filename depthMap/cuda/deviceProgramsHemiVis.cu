#include <optix_device.h>
#include "LaunchParams_HemiVis.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

using namespace osc;

namespace hemivis {

extern "C" __constant__ LaunchParams_HemiVis optixLaunchParams;

// Costanti della sequenza di Fibonacci, in doppia precisione.
// Sono replicate BIT PER BIT lato Python (_hemi_directions in images_generator.py):
// qualunque divergenza appaierebbe ogni t_hit alla direzione sbagliata, senza
// alcun sintomo visibile se non una L_j sbagliata. Vedi test_hemivis_shared.py.
#define HEMIVIS_INV_GOLDEN 0.6180339887498948482   // 1/φ = (√5 − 1)/2
#define HEMIVIS_TWO_PI     6.283185307179586477

// Frisvad 2012 branchless ONB (identico a deviceProgramsSpecCone.cu)
static __forceinline__ __device__
void buildONB(const vec3f& n, vec3f& T, vec3f& B)
{
    const float sgn = copysignf(1.0f, n.z);
    const float a   = -1.0f / (sgn + n.z);
    const float b   = n.x * n.y * a;
    T = vec3f(1.0f + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    B = vec3f(b, sgn + n.y * n.y * a, -n.y);
}

// Hash intero (lowbias32) → rotazione azimutale in [0, 1) per texel.
// Decorrela il pattern QMC tra texel vicini: senza, l'insieme di direzioni è lo
// stesso a meno della ONB e il rumore si allinea in bande visibili sull'atlante.
static __forceinline__ __device__
double rotationFromIndex(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (double)(x >> 8) * (1.0 / 16777216.0);
}

// Direzione condivisa s-esima attorno alla normale n (uniforme in angolo solido
// sull'emisfero: cosθ equispaziato, azimut sulla sequenza aurea).
// L'aritmetica di φ è in doppia precisione e ridotta in [0, 2π) PRIMA della
// trigonometria: in float32 s·goldenAngle arriva a ~4·10⁴ rad, dove un solo ULP
// vale già 0.004 rad (0.23°), abbastanza da spostare un campione di anello.
static __forceinline__ __device__
vec3f sharedDirection(int s, int numSamples, double rot,
                      const vec3f& n, const vec3f& T, const vec3f& B)
{
    const float cosT = (float)(1.0 - ((double)s + 0.5) / (double)numSamples);
    const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));

    double x = (double)s * HEMIVIS_INV_GOLDEN + rot;
    x -= floor(x);
    const float phi = (float)(x * HEMIVIS_TWO_PI);

    return T * (sinT * cosf(phi)) + B * (sinT * sinf(phi)) + n * cosT;
}

// Traccia un raggio di sola visibilità e restituisce la distanza di hit
// (0 = miss, > 0 = hit).
static __forceinline__ __device__
float traceVisibility(const vec3f& origin, const vec3f& dir)
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

    if (occluded == 0u) return 0.0f;
    // 0 è riservato al miss: un hit degenere a distanza nulla verrebbe letto come
    // cielo lato Python, quindi lo si tiene strettamente positivo.
    return fmaxf(__uint_as_float(t_hit_bits), 1e-8f);
}

// Legge posizione e normale normalizzata del texel; false se il texel non è
// utilizzabile (fuori maschera oppure normale degenere).
static __forceinline__ __device__
bool loadTexel(int global_idx, vec3f& pos, vec3f& n)
{
    if (optixLaunchParams.ium_data.ium_masks[global_idx] == 0) return false;

    pos = optixLaunchParams.ium_data.ium_positions[global_idx];

    n = optixLaunchParams.ium_data.ium_normals[global_idx];
    const float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len < 1e-8f) return false;
    n = n * (1.0f / len);
    return true;
}

// ----------------------------------------------------------------------------
// Raygen — un thread per raggio.
//   MODE_SHARED: lancio (tile_size, num_samples), raggi condivisi tra tutte le camere
//   MODE_MIRROR: lancio (tile_size, num_cams),    raggio specchio della camera j
// Ogni thread scrive sempre il proprio slot (anche i casi degeneri, con un valore
// negativo), così i buffer di output non vanno azzerati tra un tile e l'altro.
// ----------------------------------------------------------------------------
extern "C" __global__ void __raygen__hemiVis()
{
    const int local_idx = optixGetLaunchIndex().x;
    const int lane      = optixGetLaunchIndex().y;   // campione oppure camera

    if (local_idx >= optixLaunchParams.tile_size) return;

    const bool shared = (optixLaunchParams.mode == LaunchParams_HemiVis::MODE_SHARED);
    float* out = shared
        ? optixLaunchParams.t_hit_shared + (size_t)local_idx * optixLaunchParams.num_samples + lane
        : optixLaunchParams.t_hit_mirror + (size_t)local_idx * optixLaunchParams.num_cams  + lane;

    const int lanes = shared ? optixLaunchParams.num_samples
                             : optixLaunchParams.num_cams;
    vec3f* dbg = optixLaunchParams.dbg_dirs
        ? optixLaunchParams.dbg_dirs + (size_t)local_idx * lanes + lane
        : nullptr;
    if (dbg) *dbg = vec3f(0.0f, 0.0f, 0.0f);

    const int global_idx = optixLaunchParams.tile_offset + local_idx;
    if (global_idx >= optixLaunchParams.ium_data.num_pixels) { *out = -1.0f; return; }

    vec3f pos, n;
    if (!loadTexel(global_idx, pos, n)) { *out = -1.0f; return; }

    const vec3f origin = pos + n * optixLaunchParams.epsilon;

    if (shared) {
        vec3f T, B;
        buildONB(n, T, B);
        const double rot = rotationFromIndex((unsigned int)global_idx);
        const vec3f dir = sharedDirection(lane, optixLaunchParams.num_samples,
                                          rot, n, T, B);
        if (dbg) *dbg = dir;
        *out = traceVisibility(origin, dir);
    } else {
        vec3f v = optixLaunchParams.cam_pos[lane] - pos;
        const float vLen = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        if (vLen < 1e-8f) { *out = -1.0f; return; }
        v = v * (1.0f / vLen);

        const float nv = n.x * v.x + n.y * v.y + n.z * v.z;
        if (nv <= 0.0f) { *out = -1.0f; return; }   // camera dietro la superficie

        const vec3f R = n * (2.0f * nv) - v;
        if (dbg) *dbg = R;
        *out = traceVisibility(origin, R);
    }
}

// ----------------------------------------------------------------------------
// Miss — cielo visibile → occluded = 0
// ----------------------------------------------------------------------------
extern "C" __global__ void __miss__hemiVis()
{
    optixSetPayload_0(0u);
}

// ----------------------------------------------------------------------------
// Closest hit — raggio bloccato dalla geometria: occluded = 1, t_hit = tmax
// ----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__hemiVis()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

} // namespace hemivis
