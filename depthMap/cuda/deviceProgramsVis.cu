#include <optix_device.h>
#include "LaunchParams_Vis.h"

using namespace osc;

namespace vis {
    extern "C" __constant__ LaunchParams_Vis optixLaunchParams;

    // Ray payloads
    struct VisibilityPRD {
        bool visible;
    };

    static __forceinline__ __device__
    void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
    void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    //------------------------------------------------------------------------------
    // Raygen
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        // 2D grid: x = pixel index, y = camera index
        const int pixel_idx = optixGetLaunchIndex().x;
        const int camera_idx = optixGetLaunchIndex().y;

        if (pixel_idx >= optixLaunchParams.ium_data.num_pixels || 
            camera_idx >= optixLaunchParams.camera_data.num_cameras) {
            return;
        }

        const int out_idx = pixel_idx * optixLaunchParams.camera_data.num_cameras + camera_idx;

        // When IUM decided the pixel does not sit on the mesh, it is always invisible
        if (optixLaunchParams.ium_data.ium_masks[pixel_idx] == 0) {
            optixLaunchParams.results.visibility_results[out_idx] = 0;
            return;
        }

        vec3f target_pos = optixLaunchParams.ium_data.ium_positions[pixel_idx];
        CameraDef cam = optixLaunchParams.camera_data.cameras[camera_idx];

        // Ray from the camera towards the target point
        vec3f ray_dir = target_pos - cam.position;
        float distance = length(ray_dir);
        
        if(distance > 0.0f) {
            ray_dir = ray_dir / distance;
        }

        // TODO: Optional frustom cull checking here before tracing

        VisibilityPRD prd;
        prd.visible = true; // assume visible until something is hit

        uint32_t u0, u1;
        packPointer(&prd, u0, u1);

        // Trace the shadow ray from the camera towards the target point.
        // tmax is kept slightly below the real distance (-1e-3f) to avoid
        // self-intersecting the target itself on the surface.
        optixTrace(optixLaunchParams.traversable,
            cam.position,
            ray_dir,
            0.0f,               // tmin
            distance - 1e-3f,   // tmax
            0.0f,               // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, // shadow ray flags
            0,             // SBT offset
            1,             // SBT stride
            0,             // missSBTIndex
            u0, u1);

        // Write the result (1 when visible and nothing intersects in between, 0 otherwise)
        optixLaunchParams.results.visibility_results[out_idx] = prd.visible ? 1 : 0;
    }

    //------------------------------------------------------------------------------
    // Miss
    //------------------------------------------------------------------------------
    extern "C" __global__ void __miss__radiance()
    {
        VisibilityPRD* prd = getPRD<VisibilityPRD>();
        // Miss = no geometry between camera and target point -> Visible
        prd->visible = true;
    }

    //------------------------------------------------------------------------------
    // Hitgroup
    //------------------------------------------------------------------------------
    extern "C" __global__ void __closesthit__radiance()
    {
        VisibilityPRD* prd = getPRD<VisibilityPRD>();
        // Hit = geometry obstructs the view -> Occluded
        prd->visible = false;
    }
}
