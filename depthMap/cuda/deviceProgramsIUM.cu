#include <optix_device.h>

#include "LaunchParams_IUM.h"

using namespace osc;

	/*! launch parameters in constant memory, filled in by optix upon
		optixLaunch (this gets filled in from the buffer we pass to
		optixLaunch) */
	extern "C" __constant__ LaunchParams_IUM optixLaunchParams;

	// a single ray type is enough for this pass
	enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

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

	struct IUMPayload {
		vec3f position;
		vec3f normal;
		uint8_t mask;
	};


	//------------------------------------------------------------------------------
	// IUM (Inverse UV Mapping) Ray Generation Program
	// Every pixel (ix, iy) stands for a UV coordinate in the texture
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderIUM()
	{
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		const auto size = optixLaunchParams.size;

		// Convert pixel coordinates into normalised UV coordinates in [0,1].
		// The V axis is flipped with respect to the row index (iy=0 = top row of the
		// image = V=1) so that the UV origin (0,0) sits bottom left, which is the
		// Blender/OBJ/OpenGL convention.
		const vec2f pixelUV(
			(float(ix) + 0.5f) / float(size.x),
			1.0f - (float(iy) + 0.5f) / float(size.y)
		);

		// Initialise the payload
		IUMPayload prd;
		prd.position = vec3f(0.f);
		prd.normal = vec3f(0.f);
		prd.mask = 0;

		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		// Trace a ray through the GAS built in UV space.
		// That GAS is built from vertices (u, v, 0), so the ray starts at
		// (pixelUV.x, pixelUV.y, 1) and travels towards -Z.
		vec3f rayOrigin = vec3f(pixelUV.x, pixelUV.y, 1.0f);
		vec3f rayDirection = vec3f(0.f, 0.f, -1.f);

		optixTrace(optixLaunchParams.traversable,
			rayOrigin,
			rayDirection,
			0.f,              // tmin
			1e20f,            // tmax
			0.0f,             // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,  // we want the first hit
			SURFACE_RAY_TYPE,     // SBT offset
			RAY_TYPE_COUNT,       // SBT stride
			SURFACE_RAY_TYPE,     // missSBTIndex 
			u0, u1);

		// Write the result into the output buffers
		const uint32_t index = ix + iy * size.x;
		optixLaunchParams.results.positions[index] = prd.position;
		optixLaunchParams.results.normals[index] = prd.normal;
		optixLaunchParams.results.masks[index] = prd.mask;
	}

	//------------------------------------------------------------------------------
	// IUM Miss Program
	//------------------------------------------------------------------------------
	extern "C" __global__ void __miss__renderIUM()
	{
		IUMPayload& prd = *(IUMPayload*)getPRD<IUMPayload>();
		// This UV pixel maps to no triangle
		prd.position = vec3f(0.f);
		prd.normal = vec3f(0.f);
		prd.mask = 0;
	}

	//------------------------------------------------------------------------------
	// IUM Closest Hit Program
	// This is where UV space is mapped back to world space
	//------------------------------------------------------------------------------
	extern "C" __global__ void __closesthit__renderIUM()
	{
		const int primID = optixGetPrimitiveIndex();
		IUMPayload& prd = *(IUMPayload*)getPRD<IUMPayload>();

		// Barycentric coordinates of the hit point IN UV SPACE
		const float2 barycentrics = optixGetTriangleBarycentrics();
		const float u = barycentrics.x;
		const float v = barycentrics.y;
		const float w = 1.0f - u - v;

		// Vertex indices of the triangle that was hit
		const vec3i index = optixLaunchParams.data.indices[primID];

		// The REAL 3D positions of the triangle's three vertices
		const vec3f worldPos0 = optixLaunchParams.data.worldVertices[index.x];
		const vec3f worldPos1 = optixLaunchParams.data.worldVertices[index.y];
		const vec3f worldPos2 = optixLaunchParams.data.worldVertices[index.z];

		// Interpolate the 3D position with the same barycentric coordinates the hit
		// in UV space produced
		const vec3f worldPosition = w * worldPos0 + u * worldPos1 + v * worldPos2;

		// Face normal: cross product of the triangle's edges in world space.
		// Outward-facing when the OBJ winding is CCW (the tinyobj standard).
		const vec3f e1 = worldPos1 - worldPos0;
		const vec3f e2 = worldPos2 - worldPos0;
		const vec3f faceNormal = normalize(cross(e1, e2));

		// Store the result
		prd.position = worldPosition;
		prd.normal = faceNormal;
		prd.mask = 1;  // valid hit
	}

	extern "C" __global__ void __anyhit__renderIUM()
	{
		// Intentionally empty: only the first hit matters here
	}

