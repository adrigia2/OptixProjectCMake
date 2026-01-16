// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>

#include "LaunchParams.h"

using namespace osc;

namespace osc {


	/*! launch parameters in constant memory, filled in by optix upon
		optixLaunch (this gets filled in from the buffer we pass to
		optixLaunch) */
	extern "C" __constant__ LaunchParams optixLaunchParams;

	// for this simple example, we have a single ray type
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

	// Struttura per payload che include colore e depth
	struct RayPayload {
		vec3f color;
		float depth;
	};

	//------------------------------------------------------------------------------
	// closest hit and anyhit programs for radiance-type rays.
	//
	// Note eventually we will have to create one pair of those for each
	// ray type and each geometry type we want to render; but this
	// simple example doesn't use any actual geometries yet, so we only
	// create a single, dummy, set of them (we do have to have at least
	// one group of them to set up the SBT)
	//------------------------------------------------------------------------------

	extern "C" __global__ void __closesthit__radiance()
	{
		const int   primID = optixGetPrimitiveIndex();
		RayPayload& prd = *(RayPayload*)getPRD<RayPayload>();

		const vec3f O3 = optixGetWorldRayOrigin();
		const vec3f D3 = optixGetWorldRayDirection();
		const float  t = optixGetRayTmax();

		vec3f P = O3 + t * D3;

		// Calcola la distanza (depth)
		prd.depth = t;

		// ---- position -> color mapping ----
	// worldScale controlla quanto "spazio" comprimi in [0,1]
		const float worldScale = 1.0f;     // prova 1, 5, 10, 50 a seconda della scena
		vec3f c = (P / worldScale) * 0.5f + vec3f(0.5f);   // [-worldScale, +worldScale] -> [0,1]

		// clamp in [0,1]
		c.x = fminf(1.f, fmaxf(0.f, c.x));
		c.y = fminf(1.f, fmaxf(0.f, c.y));
		c.z = fminf(1.f, fmaxf(0.f, c.z));

		prd.color = c;
	}

	extern "C" __global__ void __anyhit__radiance()
	{ /*! for this simple example, this will remain empty */
	}



	//------------------------------------------------------------------------------
	// miss program that gets called for any ray that did not have a
	// valid intersection
	//
	// as with the anyhit/closest hit programs, in this example we only
	// need to have _some_ dummy function to set up a valid SBT
	// ------------------------------------------------------------------------------

	extern "C" __global__ void __miss__radiance()
	{
		RayPayload& prd = *(RayPayload*)getPRD<RayPayload>();
		// set to constant white as background color
		prd.color = vec3f(1.f);
		// Imposta depth a infinito per i miss
		prd.depth = 1e20f;
	}

	//------------------------------------------------------------------------------
	// ray gen program - the actual rendering happens in here
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderFrame()
	{
		// compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		const auto& camera = optixLaunchParams.camera;

		// our per-ray data for this example. what we initialize it to
		// won't matter, since this value will be overwritten by either
		// the miss or hit program, anyway
		RayPayload prd;
		prd.color = vec3f(0.f);
		prd.depth = 0.f;

		// the values we store the PRD pointer in:
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		// normalized screen plane position, in [0,1]^2
		const vec2f screen(vec2f(ix + .5f, iy + .5f)
			/ vec2f(optixLaunchParams.frame.size));

		// generate ray direction
		vec3f rayDir = normalize(camera.direction
			+ (screen.x - 0.5f) * camera.horizontal
			+ (screen.y - 0.5f) * -camera.vertical);

		optixTrace(optixLaunchParams.traversable,
			camera.position,
			rayDir,
			0.f,    // tmin
			1e20f,  // tmax
			0.0f,   // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
			SURFACE_RAY_TYPE,             // SBT offset
			RAY_TYPE_COUNT,               // SBT stride
			SURFACE_RAY_TYPE,             // missSBTIndex 
			u0, u1);

		const int r = int(255.99f * prd.color.x);
		const int g = int(255.99f * prd.color.y);
		const int b = int(255.99f * prd.color.z);

		// convert to 32-bit rgba value (we explicitly set alpha to 0xff
		// to make stb_image_write happy ...
		const uint32_t rgba = 0xff000000
			| (r << 0) | (g << 8) | (b << 16);

		// and write to frame buffer ...
		const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
		optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;

		// Scrivi anche la depth se il buffer è disponibile
		if (optixLaunchParams.frame.depthBuffer) {
			optixLaunchParams.frame.depthBuffer[fbIndex] = prd.depth;
			//optixLaunchParams.frame.depthBuffer[fbIndex] = (ix + iy) / float(optixLaunchParams.frame.size.x + optixLaunchParams.frame.size.y); 
		}
	}

} // ::osc