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

	struct IUMPayload {
		vec3f position;
		uint8_t mask;
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

		const auto& camera = optixLaunchParams.depth.camera;

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
			/ vec2f(optixLaunchParams.depth.frame.size));

		// generate ray direction
		vec3f rayDir = normalize(camera.direction
			+ (2.0f * screen.x - 1.0f) * camera.horizontal
			+ (1.0f - 2.0f * screen.y) * camera.vertical);


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
		const uint32_t fbIndex = ix + iy * optixLaunchParams.depth.frame.size.x;
		//optixLaunchParams.depth.frame.colorBuffer[fbIndex] = rgba;

		// Scrivi anche la depth se il buffer è disponibile
		if (optixLaunchParams.depth.frame.depthBuffer) {
			optixLaunchParams.depth.frame.depthBuffer[fbIndex] = prd.depth;
			//optixLaunchParams.depth.frame.depthBuffer[fbIndex] = (ix + iy) / float(optixLaunchParams.depth.frame.size.x + optixLaunchParams.depth.frame.size.y); 
		}
	}

	//------------------------------------------------------------------------------
	// IUM (Inverse UV Mapping) Ray Generation Program
	// Ogni pixel (ix, iy) rappresenta una coordinata UV nella texture
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderIUM()
	{
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		const auto size = optixLaunchParams.ium.size;

		// Converti coordinate pixel in coordinate UV normalizzate [0,1]
		const vec2f pixelUV(
			(float(ix) + 0.5f) / float(size.width),
			(float(iy) + 0.5f) / float(size.height)
		);

		// Inizializza il payload
		IUMPayload prd;
		prd.position = vec3f(0.f);
		prd.mask = 0;

		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		// Lancia un ray nella GAS in UV space
		// La GAS è costruita con vertici (u, v, 0), quindi lanciamo un ray
		// che parte da (pixelUV.x, pixelUV.y, 1) e va verso -Z
		vec3f rayOrigin = vec3f(pixelUV.x, pixelUV.y, 1.0f);
		vec3f rayDirection = vec3f(0.f, 0.f, -1.f);

		optixTrace(optixLaunchParams.traversable,
			rayOrigin,
			rayDirection,
			0.f,              // tmin
			1e20f,            // tmax
			0.0f,             // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,  // Vogliamo il primo hit
			SURFACE_RAY_TYPE,     // SBT offset
			RAY_TYPE_COUNT,       // SBT stride
			SURFACE_RAY_TYPE,     // missSBTIndex 
			u0, u1);

		// Scrivi risultato nei buffer di output
		const uint32_t index = ix + iy * size.width;
		optixLaunchParams.ium.positions[index] = prd.position;
		optixLaunchParams.ium.masks[index] = prd.mask;
	}

	//------------------------------------------------------------------------------
	// IUM Miss Program
	//------------------------------------------------------------------------------
	extern "C" __global__ void __miss__renderIUM()
	{
		IUMPayload& prd = *(IUMPayload*)getPRD<IUMPayload>();
		// Questo pixel UV non corrisponde a nessun triangolo
		prd.position = vec3f(0.f);
		prd.mask = 0;
	}

	//------------------------------------------------------------------------------
	// IUM Closest Hit Program
	// Qui mappiamo da UV space a World space
	//------------------------------------------------------------------------------
	extern "C" __global__ void __closesthit__renderIUM()
	{
		const int primID = optixGetPrimitiveIndex();
		IUMPayload& prd = *(IUMPayload*)getPRD<IUMPayload>();

		// Ottieni coordinate baricentriche dell'hit point IN UV SPACE
		const float2 barycentrics = optixGetTriangleBarycentrics();
		const float u = barycentrics.x;
		const float v = barycentrics.y;
		const float w = 1.0f - u - v;

		// Ottieni gli indici dei vertici del triangolo colpito
		const vec3i index = optixLaunchParams.ium.indices[primID];

		// Ottieni le posizioni 3D REALI dei tre vertici del triangolo
		const vec3f worldPos0 = optixLaunchParams.ium.worldVertices[index.x];
		const vec3f worldPos1 = optixLaunchParams.ium.worldVertices[index.y];
		const vec3f worldPos2 = optixLaunchParams.ium.worldVertices[index.z];

		// Interpola la posizione 3D usando le stesse coordinate baricentriche
		// che abbiamo ottenuto dal hit in UV space
		const vec3f worldPosition = w * worldPos0 + u * worldPos1 + v * worldPos2;

		// Salva il risultato
		prd.position = worldPosition;
		prd.mask = 1;  // Hit valido!
	}

	extern "C" __global__ void __anyhit__renderIUM()
	{
		// Lascia vuoto per ora - processiamo solo il primo hit
	}

} // ::osc