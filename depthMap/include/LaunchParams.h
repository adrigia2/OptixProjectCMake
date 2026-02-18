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

#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
	using namespace gdt;
	struct LaunchParams
	{

		struct depth
		{
			struct {
				float* depthBuffer;
				vec2i     size;
			} frame;

			struct {
				vec3f position;
				vec3f direction;
				vec3f horizontal;
				vec3f vertical;
			} camera;

		} depth;

		struct ium
		{
			// Output buffers
			vec3f* positions;
			uint8_t* masks;
			
			struct Size {
				uint32_t width;
				uint32_t height;
			} size;

			// Input geometry data per inverse UV mapping
			vec3f* worldVertices;  // Posizioni 3D reali dei vertici
			vec2f* uvVertices;     // Coordinate UV dei vertici
			vec3i* indices;        // Indici dei triangoli
			uint32_t numTriangles; // Numero di triangoli
		} ium;

		OptixTraversableHandle traversable;


	};


} // ::osc
