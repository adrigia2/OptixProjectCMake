
#pragma once
#include "gdt/math/vec.h"
#include "optix7.h"

using namespace gdt;

struct LaunchParams_DPN
{
	struct {
		bool computeDepth = false;      // when true, the raygen program computes the depth map
		bool computePositional = false; // when true, the raygen program computes the 3D positions
		bool computeNormal = false;     // when true, the raygen program computes the 3D normals
	} flags;
	vec2i size; // size of the frame to render (width, height)
	struct {
		float* depthBuffer;
		vec3f* positionalBuffer;
		vec3f* normalBuffer;
		uint8_t* maskBuffer; // per-pixel validity mask
	} results;
	struct {
		vec3f position;
		vec3f direction;
		vec3f horizontal;
		vec3f vertical;
	} camera;
	OptixTraversableHandle traversable;
};
