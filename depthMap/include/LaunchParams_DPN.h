
#pragma once
#include "gdt/math/vec.h"
#include "optix7.h"

using namespace gdt;

struct LaunchParams_DPN
{
	struct {
		bool computeDepth;
		bool computePositional;
		bool computeNormal;
	} flags;
	vec2i size; // dimensione del frame da creare (width, height)
	struct {
		float* depthBuffer;
		vec3f* positionalBuffer;
		vec3f* normalBuffer;
	} results;
	struct {
		vec3f position;
		vec3f direction;
		vec3f horizontal;
		vec3f vertical;
	} camera;
	OptixTraversableHandle traversable;
};
