// ======================================================================== //
// Camera Structure - Shared camera definition                             //
// ======================================================================== //

#pragma once

#include "gdt/math/vec.h"

namespace osc {

	using namespace gdt;

	/*! Camera structure for defining viewpoint and orientation */
	struct Camera {
		Camera(const vec3f pos, const vec3f forward, const vec3f up)
			: pos(pos), forward(forward), up(up) {
		}

	private:
		/*! camera position - *from* where we are looking */
		vec3f pos;
		/*! which direction we are looking (forward vector) */
		vec3f forward;
		/*! general up-vector */
		vec3f up;

	public:
		vec3f getPos() const { return pos; }
		vec3f getForward() const { return forward; }
		vec3f getUp() const { return up; }

		void setPos(const vec3f newPos) { pos = newPos; }
		void setForward(const vec3f newForward) { forward = newForward; }
		void setUp(const vec3f newUp) { up = newUp; }
	};

} // namespace osc
